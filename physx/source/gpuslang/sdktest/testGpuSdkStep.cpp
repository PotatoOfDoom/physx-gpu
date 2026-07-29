// RUNTIME_PLAN.md P5 milestone B: the GPU-pipeline integration test.  Builds a
// real PxScene from the unmodified PhysX CPU SDK, but wires OUR Vulkan bridge in
// as the cudaContextManager and turns on the GPU pipeline (eENABLE_GPU_DYNAMICS +
// GPU broadphase).  The scene then constructs the whole Pxg* subsystem stack
// (broadphase / AABB manager / narrowphase / dynamics / simulation controller)
// through the bridge factories in PhysXGpuVulkan.cpp, over the Vulkan backend.
//
// This is the first point where the engine drives the bridge as a real GPU module
// rather than the standalone construction seams the gpuslang suite exercises.  The
// milestone is a constructed GPU scene; stepping it is attempted and reported, as
// the per-kernel launch marshalling is the remaining frontier (RUNTIME_PLAN.md P5).

#include "PxPhysicsAPI.h"       // PxCreateCudaContextManager (public C-ABI, resolves to the bridge)

// Runtime convex cooking: the immediate-mode entry points live in
// geomutils/src/cooking (GuCookingConvexMesh.cpp:181 createConvexMesh,
// GuMeshFactory.cpp:1259 getInsertionCallback) and are compiled into the
// physx-sdk static lib by the GLOB_RECURSE over geomutils/src/*.cpp — the
// physxcooking/src/Cooking.cpp C-ABI wrappers are NOT in the build, so we
// call physx::immediateCooking:: directly via GuCooking.h (on the include
// path through physx-sdk's PUBLIC ${SRC}/geomutils/include).
#include "GuCooking.h"
#include "cooking/PxCooking.h"
#include "extensions/PxD6Joint.h"   // PxD6JointCreate + PxD6Axis/PxD6Motion (jointed scene)
#include "cudamanager/PxCudaContext.h" // PxCudaContext::memcpyHtoD/DtoH (particle buffer fill/read)

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace physx;

namespace physx { class PxPhysXGpu; }

// The bridge's statically-linked C-ABI factory (PhysXGpuVulkan.cpp).  Declared here
// rather than via the internal PxPhysXGpu.h, whose PxCreateCudaContextManager default
// arguments would clash with the public PxGpu.h declaration.
extern "C" physx::PxPhysXGpu* PxCreatePhysXGpu();

namespace physx
{
// Defined in PxPhysXGpuModuleLoader.cpp (compiled into physx-sdk).  In a non-static
// config PxvGetPhysXGpu() populates this by dlopen'ing libPhysXGpu_64.so via libcuda;
// we have neither, so we point it straight at the bridge's statically-linked factory.
// PxvGetPhysXGpu still calls PxLoadPhysxGPUModule first, but on Linux that fails to
// load libcuda and returns WITHOUT clearing this pointer, so our value survives.
typedef physx::PxPhysXGpu* (PxCreatePhysXGpu_FUNC)();
extern PxCreatePhysXGpu_FUNC* g_PxCreatePhysXGpu_Func;
}

namespace
{
class StepAllocator : public PxAllocatorCallback
{
public:
    void* allocate(size_t size, const char*, const char*, int) override
    {
        void* p = nullptr;
        if (posix_memalign(&p, 16, size ? size : 1))
            return nullptr;
        return p;
    }
    void deallocate(void* p) override { ::free(p); }
};

class StepError : public PxErrorCallback
{
public:
    void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) override
    {
        std::printf("  [PhysX %d] %s (%s:%d)\n", int(code), message, file, line);
    }
};

class SyncDispatcher : public PxCpuDispatcher
{
public:
    void submitTask(PxBaseTask& task) override { task.run(); task.release(); }
    PxU32 getWorkerCount() const override { return 1; }
};

PxFilterFlags stepFilterShader(PxFilterObjectAttributes, PxFilterData, PxFilterObjectAttributes, PxFilterData,
                               PxPairFlags& pairFlags, const void*, PxU32)
{
    pairFlags = PxPairFlag::eCONTACT_DEFAULT;
    return PxFilterFlag::eDEFAULT;
}

// A dynamic body expected to come to rest at a known centre height on the GPU.
struct RestTarget
{
    PxRigidDynamic* body;
    float startY;   // spawn centre height
    float restY;    // expected resting centre height
    const char* label;
};

// Solid-box mass-space inertia (about the centre): I_axis = m/3 * (sum of the two
// perpendicular half-extents squared).  For half 0.5 / m 1000 this is 166.67 per
// axis, matching the single-box scene.
PxVec3 boxInertia(const PxVec3& h, float mass)
{
    const float k = mass / 3.0f;
    return PxVec3(k * (h.y * h.y + h.z * h.z),
                  k * (h.x * h.x + h.z * h.z),
                  k * (h.x * h.x + h.y * h.y));
}

PxRigidStatic* addGroundBox(PxPhysics* physics, PxScene* scene, PxMaterial* material)
{
    PxRigidStatic* ground = physics->createRigidStatic(PxTransform(PxVec3(0.0f, -0.5f, 0.0f)));
    PxShape* gs = physics->createShape(PxBoxGeometry(50.0f, 0.5f, 50.0f), *material);
    ground->attachShape(*gs);
    scene->addActor(*ground);
    return ground;
}

PxRigidDynamic* addDynamic(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                           const PxGeometry& geom, const PxVec3& inertia, float y)
{
    PxRigidDynamic* b = physics->createRigidDynamic(PxTransform(PxVec3(0.0f, y, 0.0f)));
    PxShape* s = physics->createShape(geom, *material);
    b->attachShape(*s);
    b->setMass(1000.0f);
    b->setMassSpaceInertiaTensor(inertia);
    scene->addActor(*b);
    return b;
}

// PXG_SCENE=box (default): one dynamic box (half 0.5) dropped from y=5 onto a static
// ground box; rests at centre y=0.5.  The original single-body scene.
void buildBoxScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                   std::vector<RestTarget>& targets)
{
    addGroundBox(physics, scene, material);
    PxRigidDynamic* b = addDynamic(physics, scene, material, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                   boxInertia(PxVec3(0.5f), 1000.0f), 5.0f);
    targets.push_back({ b, 5.0f, 0.5f, "box" });
}

// PXG_SCENE=stack: a static ground box + three dynamic boxes spawned with gaps so
// they settle into a stack.  Exercises dynamic-vs-dynamic box-box contacts and the
// TGS partition solve path (solveBlockUnified) — the multi-body island case.
void buildStackScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                     std::vector<RestTarget>& targets)
{
    addGroundBox(physics, scene, material);
    const char* labels[3] = { "stack0", "stack1", "stack2" };
    const float restY[3]  = { 0.5f, 1.5f, 2.5f };
    const float startY[3] = { 2.0f, 3.4f, 4.8f };
    for (int i = 0; i < 3; ++i)
    {
        PxRigidDynamic* b = addDynamic(physics, scene, material, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                       boxInertia(PxVec3(0.5f), 1000.0f), startY[i]);
        targets.push_back({ b, startY[i], restY[i], labels[i] });
    }
}

// PXG_SCENE=plane: one dynamic box resting on a static infinite PxPlaneGeometry
// (y=0).  Exercises the convex/box-plane narrowphase path.
void buildPlaneScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                     std::vector<RestTarget>& targets)
{
    PxRigidStatic* ground = physics->createRigidStatic(PxTransformFromPlaneEquation(PxPlane(0.0f, 1.0f, 0.0f, 0.0f)));
    PxShape* ps = physics->createShape(PxPlaneGeometry(), *material);
    ground->attachShape(*ps);
    scene->addActor(*ground);
    PxRigidDynamic* b = addDynamic(physics, scene, material, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                   boxInertia(PxVec3(0.5f), 1000.0f), 5.0f);
    targets.push_back({ b, 5.0f, 0.5f, "boxOnPlane" });
}

// PXG_SCENE=sphere: one dynamic sphere (radius 0.5) resting on the static ground box.
// Exercises the sphere narrowphase path (sphere-box sub-case).
void buildSphereScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                      std::vector<RestTarget>& targets)
{
    addGroundBox(physics, scene, material);
    const float r = 0.5f;
    const float sphereI = 0.4f * 1000.0f * r * r; // solid sphere: 2/5 m r^2
    PxRigidDynamic* b = addDynamic(physics, scene, material, PxSphereGeometry(r),
                                   PxVec3(sphereI, sphereI, sphereI), 5.0f);
    targets.push_back({ b, 5.0f, 0.5f, "sphere" });
}

// A rough capsule mass-space inertia (axis along local X, radius r, cylinder
// half-height h): I_axis ~= 1/2 m r^2, I_transverse ~= m (h^2 + r^2).  At rest
// (~0 angular velocity) the exact tensor is immaterial; it only needs to be
// positive-definite so the solver is well-conditioned.
PxVec3 capsuleInertia(float r, float h, float mass)
{
    const float axis = 0.5f * mass * r * r;
    const float trans = mass * (h * h + r * r);
    return PxVec3(axis, trans, trans);
}

// PXG_SCENE=capsule: one dynamic capsule (radius 0.5, half-height 0.5, axis along
// local X so it lies on its side) resting on the static infinite PxPlaneGeometry.
// Routes through the sphere/"primitive" narrowphase (planeCapsule sub-case), which
// is the already-mapped SPHERE_KERNEL_MAIN — no new narrowphase kernel needed.
void buildCapsuleScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                       std::vector<RestTarget>& targets)
{
    PxRigidStatic* ground = physics->createRigidStatic(PxTransformFromPlaneEquation(PxPlane(0.0f, 1.0f, 0.0f, 0.0f)));
    PxShape* ps = physics->createShape(PxPlaneGeometry(), *material);
    ground->attachShape(*ps);
    scene->addActor(*ground);
    const float r = 0.5f, h = 0.5f;
    PxRigidDynamic* b = addDynamic(physics, scene, material, PxCapsuleGeometry(r, h),
                                   capsuleInertia(r, h, 1000.0f), 5.0f);
    targets.push_back({ b, 5.0f, r, "capsuleOnPlane" }); // rests on its side at centre y = radius
}

// PXG_SCENE=capsulebox: the capsule resting on the static ground BOX — routes
// through the convex-convex GJK narrowphase (collide(): GJK/EPA + capsule
// face/EE contact gen), unlike PXG_SCENE=capsule which rides the primitive kernel.
void buildCapsuleBoxScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                          std::vector<RestTarget>& targets)
{
    addGroundBox(physics, scene, material);
    const float r = 0.5f, h = 0.5f;
    PxRigidDynamic* b = addDynamic(physics, scene, material, PxCapsuleGeometry(r, h),
                                   capsuleInertia(r, h, 1000.0f), 5.0f);
    targets.push_back({ b, 5.0f, r, "capsuleOnBox" });
}

// PXG_SCENE=convex: a dynamic CONVEX MESH (a runtime-cooked cube hull, half
// 0.5) resting on the static ground BOX.  Box (eBOX=3) vs convex mesh
// (eCONVEXMESH=5) lands in the GPU_BUCKET_ID::eConvex bucket
// (PxgNphaseImplementationContext.cpp cndCasePrimitiveConvex) — the same
// convexConvexNphase stage1/stage2/finish kernel chain as capsulebox, but
// stage2's collide() now dispatches the convex-hull generateConvexContacts
// arm (cudaGJKEPA.cu L1458-1462): polygon-vs-polygon clipping instead of the
// capsule face/EE path.  cooking: buildGPUData = true sets the GRB edge bit
// so ConvexMesh::isGpuCompatible() (GuConvexMesh.cpp:104) holds and the pair
// is NOT demoted to the CPU fallback bucket; insertion goes through THIS
// PxPhysics' insertion callback so the mesh registers with its factory.
void buildConvexScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                      std::vector<RestTarget>& targets)
{
    addGroundBox(physics, scene, material);

    static const PxVec3 cubeVerts[8] = {
        PxVec3(-0.5f, -0.5f, -0.5f), PxVec3(-0.5f, -0.5f, 0.5f),
        PxVec3(-0.5f, 0.5f, -0.5f),  PxVec3(-0.5f, 0.5f, 0.5f),
        PxVec3(0.5f, -0.5f, -0.5f),  PxVec3(0.5f, -0.5f, 0.5f),
        PxVec3(0.5f, 0.5f, -0.5f),   PxVec3(0.5f, 0.5f, 0.5f),
    };
    PxCookingParams cp(physics->getTolerancesScale());
    cp.buildGPUData = true;
    PxConvexMeshDesc desc;
    desc.points.count = 8;
    desc.points.stride = sizeof(PxVec3);
    desc.points.data = cubeVerts;
    desc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
    PxConvexMeshCookingResult::Enum cond = PxConvexMeshCookingResult::eSUCCESS;
    PxConvexMesh* mesh = immediateCooking::createConvexMesh(
        cp, desc, physics->getPhysicsInsertionCallback(), &cond);
    if (!mesh)
    {
        std::printf("FAIL: runtime convex cooking failed (result %d)\n", int(cond));
        return; // targets stays empty; main() hard-fails on that
    }
    std::printf("  cooked convex hull: %u verts, %u polygons (GPU-compatible: %s)\n",
                mesh->getNbVertices(), mesh->getNbPolygons(),
                mesh->isGpuCompatible() ? "yes" : "NO");

    PxRigidDynamic* b = addDynamic(physics, scene, material, PxConvexMeshGeometry(mesh),
                                   boxInertia(PxVec3(0.5f), 1000.0f), 5.0f);
    targets.push_back({ b, 5.0f, 0.5f, "convexOnBox" });
}

// PXG_SCENE=joint: a dynamic box suspended from a static WORLD anchor by a
// PxD6Joint with the three LINEAR DOFs locked (angular free).  The joint pins
// the box centre to the anchor point and holds it against gravity, so the box
// hangs at a fixed rest height instead of falling.  This is the FIRST scene that
// launches constraint1DPrePrepLaunch (the D6-joint data-prep kernel): a D6 joint
// is the only joint whose rows are prepped on the GPU (all other joint types are
// prepped on the CPU), so the JOINTED path must NOT refuse.  The joint path is
// solver-shared, so this holds under BOTH the TGS and PGS cores.
void buildJointScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                     std::vector<RestTarget>& targets)
{
    const float restY = 3.0f;          // world anchor / pinned box-centre height
    const float spawnY = restY + 0.5f; // spawn slightly above so it falls onto the pin

    // A free dynamic box (no ground); its resting pose is the joint, not a contact.
    PxRigidDynamic* box = physics->createRigidDynamic(PxTransform(PxVec3(0.0f, spawnY, 0.0f)));
    PxShape* s = physics->createShape(PxBoxGeometry(0.5f, 0.5f, 0.5f), *material);
    box->attachShape(*s);
    box->setMass(1000.0f);
    box->setMassSpaceInertiaTensor(boxInertia(PxVec3(0.5f), 1000.0f));
    scene->addActor(*box);

    // actor0 = NULL -> world-attached at localFrame0; the box centre (identity
    // localFrame1) is constrained to the world point (0, restY, 0).
    PxD6Joint* joint = PxD6JointCreate(*physics, NULL, PxTransform(PxVec3(0.0f, restY, 0.0f)),
                                       box, PxTransform(PxIdentity));
    joint->setMotion(PxD6Axis::eX, PxD6Motion::eLOCKED);
    joint->setMotion(PxD6Axis::eY, PxD6Motion::eLOCKED);
    joint->setMotion(PxD6Axis::eZ, PxD6Motion::eLOCKED);
    joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eFREE);
    joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eFREE);
    joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eFREE);

    targets.push_back({ box, spawnY, restY, "boxOnD6Joint" });
}

// PXG_SCENE=particle (P6.0): the first deformable/particle scene.  A PBD particle
// cloud (4x4x4 = 64 particles, invMass = 1) is spawned HIGH above everything with
// zero initial velocity and no ground, so it is a pure free fall with NO contacts.
// The engine launches the whole per-frame ps_* sequence, but only the four P6.0
// kernels are mapped (ps_updateUnsortedArrayLaunch ingest, ps_preIntegrateLaunch
// predict, ps_integrateLaunch writeback, ps_updateUserBufferLaunch egress); every
// grid / sort / collision / solve kernel refuses (non-fatal).  Because there is no
// contact, the sorted buffers stay zero, ps_integrateLaunch is an inert no-op
// (pos4.w == 0 guard), and the observable fall is produced entirely by the ingest
// -> predict -> egress chain on the UNSORTED arrays.
struct ParticleScene
{
    PxPBDParticleSystem* system = nullptr;
    PxParticleBuffer* buffer = nullptr;
    PxU32 numParticles = 0;
    PxCudaContextManager* cudaCtx = nullptr;
};

ParticleScene buildParticleScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                 PxCudaContextManager* cudaCtx)
{
    ParticleScene out;
    out.cudaCtx = cudaCtx;

    // A minimal rigid body (box on ground) FAR BELOW the cloud.  The engine drives
    // the rigid solver core for the particle island's iteration loop; with zero
    // dynamic rigid bodies the rigid batch-sum kernel (rigidSumInternalContactAndJoint
    // Batches2) dereferences null solver descriptors and faults, so the scene needs
    // one live rigid body to keep that proven path valid.  It rests at y=0.5, ~100
    // below the particle cloud, so there is never any particle/rigid interaction.
    addGroundBox(physics, scene, rigidMat);
    // The engine drives the rigid solver core for the particle island, so one live
    // dynamic rigid body is required (else rigidSumInternalContactAndJointBatches2
    // null-derefs).  Free-fall: box at x=0,y=5 (never meets the y~100 cloud).  Bag
    // mode (PXG_PARTICLE_STARTY set): place it FAR ASIDE at x=20 so the settling
    // pile on the floor never touches it.
    if (std::getenv("PXG_PARTICLE_STARTY"))
    {
        PxRigidDynamic* aside = physics->createRigidDynamic(PxTransform(PxVec3(20.0f, 0.5f, 0.0f)));
        PxShape* s = physics->createShape(PxBoxGeometry(0.5f, 0.5f, 0.5f), *rigidMat);
        aside->attachShape(*s);
        aside->setMass(1000.0f);
        aside->setMassSpaceInertiaTensor(boxInertia(PxVec3(0.5f), 1000.0f));
        s->release();
        scene->addActor(*aside);
    }
    else
    {
        addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                   boxInertia(PxVec3(0.5f), 1000.0f), 5.0f);
    }

    // PBD material + particle system + one phase mapping to that material.
    PxPBDMaterial* pbdMat = physics->createPBDMaterial(/*friction*/ 0.05f, /*damping*/ 0.0f,
        /*adhesion*/ 0.0f, /*viscosity*/ 0.0f, /*vorticityConfinement*/ 0.0f, /*surfaceTension*/ 0.0f,
        /*cohesion*/ 0.0f, /*lift*/ 0.0f, /*drag*/ 0.0f);

    PxPBDParticleSystem* ps = physics->createPBDParticleSystem(*cudaCtx);
    const float particleSpacing = 0.5f;
    const float restOffset = 0.5f * particleSpacing;
    ps->setRestOffset(restOffset);
    ps->setContactOffset(restOffset + 0.05f);
    ps->setParticleContactOffset(restOffset + 0.05f);
    ps->setSolidRestOffset(restOffset);
    ps->setMaxVelocity(1e6f); // do not clamp the free fall

    const PxU32 phase = ps->createPhase(pbdMat,
        PxParticlePhaseFlags(PxParticlePhaseFlag::eParticlePhaseSelfCollide));

    // 4x4x4 grid, invMass 1, zero velocity, spawned centred high above y=0.
    const int G = 4;
    const PxU32 N = PxU32(G * G * G);
    // startY defaults to 100 (the P6.0 free-fall baseline all tranches regress
    // against); PXG_PARTICLE_STARTY lowers the cloud so it can reach the floor
    // (P6.5 settling bag).  When lowered, raise solver iterations for a stable pile.
    const char* startYEnv = std::getenv("PXG_PARTICLE_STARTY");
    const float startY = startYEnv ? float(atof(startYEnv)) : 100.0f;
    if (startYEnv)
        ps->setSolverIterationCounts(/*pos*/ 8, /*vel*/ 1);
    std::vector<PxVec4> hostPos(N), hostVel(N, PxVec4(0.0f, 0.0f, 0.0f, 0.0f));
    std::vector<PxU32> hostPhase(N, phase);
    PxU32 idx = 0;
    for (int x = 0; x < G; ++x)
        for (int y = 0; y < G; ++y)
            for (int z = 0; z < G; ++z)
            {
                hostPos[idx++] = PxVec4(float(x) * particleSpacing - 0.75f,
                                        startY + float(y) * particleSpacing,
                                        float(z) * particleSpacing - 0.75f, 1.0f);
            }

    PxParticleBuffer* buffer = physics->createParticleBuffer(N, /*maxVolumes*/ 0, cudaCtx);
    if (!buffer)
    {
        std::printf("FAIL: createParticleBuffer returned null (bridge stub not replaced?)\n");
        return out;
    }

    // Fill the buffer's DEVICE arrays (getPositionInvMasses() et al. are device
    // pointers on the GPU path) via host->device copies through the bridge.
    PxCudaContext* ctx = cudaCtx->getCudaContext();
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buffer->getPositionInvMasses()), hostPos.data(),
                    N * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buffer->getVelocities()), hostVel.data(),
                    N * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buffer->getPhases()), hostPhase.data(),
                    N * sizeof(PxU32));
    buffer->setNbActiveParticles(N);
    buffer->raiseFlags(PxParticleBufferFlag::eUPDATE_POSITION);
    buffer->raiseFlags(PxParticleBufferFlag::eUPDATE_VELOCITY);
    buffer->raiseFlags(PxParticleBufferFlag::eUPDATE_PHASE);
    ps->addParticleBuffer(buffer);
    scene->addActor(*ps);

    out.system = ps;
    out.buffer = buffer;
    out.numParticles = N;
    return out;
}

// DtoH-read the particle buffer and return the mean y of the cloud.
float particleMeanY(const ParticleScene& psc)
{
    std::vector<PxVec4> host(psc.numParticles);
    PxCudaContext* ctx = psc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(), reinterpret_cast<CUdeviceptr>(psc.buffer->getPositionInvMasses()),
                    psc.numParticles * sizeof(PxVec4));
    double sum = 0.0;
    for (PxU32 i = 0; i < psc.numParticles; ++i) sum += double(host[i].y);
    return float(sum / double(psc.numParticles));
}

// PXG_SCENE=cloth (P7.0): the first FEMCloth/deformable-surface scene — the exact
// analog of the PBD particle free fall (buildParticleScene above).  A small
// triangle GRID mesh (a PxDeformableSurface) is spawned HIGH with invMass = 1 and
// zero velocity, so it is a pure free fall with NO contacts.  The engine launches
// the whole per-frame cloth sequence, but only the three P7.0 orchestration
// kernels are mapped (cloth_preIntegrateLaunch gravity-predict + integrate,
// cloth_finalizeVelocitiesLaunch, cloth_sleeping wake bookkeeping); every
// membrane / bending / collision / solve kernel refuses (non-fatal).
//
// Two structural wins over particles: (1) the engine's internal cloth vertex
// arrays ALIAS the user buffer (PxgSimulationCore.cpp:1850 femCloth.mPosition_
// InvMass = ...positionInvMass), so there is NO ingest/egress kernel — the
// predictor writes the fall straight into the buffer getPositionInvMassBufferD()
// reads back; (2) the cloth device buffers allocate via the already-real GPU
// memory manager in attachShape (NpDeformableSurface.cpp:387), so NO bridge
// factory code is needed (unlike particles' createParticleBuffer).
struct ClothScene
{
    PxDeformableSurface* surface = nullptr;
    PxU32 numVerts = 0;
    PxCudaContextManager* cudaCtx = nullptr;
};

ClothScene buildClothScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                           PxCudaContextManager* cudaCtx)
{
    ClothScene out;
    out.cudaCtx = cudaCtx;

    // A minimal rigid body (box on ground) FAR BELOW the cloth.  As with the
    // particle scene, the cloth solve runs inside the rigid PGS island loop and
    // rigidSumInternalContactAndJointBatches2 null-derefs the solver descriptors
    // with zero dynamic rigids, so one live dynamic body is required to keep that
    // proven path valid.  It rests at y=0.5, well below the cloth (spawned y~5).
    addGroundBox(physics, scene, rigidMat);
    addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
               boxInertia(PxVec3(0.5f), 1000.0f), 1.5f);

    // Cook a small triangle GRID mesh.  buildGPUData + eENABLE_VERT_MAPPING are
    // MANDATORY: attachShape rejects a mesh whose getNbTriangleReferences()==0
    // (NpDeformableSurface.cpp:363), and the GPU solve needs the GPU accel data.
    const PxU32 numX = 5, numZ = 5;              // 25 verts, 32 triangles
    const float sizeX = 2.0f, sizeZ = 2.0f;
    const float spawnY = 5.0f;                   // spawn HIGH — a pure free fall
    const float sx = sizeX / float(numX - 1);
    const float sz = sizeZ / float(numZ - 1);

    std::vector<PxVec3> verts;
    verts.reserve(numX * numZ);
    for (PxU32 i = 0; i < numX; ++i)
        for (PxU32 j = 0; j < numZ; ++j)
            verts.push_back(PxVec3(float(i) * sx - 0.5f * sizeX, 0.0f, float(j) * sz - 0.5f * sizeZ));

    std::vector<PxU32> tris;
    tris.reserve(3 * 2 * (numX - 1) * (numZ - 1));
    auto vid = [numZ](PxU32 x, PxU32 y) { return x * numZ + y; };
    for (PxU32 i = 1; i < numX; ++i)
        for (PxU32 j = 1; j < numZ; ++j)
        {
            tris.push_back(vid(i - 1, j - 1)); tris.push_back(vid(i, j - 1)); tris.push_back(vid(i - 1, j));
            tris.push_back(vid(i - 1, j));     tris.push_back(vid(i, j - 1)); tris.push_back(vid(i, j));
        }

    PxTriangleMeshDesc meshDesc;
    meshDesc.points.count = PxU32(verts.size());
    meshDesc.points.stride = sizeof(PxVec3);
    meshDesc.points.data = verts.data();
    meshDesc.triangles.count = PxU32(tris.size() / 3);
    meshDesc.triangles.stride = 3 * sizeof(PxU32);
    meshDesc.triangles.data = tris.data();

    PxCookingParams ckParams(physics->getTolerancesScale());
    ckParams.buildGPUData = true;
    ckParams.meshPreprocessParams |= PxMeshPreprocessingFlag::eENABLE_VERT_MAPPING;

    PxTriangleMesh* triMesh =
        immediateCooking::createTriangleMesh(ckParams, meshDesc, physics->getPhysicsInsertionCallback());
    if (!triMesh) { std::printf("FAIL: immediateCooking::createTriangleMesh returned null\n"); return out; }

    // Deformable-surface material + a SIMULATION_SHAPE built from the tri-mesh.
    PxDeformableSurfaceMaterial* clothMat =
        physics->createDeformableSurfaceMaterial(/*youngs*/ 1.e6f, /*poissons*/ 0.3f,
            /*dynamicFriction*/ 0.5f, /*thickness*/ 0.01f, /*bendingStiffness*/ 1.e-4f);
    PxDeformableSurfaceMaterial* mats[1] = { clothMat };

    PxShapeFlags shapeFlags = PxShapeFlag::eSIMULATION_SHAPE | PxShapeFlag::eSCENE_QUERY_SHAPE;
    PxShape* shape = physics->createShape(PxTriangleMeshGeometry(triMesh),
                                          mats, PxU16(1), true, shapeFlags);
    if (!shape) { std::printf("FAIL: createShape (deformable surface) returned null\n"); return out; }

    PxDeformableSurface* surface = physics->createDeformableSurface(*cudaCtx);
    if (!surface) { std::printf("FAIL: createDeformableSurface returned null\n"); return out; }

    // NON-ADAPTIVE collision-pair updates (frequency != 0): this makes the engine
    // pass adaptiveCollisionPairUpdate = false to cloth_preIntegrateLaunch, so the
    // predictor's ONLY warp op (warpReduction in the adaptive branch) is dead code
    // — no subgroup op runs on llvmpipe from the fall driver.
    surface->setNbCollisionPairUpdatesPerTimestep(4);
    surface->setNbCollisionSubsteps(1);
    surface->setLinearDamping(0.0f);             // cleanest free fall (damping factor 1)
    surface->setMaxLinearVelocity(1.0e6f);       // do not clamp the free fall
    surface->attachShape(*shape);
    shape->release();

    scene->addActor(*surface);

    // Fill the DEVICE buffers directly (getPositionInvMassBufferD() aliases the
    // engine's internal cloth arrays).  Spawn the whole grid at y=spawnY, invMass=1
    // so every vertex free-falls; rest position = spawn shape; velocity = 0.
    const PxU32 nv = PxU32(verts.size());
    std::vector<PxVec4> posInvMass(nv), vel(nv, PxVec4(0.0f)), rest(nv);
    for (PxU32 v = 0; v < nv; ++v)
    {
        posInvMass[v] = PxVec4(verts[v].x, spawnY + verts[v].y, verts[v].z, 1.0f);
        rest[v]       = PxVec4(verts[v].x, spawnY + verts[v].y, verts[v].z, 1.0f);
    }

    PxCudaContext* ctx = cudaCtx->getCudaContext();
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getPositionInvMassBufferD()),
                    posInvMass.data(), nv * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getVelocityBufferD()),
                    vel.data(), nv * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getRestPositionBufferD()),
                    rest.data(), nv * sizeof(PxVec4));

    out.surface = surface;
    out.numVerts = nv;
    return out;
}

// DtoH-read the cloth vertex buffer and return the mean y across vertices.
float clothMeanY(const ClothScene& csc)
{
    std::vector<PxVec4> host(csc.numVerts);
    PxCudaContext* ctx = csc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(), reinterpret_cast<CUdeviceptr>(csc.surface->getPositionInvMassBufferD()),
                    csc.numVerts * sizeof(PxVec4));
    double sum = 0.0;
    for (PxU32 i = 0; i < csc.numVerts; ++i) sum += double(host[i].y);
    return float(sum / double(csc.numVerts));
}

// DtoH-read the cloth vertex buffer once and fill the mean AND the min y across all
// verts (the drape needs the min to prove no corner tunnelled below the floor).
// Returns false on any non-finite y.
bool clothMeanMinY(const ClothScene& csc, float& meanY, float& minY)
{
    std::vector<PxVec4> host(csc.numVerts);
    PxCudaContext* ctx = csc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(), reinterpret_cast<CUdeviceptr>(csc.surface->getPositionInvMassBufferD()),
                    csc.numVerts * sizeof(PxVec4));
    double sum = 0.0;
    float lo = 1.0e30f;
    bool finite = true;
    for (PxU32 i = 0; i < csc.numVerts; ++i)
    {
        const float y = host[i].y;
        sum += double(y);
        lo = PxMin(lo, y);
        if (!std::isfinite(y) || !std::isfinite(host[i].x) || !std::isfinite(host[i].z)) finite = false;
    }
    meanY = float(sum / double(csc.numVerts));
    minY = lo;
    return finite;
}

// PXG_SCENE=clothdrape (P7.1 tranche 4c): the cloth-vs-rigid CONTACT money shot — a
// HORIZONTAL FEMCloth sheet dropped a short distance ABOVE a STATIC floor, with NO pins.
// The whole cloth-vs-primitive contact chain must CATCH it: broadphase pairs the cloth
// mesh with the static PLANE (cndCasePlaneMesh -> GPU_BUCKET_ID::eFemClothPlane), the
// narrowphase dispatches testSDKFemClothPlane (cloth_planeVertContactGenLaunch), the
// contacts sort (fem_reorderRigidContactsLaunch + the shared radix sort), prepare
// (cloth_rigidContactPrepareLaunch), query ref-count + solve
// (cloth_queryRigidClothContactReferenceCount{,TGS} + cloth_solveRigidClothCollision{,TGS}),
// and the accumulated mDeltaPos folds back into the vertices via
// cloth_applyExternalDeltasLaunch.  A static PLANE is the natural target because
// cloth_planeVertContactGenLaunch is the mapped direct plane path (no midphase needed).
//
// EXPECTED PHYSICS: the sheet free-falls, the floor contact generates+solves, and the
// cloth SETTLES resting ON the floor (mean y drops then converges to ~floorY + the cloth
// contact/rest offset), NOT tunnelling to -inf and NOT free-falling through.
//
// A live dynamic rigid (a box resting on the same plane, spawned FAR from the cloth in x)
// is kept for the same reason as the free-fall/hang scenes: the cloth solve runs inside
// the rigid PGS island loop, whose descriptors must be non-null.  It is placed well away
// from the cloth footprint so it never touches the sheet.
struct ClothDrapeConfig { float floorY = 0.0f; float spawnY = 1.5f; };

ClothScene buildClothDrapeScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                PxCudaContextManager* cudaCtx, const ClothDrapeConfig& cfg)
{
    ClothScene out;
    out.cudaCtx = cudaCtx;

    // The STATIC FLOOR: an infinite plane at y = floorY, normal +Y.  This is the surface
    // the cloth contact path must catch (cloth-vertex-vs-plane, the mapped direct path).
    PxRigidStatic* floor = physics->createRigidStatic(
        PxTransformFromPlaneEquation(PxPlane(0.0f, 1.0f, 0.0f, -cfg.floorY)));
    PxShape* floorShape = physics->createShape(PxPlaneGeometry(), *rigidMat);
    floor->attachShape(*floorShape);
    scene->addActor(*floor);
    floorShape->release();

    // A live dynamic rigid resting on the SAME plane, spawned FAR from the cloth (x=8) so
    // it never interacts with the sheet.  Keeps the rigid PGS island non-empty (see the
    // free-fall/hang scenes' rationale) and rides the proven box-on-plane contact path.
    {
        PxRigidDynamic* b = physics->createRigidDynamic(PxTransform(PxVec3(8.0f, cfg.floorY + 3.0f, 0.0f)));
        PxShape* s = physics->createShape(PxBoxGeometry(0.5f, 0.5f, 0.5f), *rigidMat);
        b->attachShape(*s);
        b->setMass(1000.0f);
        b->setMassSpaceInertiaTensor(boxInertia(PxVec3(0.5f), 1000.0f));
        scene->addActor(*b);
        s->release();
    }

    // Cook the same small horizontal triangle GRID mesh as buildClothScene (buildGPUData +
    // eENABLE_VERT_MAPPING mandatory).  Sheet is planar in XZ, spawned at y = spawnY.
    const PxU32 numX = 5, numZ = 5;              // 25 verts, 32 triangles
    const float sizeX = 2.0f, sizeZ = 2.0f;
    const float sx = sizeX / float(numX - 1);
    const float sz = sizeZ / float(numZ - 1);

    std::vector<PxVec3> verts;
    verts.reserve(numX * numZ);
    for (PxU32 i = 0; i < numX; ++i)
        for (PxU32 j = 0; j < numZ; ++j)
            verts.push_back(PxVec3(float(i) * sx - 0.5f * sizeX, 0.0f, float(j) * sz - 0.5f * sizeZ));

    std::vector<PxU32> tris;
    tris.reserve(3 * 2 * (numX - 1) * (numZ - 1));
    auto vid = [numZ](PxU32 x, PxU32 y) { return x * numZ + y; };
    for (PxU32 i = 1; i < numX; ++i)
        for (PxU32 j = 1; j < numZ; ++j)
        {
            tris.push_back(vid(i - 1, j - 1)); tris.push_back(vid(i, j - 1)); tris.push_back(vid(i - 1, j));
            tris.push_back(vid(i - 1, j));     tris.push_back(vid(i, j - 1)); tris.push_back(vid(i, j));
        }

    PxTriangleMeshDesc meshDesc;
    meshDesc.points.count = PxU32(verts.size());
    meshDesc.points.stride = sizeof(PxVec3);
    meshDesc.points.data = verts.data();
    meshDesc.triangles.count = PxU32(tris.size() / 3);
    meshDesc.triangles.stride = 3 * sizeof(PxU32);
    meshDesc.triangles.data = tris.data();

    PxCookingParams ckParams(physics->getTolerancesScale());
    ckParams.buildGPUData = true;
    ckParams.meshPreprocessParams |= PxMeshPreprocessingFlag::eENABLE_VERT_MAPPING;

    PxTriangleMesh* triMesh =
        immediateCooking::createTriangleMesh(ckParams, meshDesc, physics->getPhysicsInsertionCallback());
    if (!triMesh) { std::printf("FAIL: immediateCooking::createTriangleMesh returned null\n"); return out; }

    PxDeformableSurfaceMaterial* clothMat =
        physics->createDeformableSurfaceMaterial(/*youngs*/ 1.e6f, /*poissons*/ 0.3f,
            /*dynamicFriction*/ 0.5f, /*thickness*/ 0.01f, /*bendingStiffness*/ 1.e-4f);
    PxDeformableSurfaceMaterial* mats[1] = { clothMat };

    PxShapeFlags shapeFlags = PxShapeFlag::eSIMULATION_SHAPE | PxShapeFlag::eSCENE_QUERY_SHAPE;
    PxShape* shape = physics->createShape(PxTriangleMeshGeometry(triMesh), mats, PxU16(1), true, shapeFlags);
    if (!shape) { std::printf("FAIL: createShape (deformable surface) returned null\n"); return out; }

    PxDeformableSurface* surface = physics->createDeformableSurface(*cudaCtx);
    if (!surface) { std::printf("FAIL: createDeformableSurface returned null\n"); return out; }

    surface->setNbCollisionPairUpdatesPerTimestep(4);
    surface->setNbCollisionSubsteps(1);
    surface->setLinearDamping(0.0f);
    surface->setMaxLinearVelocity(1.0e6f);
    surface->attachShape(*shape);
    shape->release();

    scene->addActor(*surface);

    // Fill the DEVICE buffers directly.  invMass=1 for every vertex (a genuine free fall
    // caught only by the floor contact — no pin, no zero-mass bypass); velocity=0; rest =
    // the flat spawn shape (membrane at rest at spawn).
    const PxU32 nv = PxU32(verts.size());
    std::vector<PxVec4> posInvMass(nv), vel(nv, PxVec4(0.0f)), rest(nv);
    for (PxU32 v = 0; v < nv; ++v)
    {
        posInvMass[v] = PxVec4(verts[v].x, cfg.spawnY + verts[v].y, verts[v].z, 1.0f);
        rest[v]       = PxVec4(verts[v].x, cfg.spawnY + verts[v].y, verts[v].z, 1.0f);
    }

    PxCudaContext* ctx = cudaCtx->getCudaContext();
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getPositionInvMassBufferD()),
                    posInvMass.data(), nv * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getVelocityBufferD()),
                    vel.data(), nv * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getRestPositionBufferD()),
                    rest.data(), nv * sizeof(PxVec4));

    out.surface = surface;
    out.numVerts = nv;
    return out;
}

// PXG_SCENE=clothhang (P7.1 tranche 3b): a HANGING / draping cloth — the visible
// "cloth is usable" money shot.  Same triangle-mesh PxDeformableSurface cook + GPU
// spawn as the free-fall 'cloth' scene, but the cloth is oriented as a VERTICAL
// curtain (XY plane) and its TOP ROW of vertices is PINNED to fixed world anchors at
// their spawn positions via the deformable-surface rigid-attachment API
// (PxPhysics::createDeformableAttachment with actor0 = the surface / eVERTEX,
// actor1 = NULL / eWORLD).  That path threads down through
// NpDeformableAttachment::addAttachment -> Sc::DeformableSurfaceCore::addRigidAttachment
// -> Sc::Scene::addRigidAttachment -> PxgSimulationController::addRigidAttachment ->
// PxgFEMRigidAttachment, so the engine's rigid-cloth-attachment buffers get populated
// and the P7.1.3a constraint kernels (cloth_rigidAttachmentPrepareLaunch +
// cloth_solveRigidClothAttachment{,TGS}Launch) dispatch each solver iteration.
//
// EXPECTED PHYSICS: the pinned top row stays at its anchor y; the free rows below
// stretch a bounded amount under gravity, then the membrane/bending energy solve
// balances the load and the drape STABILIZES (fall-then-settle, not free-fall).
//
// Each pinned vertex gets its OWN attachment (all distinct vertices — one attachment
// per vertex, no per-vertex contention), so the exact single-attachment 3a solve path
// is exercised and the deferred multi-attachment averaging kernel is NOT needed.
struct ClothHangScene
{
    PxDeformableSurface* surface = nullptr;
    PxU32 numVerts = 0;
    PxCudaContextManager* cudaCtx = nullptr;
    PxDeformableAttachment* attachment = nullptr;
    std::vector<PxU32> topRow;      // PINNED vertices (top edge of the curtain)
    std::vector<PxU32> bottomRow;   // FREE vertices tracked for the drape/settle
};

ClothHangScene buildClothHangScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                   PxCudaContextManager* cudaCtx)
{
    ClothHangScene out;
    out.cudaCtx = cudaCtx;

    // A live dynamic rigid FAR BELOW the cloth (same rationale as the free-fall scene:
    // the cloth solve runs inside the rigid PGS island loop, whose descriptors must be
    // non-null).  It rests at y=0.5, well below the curtain (top row at y~5).
    addGroundBox(physics, scene, rigidMat);
    addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
               boxInertia(PxVec3(0.5f), 1000.0f), 1.5f);

    // Cook a triangle GRID mesh (buildGPUData + eENABLE_VERT_MAPPING mandatory, as in
    // buildClothScene).  Grid indexed [col][row]; the curtain hangs in the XY plane.
    const PxU32 numCol = 5, numRow = 5;             // 25 verts, 32 triangles
    const float sizeX = 2.0f, sizeY = 2.0f;
    const float topY = 5.0f;                        // the pinned top edge height
    const float sx = sizeX / float(numCol - 1);
    const float sy = sizeY / float(numRow - 1);
    auto vid = [numRow](PxU32 c, PxU32 r) { return c * numRow + r; };

    std::vector<PxVec3> verts;
    verts.reserve(numCol * numRow);
    for (PxU32 c = 0; c < numCol; ++c)
        for (PxU32 r = 0; r < numRow; ++r)
            // row 0 = top edge (y = topY); increasing r hangs DOWNWARD; planar z = 0.
            verts.push_back(PxVec3(float(c) * sx - 0.5f * sizeX, topY - float(r) * sy, 0.0f));

    std::vector<PxU32> tris;
    tris.reserve(3 * 2 * (numCol - 1) * (numRow - 1));
    for (PxU32 c = 1; c < numCol; ++c)
        for (PxU32 r = 1; r < numRow; ++r)
        {
            tris.push_back(vid(c - 1, r - 1)); tris.push_back(vid(c, r - 1)); tris.push_back(vid(c - 1, r));
            tris.push_back(vid(c - 1, r));     tris.push_back(vid(c, r - 1)); tris.push_back(vid(c, r));
        }

    // Track the top row (pinned) and bottom row (free) for the per-row settle check.
    for (PxU32 c = 0; c < numCol; ++c) out.topRow.push_back(vid(c, 0));
    for (PxU32 c = 0; c < numCol; ++c) out.bottomRow.push_back(vid(c, numRow - 1));

    PxTriangleMeshDesc meshDesc;
    meshDesc.points.count = PxU32(verts.size());
    meshDesc.points.stride = sizeof(PxVec3);
    meshDesc.points.data = verts.data();
    meshDesc.triangles.count = PxU32(tris.size() / 3);
    meshDesc.triangles.stride = 3 * sizeof(PxU32);
    meshDesc.triangles.data = tris.data();

    PxCookingParams ckParams(physics->getTolerancesScale());
    ckParams.buildGPUData = true;
    ckParams.meshPreprocessParams |= PxMeshPreprocessingFlag::eENABLE_VERT_MAPPING;

    PxTriangleMesh* triMesh =
        immediateCooking::createTriangleMesh(ckParams, meshDesc, physics->getPhysicsInsertionCallback());
    if (!triMesh) { std::printf("FAIL: immediateCooking::createTriangleMesh returned null\n"); return out; }

    PxDeformableSurfaceMaterial* clothMat =
        physics->createDeformableSurfaceMaterial(/*youngs*/ 1.e6f, /*poissons*/ 0.3f,
            /*dynamicFriction*/ 0.5f, /*thickness*/ 0.01f, /*bendingStiffness*/ 1.e-4f);
    PxDeformableSurfaceMaterial* mats[1] = { clothMat };

    PxShapeFlags shapeFlags = PxShapeFlag::eSIMULATION_SHAPE | PxShapeFlag::eSCENE_QUERY_SHAPE;
    PxShape* shape = physics->createShape(PxTriangleMeshGeometry(triMesh), mats, PxU16(1), true, shapeFlags);
    if (!shape) { std::printf("FAIL: createShape (deformable surface) returned null\n"); return out; }

    PxDeformableSurface* surface = physics->createDeformableSurface(*cudaCtx);
    if (!surface) { std::printf("FAIL: createDeformableSurface returned null\n"); return out; }

    surface->setNbCollisionPairUpdatesPerTimestep(4);
    surface->setNbCollisionSubsteps(1);
    surface->setLinearDamping(0.0f);
    surface->setMaxLinearVelocity(1.0e6f);
    surface->attachShape(*shape);
    shape->release();

    scene->addActor(*surface);

    // Fill the DEVICE buffers directly (getPositionInvMassBufferD() aliases the engine's
    // internal cloth arrays).  invMass=1 for every vertex (the pin is a CONSTRAINT, not a
    // fixed/zero-mass vertex — the attachment solve does the holding); velocity=0; rest =
    // spawn shape (so the membrane is at rest at spawn).
    const PxU32 nv = PxU32(verts.size());
    std::vector<PxVec4> posInvMass(nv), vel(nv, PxVec4(0.0f)), rest(nv);
    for (PxU32 v = 0; v < nv; ++v)
    {
        posInvMass[v] = PxVec4(verts[v].x, verts[v].y, verts[v].z, 1.0f);
        rest[v]       = PxVec4(verts[v].x, verts[v].y, verts[v].z, 1.0f);
    }

    PxCudaContext* ctx = cudaCtx->getCudaContext();
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getPositionInvMassBufferD()),
                    posInvMass.data(), nv * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getVelocityBufferD()),
                    vel.data(), nv * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getRestPositionBufferD()),
                    rest.data(), nv * sizeof(PxVec4));

    // PIN the top row: one VERTEX->WORLD attachment per top-row vertex, anchored at the
    // vertex's own spawn position (pose = identity, so anchor = coords[i]).  Distinct
    // vertices => one attachment each => the exact single-attachment 3a path.
    static std::vector<PxU32> pinIdx;    // must outlive createDeformableAttachment's copy
    static std::vector<PxVec4> pinPos;
    pinIdx = out.topRow;
    pinPos.clear();
    for (PxU32 id : out.topRow)
        pinPos.push_back(PxVec4(verts[id].x, verts[id].y, verts[id].z, 0.0f));

    PxDeformableAttachmentData attData;
    attData.actor[0] = surface;
    attData.type[0]  = PxDeformableAttachmentTargetType::eVERTEX;
    attData.indices[0].data = pinIdx.data();
    attData.indices[0].count = PxU32(pinIdx.size());
    attData.actor[1] = NULL;                                          // world anchor
    attData.type[1]  = PxDeformableAttachmentTargetType::eWORLD;
    attData.coords[1].data = pinPos.data();
    attData.coords[1].count = PxU32(pinPos.size());
    attData.pose[1] = PxTransform(PxIdentity);

    out.attachment = physics->createDeformableAttachment(attData);
    if (!out.attachment)
        std::printf("FAIL: createDeformableAttachment (cloth vertex -> world) returned null\n");

    out.surface = surface;
    out.numVerts = nv;
    return out;
}

// DtoH-read the cloth vertex buffer once; fill the mean y of the pinned top row and the
// free bottom row (the two observables of the drape).  Returns false on non-finite y.
bool clothHangRowMeans(const ClothHangScene& csc, float& topMeanY, float& bottomMeanY)
{
    std::vector<PxVec4> host(csc.numVerts);
    PxCudaContext* ctx = csc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(), reinterpret_cast<CUdeviceptr>(csc.surface->getPositionInvMassBufferD()),
                    csc.numVerts * sizeof(PxVec4));
    double topSum = 0.0, botSum = 0.0;
    bool finite = true;
    for (PxU32 id : csc.topRow)    { topSum += double(host[id].y); if (!std::isfinite(host[id].y)) finite = false; }
    for (PxU32 id : csc.bottomRow) { botSum += double(host[id].y); if (!std::isfinite(host[id].y)) finite = false; }
    topMeanY = float(topSum / double(csc.topRow.size()));
    bottomMeanY = float(botSum / double(csc.bottomRow.size()));
    return finite;
}

// PXG_SCENE=clothselfcollide (P7.2 tranche 3): the cloth SELF-collision money shot.
//
// A single FEMCloth mesh HAIRPIN-FOLDED into two stacked layers of the SAME cloth.  The
// LOWER layer is a FIXED shelf (invMass 0, pinned at y=baseY); the UPPER layer (invMass 1)
// hangs from the crease edge and, under gravity, DRAPES DOWN onto the shelf.  The two
// layers share the same (x,z) footprint but are FAR apart in mesh topology (joined only at
// the crease), so the vertex-triangle self-collision filter does not exclude them.  The ONLY
// thing that can stop the upper layer from drooping straight THROUGH the shelf is self-
// collision — there is NO rigid body here, so this isolates the tranche-1 midphase
// (cloth_selfCollisionMidphase{VT,EE}) + tranche-2 solve (cloth_clothContactPrepare ->
// query -> cloth_solveClothClothDelta{VT,EE}, folded back by cloth_applyExternalDeltas)
// with nothing else in the loop that could catch the cloth.  (Cloth runs on the PGS solver;
// pass PXG_SOLVER=pgs — the TGS cloth path is a separate, not-yet-ported frontier.)
//
// The self-collision SWITCH is the rest-pose filter distance, whose exact semantics the
// midphase documents (femClothClothMidPhase.cu:789): a vertex-triangle pair self-collides
// only when its REST-pose distance EXCEEDS the filter distance.  The two layers sit dropH
// apart in the rest pose, so filterDistance < dropH LETS them collide (self-collision ON)
// while filterDistance > dropH FILTERS the pair out (self-collision OFF, the control).
// PXG_SELFCOLL (default "1") picks the mode; everything else is identical, so a two-run A/B
// comparison isolates self-collision as the ONLY cause of the upper layer staying up.
struct ClothSelfScene
{
    PxDeformableSurface* surface = nullptr;
    PxU32 numVerts = 0;
    PxCudaContextManager* cudaCtx = nullptr;
    std::vector<PxU32> topIdx;      // the FREE upper layer (drapes onto the shelf) — observable
    float baseY = 0.0f;            // the fixed shelf height
    bool selfCollOn = true;
};

ClothSelfScene buildClothSelfCollideScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                          PxCudaContextManager* cudaCtx, bool enableSelfCollision)
{
    ClothSelfScene out;
    out.cudaCtx = cudaCtx;
    out.selfCollOn = enableSelfCollision;

    // A live dynamic rigid FAR from the cloth keeps the rigid PGS island non-empty (every
    // cloth scene needs this — the cloth solve runs inside it).  No floor: the "shelf" the
    // upper layer lands on is made of the cloth's OWN fixed verts, so self-collision is the
    // only thing in the loop that can stop the drape.
    addGroundBox(physics, scene, rigidMat);
    addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
               boxInertia(PxVec3(0.5f), 1000.0f), 1.5f);

    // Mesh density is a knob because the inherited per-pass contact overshoot DECAYS with
    // contact density (femClothSimSelfContactManyTriangles: 1.20x at K=1 down to 1.014x at
    // K=16), so density is the lever that decides whether this fold stays put.
    const char* halfEnv = std::getenv("PXG_CLOTH_HALF");
    const char* rowsEnv = std::getenv("PXG_CLOTH_ROWS");
    const PxU32 half = halfEnv ? PxU32(atoi(halfEnv)) : 4u;   // columns per layer
    const PxU32 numX = 2 * half;
    const PxU32 numZ = rowsEnv ? PxU32(atoi(rowsEnv)) : 5u;
    const float baseY = 5.0f;             // the fixed shelf height
    // The upper layer starts ABOVE the contact rest distance (2 * restOffset = 0.10) and
    // FALLS onto the shelf, so the scene shows a real drop and a real catch.  It used to
    // start at 0.05 — INSIDE the rest distance — which handed the contact solve a 0.05
    // penetration deficit on frame one.  That mattered once the midphase grid defect was
    // fixed: the solve's inherited per-pass overshoot (mDeltaPos.w accumulates |coeff|
    // rather than 1, see femClothSimSelfContactManyTriangles — 1.2x for a lone contact)
    // is absorbed by whichever endpoint acts as the vertex side, and with correct coverage
    // twice as many vertices do.  Overshooting an artificial deficit every pass ejected the
    // fold.  Starting outside the rest distance removes the deficit; the layer settles ON
    // the rest distance instead of being pushed through it.
    const char* dropEnv = std::getenv("PXG_CLOTH_DROP");
    const float dropH = dropEnv ? float(atof(dropEnv)) : 0.30f;
    const float lenX  = 2.0f, lenZ = 2.0f;
    const float dx = lenX / float(half - 1);
    const float dz = lenZ / float(numZ - 1);
    out.baseY = baseY;

    auto vid = [numZ](PxU32 c, PxU32 r) { return c * numZ + r; };

    // Hairpin fold: columns [0,half) run x = -lenX/2 -> +lenX/2 at y=baseY (the shelf);
    // columns [half,2*half) fold BACK, x = +lenX/2 -> -lenX/2 at y=baseY+dropH (the top).
    //
    // The upper layer's (x,z) footprint is INSET relative to the shelf's.  Vertex-triangle
    // self-collision can only hold a vertex that actually sits over a shelf TRIANGLE: with
    // the two footprints identical, the upper layer's border vertices sit exactly on the
    // shelf's outer edge, have no triangle beneath them, and droop through while the
    // interior is held perfectly (measured: interior 5.0-5.5, border 4.66-4.91).  That is
    // correct cloth behaviour, not a solver gap — so the scene gives the shelf an overhang,
    // the way a real sheet resting on a table is smaller than the table.
    const float inset = 0.9f;
    std::vector<PxVec3> verts;
    verts.reserve(numX * numZ);
    for (PxU32 c = 0; c < numX; ++c)
    {
        const bool lower = (c < half);
        const float x0 = (lower ? float(c) : float(2 * half - 1 - c)) * dx - 0.5f * lenX;
        const float x = lower ? x0 : x0 * inset;
        const float y = lower ? baseY : baseY + dropH;
        for (PxU32 r = 0; r < numZ; ++r)
        {
            const float z0 = float(r) * dz - 0.5f * lenZ;
            verts.push_back(PxVec3(x, y, lower ? z0 : z0 * inset));
        }
    }

    std::vector<PxU32> tris;
    tris.reserve(3 * 2 * (numX - 1) * (numZ - 1));
    for (PxU32 c = 1; c < numX; ++c)
        for (PxU32 r = 1; r < numZ; ++r)
        {
            tris.push_back(vid(c - 1, r - 1)); tris.push_back(vid(c, r - 1)); tris.push_back(vid(c - 1, r));
            tris.push_back(vid(c - 1, r));     tris.push_back(vid(c, r - 1)); tris.push_back(vid(c, r));
        }

    // The FREE upper layer (columns [half,2*half)) is the observable — does it stay on the shelf?
    for (PxU32 c = half; c < numX; ++c)
        for (PxU32 r = 0; r < numZ; ++r)
            out.topIdx.push_back(vid(c, r));

    PxTriangleMeshDesc meshDesc;
    meshDesc.points.count = PxU32(verts.size());
    meshDesc.points.stride = sizeof(PxVec3);
    meshDesc.points.data = verts.data();
    meshDesc.triangles.count = PxU32(tris.size() / 3);
    meshDesc.triangles.stride = 3 * sizeof(PxU32);
    meshDesc.triangles.data = tris.data();

    PxCookingParams ckParams(physics->getTolerancesScale());
    ckParams.buildGPUData = true;
    ckParams.meshPreprocessParams |= PxMeshPreprocessingFlag::eENABLE_VERT_MAPPING;

    PxTriangleMesh* triMesh =
        immediateCooking::createTriangleMesh(ckParams, meshDesc, physics->getPhysicsInsertionCallback());
    if (!triMesh) { std::printf("FAIL: immediateCooking::createTriangleMesh returned null\n"); return out; }

    // A THICK cloth (0.1) gives self-collision a wide contact band so a vertex cannot skip
    // through a triangle between substeps; low bending lets the free layer settle.
    PxDeformableSurfaceMaterial* clothMat =
        physics->createDeformableSurfaceMaterial(/*youngs*/ 1.e6f, /*poissons*/ 0.3f,
            /*dynamicFriction*/ 0.5f, /*thickness*/ 0.1f, /*bendingStiffness*/ 1.e-4f);
    PxDeformableSurfaceMaterial* mats[1] = { clothMat };

    PxShapeFlags shapeFlags = PxShapeFlag::eSIMULATION_SHAPE | PxShapeFlag::eSCENE_QUERY_SHAPE;
    PxShape* shape = physics->createShape(PxTriangleMeshGeometry(triMesh), mats, PxU16(1), true, shapeFlags);
    if (!shape) { std::printf("FAIL: createShape (deformable surface) returned null\n"); return out; }
    shape->setContactOffset(0.1f);   // widen the contact band to match the thick cloth
    // The REST offset is what actually decides self-collision: the contact query keeps a
    // pair only while its separation is within `cloth0->mRestDistance + cloth1->mRestDistance`
    // (FEMClothExternalSolve.cu:870, countContact), and mRestDistance IS the shape's rest
    // offset (PxgSimulationController.cpp:1770).  Left at its 0 default every self-contact
    // is immediately marked not-in-collision, so the midphase generates pairs that the solve
    // then discards — the layers pass straight through each other.  0.05 per layer gives a
    // 0.1 separation band, which both exceeds the dropH=0.05 the layers start at (so they
    // engage from step 0) and stays <= the contact offset, as PhysX requires.
    // The rest offset sets the separation the contact solve drives the layers to; the
    // absolute per-pass overshoot scales with the deficit it has to close, so this is the
    // second lever alongside mesh density.
    const char* roEnv = std::getenv("PXG_CLOTH_RESTOFFSET");
    shape->setRestOffset(roEnv ? float(atof(roEnv)) : 0.05f);

    PxDeformableSurface* surface = physics->createDeformableSurface(*cudaCtx);
    if (!surface) { std::printf("FAIL: createDeformableSurface returned null\n"); return out; }

    surface->setNbCollisionPairUpdatesPerTimestep(4);
    // PXG_CLOTH_SUBSTEPS overrides the self-collision solve iteration count so the
    // scene's stability margin can be measured rather than assumed.
    const char* subEnv = std::getenv("PXG_CLOTH_SUBSTEPS");
    surface->setNbCollisionSubsteps(subEnv ? PxU32(atoi(subEnv)) : 8u);   // more iterations -> stiffer hold
    surface->setLinearDamping(0.2f);      // bleed off the drape momentum so the hold is clean
    surface->setMaxLinearVelocity(1.0e6f);
    // The rest-pose filter is the self-collision switch (see the scene comment): ON lets the
    // dropH-separated layers collide, OFF filters them out AND flags the body disabled.
    if (enableSelfCollision)
        surface->setSelfCollisionFilterDistance(0.02f);         // < dropH -> layers self-collide
    else
    {
        surface->setSelfCollisionFilterDistance(dropH + 0.3f);  // > dropH -> pair filtered out
        surface->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
    }
    surface->attachShape(*shape);
    shape->release();

    scene->addActor(*surface);

    // invMass: lower layer (c<half) FIXED (0) = the shelf; upper layer (c>=half) FREE (1).
    // rest = spawn shape (membrane at rest at spawn, so the ONLY driver is gravity).
    const PxU32 nv = PxU32(verts.size());
    std::vector<PxVec4> posInvMass(nv), vel(nv, PxVec4(0.0f)), rest(nv);
    for (PxU32 c = 0; c < numX; ++c)
        for (PxU32 r = 0; r < numZ; ++r)
        {
            const PxU32 v = vid(c, r);
            const float invMass = (c < half) ? 0.0f : 1.0f;
            posInvMass[v] = PxVec4(verts[v].x, verts[v].y, verts[v].z, invMass);
            rest[v]       = PxVec4(verts[v].x, verts[v].y, verts[v].z, invMass);
        }

    PxCudaContext* ctx = cudaCtx->getCudaContext();
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getPositionInvMassBufferD()),
                    posInvMass.data(), nv * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getVelocityBufferD()),
                    vel.data(), nv * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getRestPositionBufferD()),
                    rest.data(), nv * sizeof(PxVec4));

    out.surface = surface;
    out.numVerts = nv;
    return out;
}

// DtoH-read the cloth vertex buffer once; fill the min AND mean y across the FREE upper
// layer (the observable — does self-collision keep it on the shelf?).  Returns false on
// any non-finite coordinate.
bool clothTopMinMeanY(const ClothSelfScene& csc, float& minY, float& meanY)
{
    std::vector<PxVec4> host(csc.numVerts);
    PxCudaContext* ctx = csc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(), reinterpret_cast<CUdeviceptr>(csc.surface->getPositionInvMassBufferD()),
                    csc.numVerts * sizeof(PxVec4));
    double sum = 0.0;
    float lo = 1.0e30f;
    bool finite = true;
    for (PxU32 id : csc.topIdx)
    {
        const float y = host[id].y;
        sum += double(y);
        lo = PxMin(lo, y);
        if (!std::isfinite(y) || !std::isfinite(host[id].x) || !std::isfinite(host[id].z)) finite = false;
    }
    minY = lo;
    meanY = float(sum / double(csc.topIdx.size()));
    return finite;
}

// ===========================================================================
// PXG_SCENE=softbody (P8.0 tranche 7): the FEM-VOLUME money shot — a tetrahedral
// cube that falls under gravity, and (with PXG_SB_STRETCH) relaxes back toward its
// rest shape while it falls.
//
// It does NOT settle on the ground box, and deliberately does not claim to: the
// whole softbody-vs-rigid contact arc (SB_MIDPHASE_PRIMITIVES, SB_PRIMITIVES_CG,
// SB_OTHER_CONTACT_REMAP_TO_SIM, SB_RS_CONTACTPREPARE, SB_SOLVE_RIGID_SOFT_COLLISION)
// is still unmapped, so the cube passes straight through.  This is a CONTACT-FREE
// free-fall scene, exactly as PXG_SCENE=cloth was at P7.0; the settle comes with the
// contact tranche.
//
// This is the softbody analog of PXG_SCENE=cloth / clothdrape, and it is the first
// scene that can exist at all: the softbody solve loop only closed with tranche 7.
// The observable is deliberately the COLLISION mesh (getPositionInvMassBufferD()),
// not the simulation mesh, because the two are produced by different kernels:
//
//   the SIM mesh  (getSimPositionInvMassBufferD()) is what sb_gm_preIntegrateLaunch
//   predicts, what sb_gm_cp_solveTetrahedronsPartitionLaunch corrects, and what
//   sb_gm_cp_averageVertsLaunch folds back;
//   the COLLISION mesh (getPositionInvMassBufferD()) only ever moves because
//   sb_gm_updateTetModelVertsLaunch skins it onto the sim tetrahedra.
//
// Of the three tranche-7 kernels this scene drives TWO.  sb_gm_cp_averageVertsLaunch
// runs because mUsePartitionAveraging is set for any mesh with one tet per element
// (PxgSimulationCore.cpp:1420), and sb_gm_updateTetModelVertsLaunch runs twice per
// frame (preIntegrateAndUpdateBoundSoftBody and again in finalizeVelocities).
// sb_gm_applyDeformationDeltasLaunch does NOT run here: the cooker pins
// mGridModelNbPartitions to 8 for a one-tet-per-element mesh, so the Jacobi remainder
// is empty and the host never takes that branch.  It stays covered by its unit test.
//
// So a falling collision mesh is end-to-end proof of the WHOLE chain, tranche 1
// through 7, in a way no unit test can be: the skin kernel reads what the solve
// wrote, through the engine's own buffers, with the engine choosing the launch
// dimensions.  Both meshes are read back so a divergence between them localises
// immediately to the skin.
//
// The tetrahedral mesh is hand-authored rather than tetrahedralised: a regular
// 2x2x2 cell grid (27 vertices, 40 tetrahedra) with each cell cut into 5 tets on
// ALTERNATING diagonals, which is the standard conforming split — neighbouring
// cells must use opposite diagonals or their shared face is triangulated two
// different ways and the mesh is non-conforming.  Hand-authoring keeps the scene
// free of PxTetMaker (physxextensions is not linked into this harness; only
// ExtJoint/ExtD6Joint are) and makes the vertex count small enough to print.
//
// The same tet mesh is used as BOTH the simulation and the collision mesh, which
// is what immediateCooking::createDeformableVolumeMesh's "no voxels" path does;
// the vertex-to-tet mapping is left unassigned so the cooker computes it.
struct SoftBodyScene
{
    PxDeformableVolume* volume = nullptr;
    PxU32 numCollVerts = 0;
    PxU32 numSimVerts = 0;
    float restExtentY = 0.0f;   // the cube's undeformed y extent (its rest shape)
    float stretchY = 1.0f;      // the spawn-time elongation factor along y
    bool noGravity = false;     // PXG_SB_NOGRAVITY: pure constitutive relaxation
    PxCudaContextManager* cudaCtx = nullptr;
};

SoftBodyScene buildSoftBodyScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                 PxCudaContextManager* cudaCtx, float stretchY)
{
    SoftBodyScene out;
    out.cudaCtx = cudaCtx;

    // As in the cloth and particle scenes: the deformable solve runs inside the rigid
    // PGS island loop, and rigidSumInternalContactAndJointBatches2 null-derefs the
    // solver descriptors with zero dynamic rigids, so one live dynamic body is
    // required to keep that proven path valid.  It rests at y=0.5, far below.
    addGroundBox(physics, scene, rigidMat);
    addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
               boxInertia(PxVec3(0.5f), 1000.0f), 1.5f);

    // --- the tetrahedral cube -------------------------------------------------
    const PxU32 cells = 2;                        // 2x2x2 cells -> 27 verts, 40 tets
    const float size = 1.0f;                      // edge length of the whole cube
    const float h = size / float(cells);
    const PxU32 n = cells + 1;                    // verts per axis

    std::vector<PxVec3> tverts;
    tverts.reserve(n * n * n);
    for (PxU32 i = 0; i < n; ++i)
        for (PxU32 j = 0; j < n; ++j)
            for (PxU32 k = 0; k < n; ++k)
                tverts.push_back(PxVec3(float(i) * h - 0.5f * size,
                                        float(j) * h - 0.5f * size,
                                        float(k) * h - 0.5f * size));
    auto vid = [n](PxU32 i, PxU32 j, PxU32 k) { return (i * n + j) * n + k; };

    std::vector<PxU32> tets;
    tets.reserve(5 * 4 * cells * cells * cells);
    for (PxU32 i = 0; i < cells; ++i)
        for (PxU32 j = 0; j < cells; ++j)
            for (PxU32 k = 0; k < cells; ++k)
            {
                // cube corners, indexed by the (i,j,k) bit pattern
                const PxU32 c[8] = {
                    vid(i,     j,     k    ), vid(i + 1, j,     k    ),
                    vid(i,     j + 1, k    ), vid(i + 1, j + 1, k    ),
                    vid(i,     j,     k + 1), vid(i + 1, j,     k + 1),
                    vid(i,     j + 1, k + 1), vid(i + 1, j + 1, k + 1),
                };
                // ALTERNATING diagonals — neighbouring cells must disagree or the
                // shared face gets two different triangulations.
                static const PxU32 splitA[5][4] = {
                    {0,1,2,4}, {1,3,2,7}, {1,4,5,7}, {2,4,6,7}, {1,2,4,7}
                };
                static const PxU32 splitB[5][4] = {
                    {0,1,3,5}, {0,3,2,6}, {0,5,4,6}, {3,7,5,6}, {0,3,5,6}
                };
                const bool even = ((i + j + k) & 1u) == 0u;
                for (PxU32 t = 0; t < 5; ++t)
                {
                    PxU32 a = c[even ? splitA[t][0] : splitB[t][0]];
                    PxU32 b = c[even ? splitA[t][1] : splitB[t][1]];
                    PxU32 cc = c[even ? splitA[t][2] : splitB[t][2]];
                    PxU32 d = c[even ? splitA[t][3] : splitB[t][3]];
                    // Orient every tet positively: PhysX expects a non-degenerate,
                    // consistently wound tetrahedron, and the two splits do not
                    // agree on handedness.  Swapping any two corners flips the sign.
                    const PxVec3 e1 = tverts[b] - tverts[a];
                    const PxVec3 e2 = tverts[cc] - tverts[a];
                    const PxVec3 e3 = tverts[d] - tverts[a];
                    if (e1.cross(e2).dot(e3) < 0.0f) { const PxU32 tmp = cc; cc = b; b = tmp; }
                    tets.push_back(a); tets.push_back(b); tets.push_back(cc); tets.push_back(d);
                }
            }

    PxTetrahedronMeshDesc tetDesc;
    tetDesc.points.count = PxU32(tverts.size());
    tetDesc.points.stride = sizeof(PxVec3);
    tetDesc.points.data = tverts.data();
    tetDesc.tetrahedrons.count = PxU32(tets.size() / 4);
    tetDesc.tetrahedrons.stride = 4 * sizeof(PxU32);
    tetDesc.tetrahedrons.data = tets.data();

    // buildGPUData is MANDATORY — the GPU solve needs the BV32 tree that
    // sb_refitBoundLaunch (tranche 2) refits every frame.
    PxCookingParams ckParams(physics->getTolerancesScale());
    ckParams.buildGPUData = true;
    ckParams.meshWeldTolerance = 0.001f;
    ckParams.midphaseDesc = PxMeshMidPhase::eBVH34;

    // Simulation mesh == collision mesh; vertexToTet left unassigned so the cooker
    // computes the embedding itself.
    PxDeformableVolumeSimulationDataDesc simDataDesc;
    PxDeformableVolumeMesh* volMesh = immediateCooking::createDeformableVolumeMesh(
        ckParams, tetDesc, tetDesc, simDataDesc, physics->getPhysicsInsertionCallback());
    if (!volMesh) { std::printf("FAIL: immediateCooking::createDeformableVolumeMesh returned null\n"); return out; }

    // --- the actor ------------------------------------------------------------
    // elasticityDamping damps the CONSTITUTIVE response only, not the rigid-body
    // motion, so it leaves the free fall's closed form untouched while making the
    // stretched cube settle instead of ringing forever.  It also puts the tranche-6
    // solve on its DAMPED branch (the `1 + damping*dtInv` denominator plus the
    // scaledDamping*velocity term), which the undeformed case never reaches.
    PxDeformableVolumeMaterial* volMat =
        physics->createDeformableVolumeMaterial(/*youngs*/ 1.e5f, /*poissons*/ 0.3f,
                                                /*dynamicFriction*/ 0.5f,
                                                /*elasticityDamping*/ stretchY > 1.0f ? 0.05f : 0.0f);
    PxDeformableVolumeMaterial* mats[1] = { volMat };

    PxShapeFlags shapeFlags = PxShapeFlag::eSIMULATION_SHAPE | PxShapeFlag::eSCENE_QUERY_SHAPE;
    PxShape* shape = physics->createShape(PxTetrahedronMeshGeometry(volMesh->getCollisionMesh()),
                                          mats, PxU16(1), true, shapeFlags);
    if (!shape) { std::printf("FAIL: createShape (deformable volume) returned null\n"); return out; }

    PxDeformableVolume* volume = physics->createDeformableVolume(*cudaCtx);
    if (!volume) { std::printf("FAIL: createDeformableVolume returned null\n"); return out; }

    volume->attachShape(*shape);
    shape->release();
    volume->attachSimulationMesh(*volMesh->getSimulationMesh(), *volMesh->getDeformableVolumeAuxData());

    // Self-collision OFF: it would drag in the whole sb_selfCollision midphase +
    // contact-prep arc, none of which is mapped yet, and a solid cube has nothing to
    // self-collide with anyway.  This also puts the tranche-5 collision-mesh rotation
    // extractor on its skip path (it is gated on stress OR self-collision).
    volume->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
    // PXG_SB_NOGRAVITY isolates the CONSTITUTIVE response from the fall: with gravity
    // off, a cube spawned stretched relaxes toward its rest shape and nothing else
    // moves it, so the steady state is unambiguous.  It also puts the tranche-1
    // pre-integrate gravity gate (PxActorFlag::eDISABLE_GRAVITY) on its skip path --
    // the one branch its unit test could not give teeth to, because no case set it.
    out.noGravity = std::getenv("PXG_SB_NOGRAVITY") != nullptr;
    if (out.noGravity)
        volume->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
    volume->setLinearDamping(0.0f);          // cleanest free fall
    volume->setMaxLinearVelocity(1.0e6f);    // do not clamp it
    // PXG_SB_ITERS: XPBD position iterations.  Exposed because the stretched cube's
    // steady-state extent is an XPBD CONVERGENCE residual, not a fixed offset -- see
    // the note in RUNTIME_STATUS.md.  4 is the engine's usual default.
    // Called through PxDeformableBody: the same-named PxDeformableVolume override
    // (PxDeformableVolume.h:453) is PX_DEPRECATED and hides the live one.
    const char* itersEnv = std::getenv("PXG_SB_ITERS");
    static_cast<PxDeformableBody*>(volume)->setSolverIterationCounts(
        itersEnv ? PxU32(atoi(itersEnv)) : 4u, 1u);

    scene->addActor(*volume);

    // --- the host mirror ------------------------------------------------------
    // Mirrors PxDeformableVolumeExt::allocateAndInitializeHostMirror + transform +
    // copyToDevice (ExtDeformableVolumeExt.cpp), open-coded because physxextensions
    // is not linked into this harness.  The cube is spawned at spawnY with invMass
    // taken from the cooked aux data (uniform for a uniform mesh) and zero velocity,
    // so the first frames are a pure free fall with no contact at all.
    const float spawnY = 4.0f;
    const PxU32 nbSim = volume->getSimulationMesh()->getNbVertices();
    out.restExtentY = size;
    out.stretchY = stretchY;
    const PxU32 nbColl = volume->getCollisionMesh()->getNbVertices();

    std::vector<PxVec4> simPos(nbSim), simVel(nbSim), collPos(nbColl), restPos(nbColl);

    const PxVec3* simVerts = volume->getSimulationMesh()->getVertices();
    const PxReal* invMassGM = volume->getDeformableVolumeAuxData()->getGridModelInvMass();
    for (PxU32 v = 0; v < nbSim; ++v)
    {
        const PxReal im = invMassGM ? invMassGM[v] : 1.0f;
        // stretchY > 1 spawns the cube ELONGATED along y while its rest pose
        // (mSimTetraRestPoses, computed by the cooker from the undeformed mesh) still
        // describes the cube.  That is a genuine constitutive violation, so the FEM
        // solve has real work to do from frame one — see the scene header.
        simPos[v] = PxVec4(simVerts[v].x, simVerts[v].y * stretchY + spawnY, simVerts[v].z, im);
        simVel[v] = PxVec4(0.0f, 0.0f, 0.0f, im);
    }
    const PxVec3* collVerts = volume->getCollisionMesh()->getVertices();
    for (PxU32 v = 0; v < nbColl; ++v)
    {
        collPos[v] = PxVec4(collVerts[v].x, collVerts[v].y * stretchY + spawnY, collVerts[v].z, 1.0f);
        restPos[v] = collPos[v];
    }

    PxCudaContext* ctx = cudaCtx->getCudaContext();
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(volume->getSimPositionInvMassBufferD()),
                    simPos.data(), nbSim * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(volume->getSimVelocityBufferD()),
                    simVel.data(), nbSim * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(volume->getPositionInvMassBufferD()),
                    collPos.data(), nbColl * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(volume->getRestPositionBufferD()),
                    restPos.data(), nbColl * sizeof(PxVec4));
    volume->markDirty(PxDeformableVolumeDataFlag::eALL);

    // The skin-gap check below is only meaningful if the two buffers are genuinely
    // separate device allocations — if they aliased (as the CLOTH buffers do with the
    // engine's internal arrays) the gap would be identically zero for free.
    if (volume->getPositionInvMassBufferD() == volume->getSimPositionInvMassBufferD())
    {
        std::printf("FAIL: the collision and simulation position buffers ALIAS — the skin check "
                    "would have no teeth\n");
        return out;
    }

    out.volume = volume;
    out.numSimVerts = nbSim;
    out.numCollVerts = nbColl;
    std::printf("  softbody mesh: %u sim verts / %u tets, %u collision verts "
                "(coll buf %p, sim buf %p)\n",
                nbSim, volume->getSimulationMesh()->getNbTetrahedrons(), nbColl,
                static_cast<void*>(volume->getPositionInvMassBufferD()),
                static_cast<void*>(volume->getSimPositionInvMassBufferD()));
    return out;
}

// DtoH-read ONE of the two softbody vertex buffers and fill mean / min y.  `sim`
// picks the simulation mesh (what the solver moves) over the collision mesh (what
// sb_gm_updateTetModelVertsLaunch skins).  Returns false on any non-finite coord.
bool softBodyMeanMinY(const SoftBodyScene& sbs, bool sim, float& meanY, float& minY,
                      float* extentY = nullptr)
{
    const PxU32 nv = sim ? sbs.numSimVerts : sbs.numCollVerts;
    std::vector<PxVec4> host(nv);
    PxCudaContext* ctx = sbs.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(),
                    reinterpret_cast<CUdeviceptr>(sim ? sbs.volume->getSimPositionInvMassBufferD()
                                                      : sbs.volume->getPositionInvMassBufferD()),
                    nv * sizeof(PxVec4));
    double sum = 0.0;
    float lo = 1.0e30f, hi = -1.0e30f;
    bool finite = true;
    for (PxU32 i = 0; i < nv; ++i)
    {
        const float y = host[i].y;
        sum += double(y);
        lo = PxMin(lo, y);
        hi = PxMax(hi, y);
        if (!std::isfinite(y) || !std::isfinite(host[i].x) || !std::isfinite(host[i].z)) finite = false;
    }
    meanY = float(sum / double(nv));
    minY = lo;
    if (extentY) *extentY = hi - lo;
    return finite;
}

} // namespace

int main()
{
    StepAllocator alloc;
    StepError err;

    PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, alloc, err);
    if (!foundation) { std::printf("FAIL: PxCreateFoundation\n"); return 1; }

    // Route the engine's PxvGetPhysXGpu at the bridge (see the note above).
    physx::g_PxCreatePhysXGpu_Func = &PxCreatePhysXGpu;

    // Create the Vulkan bridge as the cudaContextManager.  This calls straight into
    // PhysXGpuVulkan.cpp's PxCreateCudaContextManager (extern "C"), which spins up the
    // Vulkan compute device + SPIR-V shim and wraps them in a PxgCudaContextManager.
    PxCudaContextManagerDesc cudaDesc;
    PxCudaContextManager* cudaCtx = PxCreateCudaContextManager(*foundation, cudaDesc, nullptr, false);
    if (!cudaCtx || !cudaCtx->contextIsValid())
    {
        std::printf("FAIL: Vulkan cudaContextManager not valid (GPU backend unavailable)\n");
        if (cudaCtx) cudaCtx->release();
        foundation->release();
        return 1;
    }
    std::printf("  Vulkan cudaContextManager created + valid (device '%s')\n",
                cudaCtx->getDeviceName());

    PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale(), false, nullptr);
    if (!physics) { std::printf("FAIL: PxCreatePhysics\n"); foundation->release(); return 1; }

    SyncDispatcher dispatcher;

    // GPU-enabled scene: hand it the Vulkan ccm and switch the dynamics + broadphase
    // to the GPU pipeline.  A GPU dispatcher is required alongside the cpuDispatcher.
    PxSceneDesc sceneDesc(physics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    sceneDesc.cpuDispatcher = &dispatcher;
    sceneDesc.filterShader = stepFilterShader;
    sceneDesc.cudaContextManager = cudaCtx;
    sceneDesc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
    sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
    // PXG_SOLVER=pgs switches the scene to the PGS solver (default TGS): the two
    // GPU solver paths launch DIFFERENT kernel sets (PxgCudaSolverCore vs
    // PxgTGSCudaSolverCore), so each scene can be proven on either.
    const char* solverEnv = std::getenv("PXG_SOLVER");
    const bool usePgs = solverEnv && !std::strcmp(solverEnv, "pgs");
    sceneDesc.solverType = usePgs ? PxSolverType::ePGS : PxSolverType::eTGS;
    // Keep the GPU memory footprint modest for a two-body scene.
    sceneDesc.gpuDynamicsConfig.maxRigidContactCount = 1024 * 4;
    sceneDesc.gpuDynamicsConfig.maxRigidPatchCount = 1024;

    std::printf("  creating GPU scene (eENABLE_GPU_DYNAMICS + eGPU broadphase, %s)...\n",
                usePgs ? "PGS" : "TGS");
    PxScene* scene = physics->createScene(sceneDesc);
    if (!scene)
    {
        std::printf("[ FAIL ] gpu-sdk-step: createScene returned null (GPU pipeline construction failed)\n");
        physics->release();
        cudaCtx->release();
        foundation->release();
        return 1;
    }
    std::printf("  [  OK  ] GPU scene constructed: the Pxg* pipeline is live over Vulkan\n");

    PxMaterial* material = physics->createMaterial(0.5f, 0.5f, 0.1f);

    // Scene selection: PXG_SCENE picks which rigid scene to build (default "box").
    // Each builder adds its actors and pushes the dynamic bodies it wants rest-checked.
    const char* sceneName = std::getenv("PXG_SCENE");
    if (!sceneName) sceneName = "box";

    // PXG_SCENE=particle (P6.0): the deformable/particle free-fall path.  Wholly
    // separate observable from the rigid RestTarget path — a PBD cloud that falls
    // under gravity, verified by DtoH-reading its mean y across steps.
    if (!std::strcmp(sceneName, "particle"))
    {
        ParticleScene psc = buildParticleScene(physics, scene, material, cudaCtx);
        if (!psc.system || !psc.buffer)
        {
            std::printf("[ FAIL ] gpu-sdk-step: particle scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'particle' built: %u PBD particles free-falling from y~100 on the GPU\n",
                    psc.numParticles);

        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: particle GPU pipeline CONSTRUCTED + live over Vulkan "
                        "(set PXG_GPU_STEP=1 to step the free fall)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_PARTICLE_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 120;
        const bool bagMode = std::getenv("PXG_PARTICLE_STARTY") != nullptr;
        const float y0 = particleMeanY(psc);
        std::printf("  stepping the particle cloud (%d x %.4fs), initial mean y=%.4f...\n", steps, dt, y0);

        float prevY = y0;
        int nonMonotonic = 0;
        float restMaxDy = 0.0f;   // max |dy| over the final quarter (rest convergence)
        const int restWindow = steps / 4;
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            const float y = particleMeanY(psc);
            if (i < 5 || i == steps - 1 || (bagMode && (i % 15 == 0)))
                std::printf("    step %3d: mean y = %.5f (dy = %+.5f)\n", i, y, y - prevY);
            // Free fall: mean y must not increase (allow a tiny float slack).
            if (y > prevY + 1e-3f) ++nonMonotonic;
            if (i >= steps - restWindow) restMaxDy = PxMax(restMaxDy, PxAbs(y - prevY));
            prevY = y;
        }
        const float yEnd = particleMeanY(psc);
        const float drop = y0 - yEnd;

        int failures = 0;
        if (bagMode)
        {
            // Settling bag (P6.5): the cloud must FALL, be CAUGHT on the floor, and
            // come to REST.  restMaxDy over the final quarter must decay to ~0, yEnd
            // must sit just above the floor (top y=0, particles rest near solidRest
            // offset), and everything must stay finite.
            std::printf("  particle SETTLING BAG: mean y %.4f -> %.4f (dropped %.4f over %.2fs); "
                        "rest |dy| over last %d steps = %.6f\n", y0, yEnd, drop, steps * dt, restWindow, restMaxDy);
            if (!(drop > 0.5f))          { std::printf("FAIL: cloud did not fall (drop=%.4f)\n", drop); ++failures; }
            if (!(restMaxDy < 0.01f))    { std::printf("FAIL: cloud did not come to rest (rest |dy|=%.6f)\n", restMaxDy); ++failures; }
            if (!(yEnd > -0.5f && yEnd < 1.5f)) { std::printf("FAIL: cloud did not settle near the floor (yEnd=%.4f)\n", yEnd); ++failures; }
            if (!std::isfinite(yEnd))    { std::printf("FAIL: mean y is not finite (%.4f)\n", yEnd); ++failures; }
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            if (failures == 0)
            {
                std::printf("[  OK  ] gpu-sdk-step: PBD particle bag fell, was caught by the floor, and "
                            "SETTLED TO REST on Vulkan (the money shot)\n");
                return 0;
            }
            std::printf("[ FAIL ] gpu-sdk-step: particle settling-bag checks failed (%d)\n", failures);
            return 2;
        }

        std::printf("  particle free fall: mean y %.4f -> %.4f (dropped %.4f over %.2fs)\n",
                    y0, yEnd, drop, steps * dt);
        if (!(yEnd < y0 - 0.05f)) { std::printf("FAIL: cloud did not fall (drop=%.4f)\n", drop); ++failures; }
        if (nonMonotonic != 0)    { std::printf("FAIL: mean y increased on %d step(s)\n", nonMonotonic); ++failures; }
        if (!std::isfinite(yEnd)) { std::printf("FAIL: mean y is not finite (%.4f)\n", yEnd); ++failures; }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: particle cloud fell monotonically under gravity on Vulkan "
                        "(the first live particle frame)\n");
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: particle free-fall checks failed (%d)\n", failures);
        return 2;
    }

    // PXG_SCENE=cloth (P7.0): the FEMCloth free-fall path — the exact analog of the
    // particle scene above.  A triangle-mesh cloth falls under gravity, verified by
    // DtoH-reading its mean vertex y across steps.
    if (!std::strcmp(sceneName, "cloth"))
    {
        ClothScene csc = buildClothScene(physics, scene, material, cudaCtx);
        if (!csc.surface || csc.numVerts == 0)
        {
            std::printf("[ FAIL ] gpu-sdk-step: cloth scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'cloth' built: %u-vertex FEMCloth free-falling from y~5 on the GPU\n",
                    csc.numVerts);

        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: cloth GPU pipeline CONSTRUCTED + live over Vulkan "
                        "(set PXG_GPU_STEP=1 to step the free fall)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_CLOTH_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 60;
        const float y0 = clothMeanY(csc);
        std::printf("  stepping the cloth (%d x %.4fs), initial mean y=%.4f...\n", steps, dt, y0);

        float prevY = y0;
        int nonMonotonic = 0;
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            const float y = clothMeanY(csc);
            if (i < 5 || i == steps - 1 || (i % 10 == 0))
                std::printf("    step %3d: mean y = %.5f (dy = %+.5f)\n", i, y, y - prevY);
            if (y > prevY + 1e-3f) ++nonMonotonic;   // free fall: must not rise
            prevY = y;
        }
        const float yEnd = clothMeanY(csc);
        const float drop = y0 - yEnd;

        int failures = 0;
        std::printf("  cloth free fall: mean y %.4f -> %.4f (dropped %.4f over %.2fs)\n",
                    y0, yEnd, drop, steps * dt);
        if (!(yEnd < y0 - 0.05f)) { std::printf("FAIL: cloth did not fall (drop=%.4f)\n", drop); ++failures; }
        if (nonMonotonic != 0)    { std::printf("FAIL: mean y increased on %d step(s)\n", nonMonotonic); ++failures; }
        if (!std::isfinite(yEnd)) { std::printf("FAIL: mean y is not finite (%.4f)\n", yEnd); ++failures; }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: FEMCloth fell monotonically under gravity on Vulkan "
                        "(the first live FEMCloth frame)\n");
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: cloth free-fall checks failed (%d)\n", failures);
        return 2;
    }

    // PXG_SCENE=clothhang (P7.1 tranche 3b): the HANGING / draping cloth money shot.
    // A vertical FEMCloth curtain pinned along its top edge to fixed world anchors; it
    // must DRAPE below the pins and SETTLE (fall-then-stabilize), mirroring the P6.5
    // particle settling-bag pass criterion but per-row: pinned top row ~constant y, free
    // bottom row drops a bounded amount then converges.
    if (!std::strcmp(sceneName, "clothhang"))
    {
        ClothHangScene csc = buildClothHangScene(physics, scene, material, cudaCtx);
        if (!csc.surface || csc.numVerts == 0 || !csc.attachment)
        {
            std::printf("[ FAIL ] gpu-sdk-step: clothhang scene build failed "
                        "(surface=%p verts=%u attachment=%p)\n",
                        (void*)csc.surface, csc.numVerts, (void*)csc.attachment);
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'clothhang' built: %u-vertex FEMCloth curtain, top row (%zu verts) "
                    "PINNED to world anchors\n", csc.numVerts, csc.topRow.size());

        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: clothhang GPU pipeline CONSTRUCTED + live over Vulkan "
                        "(set PXG_GPU_STEP=1 to step the drape)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_CLOTH_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 120;
        float top0 = 0.0f, bot0 = 0.0f;
        bool finite0 = clothHangRowMeans(csc, top0, bot0);
        std::printf("  stepping the curtain (%d x %.4fs): top row y0=%.4f, bottom row y0=%.4f\n",
                    steps, dt, top0, bot0);

        float prevBot = bot0;
        float topMaxDev = 0.0f;         // max |top mean y - top0| over the whole run
        float botRestMaxDy = 0.0f;      // max |d(bottom mean y)| over the final quarter (convergence)
        bool allFinite = finite0;
        const int restWindow = steps / 4;
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            float top = 0.0f, bot = 0.0f;
            allFinite = clothHangRowMeans(csc, top, bot) && allFinite;
            topMaxDev = PxMax(topMaxDev, PxAbs(top - top0));
            if (i >= steps - restWindow) botRestMaxDy = PxMax(botRestMaxDy, PxAbs(bot - prevBot));
            if (i < 6 || i == steps - 1 || (i % 10 == 0))
                std::printf("    step %3d: top y = %.5f (dev %+.5f) | bottom y = %.5f (dy %+.5f)\n",
                            i, top, top - top0, bot, bot - prevBot);
            prevBot = bot;
        }
        float topEnd = 0.0f, botEnd = 0.0f;
        clothHangRowMeans(csc, topEnd, botEnd);
        const float botDrop = bot0 - botEnd;

        std::printf("  clothhang: top row y %.4f -> %.4f (max |dev|=%.5f) | bottom row y %.4f -> %.4f "
                    "(dropped %.4f); bottom rest |dy| over last %d steps = %.6f\n",
                    top0, topEnd, topMaxDev, bot0, botEnd, botDrop, restWindow, botRestMaxDy);

        int failures = 0;
        // (1) the PINNED top row must stay put (near its anchor y).
        if (!(topMaxDev < 0.05f)) { std::printf("FAIL: pinned top row moved (max |dev|=%.5f)\n", topMaxDev); ++failures; }
        // (2) the free bottom row must DRAPE below by a BOUNDED amount (not free-fall).
        if (!(botDrop > 0.0f))    { std::printf("FAIL: bottom row did not drape down (drop=%.5f)\n", botDrop); ++failures; }
        if (!(botDrop < 2.0f))    { std::printf("FAIL: bottom row fell unbounded (drop=%.5f) — not hanging\n", botDrop); ++failures; }
        // (3) the drape must CONVERGE (bottom row stops moving over the final quarter).
        if (!(botRestMaxDy < 0.01f)) { std::printf("FAIL: drape did not settle (rest |dy|=%.6f)\n", botRestMaxDy); ++failures; }
        // (4) nothing NaN/inf.
        if (!allFinite || !std::isfinite(topEnd) || !std::isfinite(botEnd))
        { std::printf("FAIL: cloth y went non-finite\n"); ++failures; }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: FEMCloth HANGS from its pinned edge and drapes under gravity\n");
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: clothhang drape checks failed (%d)\n", failures);
        return 2;
    }

    // PXG_SCENE=clothdrape (P7.1 tranche 4c): the cloth-vs-rigid CONTACT money shot.
    // A horizontal FEMCloth sheet dropped above a STATIC plane floor with NO pins; the
    // floor contact chain must catch it and it must SETTLE resting on the floor (mean y
    // drops then converges to ~floorY), NOT tunnel to -inf and NOT free-fall through.
    if (!std::strcmp(sceneName, "clothdrape"))
    {
        const ClothDrapeConfig cfg;   // floorY = 0.0, spawnY = 1.5
        ClothScene csc = buildClothDrapeScene(physics, scene, material, cudaCtx, cfg);
        if (!csc.surface || csc.numVerts == 0)
        {
            std::printf("[ FAIL ] gpu-sdk-step: clothdrape scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'clothdrape' built: %u-vertex FEMCloth sheet spawned at y~%.2f above a "
                    "STATIC plane floor at y=%.2f (no pins)\n", csc.numVerts, cfg.spawnY, cfg.floorY);

        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: clothdrape GPU pipeline CONSTRUCTED + live over Vulkan "
                        "(set PXG_GPU_STEP=1 to step the drop)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_CLOTH_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 180;
        float mean0 = 0.0f, min0 = 0.0f;
        bool finite0 = clothMeanMinY(csc, mean0, min0);
        std::printf("  stepping the drop (%d x %.4fs): initial mean y=%.4f (min %.4f), floor y=%.2f...\n",
                    steps, dt, mean0, min0, cfg.floorY);

        float prevMean = mean0;
        float restMaxDy = 0.0f;         // max |d(mean y)| over the final quarter (convergence)
        bool allFinite = finite0;
        const int restWindow = steps / 4;
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            float mean = 0.0f, mn = 0.0f;
            allFinite = clothMeanMinY(csc, mean, mn) && allFinite;
            if (i >= steps - restWindow) restMaxDy = PxMax(restMaxDy, PxAbs(mean - prevMean));
            if (i < 6 || i == steps - 1 || (i % 10 == 0))
                std::printf("    step %3d: mean y = %.5f (dy %+.5f) | min y = %.5f\n",
                            i, mean, mean - prevMean, mn);
            prevMean = mean;
        }
        float meanEnd = 0.0f, minEnd = 0.0f;
        clothMeanMinY(csc, meanEnd, minEnd);
        const float drop = mean0 - meanEnd;

        std::printf("  clothdrape: mean y %.4f -> %.4f (dropped %.4f over %.2fs) | final min y = %.4f; "
                    "rest |dy| over last %d steps = %.6f\n",
                    mean0, meanEnd, drop, steps * dt, minEnd, restWindow, restMaxDy);

        int failures = 0;
        // (1) the cloth must FALL a real distance under gravity (not stuck at spawn).
        if (!(drop > 0.5f))          { std::printf("FAIL: cloth did not fall (drop=%.4f)\n", drop); ++failures; }
        // (2) the drop must CONVERGE — the sheet stops moving over the final quarter.
        if (!(restMaxDy < 0.01f))    { std::printf("FAIL: cloth did not settle (rest |dy|=%.6f)\n", restMaxDy); ++failures; }
        // (3) it must settle ON the floor: mean near floorY and NOT tunnelled below it.
        if (!(meanEnd < cfg.floorY + 0.5f)) { std::printf("FAIL: cloth did not reach the floor (meanY=%.4f)\n", meanEnd); ++failures; }
        if (!(minEnd > cfg.floorY - 0.1f))  { std::printf("FAIL: cloth tunnelled below the floor (minY=%.4f)\n", minEnd); ++failures; }
        // (4) nothing NaN/inf.
        if (!allFinite || !std::isfinite(meanEnd)) { std::printf("FAIL: cloth y went non-finite\n"); ++failures; }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: FEMCloth fell onto the floor and SETTLED under contact\n");
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: clothdrape settle checks failed (%d)\n", failures);
        return 2;
    }

    // PXG_SCENE=clothselfcollide (P7.2 tranche 3): the cloth SELF-collision money shot.
    // A hairpin-folded cloth whose LOWER layer is a fixed shelf (invMass 0) and whose UPPER
    // layer (invMass 1) drapes down onto it under gravity.  With self-collision ON
    // (PXG_SELFCOLL=1, default) the upper layer must be CAUGHT on the shelf (min y ~ baseY);
    // with it OFF (PXG_SELFCOLL=0, the control) it droops straight THROUGH (min y well below
    // baseY).  There is no rigid body in the loop, so self-collision is the ONLY thing that
    // can catch it.  Cloth runs on PGS — pass PXG_SOLVER=pgs.  Run the binary twice to see
    // the A/B difference (the verification does exactly that).
    if (!std::strcmp(sceneName, "clothselfcollide"))
    {
        const char* scEnv = std::getenv("PXG_SELFCOLL");
        const bool selfOn = !scEnv || scEnv[0] != '0';

        ClothSelfScene csc = buildClothSelfCollideScene(physics, scene, material, cudaCtx, selfOn);
        if (!csc.surface || csc.numVerts == 0)
        {
            std::printf("[ FAIL ] gpu-sdk-step: clothselfcollide scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'clothselfcollide' built: %u-vertex hairpin-folded FEMCloth (upper layer "
                    "drapes onto a fixed shelf at y=%.2f), self-collision %s\n",
                    csc.numVerts, csc.baseY, selfOn ? "ON" : "OFF (control)");

        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: clothselfcollide GPU pipeline CONSTRUCTED + live over Vulkan "
                        "(set PXG_GPU_STEP=1 to step the drape)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_CLOTH_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 200;
        float min0 = 0.0f, mean0 = 0.0f;
        bool finite = clothTopMinMeanY(csc, min0, mean0);
        float maxExcursion = 0.0f;   // largest |mean - mean0| seen at ANY step
        std::printf("  stepping the drape (%d x %.4fs, self-collision %s): upper layer starts at mean y=%.4f "
                    "(min %.4f); shelf y=%.2f...\n", steps, dt, selfOn ? "ON" : "OFF", mean0, min0, csc.baseY);

        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            float mn = 0.0f, mean = 0.0f;
            finite = clothTopMinMeanY(csc, mn, mean) & finite;
            const float excursion = PxAbs(mean - mean0);
            if (excursion > maxExcursion) maxExcursion = excursion;
            if (i < 6 || i == steps - 1 || (i % 20 == 0))
                std::printf("    step %3d: upper layer min y=%.5f mean=%.5f (shelf y=%.2f)\n",
                            i, mn, mean, csc.baseY);
        }
        float minE = 0.0f, meanE = 0.0f;
        clothTopMinMeanY(csc, minE, meanE);

        // PXG_CLOTH_DUMP=1: per-vertex y of the free layer — how a droop localises (a
        // border vertex that lost its contact band vs a whole layer falling) is invisible
        // in min/mean alone.
        if (std::getenv("PXG_CLOTH_DUMP"))
        {
            std::vector<PxVec4> hostD(csc.numVerts);
            csc.cudaCtx->getCudaContext()->memcpyDtoH(hostD.data(),
                reinterpret_cast<CUdeviceptr>(csc.surface->getPositionInvMassBufferD()),
                csc.numVerts * sizeof(PxVec4));
            std::printf("  [dump] upper layer y by index:\n   ");
            for (size_t i = 0; i < csc.topIdx.size(); ++i)
            {
                std::printf(" v%2u=%.3f", csc.topIdx[i], hostD[csc.topIdx[i]].y);
                if ((i % 5) == 4) std::printf("\n   ");
            }
            std::printf("\n");
        }
        std::printf("  clothselfcollide (self-collision %s): final upper layer min y=%.4f (mean %.4f); shelf y=%.2f\n",
                    selfOn ? "ON" : "OFF", minE, meanE, csc.baseY);

        int failures = 0;
        // The layer must MOVE — it is dropped from 0.30 above the shelf onto a rest
        // distance of 0.10, so a working run travels ~0.19 in the mean.  Measured over the
        // whole run rather than start-vs-end, because a good catch settles back close to
        // where it was caught.
        if (!(maxExcursion > 0.05f))
        { std::printf("FAIL: upper layer never moved (max mean excursion %.5f from %.4f) — sim frozen?\n",
                      maxExcursion, mean0); ++failures; }
        if (!finite) { std::printf("FAIL: cloth y went non-finite\n"); ++failures; }
        if (selfOn)
        {
            // Self-collision ON: the layer is CAUGHT and comes to rest exactly ON the
            // contact rest distance (shelf + 2 * restOffset = baseY + 0.10).  This is a much
            // sharper statement than the old "it stayed roughly up there": the run is now
            // fully DETERMINISTIC (bit-identical over 5 runs: min 5.0999, mean 5.1105),
            // because the mDeltaPos accumulators use real atomics and the scene no longer
            // hands the solve an artificial penetration deficit to overshoot.
            const float restBand = csc.baseY + 0.10f;
            if (!(PxAbs(minE - restBand) < 0.03f))
            { std::printf("FAIL: self-collision ON but the layer did not settle ON the rest "
                          "distance (min y=%.4f, expected %.4f +/- 0.03)\n", minE, restBand); ++failures; }
            if (!(meanE > csc.baseY + 0.05f))
            { std::printf("FAIL: self-collision ON but the upper layer sank toward the shelf "
                          "(mean y=%.4f, shelf=%.2f)\n", meanE, csc.baseY); ++failures; }
        }
        else
        {
            // Control: without self-collision nothing can catch the layer and it drops far
            // below the shelf.  Measured 3.118 / 3.875; both thresholds clear the ON case by
            // more than a full unit, so each separates ON from OFF on its own.
            if (!(minE < csc.baseY - 1.0f))
            { std::printf("FAIL: self-collision OFF but the upper layer stayed near the shelf "
                          "(min y=%.4f, shelf=%.2f) — control invalid\n", minE, csc.baseY); ++failures; }
            if (!(meanE < csc.baseY - 0.5f))
            { std::printf("FAIL: self-collision OFF but the upper layer still rested on the shelf "
                          "(mean y=%.4f, shelf=%.2f) — control invalid\n", meanE, csc.baseY); ++failures; }
        }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        if (failures == 0)
        {
            if (selfOn)
                std::printf("[  OK  ] gpu-sdk-step: FEMCloth self-collision CAUGHT the falling layer and "
                            "settled it ON the contact rest distance (min y=%.4f vs %.4f expected, "
                            "mean y=%.4f, shelf=%.2f)\n", minE, csc.baseY + 0.10f, meanE, csc.baseY);
            else
                std::printf("[  OK  ] gpu-sdk-step: control run — WITHOUT self-collision the upper layer drooped "
                            "through the shelf (min y=%.4f, shelf=%.2f)\n", minE, csc.baseY);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: clothselfcollide checks failed (%d)\n", failures);
        return 2;
    }

    // PXG_SCENE=softbody (P8.0 tranche 7): the FEM-VOLUME money shot — a tetrahedral cube
    // free-falling under gravity.  The pass criterion is deliberately read off the
    // COLLISION mesh, which only moves because sb_gm_updateTetModelVertsLaunch skins it
    // onto the simulation tetrahedra; both meshes are reported every step so a divergence
    // between them localises straight to that kernel.  Softbody runs on PGS — pass
    // PXG_SOLVER=pgs.
    if (!std::strcmp(sceneName, "softbody"))
    {
        // PXG_SB_STRETCH elongates the cube along y at spawn while leaving its REST pose
        // (which the cooker computed from the undeformed mesh) alone.  With the default
        // 1.0 the cube is spawned at rest and the constitutive solve produces a ZERO
        // delta all the way down, so the fall is a pure integrate test.  With a value > 1
        // the FEM solve has to pull the cube back toward its rest shape while it falls,
        // which is the only way this scene exercises the tranche-6 constraint solve and
        // the tranche-7 partition-copy MERGE at all -- see the extra check below.
        const char* stretchEnv = std::getenv("PXG_SB_STRETCH");
        const float stretchY = stretchEnv ? float(atof(stretchEnv)) : 1.0f;

        SoftBodyScene sbs = buildSoftBodyScene(physics, scene, material, cudaCtx, stretchY);
        if (!sbs.volume || sbs.numCollVerts == 0)
        {
            std::printf("[ FAIL ] gpu-sdk-step: softbody scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'softbody' built: %u-vertex FEM-volume cube at y~4 on the GPU "
                    "(%s, spawn stretch %.2fx)\n", sbs.numCollVerts,
                    sbs.noGravity ? "gravity OFF" : "free-falling", sbs.stretchY);

        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: softbody GPU pipeline CONSTRUCTED + live over Vulkan "
                        "(set PXG_GPU_STEP=1 to step the free fall)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_SB_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 60;

        float meanC0 = 0.0f, minC0 = 0.0f, meanS0 = 0.0f, minS0 = 0.0f, extS0 = 0.0f;
        bool finite = softBodyMeanMinY(sbs, /*sim*/ false, meanC0, minC0);
        finite = softBodyMeanMinY(sbs, /*sim*/ true, meanS0, minS0, &extS0) && finite;
        std::printf("  stepping the softbody (%d x %.4fs), initial coll mean y=%.4f / sim mean y=%.4f, "
                    "sim y-extent %.4f (rest %.4f)...\n",
                    steps, dt, meanC0, meanS0, extS0, sbs.restExtentY);

        float prevC = meanC0;
        int nonMonotonic = 0;
        float meanC = meanC0, minC = minC0, meanS = meanS0, minS = minS0, extS = extS0;
        float maxSkinGap = 0.0f;
        float extSettleRef = extS0;
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            finite = softBodyMeanMinY(sbs, /*sim*/ false, meanC, minC) && finite;
            finite = softBodyMeanMinY(sbs, /*sim*/ true, meanS, minS, &extS) && finite;
            // The collision mesh is a barycentric skin over the sim tets, so for a rigid
            // free fall the two means must track each other closely.  A growing gap is the
            // skin kernel failing to keep up (or not running at all).
            maxSkinGap = PxMax(maxSkinGap, PxAbs(meanC - meanS));
            if (i == steps - 21) extSettleRef = extS;   // 20 steps before the end
            if (i < 5 || i == steps - 1 || (i % 10 == 0))
                std::printf("    step %3d: coll mean y = %.5f (dy = %+.5f), sim mean y = %.5f, "
                            "skin gap = %.5f, sim y-extent = %.5f\n",
                            i, meanC, meanC - prevC, meanS, PxAbs(meanC - meanS), extS);
            if (meanC > prevC + 1e-3f) ++nonMonotonic;   // free fall: must not rise
            prevC = meanC;
        }
        const float drop = meanC0 - meanC;

        int failures = 0;
        // The engine integrates gravity semi-implicitly (v += g*dt; x += v*dt), so after n
        // steps a contact-free body has dropped exactly g*dt^2*n(n+1)/2 -- NOT the
        // continuous 0.5*g*t^2.  Used twice below: as the closed form the undeformed fall
        // must match, and as the yardstick the no-gravity run must come nowhere near.
        const float gAccel = 9.81f;
        const float expectDrop = gAccel * dt * dt * float(steps) * float(steps + 1) * 0.5f;
        std::printf("  softbody free fall: coll mean y %.4f -> %.4f (dropped %.4f over %.2fs), "
                    "sim mean y %.4f -> %.4f\n", meanC0, meanC, drop, steps * dt, meanS0, meanS);
        if (sbs.noGravity)
        {
            // PXG_SB_NOGRAVITY puts the tranche-1 pre-integrate gravity gate
            // (PxActorFlag::eDISABLE_GRAVITY) on its SKIP path -- the one branch its unit
            // test could not give teeth to, because no case set the flag.  If the gate did
            // not take, the cube falls and this fires.
            // Budgeted against what gravity WOULD have done, not against zero: a stretched
            // cube's relaxation shifts its centroid slightly (the alternating 5-tet split is
            // not mirror-symmetric about the mid-plane, and the orientation flips break what
            // symmetry remains).  Measured at 120 steps: -0.018 against a 19.8 free fall, so
            // a 2% budget still catches a gravity leak by a factor of ~100.
            if (!(PxAbs(drop) < 0.02f * expectDrop))
            { std::printf("FAIL: PXG_SB_NOGRAVITY is set but the cube still moved %.5f in y "
                          "(gravity would have moved it %.5f) — the pre-integrate gravity gate "
                          "did not take\n", drop, expectDrop); ++failures; }
        }
        else if (!(meanC < meanC0 - 0.05f))
        { std::printf("FAIL: softbody collision mesh did not fall (drop=%.4f)\n", drop); ++failures; }
        // QUANTITATIVE free fall: checking the closed form rather than "it went down"
        // gives the scene teeth on the WHOLE integrate chain (sb_gm_preIntegrateLaunch's
        // gravity predict, sb_gm_stepSoftbodyLaunch, sb_gm_finalizeVelocitiesLaunch)
        // instead of only on its sign.  The band is 1%; the measured error is 0.0000%.
        if (sbs.stretchY != 1.0f || sbs.noGravity)
            std::printf("  (closed-form free-fall check skipped: %s)\n",
                        sbs.noGravity ? "gravity is disabled"
                                      : "the cube was spawned stretched, so its elastic response "
                                        "moves the mean too");
        else if (!(PxAbs(drop - expectDrop) < 0.01f * expectDrop))
        { std::printf("FAIL: free fall does not match semi-implicit Euler: dropped %.5f, "
                      "expected %.5f (g*dt^2*n(n+1)/2)\n", drop, expectDrop); ++failures; }
        else
            std::printf("  free fall matches semi-implicit Euler: %.5f vs %.5f expected "
                        "(%.4f%% off)\n", drop, expectDrop,
                        100.0f * PxAbs(drop - expectDrop) / expectDrop);
        if (!sbs.noGravity && !(meanS < meanS0 - 0.05f))
        { std::printf("FAIL: softbody simulation mesh did not fall (%.4f -> %.4f)\n", meanS0, meanS); ++failures; }
        // Monotonicity and the closed form are INTEGRATE assertions: they hold only when
        // the constitutive solve contributes nothing.  A cube spawned deformed is pulled
        // by its own elastic response as well as by gravity, and that legitimately moves
        // the mean both ways -- so these two are gated on the undeformed case.
        if (sbs.stretchY == 1.0f && !sbs.noGravity && nonMonotonic != 0)
        { std::printf("FAIL: collision mean y increased on %d step(s)\n", nonMonotonic); ++failures; }
        if (!finite)
        { std::printf("FAIL: softbody y went non-finite\n"); ++failures; }
        // THE tranche-7 assertion: the skin must track the solve.  The cube is spawned
        // undeformed and falls without contact, so the two meshes are the same point set;
        // any drift beyond a small band means updateTetModelVerts is not doing its job.
        if (!(maxSkinGap < 0.05f))
        { std::printf("FAIL: collision mesh drifted from the simulation mesh (max gap %.5f) — "
                      "the barycentric skin is not tracking the solve\n", maxSkinGap); ++failures; }

        // THE tranche-6 / partition-merge assertion, only meaningful when the cube was
        // spawned deformed.  Gravity is uniform, so it can never change the cube's
        // EXTENT — only the constitutive solve can.  A cube spawned 1.6x too tall must
        // therefore contract measurably toward its rest height, and it must not blow up.
        if (sbs.stretchY > 1.0f)
        {
            const float spawnExtent = sbs.restExtentY * sbs.stretchY;
            const float recovered = (spawnExtent - extS) / (spawnExtent - sbs.restExtentY);
            std::printf("  sim y-extent: %.5f -> %.5f (rest %.5f, spawned %.5f) — the FEM solve "
                        "recovered %.1f%% of the spawn stretch\n",
                        extS0, extS, sbs.restExtentY, spawnExtent, 100.0f * recovered);

            // Whatever else happens, the solve must never ADD stretch.
            if (!(extS < spawnExtent * 1.05f))
            { std::printf("FAIL: the stretched cube EXPANDED instead of contracting "
                          "(%.5f -> %.5f) — the solve is unstable\n", spawnExtent, extS); ++failures; }

            if (sbs.noGravity)
            {
                // The CLEAN constitutive test: nothing but the solve can move the cube, so it
                // must relax essentially all the way back and then HOLD STILL.  Measured over
                // 120 steps: the extent converges to 1.03481 (94.2% recovered) by step ~60 and
                // is then constant to five decimals.  The 3.5% residual is XPBD's finite
                // compliance, not drift — 4x the iterations barely moves it (1.128 -> 1.109 in
                // the falling case at step 119).
                if (!(recovered > 0.85f))
                { std::printf("FAIL: with gravity off the solve must relax the cube nearly all "
                              "the way back (only %.1f%%)\n", 100.0f * recovered); ++failures; }
                if (!(PxAbs(extS - extSettleRef) < 1.0e-3f))
                { std::printf("FAIL: the cube never SETTLED — y-extent still moving over the last "
                              "20 steps (%.5f -> %.5f)\n", extSettleRef, extS); ++failures; }
                else
                    std::printf("  and it SETTLED: y-extent constant to %.1e over the last 20 steps\n",
                                PxAbs(extS - extSettleRef));
            }
            else
            {
                // WITH gravity the equilibrium creeps outward as the fall speeds up: the
                // constitutive damping term couples to the vertices' ABSOLUTE velocity, and in
                // free fall that grows without bound.  Measured: the extent relaxes to ~1.034
                // by step 20 then creeps to ~1.13 by step 120, decelerating.  That is engine
                // behaviour, not a port defect — the PXG_SB_NOGRAVITY run isolates the same
                // solve and shows it dead steady.  So the falling case only asserts that a
                // substantial contraction happened at all.
                if (!(recovered > 0.10f))
                { std::printf("FAIL: the FEM solve did not contract the stretched cube "
                              "(extent %.5f -> %.5f, %.1f%% of the way back to rest %.5f)\n",
                              extS0, extS, 100.0f * recovered, sbs.restExtentY); ++failures; }
            }
        }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        if (failures == 0)
        {
            if (sbs.noGravity)
                std::printf("[  OK  ] gpu-sdk-step: FEM-volume softbody RELAXED from a %.2fx stretch back "
                            "to within %.1f%% of its rest shape and held there — the constitutive solve "
                            "converges on Vulkan\n", sbs.stretchY,
                            100.0f * (extS / sbs.restExtentY - 1.0f));
            else if (sbs.stretchY > 1.0f)
                std::printf("[  OK  ] gpu-sdk-step: FEM-volume softbody fell under gravity on Vulkan while "
                            "the solve pulled it back from a %.2fx stretch, collision mesh skinned to the "
                            "solve (max gap %.5f)\n", sbs.stretchY, maxSkinGap);
            else
                std::printf("[  OK  ] gpu-sdk-step: FEM-volume softbody fell monotonically under gravity on "
                            "Vulkan, collision mesh skinned to the solve (max gap %.5f) — the first live "
                            "softbody frame\n", maxSkinGap);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: softbody free-fall checks failed (%d)\n", failures);
        return 2;
    }

    std::vector<RestTarget> targets;
    if      (!std::strcmp(sceneName, "box"))       buildBoxScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "stack"))     buildStackScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "plane"))     buildPlaneScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "sphere"))    buildSphereScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "capsule"))   buildCapsuleScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "capsulebox")) buildCapsuleBoxScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "convex"))    buildConvexScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "joint"))     buildJointScene(physics, scene, material, targets);
    else { std::printf("FAIL: unknown PXG_SCENE='%s' (box|stack|plane|sphere|capsule|capsulebox|convex|joint|particle|cloth|clothhang|clothdrape|clothselfcollide|softbody)\n", sceneName);
           scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1; }
    if (targets.empty())
    {
        std::printf("[ FAIL ] gpu-sdk-step: scene '%s' built no dynamic bodies\n", sceneName);
        scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
    }
    std::printf("  [  OK  ] scene '%s' built: %zu dynamic body(ies) + ground on the GPU\n",
                sceneName, targets.size());

    // The asserted milestone is a CONSTRUCTED, live GPU pipeline (above).  Actually
    // STEPPING it is RUNTIME_PLAN.md P5 Step 4 — a per-kernel launch-ABI bring-up that
    // is only partly done.  Kernels are converted to the BDA launch model one at a time
    // and each is verified in isolation by the gpuslang suite (e.g. MemCopyBalanced).
    // A whole-scene step can only succeed once the ENTIRE rigid chain is up: a kernel
    // that launches while its upstream producers do not reads garbage device memory and
    // its BDA dereferences hit wild addresses (a GPU fault, not a host-catchable error).
    // So stepping stays behind PXG_GPU_STEP until the chain is complete; the default
    // test verifies construction and exits clean.
    if (!std::getenv("PXG_GPU_STEP"))
    {
        std::printf("[  OK  ] gpu-sdk-step: GPU pipeline CONSTRUCTED + live over Vulkan "
                    "(set PXG_GPU_STEP=1 to attempt stepping — P5 Step 4, WIP)\n");
        scene->release();
        physics->release();
        cudaCtx->release();
        foundation->release();
        return 0;
    }

    const float dt = 1.0f / 60.0f;
    const int steps = 240;
    std::vector<float> yAfterOne(targets.size());
    for (size_t t = 0; t < targets.size(); ++t) yAfterOne[t] = targets[t].startY;
    std::printf("  stepping the GPU scene '%s' (%d x %.4fs)...\n", sceneName, steps, dt);
    for (int i = 0; i < steps; ++i)
    {
        scene->simulate(dt);
        scene->fetchResults(true);
        if (i == 0)
            for (size_t t = 0; t < targets.size(); ++t)
                yAfterOne[t] = targets[t].body->getGlobalPose().p.y;
    }

    // Per-body rest checks: began falling, reached its resting band (restY within a
    // generous tolerance — stacks settle with some jitter), did not sink through, at rest.
    int failures = 0;
    for (size_t t = 0; t < targets.size(); ++t)
    {
        const RestTarget& tg = targets[t];
        const float finalY = tg.body->getGlobalPose().p.y;
        const float speed = tg.body->getLinearVelocity().magnitude();
        std::printf("  [%s] start y=%.3f, y after 1 step=%.4f, final y=%.4f (rest %.3f), |v|=%.4f\n",
                    tg.label, tg.startY, yAfterOne[t], finalY, tg.restY, speed);
        if (!(yAfterOne[t] < tg.startY)) { std::printf("FAIL: [%s] did not begin falling\n", tg.label); ++failures; }
        if (!(finalY < tg.restY + 0.75f)) { std::printf("FAIL: [%s] above rest band (finalY=%.3f)\n", tg.label, finalY); ++failures; }
        if (!(finalY > tg.restY - 0.75f)) { std::printf("FAIL: [%s] below rest band / sank (finalY=%.3f)\n", tg.label, finalY); ++failures; }
        if (!(speed < 0.75f)) { std::printf("FAIL: [%s] not at rest (|v|=%.3f)\n", tg.label, speed); ++failures; }
    }

    scene->release();
    physics->release();
    cudaCtx->release();
    foundation->release();

    if (failures == 0)
    {
        std::printf("[  OK  ] gpu-sdk-step: scene '%s' stepped on the Vulkan GPU pipeline "
                    "(all %zu body(ies) fell + rested)\n", sceneName, targets.size());
        return 0;
    }
    std::printf("[ PARTIAL ] gpu-sdk-step: scene '%s' constructed; %d step check(s) failed "
                "(kernel-launch marshalling frontier)\n", sceneName, failures);
    return 2;
}
