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

    std::vector<RestTarget> targets;
    if      (!std::strcmp(sceneName, "box"))       buildBoxScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "stack"))     buildStackScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "plane"))     buildPlaneScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "sphere"))    buildSphereScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "capsule"))   buildCapsuleScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "capsulebox")) buildCapsuleBoxScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "convex"))    buildConvexScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "joint"))     buildJointScene(physics, scene, material, targets);
    else { std::printf("FAIL: unknown PXG_SCENE='%s' (box|stack|plane|sphere|capsule|capsulebox|convex|joint|particle|cloth|clothhang)\n", sceneName);
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
