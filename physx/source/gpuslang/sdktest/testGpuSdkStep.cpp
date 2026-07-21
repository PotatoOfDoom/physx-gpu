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
    std::vector<RestTarget> targets;
    if      (!std::strcmp(sceneName, "box"))       buildBoxScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "stack"))     buildStackScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "plane"))     buildPlaneScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "sphere"))    buildSphereScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "capsule"))   buildCapsuleScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "capsulebox")) buildCapsuleBoxScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "convex"))    buildConvexScene(physics, scene, material, targets);
    else { std::printf("FAIL: unknown PXG_SCENE='%s' (box|stack|plane|sphere|capsule|capsulebox|convex)\n", sceneName);
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
