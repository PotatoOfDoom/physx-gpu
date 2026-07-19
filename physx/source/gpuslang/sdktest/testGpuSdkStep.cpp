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

#include <cstdio>
#include <cmath>

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
    sceneDesc.solverType = PxSolverType::eTGS;
    // Keep the GPU memory footprint modest for a two-body scene.
    sceneDesc.gpuDynamicsConfig.maxRigidContactCount = 1024 * 4;
    sceneDesc.gpuDynamicsConfig.maxRigidPatchCount = 1024;

    std::printf("  creating GPU scene (eENABLE_GPU_DYNAMICS + eGPU broadphase, TGS)...\n");
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

    PxRigidStatic* ground = physics->createRigidStatic(PxTransform(PxVec3(0.0f, -0.5f, 0.0f)));
    PxShape* groundShape = physics->createShape(PxBoxGeometry(50.0f, 0.5f, 50.0f), *material);
    ground->attachShape(*groundShape);
    scene->addActor(*ground);

    const float startY = 5.0f;
    PxRigidDynamic* box = physics->createRigidDynamic(PxTransform(PxVec3(0.0f, startY, 0.0f)));
    PxShape* boxShape = physics->createShape(PxBoxGeometry(0.5f, 0.5f, 0.5f), *material);
    box->attachShape(*boxShape);
    box->setMass(1000.0f);
    box->setMassSpaceInertiaTensor(PxVec3(166.67f, 166.67f, 166.67f));
    scene->addActor(*box);
    std::printf("  [  OK  ] two actors added to the GPU scene (static ground + dynamic box)\n");

    // Attempt to step.  The GPU rigid pipeline launches its kernels here; the marshalling
    // frontier may surface, so this is reported rather than asserted as the milestone.
    const float dt = 1.0f / 60.0f;
    const int steps = 240;
    float yAfterOne = startY;
    std::printf("  stepping the GPU scene (%d x %.4fs)...\n", steps, dt);
    for (int i = 0; i < steps; ++i)
    {
        scene->simulate(dt);
        scene->fetchResults(true);
        if (i == 0)
            yAfterOne = box->getGlobalPose().p.y;
    }

    const float finalY = box->getGlobalPose().p.y;
    const PxVec3 finalV = box->getLinearVelocity();
    std::printf("  box start y=%.3f, y after 1 step=%.4f, final y=%.4f, final |v|=%.4f\n",
                startY, yAfterOne, finalY, finalV.magnitude());

    int failures = 0;
    if (!(yAfterOne < startY)) { std::printf("FAIL: box did not begin falling on the GPU\n"); ++failures; }
    if (!(finalY < 1.5f))      { std::printf("FAIL: box did not reach the ground (finalY=%.3f)\n", finalY); ++failures; }
    if (!(finalY > 0.0f))      { std::printf("FAIL: box fell through the ground (finalY=%.3f)\n", finalY); ++failures; }
    if (!(finalV.magnitude() < 0.5f)) { std::printf("FAIL: box not at rest (|v|=%.3f)\n", finalV.magnitude()); ++failures; }

    scene->release();
    physics->release();
    cudaCtx->release();
    foundation->release();

    if (failures == 0)
    {
        std::printf("[  OK  ] gpu-sdk-step: real PhysX scene stepped on the Vulkan GPU pipeline (box fell + rested)\n");
        return 0;
    }
    std::printf("[ PARTIAL ] gpu-sdk-step: GPU pipeline constructed; %d step check(s) failed "
                "(kernel-launch marshalling frontier)\n", failures);
    return 2;
}
