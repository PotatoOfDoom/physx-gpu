// RUNTIME_PLAN.md P5: the full-SDK integration smoke test.  Builds a real PxScene
// from the unmodified PhysX CPU SDK (compiled from source against our foundation,
// no packman, no CUDA toolkit) and STEPS it: a dynamic box dropped above a static
// ground must fall under gravity and come to rest on the ground.  This proves the
// engine's CPU pipeline links and runs in the harness; the GPU pipeline (our Vulkan
// bridge as the cudaContextManager) is wired in the next milestone.

#include "PxPhysicsAPI.h"

#include <cstdio>
#include <cmath>

using namespace physx;

namespace
{
// A minimal allocator/error callback (we don't depend on PxDefault* from extensions).
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

// A synchronous single-thread CPU dispatcher: PhysX only submits a task once its
// dependencies are satisfied, so running it inline is correct for a simple scene.
class SyncDispatcher : public PxCpuDispatcher
{
public:
    void submitTask(PxBaseTask& task) override
    {
        task.run();
        task.release();
    }
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

    PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale(), false, nullptr);
    if (!physics) { std::printf("FAIL: PxCreatePhysics\n"); foundation->release(); return 1; }

    SyncDispatcher dispatcher;

    PxSceneDesc sceneDesc(physics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    sceneDesc.cpuDispatcher = &dispatcher;
    sceneDesc.filterShader = stepFilterShader;
    PxScene* scene = physics->createScene(sceneDesc);
    if (!scene) { std::printf("FAIL: createScene\n"); physics->release(); foundation->release(); return 1; }

    PxMaterial* material = physics->createMaterial(0.5f, 0.5f, 0.1f);

    // Static ground: a large thin box whose top surface is at y = 0.
    PxRigidStatic* ground = physics->createRigidStatic(PxTransform(PxVec3(0.0f, -0.5f, 0.0f)));
    PxShape* groundShape = physics->createShape(PxBoxGeometry(50.0f, 0.5f, 50.0f), *material);
    ground->attachShape(*groundShape);
    scene->addActor(*ground);

    // Dynamic box dropped from y = 5 (half-extent 0.5 -> resting centre y ~= 0.5).
    const float startY = 5.0f;
    PxRigidDynamic* box = physics->createRigidDynamic(PxTransform(PxVec3(0.0f, startY, 0.0f)));
    PxShape* boxShape = physics->createShape(PxBoxGeometry(0.5f, 0.5f, 0.5f), *material);
    box->attachShape(*boxShape);
    // Set mass + inertia via core methods (no extensions): solid box, half-extent
    // 0.5, mass 1000 -> I = m/12 * (a^2 + b^2) = 1000/12 * (1 + 1) ~= 166.67 per axis.
    box->setMass(1000.0f);
    box->setMassSpaceInertiaTensor(PxVec3(166.67f, 166.67f, 166.67f));
    scene->addActor(*box);

    // Step the simulation.
    const float dt = 1.0f / 60.0f;
    const int steps = 240;  // 4 seconds
    float yAfterOne = startY;
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

    // Assertions: it fell during the first step, dropped a long way, and settled near
    // the ground (centre ~0.5) essentially at rest.
    int failures = 0;
    if (!(yAfterOne < startY)) { std::printf("FAIL: box did not begin falling\n"); ++failures; }
    if (!(finalY < 1.5f))      { std::printf("FAIL: box did not reach the ground (finalY=%.3f)\n", finalY); ++failures; }
    if (!(finalY > 0.0f))      { std::printf("FAIL: box fell through the ground (finalY=%.3f)\n", finalY); ++failures; }
    if (!(finalV.magnitude() < 0.5f)) { std::printf("FAIL: box not at rest (|v|=%.3f)\n", finalV.magnitude()); ++failures; }

    scene->release();
    physics->release();
    foundation->release();

    if (failures == 0)
    {
        std::printf("[  OK  ] sdk-step: real PhysX CPU scene stepped on the harness (box fell + rested)\n");
        return 0;
    }
    std::printf("[ FAIL ] sdk-step: %d check(s) failed\n", failures);
    return 1;
}
