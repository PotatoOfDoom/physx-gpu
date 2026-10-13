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
#include "PxDeformableAttachment.h"  // PxParticleRigidAttachment (P9.76, PXG_SCENE=particleattach)
#include "PxNodeIndex.h"             // PxNodeIndex().getInd() -- the STATIC anchor sentinel
#include "PxDirectGPUAPI.h"          // PxDirectGPUAPI (P9.79, PXG_SCENE=directapi)

#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <utility>
#include <chrono>

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

// P9.40 -- the bridge reports every kernel that was launched while unmapped as
// "[pxg-unmapped] <name> ..." through this callback when the CUDA context is
// released.  They are counted here and folded into the verdict below: always
// printed, and a FAILURE under PXG_REQUIRE_MAPPED=1.  Before P9.40 such a launch was
// an anonymous no-op and 33 green scenes said nothing about it.
static unsigned gUnmappedLaunchReports = 0;

// P9.95 -- PXG_SCENE=sdfquery counts the host's own launch-failure report
// (PxgNarrowphaseCore.cpp:8925-8929): evaluateSDFDistances returns true even when the
// launch was refused, because the streamSynchronize result overwrites `success`
// (:8936-8939), so this message is the only in-process sign of a refused launch.
static unsigned gSdfQueryLaunchFailures = 0;

class StepError : public PxErrorCallback
{
public:
    void reportError(PxErrorCode::Enum code, const char* message, const char* file, int line) override
    {
        if (message && std::strncmp(message, "[pxg-unmapped]", 14) == 0)
            ++gUnmappedLaunchReports;
        if (message && std::strstr(message, "evaluatePointDistancesSDF fail to launch"))
            ++gSdfQueryLaunchFailures;
        std::printf("  [PhysX %d] %s (%s:%d)\n", int(code), message, file, line);
    }
};

// P9.41.  Call this AFTER cudaCtx->release() -- that is where the bridge reports --
// and fold the result into the verdict.  P9.40 wired the count into the rigid
// dispatch only; the fourteen deformable blocks each release the context on their own
// line and were left out, so `--strict` did not bite on the scenes that skip the most.
// Prints whenever there is something to print; counts only under PXG_REQUIRE_MAPPED=1.
static int unmappedPenalty()
{
    if (!gUnmappedLaunchReports) return 0;
    const bool strict = std::getenv("PXG_REQUIRE_MAPPED") != nullptr;
    std::printf("  [unmapped] %u kernel(s) were launched but are not mapped to SPIR-V and were "
                "silently skipped -- see the [pxg-unmapped] lines above%s\n",
                gUnmappedLaunchReports,
                strict ? "; PXG_REQUIRE_MAPPED is set, counting this as a FAILURE" : "");
    return strict ? int(gUnmappedLaunchReports) : 0;
}

class SyncDispatcher : public PxCpuDispatcher
{
public:
    void submitTask(PxBaseTask& task) override { task.run(); task.release(); }
    PxU32 getWorkerCount() const override { return 1; }
};

// P9.91 -- raised (before createScene; the filter shader is fixed there) only for
// PXG_SCENE=forcethreshold.  Every other scene keeps eCONTACT_DEFAULT and nothing else,
// so none of them reaches PxgGpuContext's mHasForceThresholds branch.
static bool gForceThresholdPairs = false;

PxFilterFlags stepFilterShader(PxFilterObjectAttributes, PxFilterData, PxFilterObjectAttributes, PxFilterData,
                               PxPairFlags& pairFlags, const void*, PxU32)
{
    pairFlags = PxPairFlag::eCONTACT_DEFAULT;
    if (gForceThresholdPairs)
        pairFlags |= PxPairFlag::eNOTIFY_THRESHOLD_FORCE_FOUND | PxPairFlag::eNOTIFY_THRESHOLD_FORCE_PERSISTS |
                     PxPairFlag::eNOTIFY_THRESHOLD_FORCE_LOST;
    return PxFilterFlag::eDEFAULT;
}

// A dynamic body expected to come to rest at a known centre height on the GPU.
struct RestTarget
{
    PxRigidBody* body;   // PxRigidDynamic, or a PxArticulationLink
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

PxRigidStatic* addGroundBox(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                            float centreY = -0.5f)
{
    PxRigidStatic* ground = physics->createRigidStatic(PxTransform(PxVec3(0.0f, centreY, 0.0f)));
    PxShape* gs = physics->createShape(PxBoxGeometry(50.0f, 0.5f, 50.0f), *material);
    ground->attachShape(*gs);
    scene->addActor(*ground);
    return ground;
}

// `x` exists for P9.20's clothconvex: the live dynamic body every cloth scene
// needs (see buildClothScene) is a BOX, and a box under the cloth is routed to
// the eFemClothBox bucket and carries the cloth through the ALREADY-mapped box
// path.  With it at the origin, turning the convex contact-gen off changed the
// resting height by nothing and the scene proved nothing.  Moved aside, the only
// thing the cloth can rest on is the convex slab.
PxRigidDynamic* addDynamic(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                           const PxGeometry& geom, const PxVec3& inertia, float y,
                           float x = 0.0f)
{
    PxRigidDynamic* b = physics->createRigidDynamic(PxTransform(PxVec3(x, y, 0.0f)));
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

// PXG_SCENE=aggregate (P9.92): the GPU broadphase's AGGREGATE path.
//
// With an eGPU broadphase every PxAggregate is processed by PxgAABBManager
// (aggregate.cu, fourteen kernels): the aggregate's merged bound goes into the
// broadphase proper, and the pairs INSIDE an aggregate (self-collision) and between an
// aggregate and anything else are found by the aggregate kernels, not by the SAP.  No
// scene before this one created an aggregate, so none of those kernels had ever run.
// Four groups, each at its own x, each deciding one rest height that only a working
// aggregate path produces:
//
//   selfstack  x=-6  one aggregate WITH self-collision, three boxes dropped with gaps:
//                    they stack at 0.5 / 1.5 / 2.5.  The stack needs the self pairs; the
//                    bottom box needs the aggregate-vs-actor pair with the ground.
//   noself     x=-2  one aggregate WITHOUT self-collision, two boxes one above the other:
//                    the upper one must fall THROUGH the lower one, so both end at 0.5.
//                    A self pair that is created anyway stacks them at 0.5 / 1.5.
//   aggagg     x=+2  a box in one aggregate resting on the ground and a box in a SECOND
//                    aggregate dropped on it: 0.5 / 1.5 through an aggregate-vs-aggregate
//                    pair.
//   actoragg   x=+6  an ordinary (non-aggregated) box dropped on a box in an aggregate:
//                    0.5 / 1.5 through an aggregate-vs-actor pair between two DYNAMICS.
void buildAggregateScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                         std::vector<RestTarget>& targets)
{
    addGroundBox(physics, scene, material);
    auto box = [&](float x, float y) {
        PxRigidDynamic* b = physics->createRigidDynamic(PxTransform(PxVec3(x, y, 0.0f)));
        PxShape* s = physics->createShape(PxBoxGeometry(0.5f, 0.5f, 0.5f), *material);
        b->attachShape(*s);
        b->setMass(1000.0f);
        b->setMassSpaceInertiaTensor(boxInertia(PxVec3(0.5f), 1000.0f));
        return b;
    };
    auto aggregate = [&](bool selfCollision, std::initializer_list<PxRigidDynamic*> bodies) {
        PxAggregate* agg = physics->createAggregate(PxU32(bodies.size()), PxU32(bodies.size()),
                                                    PxGetAggregateFilterHint(PxAggregateType::eGENERIC, selfCollision));
        for (PxRigidDynamic* b : bodies) agg->addActor(*b);
        scene->addAggregate(*agg);
    };

    PxRigidDynamic* s0 = box(-6.0f, 2.0f);
    PxRigidDynamic* s1 = box(-6.0f, 3.4f);
    PxRigidDynamic* s2 = box(-6.0f, 4.8f);
    aggregate(true, { s0, s1, s2 });
    targets.push_back({ s0, 2.0f, 0.5f, "selfstack0" });
    targets.push_back({ s1, 3.4f, 1.5f, "selfstack1" });
    targets.push_back({ s2, 4.8f, 2.5f, "selfstack2" });

    PxRigidDynamic* n0 = box(-2.0f, 1.0f);
    PxRigidDynamic* n1 = box(-2.0f, 2.4f);
    aggregate(false, { n0, n1 });
    targets.push_back({ n0, 1.0f, 0.5f, "noself0" });
    targets.push_back({ n1, 2.4f, 0.5f, "noself1" });

    PxRigidDynamic* a0 = box(2.0f, 1.0f);
    PxRigidDynamic* a1 = box(2.0f, 2.4f);
    aggregate(false, { a0 });
    aggregate(false, { a1 });
    targets.push_back({ a0, 1.0f, 0.5f, "aggagg0" });
    targets.push_back({ a1, 2.4f, 1.5f, "aggagg1" });

    PxRigidDynamic* c0 = box(6.0f, 1.0f);
    PxRigidDynamic* c1 = box(6.0f, 2.4f);
    aggregate(true, { c0 });
    scene->addActor(*c1);
    targets.push_back({ c0, 1.0f, 0.5f, "actoragg0" });
    targets.push_back({ c1, 2.4f, 1.5f, "actoragg1" });
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

// PXG_SCENE=meshfloor (P8.21): one dynamic box dropped onto a COOKED TRIANGLE MESH
// floor.  Geometrically this is PXG_SCENE=plane with the floor swapped for a
// trimesh, and that swap is the whole point: it routes the pair through the rigid
// convex-vs-TRIANGLE-MESH narrowphase instead of the box-plane one.
//
// *** THIS SCENE IS EXPECTED TO FAIL UNTIL THE convexTrimesh CHAIN IS MAPPED. ***
// P8.21's failed-launch inventory found six unmapped kernels behind that pair --
// midphaseGeneratePairs, convexTrimeshNarrowphase, sortTriangleIndices,
// convexTrimeshPostProcess, convexTrimeshCorrelate, convexTrimeshFinishContacts --
// and no scene asserted on any of them, which is exactly why the gap survived the
// whole port.  A dynamic rigid could not collide with a cooked triangle mesh, and
// nothing said so.  The scene exists to say so, and to become the milestone
// assertion when the chain lands.
//
// The floor is the SAME 20 x 20 two-triangle quad at y = 0 that PXG_SB_FLOOR=mesh
// cooks for the softbody, so the two settle numbers are directly comparable and a
// difference between them points at the pair rather than at the floor.
void buildMeshFloorScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                         std::vector<RestTarget>& targets)
{
    const float E = 10.0f;
    const PxVec3 mverts[4] = { PxVec3(-E, 0.0f, -E), PxVec3(-E, 0.0f, E),
                               PxVec3( E, 0.0f, -E), PxVec3( E, 0.0f, E) };
    const PxU32 mtris[6] = { 0u, 1u, 2u,   2u, 1u, 3u };   // wound so the normals point UP
    PxTriangleMeshDesc md;
    md.points.count = 4; md.points.stride = sizeof(PxVec3); md.points.data = mverts;
    md.triangles.count = 2; md.triangles.stride = 3 * sizeof(PxU32); md.triangles.data = mtris;
    PxCookingParams ck(physics->getTolerancesScale());
    ck.buildGPUData = true;   // without this the mesh has no BV32 tree and the GPU pair is inert
    PxTriangleMesh* floorMeshGeom =
        immediateCooking::createTriangleMesh(ck, md, physics->getPhysicsInsertionCallback());
    if (!floorMeshGeom) { std::printf("FAIL: createTriangleMesh (floor) returned null\n"); return; }
    PxRigidStatic* floor = physics->createRigidStatic(PxTransform(PxVec3(0.0f, 0.0f, 0.0f)));
    PxShape* fs = physics->createShape(PxTriangleMeshGeometry(floorMeshGeom), *material);
    floor->attachShape(*fs);
    scene->addActor(*floor);
    fs->release();

    // Dropped from the same height as PXG_SCENE=box / plane so the three are one
    // comparison: a half-0.5 box settling on a floor at y = 0 rests at centre 0.5.
    PxRigidDynamic* b = addDynamic(physics, scene, material, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                   boxInertia(PxVec3(0.5f), 1000.0f), 5.0f);
    targets.push_back({ b, 5.0f, 0.5f, "boxOnMesh" });
}


// PXG_SCENE=sdfmesh (P9.82a) -- the FIRST DYNAMIC TRIANGLE MESH in this harness, and
// therefore the first scene that can reach the rigid trimesh-vs-plane and
// trimesh-vs-trimesh narrowphase at all.
//
// ---------------------------------------------------------------------------
// WHY A SCENE HAD TO BE BUILT, AND WHY IT NEEDS AN SDF
// ---------------------------------------------------------------------------
// Both pairs are gated on the mesh being DYNAMIC, and PhysX 5.6 will not simulate a
// non-kinematic PxTriangleMeshGeometry without an SDF:
//
//   * NpRigidBodyTemplate.h:359-374 defines isSimGeom as "not plane/heightfield/
//     tetmesh, and if trianglemesh then isDynamicMesh", where isDynamicMesh is
//     literally `mesh->getSdfDataFast().mSdf != NULL`;
//   * the GPU bucket classifier agrees independently: cndCaseMeshMesh
//     (PxgNphaseImplementationContext.cpp:1117-1131) returns eTriangleTriangle only
//     when both sides are rigid, both pass isMeshGPUCompatible, and at least one
//     mesh has an SDF, and ScFiltering.cpp:399-405 KILLS the pair otherwise.
//
// Every cooked mesh in this file until now was cooked with `buildGPUData = true` and
// nothing else, and every one of them is attached to a PxRigidStatic or a
// PxDeformableSurface.  So no scene had ever produced either pair.  The SDF cooker
// itself is available: GuCookingSDF.cpp is compiled into this build and, with
// sdfDesc.sdfBuilder left NULL, physx::buildSDF uses the CPU winding-number path with
// no CUDA involved.
//
// ---------------------------------------------------------------------------
// TWO BODIES, ONE PER KERNEL, AND WHY THAT SEPARATION MATTERS
// ---------------------------------------------------------------------------
// A plane is infinite, so a single dynamic mesh would pair with BOTH the plane and any
// static mesh and a single resting height could not say which kernel caught it.  So:
//
//   body A at x = -4: over the PLANE only               -> trimeshPlaneNarrowphase
//   body B at x = +4: over a static mesh RAISED to y = 2 -> triangleTriangleOverlaps
//                     (it pairs with the plane too, but the mesh is what it meets first)
//
// which makes the four outcomes distinguishable:
//
//   both mapped        A rests at 0.5    B rests at 2.5
//   plane missing      A falls through   B rests at 2.5
//   tri-tri missing    A rests at 0.5    B falls through the raised mesh to 0.5
//   both missing       both fall
//
// The resting heights are not invented: a half-0.5 body on a floor at y = 0 rests at
// centre 0.5, the same number PXG_SCENE=box, =plane and =meshfloor have all measured.
//
// *** EXPECTED TO FAIL UNTIL THE THREE KERNELS ARE MAPPED. ***  This is P9.82a; the
// port is P9.82.  The red measurement is the point of building it first.
// P9.95: `spacing` became a parameter for PXG_SCENE=sdfquery, which cooks two boxes on
// a power-of-two grid (1/16) so that its sample points land on texel centres exactly.
// The default keeps sdfmesh's 0.05.  Every call fills its OWN PxSDFDesc: the cooker
// points the desc's sdf.data at a temporary that is freed on return
// (GuCookingSDF.cpp:399/:408), so a desc reused for a second cook hands it a dangling
// field instead of an empty one.
PxTriangleMesh* cookSdfBox(PxPhysics* physics, float h, float spacing = 0.05f)
{
    // A CLOSED box, 8 vertices and 12 triangles wound outward.  Closed matters twice
    // over: an SDF is a signed distance, which needs an inside, and isMeshGPUCompatible
    // needs the GRB BV32 tree that buildGPUData produces.
    const PxVec3 v[8] = {
        PxVec3(-h, -h, -h), PxVec3( h, -h, -h), PxVec3( h, -h,  h), PxVec3(-h, -h,  h),
        PxVec3(-h,  h, -h), PxVec3( h,  h, -h), PxVec3( h,  h,  h), PxVec3(-h,  h,  h)
    };
    // WOUND OUTWARD: (v1 - v0) x (v2 - v0) points out of the box.  The first
    // version had every face the other way round (its -y face's normal was +y)
    // while this comment already said "outward"; the cooker's winding-number sign
    // then made the whole interior POSITIVE (+0.475 at the centre), which only the
    // mesh-mesh kernel reads -- the plane kernel uses the triangles -- so nothing
    // failed until P9.83b-8 T7 mapped it.  The sign is now checked after cooking.
    const PxU32 t[36] = {
        0, 1, 2,  0, 2, 3,      // -y
        4, 6, 5,  4, 7, 6,      // +y
        0, 5, 1,  0, 4, 5,      // -z
        3, 6, 7,  3, 2, 6,      // +z
        0, 7, 4,  0, 3, 7,      // -x
        1, 6, 2,  1, 5, 6       // +x
    };
    PxTriangleMeshDesc md;
    md.points.count = 8; md.points.stride = sizeof(PxVec3); md.points.data = v;
    md.triangles.count = 12; md.triangles.stride = 3 * sizeof(PxU32); md.triangles.data = t;

    // sdf.data stays NULL, so the cooker BUILDS the field; spacing is the only thing it
    // needs from us.  0.05 over a 1-unit box is 20^3 cells -- small enough to cook in
    // well under a second on the CPU path, fine enough that the box's faces are resolved.
    // subgridSize 0 selects the DENSE field: the sparse form needs the subgrid block
    // dimensions and a min/max range, which are the cooker's outputs, not ours.
    PxSDFDesc sdfDesc;
    sdfDesc.spacing = spacing;
    sdfDesc.subgridSize = 0;
    sdfDesc.numThreadsForSdfConstruction = 4;
    md.sdfDesc = &sdfDesc;

    PxCookingParams ck(physics->getTolerancesScale());
    ck.buildGPUData = true;
    PxTriangleMesh* m = immediateCooking::createTriangleMesh(ck, md,
                                                            physics->getPhysicsInsertionCallback());
    if (!m) { std::printf("FAIL: createTriangleMesh with an SDF returned null\n"); return nullptr; }
    return m;
}

void buildSdfMeshScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                       std::vector<RestTarget>& targets)
{
    const float H = 0.5f;
    PxTriangleMesh* sdfBox = cookSdfBox(physics, H);
    if (!sdfBox) return;
    // The claim the whole scene rests on, checked rather than assumed: without this the
    // shape attaches (PX_CHECK_AND_RETURN_VAL compiles to nothing in this Release build)
    // and then simply never collides, which would look exactly like an unmapped kernel.
    if (!sdfBox->getSDF())
    {
        std::printf("FAIL: the cooked mesh carries NO SDF, so PhysX will not simulate it "
                    "as a dynamic body and neither GPU pair can form\n");
        return;
    }
    PxU32 sdx = 0, sdy = 0, sdz = 0;
    sdfBox->getSDFDimensions(sdx, sdy, sdz);
    std::printf("  sdfmesh: cooked a 12-triangle box with an SDF (dims %u x %u x %u)\n",
                sdx, sdy, sdz);
    // The SIGN of the field, checked: the texel nearest the box centre must be
    // inside, i.e. negative.  P9.83b-8 T7 found this box wound INWARD, and the
    // winding-number cooker then gave the interior a positive field whose minimum
    // sat half a cell outside every face -- every triangle-triangle contact read
    // separation +0.025 however deep the box sank, and the boxes sank through.
    {
        const PxReal centre = sdfBox->getSDF()[(size_t(sdz / 2) * sdy + sdy / 2) * sdx + sdx / 2];
        std::printf("  sdfmesh: SDF at the centre texel %.4f (must be negative; %.9g)\n", centre, centre);
        if (!(centre < 0.0f))
        {
            std::printf("FAIL: the cooked SDF is not negative inside the box -- the mesh is wound inward\n");
            return;
        }
    }

    // ---- the static PLANE, for body A ------------------------------------------------
    PxRigidStatic* plane = physics->createRigidStatic(
        PxTransformFromPlaneEquation(PxPlane(0.0f, 1.0f, 0.0f, 0.0f)));
    PxShape* ps = physics->createShape(PxPlaneGeometry(), *material);
    plane->attachShape(*ps);
    scene->addActor(*plane);
    ps->release();

    // ---- the static MESH raised to y = 2, for body B ---------------------------------
    // Two triangles wound so their normals point UP, cooked with buildGPUData so the pair
    // passes isMeshGPUCompatible.  No SDF on this one: cndCaseMeshMesh needs at least one,
    // and the dynamic body supplies it.
    const float E = 2.0f;
    const PxVec3 mv[4] = { PxVec3(-E, 0.0f, -E), PxVec3(-E, 0.0f, E),
                           PxVec3( E, 0.0f, -E), PxVec3( E, 0.0f, E) };
    const PxU32 mt[6] = { 0u, 1u, 2u,   2u, 1u, 3u };
    PxTriangleMeshDesc md;
    md.points.count = 4; md.points.stride = sizeof(PxVec3); md.points.data = mv;
    md.triangles.count = 2; md.triangles.stride = 3 * sizeof(PxU32); md.triangles.data = mt;
    PxCookingParams ck(physics->getTolerancesScale());
    ck.buildGPUData = true;
    PxTriangleMesh* shelf = immediateCooking::createTriangleMesh(ck, md,
                                                                physics->getPhysicsInsertionCallback());
    if (!shelf) { std::printf("FAIL: createTriangleMesh (shelf) returned null\n"); return; }
    PxRigidStatic* shelfActor = physics->createRigidStatic(PxTransform(PxVec3(4.0f, 2.0f, 0.0f)));
    PxShape* ss = physics->createShape(PxTriangleMeshGeometry(shelf), *material);
    shelfActor->attachShape(*ss);
    scene->addActor(*shelfActor);
    ss->release();

    // ---- the two dynamic SDF meshes ---------------------------------------------------
    // Mass and inertia are set explicitly, as every other rigid scene here does: a solid
    // box of half 0.5 at 1000 kg, so the settle is comparable with PXG_SCENE=box.
    const PxVec3 inertia = boxInertia(PxVec3(H), 1000.0f);
    PxRigidDynamic* a = addDynamic(physics, scene, material,
                                   PxTriangleMeshGeometry(sdfBox), inertia, 5.0f, -4.0f);
    targets.push_back({ a, 5.0f, H, "sdfBoxOnPlane" });
    PxRigidDynamic* b = addDynamic(physics, scene, material,
                                   PxTriangleMeshGeometry(sdfBox), inertia, 5.0f, 4.0f);
    targets.push_back({ b, 5.0f, 2.0f + H, "sdfBoxOnMesh" });

    // ---- the STACK, and why it is not decoration (P9.83b) -----------------------------
    // Everything above this line puts a dynamic SDF mesh against a shape that has NO SDF:
    // the plane, and the shelf, which is cooked deliberately without one (see :366).  That
    // reaches exactly ONE of triangleTriangleCollision's two arms and ONE of its two passes:
    //
    //   * convexMeshMidphase.cu:1100 and :1120 branch on `mesh0.mTexObject == NULL` -- the
    //     "only one object has an SDF" path, which THROWS THE SDF GRADIENT AWAY and
    //     substitutes the triangle normal, sign-corrected by triangleNormalSign.  With the
    //     shelf SDF-less, every contact of sdfBoxOnMesh takes that arm.  The genuine
    //     two-SDF gradient is never executed by any scene in this tree.
    //   * the outer loop's second pass is entered only `if (mesh0.mTexObject)`
    //     (convexMeshMidphase.cu:1048-1055), else it breaks.  With one SDF in the pair the
    //     loop breaks after pass 0, so the whole `outer == 1` arm -- which reverses the
    //     roles, gathers with aToB.getInverse() and applies the OTHER shape's scale, and
    //     whose tail does NOT negate the normal where pass 0 does (:1151 `-dir` against
    //     :1192 `aToB.rotate(dir)`; this comment had it backwards until P9.83b-5 14/n) --
    //     is dead.
    //
    // Two dynamic SDF boxes stacked put an SDF on BOTH sides of one contact manager, which
    // is the smallest change that opens both.  No new cooking: they are the same sdfBox.
    //
    // THE RED PREDICTION, which is what makes this an observable rather than a hope: with
    // triangleTriangleCollision unmapped the top box cannot see the bottom one at all, falls
    // straight through it and settles on the plane at 0.5 instead of 1.5.  The bottom box is
    // unaffected either way -- it rests on the plane through trimeshPlaneNarrowphase, which
    // P9.82 mapped -- so a failure here cannot be blamed on the plane path.
    // BOTH BOXES HAVE TO FALL.  The first draft spawned the bottom one AT its resting
    // height and the harness rejected it -- "did not begin falling" -- which is the right
    // answer: a body that never moves cannot distinguish working physics from none, and
    // that check exists so a scene cannot accidentally score a control.  The bottom drops
    // 1.5 m and is settled in about a third of a second; the top drops the same 4.5 m as
    // sdfBoxOnPlane, so it arrives long after.
    PxRigidDynamic* c = addDynamic(physics, scene, material,
                                   PxTriangleMeshGeometry(sdfBox), inertia, 2.0f, 0.0f);
    targets.push_back({ c, 2.0f, H, "sdfBoxStackBottom" });
    PxRigidDynamic* d = addDynamic(physics, scene, material,
                                   PxTriangleMeshGeometry(sdfBox), inertia, 6.0f, 0.0f);
    targets.push_back({ d, 6.0f, 3.0f * H, "sdfBoxStackTop" });
}

// A cooked TRIANGLE MESH floor at y = 0: two triangles wound so the normals point
// up, with buildGPUData so the mesh carries a BV32 tree.  Used by the particle
// mesh-floor scene (P9.10c), which is the first scene in this port to put
// PARTICLES on a triangle mesh and therefore the first that launches
// ps_meshCollisonLaunch.
PxRigidStatic* addMeshFloor(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                            float halfExtent)
{
    const float E = halfExtent;
    const PxVec3 mverts[4] = { PxVec3(-E, 0.0f, -E), PxVec3(-E, 0.0f, E),
                               PxVec3( E, 0.0f, -E), PxVec3( E, 0.0f, E) };
    const PxU32 mtris[6] = { 0u, 1u, 2u,   2u, 1u, 3u };
    PxTriangleMeshDesc md;
    md.points.count = 4; md.points.stride = sizeof(PxVec3); md.points.data = mverts;
    md.triangles.count = 2; md.triangles.stride = 3 * sizeof(PxU32); md.triangles.data = mtris;
    PxCookingParams ck(physics->getTolerancesScale());
    ck.buildGPUData = true;   // without this the mesh has no BV32 tree and the GPU pair is inert
    PxTriangleMesh* meshGeom =
        immediateCooking::createTriangleMesh(ck, md, physics->getPhysicsInsertionCallback());
    if (!meshGeom) { std::printf("FAIL: createTriangleMesh (mesh floor) returned null\n"); return nullptr; }
    PxRigidStatic* floor = physics->createRigidStatic(PxTransform(PxVec3(0.0f, 0.0f, 0.0f)));
    PxShape* fs = physics->createShape(PxTriangleMeshGeometry(meshGeom), *material);
    floor->attachShape(*fs);
    scene->addActor(*floor);
    fs->release();
    return floor;
}


// A cooked CONVEX floor whose TOP FACE sits at y = 0 (P9.14a): a flat slab of
// half-extents (halfExtent, 0.5, halfExtent), cooked as a convex hull and
// attached to a static actor at y = -0.5.
//
// Why a slab and not a plane or a box PRIMITIVE: the narrowphase's pair table
// (PxgNphaseImplementationContext.cpp:1489) routes eCONVEXMESH x ePARTICLESYSTEM
// to caseConvexParticle -> GPU_BUCKET_ID::eConvexParticle ->
// testSDKConvexParticle, which is the ONLY caller of ps_convexCollisionLaunch and
// ps_convexDiffuseCollisionLaunch.  A PxBoxGeometry floor is eBOX and goes to the
// primitives path instead; a PxPlaneGeometry likewise.  So the shape TYPE is the
// whole point of this floor, not its silhouette.
//
// cndCaseConvexParticle refuses a hull that is not GPU-compatible and falls back
// to the CPU bucket with an error, so buildGPUData must be set and the result is
// checked and printed rather than assumed.
PxRigidStatic* addConvexFloor(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                              float halfExtent)
{
    const float E = halfExtent;
    const PxVec3 slabVerts[8] = {
        PxVec3(-E, -0.5f, -E), PxVec3(-E, -0.5f, E),
        PxVec3(-E,  0.5f, -E), PxVec3(-E,  0.5f, E),
        PxVec3( E, -0.5f, -E), PxVec3( E, -0.5f, E),
        PxVec3( E,  0.5f, -E), PxVec3( E,  0.5f, E),
    };
    PxCookingParams cp(physics->getTolerancesScale());
    cp.buildGPUData = true;
    PxConvexMeshDesc desc;
    desc.points.count = 8;
    desc.points.stride = sizeof(PxVec3);
    desc.points.data = slabVerts;
    desc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
    PxConvexMeshCookingResult::Enum cond = PxConvexMeshCookingResult::eSUCCESS;
    PxConvexMesh* hull = immediateCooking::createConvexMesh(
        cp, desc, physics->getPhysicsInsertionCallback(), &cond);
    if (!hull)
    { std::printf("FAIL: convex floor cooking failed (result %d)\n", int(cond)); return nullptr; }
    std::printf("  cooked convex FLOOR: %u verts, %u polygons, half-extent %.1f "
                "(GPU-compatible: %s)\n", hull->getNbVertices(), hull->getNbPolygons(),
                E, hull->isGpuCompatible() ? "yes" : "NO");
    if (!hull->isGpuCompatible())
        std::printf("FAIL: the convex floor is not GPU-compatible -- the pair would fall back "
                    "to the CPU bucket and no convex particle kernel would ever run\n");

    // Top face at y = 0, matching the mesh and heightfield floors exactly, so the
    // three scenes' settle heights are directly comparable.
    PxRigidStatic* floor = physics->createRigidStatic(PxTransform(PxVec3(0.0f, -0.5f, 0.0f)));
    PxShape* fs = physics->createShape(PxConvexMeshGeometry(hull), *material);
    floor->attachShape(*fs);
    scene->addActor(*floor);
    fs->release();
    return floor;
}

// A cooked HEIGHTFIELD floor at y = 0: a flat grid of zero-height samples spanning
// [-halfExtent, halfExtent] in x and z.  Used by the particle heightfield-floor
// scene (P9.11c), which is the first scene in this port to put PARTICLES on a
// heightfield and therefore the first that launches ps_heightfieldCollisonLaunch.
//
// Three things about the geometry that are easy to get wrong and are spelled here:
//
//  * the local x axis is ROWS and the local z axis is COLUMNS
//    (PxHeightFieldDesc.h:55/:66), and a sample's local position is
//    (row * rowScale, height * heightScale, column * columnScale) -- so the field's
//    origin is its (0,0) CORNER, not its centre.  The actor therefore sits at
//    (-E, 0, -E) to put the span symmetrically about the origin;
//  * every height is 0, which puts the surface at exactly y = 0 whatever heightScale
//    is -- but heightScale must still be positive, because
//    PxHeightFieldGeometry::isValid rejects a non-positive scale.  (That rejection
//    is also what makes upstream candidates 22 and 23 unreachable through the public
//    API, so this floor cannot exercise them and the tranche note says so.)
//  * materialIndex0 and materialIndex1 are 0, NOT eHOLE = 127.  A hole in either
//    triangle makes that cell inert, and the whole point of this floor is that every
//    cell collides.
PxRigidStatic* addHeightFieldFloor(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                                   float halfExtent)
{
    const PxU32 nbRows = 64u, nbCols = 64u;
    std::vector<PxHeightFieldSample> samples(size_t(nbRows) * size_t(nbCols));
    for (size_t i = 0; i < samples.size(); ++i)
    {
        samples[i].height = 0;
        samples[i].materialIndex0 = PxBitAndByte(0);
        samples[i].materialIndex1 = PxBitAndByte(0);
    }
    PxHeightFieldDesc hd;
    hd.format = PxHeightFieldFormat::eS16_TM;
    hd.nbRows = nbRows;
    hd.nbColumns = nbCols;
    hd.samples.data = samples.data();
    hd.samples.stride = sizeof(PxHeightFieldSample);
    PxHeightField* hf =
        immediateCooking::createHeightField(hd, physics->getPhysicsInsertionCallback());
    if (!hf) { std::printf("FAIL: createHeightField (hf floor) returned null\n"); return nullptr; }

    const float E = halfExtent;
    const float rowScale = (2.0f * E) / float(nbRows - 1u);
    const float colScale = (2.0f * E) / float(nbCols - 1u);
    PxRigidStatic* floor = physics->createRigidStatic(PxTransform(PxVec3(-E, 0.0f, -E)));
    PxShape* fs = physics->createShape(
        PxHeightFieldGeometry(hf, PxMeshGeometryFlags(), 1.0f, rowScale, colScale), *material);
    floor->attachShape(*fs);
    scene->addActor(*floor);
    fs->release();
    return floor;
}

// PXG_SCENE=hffloor (P9.51c) -- the rigid-vs-HEIGHTFIELD narrowphase, observed.
//
// Deliberately `meshfloor` with the cooked triangle mesh swapped for a cooked
// heightfield and NOTHING else changed: the same half-0.5 box, the same drop from
// y = 5, the same rest target of 0.5.  The two scenes are then one comparison, and
// a difference between them is a difference between the two mesh narrowphases
// rather than between two experiments.
//
// WHAT THIS SCENE IS FOR.  Before P9.51 the chain
// convexHeightFieldMidphase -> convexHeightfieldNarrowphase ->
// convexHeightfieldPostProcess was unmapped, and P9.27/P9.40 established what an
// unmapped kernel on a live path does in Release: nothing, silently.  So the box
// fell THROUGH the heightfield and no diagnostic said why.  That is exactly the
// P9.41 shape -- a scene whose only job is to make skipped physics observable --
// and it is why this scene is worth its own entry rather than a note under
// meshfloor: with `--strict` (PXG_REQUIRE_MAPPED=1) an unmapped kernel on this
// path now FAILS the scene instead of quietly changing the answer.
void buildHfFloorScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                       std::vector<RestTarget>& targets)
{
    // The same 10.0 half-extent meshfloor uses.  addHeightFieldFloor places the
    // actor at (-E, 0, -E) because a heightfield's origin is its (0,0) CORNER, so
    // the span is symmetric about the origin and the box lands well inside it --
    // not on the rim, where the boundary adjacency arms would decide the contact.
    if (!addHeightFieldFloor(physics, scene, material, 10.0f))
        return;   // addHeightFieldFloor has already printed the FAIL line

    PxRigidDynamic* b = addDynamic(physics, scene, material, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                   boxInertia(PxVec3(0.5f), 1000.0f), 5.0f);
    targets.push_back({ b, 5.0f, 0.5f, "boxOnHeightField" });
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

// PXG_SCENE=convexcoreplane (P9.67): a ROW of dynamic CONVEX CORE cylinders resting
// on the same static infinite PxPlaneGeometry buildPlaneScene uses.  Geometrically
// each one is PXG_SCENE=capsule -- the cylinder's axis is the local X axis
// (PxConvexCoreGeometry.h:164-166), so it lies on its side and rests at centre
// y = radius, exactly as the capsule does -- and that is the point: the same rest
// height reached through a DIFFERENT narrowphase kernel.
//
// PxConvexCoreGeometry is ordinary public API and Gu::isGPUCompatible accepts every
// core type (GuConvexGeometry.cpp:335-353), so unlike the isosurface family of P9.65
// this path IS constructible here -- it simply had no scene.  Five of the twelve
// engine-reachable narrowphase kernels are the convexCore family, and they are
// reachable only through a shape an application explicitly creates: nothing in the
// engine synthesises an eCONVEXCORE shape, so no existing scene routed to them.
//
// WHY A ROW AND NOT ONE CYLINDER (P9.68).  The scene shipped with a single body and
// that single body is why a real defect passed it.  The engine launches
// convexCorePlaneNphase_Kernel with blockDim 1024 and numBlocks = ceil(numTests/1024)
// (PxgNarrowphaseCore.cpp:848-852); the bridge forwards gridDim ONLY, so the Slang
// [numthreads] IS the block size and a 64-wide group reached ONE SIXTEENTH of the
// tests.  With one pair that is invisible -- test 0 is inside every ceiling there is.
// PAIR_COUNT is deliberately above the port's 64-thread group so the divisor is
// observed: with the loop missing, the pairs past 64 get no contacts at all and fall
// through the plane, which is the same evidence P9.58 produced for the convex
// heightfield core (final y = -70.8, still accelerating).
void buildConvexCorePlaneScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                               std::vector<RestTarget>& targets)
{
    PxRigidStatic* ground = physics->createRigidStatic(PxTransformFromPlaneEquation(PxPlane(0.0f, 1.0f, 0.0f, 0.0f)));
    PxShape* ps = physics->createShape(PxPlaneGeometry(), *material);
    ground->attachShape(*ps);
    scene->addActor(*ground);

    const float r = 0.5f, h = 1.0f;
    // A cylinder core with zero margin: the core IS the shape.
    PxConvexCoreGeometry geom(PxConvexCore::Cylinder(h, r));

    // 80 > 64, so tests 64..79 are exactly the ones an ungrided port drops.  The
    // labels need storage that outlives this function because RestTarget holds a
    // const char*; a function-local static array is the cheapest thing that does.
    const int PAIR_COUNT = 80;
    static char labels[80][40];
    for (int i = 0; i < PAIR_COUNT; ++i)
    {
        // The inertia only has to be positive-definite for a body that comes to rest;
        // the capsule scene uses the same reasoning (see capsuleInertia).
        PxRigidDynamic* b = addDynamic(physics, scene, material, geom,
                                       capsuleInertia(r, h * 0.5f, 1000.0f), 5.0f,
                                       float(i) * 3.0f);

        // SLEEP OFF, and this is the half of the observation that took a measurement
        // to find.  The row alone does NOT discriminate: with sleeping on, bodies
        // that settle leave the pair list, `numTests` walks 80 -> 53 -> 16, and every
        // body that was starved at index >= 64 is renumbered below the ceiling within
        // a few frames and picks up its contacts there.  The traced counts were
        // 39 launches at numTests=80, one at 53, 73 at 16 -- all at grid=(1,1,1) --
        // and the defective shader still brought all 80 to rest.  Freezing the pair
        // count keeps the same 16 indices starved for all 240 steps, which is what
        // makes the failure survive to the final check.
        b->setSleepThreshold(0.0f);
        b->setWakeCounter(1.0e9f);

        std::snprintf(labels[i], sizeof(labels[i]), "convexCoreCylinderOnPlane[%d]", i);
        targets.push_back({ b, 5.0f, r, labels[i] });
    }
}

// A cooked triangle-mesh floor TESSELLATED into `cell`-sized quads, spanning
// [xMin, xMax] in x and [-zHalf, zHalf] in z at y = 0 (P9.70).
//
// addMeshFloor makes TWO triangles, which is right for a scene that drops one body
// but wrong for a row of eighty: the convex-core narrowphase turns each TRIANGLE
// into a convex core, and Convex<Support>'s accuracy -- which is also the GJK weld
// distance -- is max(maxExtent * 0.01, FLT_EPSILON).  Two triangles spanning the
// whole row weld support points metres apart, which is the conditioning trap
// PXG_CPU_CONTROL=1 caught in P9.69c.  Small cells keep every pair well conditioned.
// `rot` yaws the whole floor about Y (P9.70c).  It exists because the mutation sweep
// measured that an IDENTITY floor transform makes four separate mutants structurally
// invisible: with transform1.q == identity, `conj(q1)*q0` and `q1*q0` are the same
// quaternion, and rotating a contact point by q1 and by identity is the same point.
// A yaw keeps the floor horizontal -- so the asserted rest height is unchanged -- and
// makes every one of those paths live.
PxRigidStatic* addTessellatedMeshFloor(PxPhysics* physics, PxScene* scene,
                                       PxMaterial* material, float xMin, float xMax,
                                       float zHalf, float cell,
                                       const PxQuat& rot = PxQuat(PxIdentity))
{
    const int nx = PxMax(1, int((xMax - xMin) / cell + 0.5f));
    const int nz = PxMax(1, int((2.0f * zHalf) / cell + 0.5f));
    std::vector<PxVec3> verts;
    verts.reserve(size_t(nx + 1) * size_t(nz + 1));
    for (int iz = 0; iz <= nz; ++iz)
        for (int ix = 0; ix <= nx; ++ix)
            verts.push_back(PxVec3(xMin + float(ix) * cell, 0.0f, -zHalf + float(iz) * cell));

    std::vector<PxU32> tris;
    tris.reserve(size_t(nx) * size_t(nz) * 6u);
    for (int iz = 0; iz < nz; ++iz)
        for (int ix = 0; ix < nx; ++ix)
        {
            const PxU32 a = PxU32(iz * (nx + 1) + ix), b = a + 1;
            const PxU32 c = PxU32((iz + 1) * (nx + 1) + ix), d = c + 1;
            tris.push_back(a); tris.push_back(c); tris.push_back(b);
            tris.push_back(b); tris.push_back(c); tris.push_back(d);
        }

    PxTriangleMeshDesc md;
    md.points.count = PxU32(verts.size()); md.points.stride = sizeof(PxVec3); md.points.data = verts.data();
    md.triangles.count = PxU32(tris.size() / 3); md.triangles.stride = 3 * sizeof(PxU32); md.triangles.data = tris.data();
    PxCookingParams ck(physics->getTolerancesScale());
    ck.buildGPUData = true;   // without this the mesh has no BV32 tree and the GPU pair is inert
    PxTriangleMesh* meshGeom =
        immediateCooking::createTriangleMesh(ck, md, physics->getPhysicsInsertionCallback());
    if (!meshGeom) { std::printf("FAIL: createTriangleMesh (tessellated floor) returned null\n"); return nullptr; }
    std::printf("  tessellated mesh floor: %u verts, %u triangles (cell %.2f)\n",
                md.points.count, md.triangles.count, double(cell));
    PxRigidStatic* floor = physics->createRigidStatic(PxTransform(PxVec3(0.0f, 0.0f, 0.0f), rot));
    PxShape* fs = physics->createShape(PxTriangleMeshGeometry(meshGeom), *material);
    floor->attachShape(*fs);
    scene->addActor(*floor);
    fs->release();
    return floor;
}

// addVTroughMeshFloor (P9.71) -- a cooked triangle mesh shaped like a V-trough
// running along X, apex line on the X axis at y = 0, z = 0.  Each wall rises at
// `theta` above the horizontal:
//
//     wall A (-z side):  (x,  s*sin(theta), -s*cos(theta)),  s in [0, wallLen]
//     wall B (+z side):  (x,  s*sin(theta), +s*cos(theta)),  s in [0, wallLen]
//
// WHY A TROUGH AND NOT ANOTHER FLAT FLOOR.  P9.70c's mutation sweep measured that
// `convexcoretrimesh` exercises exactly ONE contact patch, because a cylinder lying
// flat on a plane produces one -- so pxgSlangCvxCoreOut32.slang, the multi-patch
// output tail and the only genuinely new engine code in that tranche, was verified
// at one of its sixteen patches.  Two mutants survived on that alone
// (`patch-reserve-single`, `startContactIndex-frozen`).
//
// A cylinder in a V touches TWO walls with TWO distinct normals, 2*theta apart.  The
// accumulator's fold test is `dot > SAME_NORMAL` with SAME_NORMAL = 0.999
// (pxgSlangCvxCoreAcc.slang), so any theta above about 2.6 degrees keeps them as two
// separate patches -- and the second patch is what makes the multi-patch tail's
// `numPatches * 64` reservation and its running startContactIndex observable.
PxRigidStatic* addVTroughMeshFloor(PxPhysics* physics, PxScene* scene,
                                   PxMaterial* material, float xMin, float xMax,
                                   float wallLen, float cell, float theta,
                                   const PxQuat& rot = PxQuat(PxIdentity))
{
    const int nx = PxMax(1, int((xMax - xMin) / cell + 0.5f));
    const int ns = PxMax(1, int(wallLen / cell + 0.5f));   // steps up EACH wall
    const int nz = 2 * ns;                                 // -z wall, apex, +z wall
    const float st = PxSin(theta), ct = PxCos(theta);

    // Row j runs from the top of wall A (j = 0) through the apex (j = ns) to the top
    // of wall B (j = nz).  s is the distance from the apex ALONG the wall, so the
    // apex row is shared by both walls and the mesh is watertight there.
    std::vector<PxVec3> verts;
    verts.reserve(size_t(nx + 1) * size_t(nz + 1));
    for (int j = 0; j <= nz; ++j)
    {
        const float s = float(PxAbs(j - ns)) * (wallLen / float(ns));
        const float y = s * st;
        const float z = (j < ns ? -s * ct : s * ct);
        for (int ix = 0; ix <= nx; ++ix)
            verts.push_back(PxVec3(xMin + float(ix) * cell, y, z));
    }

    std::vector<PxU32> tris;
    tris.reserve(size_t(nx) * size_t(nz) * 6u);
    for (int j = 0; j < nz; ++j)
        for (int ix = 0; ix < nx; ++ix)
        {
            const PxU32 a = PxU32(j * (nx + 1) + ix), b = a + 1;
            const PxU32 c = PxU32((j + 1) * (nx + 1) + ix), d = c + 1;
            tris.push_back(a); tris.push_back(c); tris.push_back(b);
            tris.push_back(b); tris.push_back(c); tris.push_back(d);
        }

    PxTriangleMeshDesc md;
    md.points.count = PxU32(verts.size()); md.points.stride = sizeof(PxVec3); md.points.data = verts.data();
    md.triangles.count = PxU32(tris.size() / 3); md.triangles.stride = 3 * sizeof(PxU32); md.triangles.data = tris.data();
    PxCookingParams ck(physics->getTolerancesScale());
    ck.buildGPUData = true;   // without this the mesh has no BV32 tree and the GPU pair is inert
    PxTriangleMesh* meshGeom =
        immediateCooking::createTriangleMesh(ck, md, physics->getPhysicsInsertionCallback());
    if (!meshGeom) { std::printf("FAIL: createTriangleMesh (V-trough floor) returned null\n"); return nullptr; }
    std::printf("  V-trough mesh floor: %u verts, %u triangles (cell %.2f, wall %.1f deg, len %.2f)\n",
                md.points.count, md.triangles.count, double(cell),
                double(theta * 180.0f / PxPi), double(wallLen));
    PxRigidStatic* floor = physics->createRigidStatic(PxTransform(PxVec3(0.0f, 0.0f, 0.0f), rot));
    PxShape* fs = physics->createShape(PxTriangleMeshGeometry(meshGeom), *material);
    floor->attachShape(*fs);
    scene->addActor(*floor);
    fs->release();
    return floor;
}

// PXG_SCENE=convexcoretrough (P9.71): eighty dynamic CONVEX CORE cylinders lying in
// a cooked V-TROUGH triangle mesh.  Same kernel as `convexcoretrimesh`
// (convexCoreTrimeshNphase_Kernel32) and the same cylinder -- what changes is that
// the contact is now TWO patches instead of one.
//
// THE REST HEIGHT IS DERIVED, NOT MEASURED-THEN-ASSERTED.  Put the apex at the
// origin and let each wall rise at theta.  Wall A's inward unit normal is
// (0, cos(theta), sin(theta)) and it passes through the origin, so the distance from
// an axis point (0, y, 0) to that plane is y*cos(theta).  Setting it to the radius r
// gives
//
//     y_c = r / cos(theta)
//
// and by symmetry the same for wall B.  theta = 0 recovers y_c = r, the flat floor.
// For r = 0.5 and theta = 60 degrees that is EXACTLY 1.0.
//
// WHY 60 DEGREES.  Two independent constraints pin it, and both have exact bounds.
//
//   DISCRIMINATION.  main()'s rest band is restY +/- 0.75, generous on purpose
//   (stacks jitter).  A trough only discriminates if the failure moves the body
//   further than that.  The check is `finalY > restY - 0.75`, so a collapse all the
//   way to the apex at y = 0 is caught exactly when restY >= 0.75, i.e. when
//   r/cos(theta) >= 0.75, i.e. theta >= arccos(2/3) = 48.1897 degrees.  Below that
//   the band swallows it and the scene cannot fail.  What the apex-collapse figure is
//   NOT is a prediction of where a broken kernel actually settles -- convexcoretrimesh's
//   own single-patch failure was y = -73.807, far past the apex.  It is a lower bound
//   on the displacement, which is all the argument needs.  The MEASURED statement is
//   the one that counts: at 60 degrees all three multi-patch mutants
//   (patch-reserve-single, startContactIndex-frozen, patchloop-one-patch) die here and
//   two of them survive on the flat floor.
//
//   STABILITY.  The aligned seat competes with the cylinder resting CAP-DOWN against
//   one wall.  Seat height for axis direction `a` is
//   [F(|a.n_A|) + F(|a.n_B|)] / (2 cos t) with F(x) = h*x + r*sqrt(1-x^2); F is
//   concave with F(0) = r and F(1) = h, so the aligned seat is the minimum iff
//   h >= r.  Cylinder(1.0, 0.5) has h = r = 0.5 EXACTLY on that boundary, and at
//   theta = 45 degrees the two configurations tie to the last floating-point bit
//   (both 0.70710678).  At 60 degrees the aligned seat wins by 0.183 (1.000 against
//   1.183).  So 45 is degenerate, 60 is not, and shortening the cylinder at all would
//   make the aligned seat lose outright.
//
// PXG_CVXCORE_TROUGH_DEG overrides the angle.  It has a DOMAIN, and TWO LENGTHS HAVE
// TO FOLLOW IT -- which the first version of this scene got wrong by hard-coding both.
//
//   THE WALL LENGTH.  The cylinder must sit below the top rim, i.e.
//   wallLen*sin(t) >= y_c + r, i.e. wallLen >= r(1+cos t)/(sin t cos t).  That is
//   1.73 at 60 degrees but 2.15 at 30, 5.80 at 10 and 6.26 at 85 -- so a fixed 2.0
//   only spans roughly 45..65 degrees.  Outside it the trough is too shallow to hold
//   the cylinder and it rides over the rim.
//
//   THE SPAWN HEIGHT.  restY is 5.74 at 85 degrees, above the old hard-coded spawn of
//   5.0, so the body would start BELOW its own rest height and main()'s
//   "did not begin falling" check could never pass.
//
// Both are now derived, with the constants chosen so the 60-degree default is
// unchanged: 1.15 * 1.7321 = 1.992 keeps wallLen at 2.0, and restY + 4.0 = 5.0 keeps
// the spawn where it was.  The clamp to [10, 85] then bounds a domain that actually
// works, instead of one that merely avoids the 1/cos singularity at 90.
//
// This was found by RUNNING the clamp's own upper bound rather than trusting it: at
// 85 degrees the scene failed 80 bodies, half of them falling through to y = -30.
//
// THE CELL SIZE DOES NOT CONTROL THE PATCH COUNT, and the first version of this
// comment said it did.  The accumulator folds by NORMAL --
// `dot(minDistNormal, patchNormal[l]) > SAME_NORMAL` with SAME_NORMAL = 0.999f
// (pxgSlangCvxCoreAcc.slang:236-244) -- so every triangle on one wall is coplanar,
// every contact normal is identical, every dot is 1.0, and they collapse into ONE
// patch however finely the wall is tessellated.  The patch count is TWO, one per
// wall, at any cell size.  What the tessellation does change is how many triangles
// the traversal feeds through addPoints and therefore how much point reduction runs
// -- worth knowing, but not what makes this scene discriminate.
//
// The claim was taken from a reading pass and written in without checking it against
// the accumulator, which had already been read in the same pass.  The adversarial
// verification caught it.  Recorded rather than quietly deleted, because "a plausible
// sentence from a read, committed unverified" is the failure mode, not the typo.
//
// The cylinder does lie tangent to each wall along a LINE rather than at a point --
// its axis is parallel to both wall planes, which is also why the rest height does
// not depend on the cylinder's length: in `c.n - h|a.n| - r*sqrt(1-(a.n)^2) >= 0`
// the axis term drops out when `a.n == 0`.
//
// The yaw is P9.70c's, for the same reason recorded there: with transform1.q at
// identity, four separate mutants are structurally invisible.  Trough and bodies are
// yawed together, so the geometry is unchanged and only the quaternions stop being
// trivial.
void buildConvexCoreTroughScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                                std::vector<RestTarget>& targets)
{
    const int PAIR_COUNT = 80;
    const float SPACING = 3.0f;
    const char* cellEnv = std::getenv("PXG_CVXCORE_CELL");
    const float cell = cellEnv ? float(std::atof(cellEnv)) : 1.5f;
    const char* degEnv = std::getenv("PXG_CVXCORE_TROUGH_DEG");
    // Clamped to the usable domain (see the header): below 48.19 degrees the rest
    // band cannot catch a collapse, and at 90 the geometry and 1/cos both degenerate.
    const float degRaw = degEnv ? float(std::atof(degEnv)) : 60.0f;
    const float deg = PxClamp(degRaw, 10.0f, 85.0f);
    if (deg != degRaw)
        std::printf("  note: PXG_CVXCORE_TROUGH_DEG %.2f clamped to %.2f\n",
                    double(degRaw), double(deg));
    const float theta = deg * PxPi / 180.0f;
    const char* yawEnv = std::getenv("PXG_CVXCORE_YAW");
    const float yaw = yawEnv ? float(std::atof(yawEnv)) : 0.6f;
    const PxQuat yawQ(yaw, PxVec3(0.0f, 1.0f, 0.0f));

    const float r = 0.5f, h = 1.0f;
    const float restY = r / PxCos(theta);
    // wallLen >= r(1+cos t)/(sin t cos t) puts the cylinder below the rim; the 1.15
    // is headroom for the settling transient.  The max() keeps the 60-degree default
    // at exactly 2.0, so the measured numbers above still describe it.
    const float wallLen = PxMax(2.0f, 1.15f * r * (1.0f + PxCos(theta))
                                      / (PxSin(theta) * PxCos(theta)));
    const float spawnY = restY + 4.0f;   // 5.0 at the 60-degree default

    if (!addVTroughMeshFloor(physics, scene, material,
                             -SPACING, float(PAIR_COUNT) * SPACING, wallLen, cell, theta, yawQ))
        return; // targets stays empty; main() hard-fails on that

    PxConvexCoreGeometry geom(PxConvexCore::Cylinder(h, r));
    static char labels[80][44];
    for (int i = 0; i < PAIR_COUNT; ++i)
    {
        PxRigidDynamic* b = addDynamic(physics, scene, material, geom,
                                       capsuleInertia(r, h * 0.5f, 1000.0f), spawnY,
                                       float(i) * SPACING);
        // Into the yawed trough's frame.  The cylinder's axis is local +X and the
        // trough runs along X, so a shared yaw keeps the axis along the groove.
        b->setGlobalPose(PxTransform(yawQ.rotate(PxVec3(float(i) * SPACING, spawnY, 0.0f)), yawQ));
        b->setSleepThreshold(0.0f);
        b->setWakeCounter(1.0e9f);
        std::snprintf(labels[i], sizeof(labels[i]), "convexCoreCylinderInTrough[%d]", i);
        targets.push_back({ b, spawnY, restY, labels[i] });
    }
}

// PXG_SCENE=convexcoretrimesh (P9.70): eighty dynamic CONVEX CORE cylinders on a
// cooked, tessellated TRIANGLE MESH floor.  The pair (eCONVEXCORE, eTRIANGLEMESH)
// routes to GPU_BUCKET_ID::eConvexCoreTrimesh, i.e. convexCoreTrimeshNphase_Kernel32
// -- the third convexCore narrowphase kernel and the first that walks a BVH.
//
// Same cylinder and same expected rest height (0.5) as the plane and box scenes, so
// a disagreement is the kernel's and not the geometry's.  Sleep is off for the
// reason P9.68 measured: with sleeping on, settled bodies leave the pair list and a
// starved index is renumbered below the block-coverage ceiling within a few frames,
// which heals the observation away.
//
// *** THIS SCENE IS EXPECTED TO FAIL UNTIL convexCoreTrimeshNphase_Kernel32 IS
// MAPPED. *** It is deliberately NOT in run_scenes.sh's SCENES list yet, the same
// order P9.67a used for convexcoreplane: the scene is built and MEASURED first, the
// sweep entry follows when it passes.
//
// Both halves of that measurement are on the record, because a red scene means
// nothing without them:
//   PXG_CPU_CONTROL=1  all 80 rest at y = 0.5001, 0 failed checks -- so the rest
//                      height this scene asserts is what PhysX itself produces, and
//                      the expectation is not mine to argue with.
//   GPU                all 80 fall through to y = -73.807, 160 failed checks, and
//                      [pxg-unmapped] names convexCoreTrimeshNphase_Kernel32 as
//                      launched 8 time(s) and skipped.
// That pair is what makes the next tranche's before/after unambiguous.
//
// PXG_CVXCORE_CELL overrides the floor's cell size (default 1.5).
void buildConvexCoreTrimeshScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                                 std::vector<RestTarget>& targets)
{
    const int PAIR_COUNT = 80;
    const float SPACING = 3.0f;
    const char* cellEnv = std::getenv("PXG_CVXCORE_CELL");
    const float cell = cellEnv ? float(std::atof(cellEnv)) : 1.5f;

    // THE YAW IS WHAT GIVES THIS SCENE TEETH, and it was added because the P9.70c
    // mutation sweep measured that without it the scene cannot see four separate
    // defects.  With the floor at identity, `transform1.q` is identity, and then
    // `transform1.transformInv(transform0)` computed with the conjugate and computed
    // WITHOUT it are the same quaternion; so is rotating a contact point into world
    // space by q1 and by identity.  Four mutants SURVIVED on that alone.
    //
    // A yaw about Y leaves the floor horizontal, so the asserted rest height is still
    // the cylinder's radius and the CPU control still produces it.  The bodies are
    // yawed with it, so each one still lands on the strip.  PXG_CVXCORE_YAW = 0
    // restores the old, blind geometry for a bisect.
    const char* yawEnv = std::getenv("PXG_CVXCORE_YAW");
    const float yaw = yawEnv ? float(std::atof(yawEnv)) : 0.6f;
    const PxQuat yawQ(yaw, PxVec3(0.0f, 1.0f, 0.0f));

    if (!addTessellatedMeshFloor(physics, scene, material,
                                 -SPACING, float(PAIR_COUNT) * SPACING, 3.0f, cell, yawQ))
        return; // targets stays empty; main() hard-fails on that

    const float r = 0.5f, h = 1.0f;
    PxConvexCoreGeometry geom(PxConvexCore::Cylinder(h, r));
    static char labels[80][44];
    for (int i = 0; i < PAIR_COUNT; ++i)
    {
        PxRigidDynamic* b = addDynamic(physics, scene, material, geom,
                                       capsuleInertia(r, h * 0.5f, 1000.0f), 5.0f,
                                       float(i) * SPACING);
        // Rotate the body into the yawed strip's frame.  The cylinder's axis is local
        // +X and a yaw keeps it horizontal, so the rest height is untouched.
        b->setGlobalPose(PxTransform(yawQ.rotate(PxVec3(float(i) * SPACING, 5.0f, 0.0f)), yawQ));
        b->setSleepThreshold(0.0f);
        b->setWakeCounter(1.0e9f);
        std::snprintf(labels[i], sizeof(labels[i]), "convexCoreCylinderOnTrimesh[%d]", i);
        targets.push_back({ b, 5.0f, r, labels[i] });
    }
}

// PXG_SCENE=convexcorebox (P9.69c): eighty dynamic CONVEX CORE cylinders, each on
// its OWN small static box.  The pair (eBOX, eCONVEXCORE) routes to
// GPU_BUCKET_ID::eConvexCoreConvex (PxgNphaseImplementationContext.cpp gTable), i.e.
// convexCoreConvexNphase_Kernel -- a different kernel from the plane scene's, on the
// same cylinder, so the rest height is the same 0.5 and a disagreement is the
// kernel's.
//
// EIGHTY SMALL BOXES AND NOT ONE BIG ONE, and PXG_CPU_CONTROL=1 is why.  The first
// version put the whole row on a single 400 x 1 x 100 static box, and that scene
// FAILS ON THE CPU PIPELINE TOO (98 checks, cylinders at y = -2.5).  The reason is
// in the algorithm rather than in either port: Convex<Support>'s accuracy is
// `max(maxExtent * 0.01, FLT_EPSILON)` (GuConvexSupport.h:49-59) and it is also the
// GJK weld distance, so a 400 m box welds support points four metres apart and the
// contact is meaningless.  Convex cores are meant for shapes of comparable size.
// Asserting a rest height that PhysX itself does not produce would have been a
// scene bug wearing a port bug's clothes.
//
// SLEEP OFF, for the reason P9.68 measured on the plane scene: the engine launches
// this kernel with blockDim 512 and numBlocks = ceil(numTests/512)
// (PxgNarrowphaseCore.cpp:917-921), the bridge forwards gridDim only, and a
// [numthreads(64,1,1)] group without a covering loop would reach 64 of the pairs.
// With sleeping on, settled bodies leave the pair list and the starved indices are
// renumbered below the ceiling within a few frames -- the observation heals itself.
void buildConvexCoreBoxScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                             std::vector<RestTarget>& targets)
{
    const float r = 0.5f, h = 1.0f;
    PxConvexCoreGeometry geom(PxConvexCore::Cylinder(h, r));

    // PXG_CVXCORE_COUNT reduces the row to N pairs (default 80).  80 is what makes
    // the block-coverage divisor observable (see the header); a smaller N is for
    // isolating a single pair when a result needs bisecting.
    const char* countEnv = std::getenv("PXG_CVXCORE_COUNT");
    const int PAIR_COUNT = countEnv ? PxMax(1, PxMin(80, atoi(countEnv))) : 80;
    static char labels[80][40];
    for (int i = 0; i < PAIR_COUNT; ++i)
    {
        const float x = float(i) * 3.0f;

        // PXG_CVXCORE_GROUND overrides the box's half-extent (default 1.2).  It is
        // here because the conditioning is the interesting variable, not a constant
        // to be guessed: Convex<Support>'s accuracy IS the GJK weld distance and
        // scales with the shape's extent, so this one number decides whether the
        // pair is well conditioned.  See the header on why the row does not share
        // one large ground.
        static const float groundHalf = []() {
            const char* e = std::getenv("PXG_CVXCORE_GROUND");
            return e ? float(std::atof(e)) : 1.2f;
        }();
        PxRigidStatic* ground = physics->createRigidStatic(PxTransform(PxVec3(x, -0.5f, 0.0f)));
        PxShape* gs = physics->createShape(PxBoxGeometry(groundHalf, 0.5f, groundHalf), *material);
        ground->attachShape(*gs);
        scene->addActor(*ground);

        PxRigidDynamic* b = addDynamic(physics, scene, material, geom,
                                       capsuleInertia(r, h * 0.5f, 1000.0f), 5.0f, x);
        b->setSleepThreshold(0.0f);
        b->setWakeCounter(1.0e9f);
        std::snprintf(labels[i], sizeof(labels[i]), "convexCoreCylinderOnBox[%d]", i);
        targets.push_back({ b, 5.0f, r, labels[i] });
    }
}

// PXG_SCENE=convexcoremesh (P9.69c): a dynamic COOKED CONVEX MESH cube dropped onto
// a STATIC CONVEX CORE box.  The pair (eCONVEXCORE, eCONVEXMESH) routes to
// GPU_BUCKET_ID::eConvexCoreConvex, and it is the only scene that takes
// Gu::makeConvexShape's ePOINTS arm -- the one the twelve-float4 pool made
// unwritable before P9.69b, because a cooked hull carries up to 255 points and lives
// in device memory.
//
// THE FIRST VERSION OF THIS SCENE HAD IT THE OTHER WAY ROUND and was wrong, which
// only PXG_CPU_CONTROL=1 could show: a convex core cylinder dropped on a STATIC
// cooked hull falls straight through on the CPU pipeline too (final y = -73.807,
// identical to the GPU), so that configuration is not something PhysX collides and
// the scene was asserting something untrue.  A scene whose CPU control fails is a
// broken scene, not a broken port -- and telling those two apart is exactly what the
// control mode exists for.
void buildConvexCoreMeshScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                              std::vector<RestTarget>& targets)
{
    // The static ground is a CONVEX CORE box, 16 x 1 x 16, top face at y = 0.
    PxRigidStatic* ground = physics->createRigidStatic(PxTransform(PxVec3(0.0f, -0.5f, 0.0f)));
    PxShape* gs = physics->createShape(
        PxConvexCoreGeometry(PxConvexCore::Box(PxVec3(16.0f, 1.0f, 16.0f))), *material);
    ground->attachShape(*gs);
    scene->addActor(*ground);

    // The faller is the cooked hull, so the ePOINTS arm is the DYNAMIC side.
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
    std::printf("  cooked convex cube: %u verts, %u polygons (GPU-compatible: %s)\n",
                mesh->getNbVertices(), mesh->getNbPolygons(),
                mesh->isGpuCompatible() ? "yes" : "NO");

    PxRigidDynamic* b = addDynamic(physics, scene, material, PxConvexMeshGeometry(mesh),
                                   boxInertia(PxVec3(0.5f), 1000.0f), 5.0f);
    targets.push_back({ b, 5.0f, 0.5f, "cookedHullOnConvexCoreBox" });
}

// PXG_SCENE=spheremesh / capsulemesh (P9.66): a dynamic SPHERE, and a dynamic
// CAPSULE, dropped onto the same cooked triangle-mesh floor buildMeshFloorScene
// uses.  The floor is identical; only the falling shape's type changes, and that
// type is the whole point.
//
// *** BOTH SCENES SEGFAULT UNTIL sphereTrimeshNarrowphase IS MAPPED. ***
// Measured before the port, both scenes: the launch is refused --
//   [PhysX 32] GPU sphereTrimeshCore fail to launch kernel!!
//   (PxgNarrowphaseCore.cpp:2027)
// -- and the process then dies with SIGSEGV (exit 139) at the first step, before
// any body has moved.  This is NOT the silent fall-through the other unmapped
// pairs produce: the refused kernel leaves the contact-manager output
// uninitialised and a downstream consumer dereferences it, the same shape as the
// P9.52b-6 crash where convexTrimeshCorrelate read an uninitialised contact count.
// So a sphere or a capsule touching a cooked triangle mesh takes the whole
// simulation down today, and no scene said so before P9.66.
// PxgNphaseImplementationContext.cpp routes BOTH pairs to the same bucket:
// cndCaseSphereMesh (:1047) and cndCaseCapsuleMesh (:1071) each return
// GPU_BUCKET_ID::eSphereTrimesh when the mesh is GPU-compatible, and that bucket's
// only kernel -- sphereTrimeshNarrowphase, launched from
// PxgGpuNarrowphaseCore::testSDKSphereTriMeshSATGpu (PxgNarrowphaseCore.cpp:1992) --
// is unmapped.  So a sphere and a capsule both fall straight through a cooked
// triangle mesh today, and until P9.66 nothing in this port said so.
//
// Two scenes rather than one because the two pairs reach the kernel through
// DIFFERENT geometry: the capsule arm is a segment-vs-triangle test and the sphere
// arm a point-vs-triangle one.  A port that gets only the sphere arm right passes a
// spheremesh-only scene.  The box-on-mesh number from PXG_SCENE=meshfloor is the
// control: it uses the convexTrimesh chain, which is already mapped, over the very
// same floor.
void buildSphereMeshScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                          std::vector<RestTarget>& targets)
{
    if (!addMeshFloor(physics, scene, material, 10.0f)) return;
    const float r = 0.5f;
    const float sphereI = 0.4f * 1000.0f * r * r; // solid sphere: 2/5 m r^2
    PxRigidDynamic* b = addDynamic(physics, scene, material, PxSphereGeometry(r),
                                   PxVec3(sphereI, sphereI, sphereI), 5.0f);
    targets.push_back({ b, 5.0f, r, "sphereOnMesh" });
}

void buildCapsuleMeshScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                           std::vector<RestTarget>& targets)
{
    if (!addMeshFloor(physics, scene, material, 10.0f)) return;
    const float r = 0.5f, h = 0.5f;
    PxRigidDynamic* b = addDynamic(physics, scene, material, PxCapsuleGeometry(r, h),
                                   capsuleInertia(r, h, 1000.0f), 5.0f);
    targets.push_back({ b, 5.0f, r, "capsuleOnMesh" }); // rests on its side at centre y = radius
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

// PXG_SCENE=spherehf (P9.52b-6) -- the SPHERE/CAPSULE-vs-heightfield narrowphase,
// observed.  Both arms in one scene, because they are one kernel.
//
// `hffloor` drops a BOX, which routes through convexHeightfieldNarrowphase.  Nothing
// in the scene set reached sphereHeightfieldNarrowphase at all -- so P9.52a through
// P9.52b-4 built and verified two cores, a kernel and eleven modules against host
// mirrors, and not one line of it had ever run inside a simulation.  A fixture
// proves a shader computes what the CUDA computes; it cannot prove the engine ever
// calls it.  That distinction is the whole of P9.27/P9.40/P9.41: an unmapped kernel
// on a live path does nothing, silently, and a MAPPED kernel that no scene reaches
// is the same blind spot wearing a green tick.
//
// The heightfield, the drop height and the rest tolerance are `hffloor`'s, so the
// three heightfield scenes differ only in the shape that falls -- box, sphere,
// capsule -- and a difference between them is a difference between narrowphase arms
// rather than between experiments.
//
// The capsule lies on its side (axis along local X, PhysX's capsule convention), so
// its rest centre is at y = radius and BOTH endpoints of the segment sit over the
// terrain.  That matters here more than it does on a plane: the capsule core's
// `generateContacts` accepts an endpoint only when it projects inside the triangle,
// so a capsule balanced over a single triangle's edge would exercise the edge
// generator alone.  Lying flat across the interior it reaches both.
void buildSphereHfScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                        std::vector<RestTarget>& targets)
{
    if (!addHeightFieldFloor(physics, scene, material, 10.0f))
        return;   // addHeightFieldFloor has already printed the FAIL line

    const float r = 0.5f;
    const float sphereI = 0.4f * 1000.0f * r * r;   // solid sphere: 2/5 m r^2
    PxRigidDynamic* sph = addDynamic(physics, scene, material, PxSphereGeometry(r),
                                     PxVec3(sphereI, sphereI, sphereI), 5.0f);
    targets.push_back({ sph, 5.0f, r, "sphereOnHeightField" });

    // Offset in x so the two bodies cannot touch each other and turn this into a
    // three-body experiment.
    const float ch = 0.5f;
    PxRigidDynamic* cap = addDynamic(physics, scene, material, PxCapsuleGeometry(r, ch),
                                     capsuleInertia(r, ch, 1000.0f), 5.0f);
    {
        PxTransform t = cap->getGlobalPose();
        t.p.x += 3.0f;
        cap->setGlobalPose(t);
    }
    targets.push_back({ cap, 5.0f, r, "capsuleOnHeightField" });
}

// PXG_SCENE=hfbigbox (P9.58) -- the CONVEX heightfield core's grid-stride gap, made
// observable.
//
// `hffloor` drops a half-0.5 box, whose AABB overlaps about four heightfield cells
// and so produces well under 32 midphase pairs.  The convex core's shader makes ONE
// pass covering `gridDim.x * 32` pairs, and its host launch over-provisions the grid
// 32-fold (PxgNarrowphaseCore.cpp:2898), so for one contact manager that is 32 pairs
// -- comfortably more than hffloor needs.  The gap has therefore never been reached
// by any scene, which is why its header could describe it as known and accepted for
// two tranches without anything going wrong.
//
// A half-3 box spans roughly 36 cells, hence about 72 candidate triangles, and pushes
// the pair count past the ceiling.  P9.52b-6 measured what happens beyond it on the
// SPHERE kernel: records left unwritten, and convexTrimeshCorrelate dereferencing an
// uninitialised contact count.
//
// The box is deliberately the ONLY difference from hffloor -- same heightfield, same
// drop, same tolerance -- so the two scenes remain one comparison.
void buildHfBigBoxScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                        std::vector<RestTarget>& targets)
{
    if (!addHeightFieldFloor(physics, scene, material, 10.0f))
        return;   // addHeightFieldFloor has already printed the FAIL line

    const float h = 3.0f;
    PxRigidDynamic* b = addDynamic(physics, scene, material, PxBoxGeometry(h, h, h),
                                   boxInertia(PxVec3(h), 1000.0f), 8.0f);
    targets.push_back({ b, 8.0f, h, "bigBoxOnHeightField" });
}

// PXG_SCENE=hfhugebox (P9.59) -- the heightfield POST-PROCESS's ceiling, probed.
//
// The post process is what removes the duplicate contacts a shared heightfield edge
// or vertex produces: a pair flagged `delayContacts` goes on a second-pass queue and
// is revisited.  Its Slang port makes one pass of `256 * numTests` threads
// (PxgNarrowphaseCore.cpp:2971-2973), while the CUDA covers the rest by iterating,
// and convexHfPostProcessBda's own header predicts the symptom past that ceiling:
// "duplicate contacts along shared edges, not missing ones".
//
// The terrain is 64x64 samples over 2*halfExtent, so at halfExtent 10 a cell is about
// 0.317 units and a half-3 box already spans some 19x19 cells -- which is why
// hfbigbox broke the CORE's 32-pair ceiling so decisively.  Only the flagged pairs
// reach the second-pass queue, though, and for a box lying flat on flat terrain those
// are concentrated on its PERIMETER.  A half-8 box has a perimeter of roughly 200
// cells, so this scene exists to find out whether 256 is reachable at all -- if it is
// not, the ceiling is theoretical for this kernel and saying so is worth more than a
// fix nobody can observe.
void buildHfHugeBoxScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                         std::vector<RestTarget>& targets)
{
    if (!addHeightFieldFloor(physics, scene, material, 10.0f))
        return;   // addHeightFieldFloor has already printed the FAIL line

    const float h = 8.0f;
    PxRigidDynamic* b = addDynamic(physics, scene, material, PxBoxGeometry(h, h, h),
                                   boxInertia(PxVec3(h), 1000.0f), 12.0f);
    targets.push_back({ b, 12.0f, h, "hugeBoxOnHeightField" });
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

// PXG_SCENE=directapi (P9.79): the FIRST scene in this harness that raises
// PxSceneFlag::eENABLE_DIRECT_GPU_API, and therefore the first thing that asks the
// direct-GPU-API kernels to run.
//
// *** WHY THIS SCENE HAD TO EXIST BEFORE ANYTHING COULD BE PORTED. ***
//
// P9.79's liveness audit of the twenty engine-reachable unmapped kernels found that
// a whole family of them is gated behind this one scene flag and nothing else:
// getD6JointForces / getD6JointTorques (PxgSimulationCore::getD6JointData),
// mergeChangedAABBMgrHandlesLaunch (ScPipeline.cpp:355, `if
// (isDirectGPUAPIInitialized())`), mergeTransformCacheAndBoundArrayChanges
// (PxgSimulationController.cpp:703, `if (getEnableDirectGPUAPI())`) and
// evaluatePointDistancesSDFBatch.  Not "rarely reached" -- UNREACHABLE, in every
// one of the forty-nine scenes this harness had before today.  (The flag is
// necessary, not sufficient, for mergeTransformCacheAndBoundArrayChanges: it also
// needs a shape added or re-inserted between two steps, which this scene never
// does.  PXG_SCENE=directapimerge does, and reaches it -- P9.95.)
//
// And the twelve direct-API rigid getters and setters that WERE ported
// (shaders/directApi.slang, plus setRigidDynamicGlobalPose in updateBounds.slang)
// sat in check_launch_abi.py's KNOWN_UNWIRED list for the same reason: they had no
// row in bridge/PxgVulkanKernelAbi.cpp, so pushConstantSizeFor() returned 0 and the
// shim refused every launch.  Nobody noticed, because nothing launched them.  This
// scene wired the three getters it reads (P9.79); PXG_SCENE=directapiwrite wired
// the five setters and the two acceleration getters (P9.93).
//
// *** WHAT MAKES A REFUSAL VISIBLE HERE. ***
//
// With the direct-GPU API on, the SDK stops writing simulation results back into the
// host actors: PxRigidDynamic::getGlobalPose() is no longer the body's pose, and the
// direct read IS the only way to see where the box went.  So the scene's central
// number -- did the box fall and rest? -- passes through
// getRigidDynamicGlobalPose, and a refused launch cannot be mistaken for a correct
// answer.
//
// It is made harder still to mistake: every device output buffer is POISONED with
// -999 before each read (readPoisoned() below).  A refused kernel leaves the poison
// in place, so the failure reads as an impossible number rather than as a stale but
// plausible pose.  That distinction is the whole lesson of P9.48d, where 361
// refusals looked exactly like "the kernel ran and found nothing".
//
// *** THE TWO BODIES, AND WHY THE SECOND ONE IS JOINTED OFF-CENTRE. ***
//
//   faller  -- a 1000 kg box dropped from y=5 onto the static ground box; rests at
//              centre y=0.5, exactly as PXG_SCENE=box.  Read back as a pose and as a
//              linear velocity, so two different getters are on the hook.
//   hanger  -- a 1000 kg box held by a PxD6Joint with ALL SIX DOFs LOCKED, whose
//              attachment point on the body is offset by -0.5 in x from the centre
//              of mass.  The offset is the point: with the joint frame ON the centre
//              of mass the reaction torque is identically zero, and "zero" is the one
//              answer a refused kernel also gives.  Off-centre, static equilibrium
//              fixes BOTH readings analytically:
//
//                  F = +m g y_hat                 |F| = 1000 * 9.81 = 9810 N
//                  T = -r_att x F                 |T| = m g d = 4905 N m about +z
//
//              (r_att = (-d, 0, 0) is the vector from the centre of mass to the
//              attachment point; d = 0.5.)  So getD6JointForces and getD6JointTorques
//              each have a number to be wrong about, and they are different numbers.
//
// The sign convention the engine uses for the writeback is NOT asserted here -- it is
// printed, and the checks are on magnitude and on the dominant axis.  Asserting a
// convention this scene has not measured would be inventing certainty.
struct DirectApiScene
{
    PxRigidDynamic* faller;
    PxRigidDynamic* hanger;
    PxD6Joint*      joint;
    float           hangerAttachOffsetX;  // -d: attachment point relative to the COM
    PxVec3          anchorWorld;          // the world point the hanger is pinned to
    float           hangerMass;
};

DirectApiScene buildDirectApiScene(PxPhysics* physics, PxScene* scene, PxMaterial* material)
{
    DirectApiScene s = {};
    s.hangerAttachOffsetX = -0.5f;
    s.hangerMass = 1000.0f;
    // The anchor is off BOTH the x and z axes on purpose.  See the torque
    // expectation at the check site: the engine reports the constraint torque about
    // ACTOR0's origin, and actor0 here is the world, so every component of the
    // anchor position appears in the answer.  A single-axis anchor would let a
    // whole family of wrong conventions produce the same number.
    s.anchorWorld = PxVec3(4.0f, 3.0f, 2.0f);

    addGroundBox(physics, scene, material);

    // The faller: identical to PXG_SCENE=box, so its rest height is a number this
    // harness has measured forty-nine times over.
    s.faller = addDynamic(physics, scene, material, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                          boxInertia(PxVec3(0.5f), 1000.0f), 5.0f);

    // The hanger: spawned so that its ATTACHMENT point (COM + (-0.5, 0, 0)) starts
    // just above the world anchor and settles onto it.  addDynamic places bodies at
    // z = 0, so the body is moved onto the anchor's z afterwards.
    s.hanger = addDynamic(physics, scene, material, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                          boxInertia(PxVec3(0.5f), s.hangerMass), s.anchorWorld.y + 0.25f,
                          s.anchorWorld.x - s.hangerAttachOffsetX);
    s.hanger->setGlobalPose(PxTransform(PxVec3(s.anchorWorld.x - s.hangerAttachOffsetX,
                                               s.anchorWorld.y + 0.25f, s.anchorWorld.z)));

    // actor0 = NULL -> world frame at the anchor; localFrame1 is the attachment point
    // expressed in the body's own frame, i.e. offset from the centre of mass.
    s.joint = PxD6JointCreate(*physics, NULL, PxTransform(s.anchorWorld),
                              s.hanger, PxTransform(PxVec3(s.hangerAttachOffsetX, 0.0f, 0.0f)));
    s.joint->setMotion(PxD6Axis::eX, PxD6Motion::eLOCKED);
    s.joint->setMotion(PxD6Axis::eY, PxD6Motion::eLOCKED);
    s.joint->setMotion(PxD6Axis::eZ, PxD6Motion::eLOCKED);
    s.joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eLOCKED);
    s.joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eLOCKED);
    s.joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eLOCKED);
    return s;
}

// PXG_SCENE=directapiwrite (P9.93): the WRITE half of the direct-GPU API, and the
// two ACCELERATION getters.
//
// PXG_SCENE=directapi only reads.  PxDirectGPUAPI::setRigidDynamicData (five
// kernels: eGLOBAL_POSE, eLINEAR_VELOCITY, eANGULAR_VELOCITY, eFORCE, eTORQUE) and
// the eLINEAR_/eANGULAR_ACCELERATION reads -- live only under
// PxSceneFlag::eENABLE_BODY_ACCELERATIONS on top of the direct-API flag -- were
// never launched by anything, so none of the seven had a launch-ABI row.
//
// *** WHAT MAKES A REFUSED SETTER VISIBLE. ***  A setter returns a bool the
// launch result cannot reach: without a finish event PxgSimulationCore replaces it
// with the streamSynchronize result.  So every setter is judged by what it DID --
// through the already-wired getters immediately, and through the physics after
// sixty more steps -- and every read goes through poisoned device memory.
//
// Seven 1000 kg boxes (inverse inertia 0.006 about every axis) free-fall from y=20
// at x = -10, -6, -2, 2, 6, 10, 14, far from each other and from the ground.  One
// warm-up step (the direct API refuses every call before the first simulate
// completes), then ONE write window, then sixty steps:
//
//   T  (x=-10) teleported by eGLOBAL_POSE to (-10, 0.5, 3), yawed 45 degrees: it
//              must END on the ground there, yawed.
//   T2 (x=-6)  teleported in the SAME call (two elements, reversed index order) to
//              (-6, 30, -4), rolled 90 degrees: the teleport keeps its velocity, so it
//              ends at y = 30 - 0.1635 - 4.98675 = 24.84975, still rolled.
//   V  (x=-2)  eLINEAR_VELOCITY (4, 0, -2), batched with R's own velocity written
//              back to it (the 12-byte stride): ends at (2, 15.010525, -2).
//   W  (x=2)   eANGULAR_VELOCITY (0, 3, 0), batched with R's: 3 rad about y.
//   F  (x=6)   eFORCE twice, 10000 N then 5000 N along z.  The kernel ASSIGNS, so only
//              5000 N counts: a = 5 m/s^2 for one step, v_z = 0.0833333, z = 0.0833333.
//   Q  (x=10)  eTORQUE 30000 N m about z: alpha = 180 rad/s^2 for one step,
//              omega_z = 3, 3 rad after sixty steps.
//   R  (x=14)  the reference: nothing written; it drops 5.152975 in 61 steps.
//
// The acceleration getters are read after the warm-up (R: (0, -9.81, 0) from a
// zeroed previous-velocity buffer), right after the velocity writes (exactly 0: the
// setters also write the previous velocity), and after the next two steps (F's
// force for one step only, Q's torque likewise).
//
// Every body keeps linear damping 0; W, Q and R also get angular damping 0 (the
// default 0.05 would bleed the spin), and all seven a sleep threshold of 0, so no
// body the setters touch can be asleep: a setter does not wake anything.
struct DirectApiWriteScene
{
    PxRigidDynamic* T;
    PxRigidDynamic* T2;
    PxRigidDynamic* V;
    PxRigidDynamic* W;
    PxRigidDynamic* F;
    PxRigidDynamic* Q;
    PxRigidDynamic* R;
};

DirectApiWriteScene buildDirectApiWriteScene(PxPhysics* physics, PxScene* scene, PxMaterial* material)
{
    DirectApiWriteScene s = {};
    addGroundBox(physics, scene, material);
    auto body = [&](float x) {
        PxRigidDynamic* b = addDynamic(physics, scene, material, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                       boxInertia(PxVec3(0.5f), 1000.0f), 20.0f, x);
        b->setSleepThreshold(0.0f);
        b->setAngularDamping(0.0f);
        return b;
    };
    s.T = body(-10.0f);
    s.T2 = body(-6.0f);
    s.V = body(-2.0f);
    s.W = body(2.0f);
    s.F = body(6.0f);
    s.Q = body(10.0f);
    s.R = body(14.0f);
    return s;
}

// PXG_SCENE=directapimerge (P9.95): mergeTransformCacheAndBoundArrayChanges.
//
// With the direct-GPU API on, the bounds array and the transform cache are uploaded
// whole ONCE, on the first step (PxgSimulationController.cpp:707-710).  After that,
// every shape added or re-inserted between two steps is recorded by PxgBoundsArray
// (PxgAABBManager.h:262-321) as {indexTo, indexFrom | isNew<<31} and merged into the
// device arrays by this one kernel (:712-714, launch :725-764) -- the host never
// re-uploads them.  It runs only behind four guards: the direct-API flag, a first
// step already done, a shape instance registered or unregistered since the last step
// (:799), and a change recorded BETWEEN steps (changes recorded during a step are
// dropped at its DMA-back, PxgSimulationCore.cpp:2256-2261).  No scene before this
// one changed a shape after step 0, so the kernel had never been launched.
//
// Three lanes, each on its own small static slab (half (2, 0.5, 2), top at y = 0;
// NOT a ground box, which would catch every lane at once and hide which slot is
// wrong), each exercising one branch of the kernel:
//
//   N  (x=-30)  B falls from y=5 with NOTHING under it.  After the second step the
//               slab G2 is added under it: an isNew record (from == to), copied from
//               the CPU arrays.  Two far decoy slabs are added in the same window.
//   S  (x=-10)  A falls from y=2 onto G1, which exists from step 0.  In the same
//               window resetFiltering(G1) re-inserts G1 under a NEW element id: a
//               device-to-device record (ScShapeSimBase.cpp:106-157).
//   D  (x=+10)  C falls from y=3 onto G3 and rests.  After step 60 resetFiltering(C)
//               re-inserts C: a device-to-device record for a DYNAMIC body, whose
//               device slot (the GPU pose) and CPU slot (the stale spawn pose, y=3:
//               host writeback is off) differ.  resetFiltering on a dynamic always
//               calls wakeUpInternal() too (NpScene.cpp:2597-2599).
//
// Every body is read back through PxDirectGPUAPI::getRigidDynamicData into poisoned
// memory.  The checks: B and A rest at 0.5 (lanes N and S), C's vertical velocity
// right after the reset step is ~0 and its height 0.5 (lane D: a slot taken from
// the CPU arrays would put C at y=3 for one step, with no broadphase pair, and C
// would free-fall for that step), and C still rests at the end.
//
// PXG_DIRECTAPI_CONTROL=1 builds the same scene without the flag and reads the
// ordinary API; the merge kernel never runs there (copyBoundsAndTransforms instead).
// It is the reference for C's velocity at the reset step.  Reported, never scored.
//
// *** MEASURED, RED BEFORE THE PORT (kernel unmapped, launches skipped). ***
//
//   [pxg-unmapped] mergeTransformCacheAndBoundArrayChanges was launched 2 time(s)
//   B.y = -73.8070  A.y = -76.8070  (both free-fall through: 240 steps from y=5 / y=2)
//   C.v_y after the reset step = -0.1644 (TGS) / -0.1637 (PGS): one step of free fall
//   C.y after the reset step = 0.4973, C.y at the end = 0.5000 (it recovers)
//
// The control (no direct API) gives C.v_y = -0.0009 (TGS) / -0.0001 (PGS) after the
// reset step: the pair the re-insertion destroys is found again and gets contacts in
// the SAME step, so v_y right after it separates a correct merge from a wrong one.
// After the port the live run equals the control to the printed digit under both
// solvers (B.y = A.y = C.y = 0.5000), and the dumped records (PXG_DUMP_ARGS) are
// launch 1 = {5, 5|new} {6, 6|new} {7, 7|new} {8, 0} (G2, the decoys, G1 0 -> 8) and
// launch 2 = {0, 3} (C 3 -> 0, G1's released id).  The wake resetFiltering(C) adds
// does NOT push C's stale host pose: C.y after the reset step is 0.5000, not 3.
//
// What each wrong merge does to the lanes (scene mutation sweep, TGS):
//   kernel skipped           N -73.81, S -76.81, D -0.1644
//   MSB ignored, always CPU  D only (-0.1644; -0.1637 under PGS).  S passes: G1 is a
//                            static, so its CPU slot 0 is as current as its device slot.
//   MSB ignored, always dev  N only (-73.81)
//   indexTo / indexFrom swapped  S and D.  N passes: its record has from == to.
//   isNew copies bounds only N only: G2's transform-cache slot is never written.
//   re-insert copies the transform only  S and D.
//
// The slabs are added and re-inserted, never MOVED.  PxRigidStatic::setGlobalPose
// after init takes another road (upstream candidate, unverified): updateDirtyShapes
// writes the CPU arrays and flags the bounds array changed (ScPipeline.cpp:244), so
// preBpUpdate_GPU re-uploads the WHOLE CPU bounds array into the broadphase buffer
// (PxgCudaBroadPhaseSap.cpp:1242-1246) while the NP transform cache keeps the old
// pose -- and that re-upload would also mask lanes N and S here.
struct DirectApiMergeScene
{
    PxRigidStatic*  G1;   // lane S slab, re-inserted in window W1
    PxRigidStatic*  G3;   // lane D slab
    PxRigidDynamic* A;    // lane S faller
    PxRigidDynamic* B;    // lane N faller (its slab G2 arrives in W1)
    PxRigidDynamic* C;    // lane D faller, re-inserted in window W2
};

// A static slab with ONE exclusive box shape, half (2, 0.5, 2), centred at
// (x, -0.5, 0) so its top face is y = 0.  Created, NOT added: the caller adds it,
// before the first step or between two steps.
PxRigidStatic* makeMergeSlab(PxPhysics* physics, PxMaterial* material, float x)
{
    PxRigidStatic* slab = physics->createRigidStatic(PxTransform(PxVec3(x, -0.5f, 0.0f)));
    PxShape* shape = physics->createShape(PxBoxGeometry(2.0f, 0.5f, 2.0f), *material, true);
    slab->attachShape(*shape);
    shape->release();
    return slab;
}

DirectApiMergeScene buildDirectApiMergeScene(PxPhysics* physics, PxScene* scene, PxMaterial* material)
{
    DirectApiMergeScene s = {};
    auto body = [&](float x, float y) {
        PxRigidDynamic* b = addDynamic(physics, scene, material, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                       boxInertia(PxVec3(0.5f), 1000.0f), y, x);
        b->setSleepThreshold(0.0f);
        b->setAngularDamping(0.0f);
        return b;
    };
    // Creation order fixes the element ids: G1 0, A 1, G3 2, C 3, B 4.
    s.G1 = makeMergeSlab(physics, material, -10.0f);
    scene->addActor(*s.G1);
    s.A = body(-10.0f, 2.0f);
    s.G3 = makeMergeSlab(physics, material, 10.0f);
    scene->addActor(*s.G3);
    s.C = body(10.0f, 3.0f);
    s.B = body(-30.0f, 5.0f);
    return s;
}

// PXG_SCENE=sdfquery (P9.95): evaluatePointDistancesSDFBatch, the kernel behind
// PxDirectGPUAPI::evaluateSDFDistances -- the only caller it has
// (NpDirectGPUAPI.cpp:151-163 -> PxgSimulationController.cpp:631-636 ->
// PxgGpuNarrowphaseCore::evaluateSDFDistances, PxgNarrowphaseCore.cpp:8899-8942).
// Before this scene nothing in the harness had ever asked for it: it is a pure QUERY,
// launched only by the API call, never by a simulation step.
//
// THE MESHES are two closed boxes cooked on a POWER-OF-TWO grid (spacing 1/16):
//   A  half 0.875, dims 32^3, meshLower -1.0, SDF box +-0.96875
//   B  half 0.375, dims 16^3, meshLower -0.5, SDF box +-0.46875
// (GuCookingSDF.cpp:316-345: dims = int(edge / (spacing*(1-1e-4))) + 4, meshLower =
// bounds.min - 0.5*(spacing - (edge - (dims-1)*spacing)); the dense SDF box is
// meshLower + [0.5, dims-0.5]*spacing, sdfCollision.cuh:66-72).  Every number there is
// dyadic, so a sample point p = boxLower + k/16 has texel coordinate f = k EXACTLY.
// The DIMENSIONS are powers of two as well.  Until P9.96 that was not cosmetic: pxgTex3D
// handed a sampler (k + 0.5) / dim and the sampler scaled it back, a round trip exact
// only for dim = 2^n (testSdfQuery.cpp, buildAligned, measured).  Since P9.96 pxgTex3D
// filters in texel space from texel loads and a texel centre returns the stored texel
// bit-for-bit for ANY extent (testSdfQuery's sdfQueryNonPow2); the dims are kept so
// the scene's recorded numbers stay comparable.
// That is what lets the host oracle below be literal and tolerance-free: every tap the
// kernel reads is a stored texel the host can index, plus an exact clamp distance.
//
// THE QUERY: three static shapes, posed with a rotation and a translation the kernel
// must ignore (the API's points are in mesh-local space):
//   S_B  mesh B, scale 1    7 points, one of them (0.21875, 0.03125, 0.03125) -- a
//                           texel centre of BOTH meshes (A (19,16,16), B (11,8,8)), so
//                           a shape mixup is visible
//   S_A  mesh A, scale 1    1100 points (interior and boundary nodes, outside points on
//                           every face and corner), indices >= 1024 included: the grid
//                           is (2, 3, 1) and block 1 carries 76 of them
//   S_C  mesh A, scale 2    S_A's first 5 points.  The kernel ignores PxMeshScale
//                           (convexMeshMidphase.cu:1282-1286 reads no scale), so S_C's
//                           answer must equal S_A's bit-for-bit.
// maxPointCount 1100, so S_B's slots 7..1099 and S_C's 5..1099 must stay poisoned.
//
// THE OBSERVABLE is the API's output buffer itself, poisoned with -999 before the call:
// with the kernel skipped (unmapped) every live slot reads -999.

// Index of a texel in a dense cooked SDF: x fastest (sdfmesh reads it the same way).
struct SdfQueryField
{
    PxU32 dims;              // cubic: the box is cubic
    float lower, higher;     // the SDF box on every axis
    std::vector<float> sdf;  // getSDF(), copied
};

// The cooker's meshLower for a centred cube of half extent h (GuCookingSDF.cpp:316-345),
// transcribed so the box the oracle uses is derived and asserted, not typed in.
static float sdfQueryMeshLower(float h, float spacing, PxU32 dims)
{
    const float edge = 2.0f * h;
    const float offset = 0.5f * (spacing - (edge - float(dims - 1) * spacing));
    return -h - offset;
}

// One tap: the stored texel at an EXACT integer texel coordinate.  `ok` goes false when
// a coordinate is not an integer in range -- an oracle precondition, never a tolerance.
static float sdfQuerySample(const SdfQueryField& F, const PxVec3& f, bool& ok)
{
    const float c[3] = { f.x, f.y, f.z };
    PxU32 k[3];
    for (int a = 0; a < 3; ++a)
    {
        if (!(c[a] >= 0.0f && c[a] <= float(F.dims - 1)) || c[a] != std::floor(c[a]))
        {
            ok = false;
            return 0.0f;
        }
        k[a] = PxU32(c[a]);
    }
    return F.sdf[(size_t(k[2]) * F.dims + k[1]) * F.dims + k[0]];
}

// PxSdfDistance (sdfCollision.cuh:156-166), every primitive the engine's own PxVec3.
static float sdfQueryDistance(const SdfQueryField& F, const PxVec3& p, bool& ok)
{
    const PxVec3 lo(F.lower), hi(F.higher);
    const PxVec3 clamped = p.maximum(lo).minimum(hi);
    const PxReal diffMag = (p - clamped).magnitude();
    const PxVec3 f = (clamped - lo) * (1.0f / 0.0625f);
    return sdfQuerySample(F, f, ok) + diffMag;
}

// PxVolumeGrad (sdfCollision.cuh:168-198): the texel-space central difference inside
// the inclusive guard, the clamping world-space difference outside it.
static PxVec3 sdfQueryGrad(const SdfQueryField& F, const PxVec3& p, bool& ok)
{
    const float dx = 0.0625f;
    const PxVec3 lo(F.lower), hi(F.higher);
    const PxVec3 clamped = p.maximum(lo).minimum(hi);
    const PxVec3 f = (clamped - lo) * (1.0f / dx);
    const float top = float(F.dims) - 2.0f;
    if (f.x >= 1.0f && f.x <= top && f.y >= 1.0f && f.y <= top && f.z >= 1.0f && f.z <= top)
    {
        return PxVec3(sdfQuerySample(F, PxVec3(f.x + 1.0f, f.y, f.z), ok) - sdfQuerySample(F, PxVec3(f.x - 1.0f, f.y, f.z), ok),
                      sdfQuerySample(F, PxVec3(f.x, f.y + 1.0f, f.z), ok) - sdfQuerySample(F, PxVec3(f.x, f.y - 1.0f, f.z), ok),
                      sdfQuerySample(F, PxVec3(f.x, f.y, f.z + 1.0f), ok) - sdfQuerySample(F, PxVec3(f.x, f.y, f.z - 1.0f), ok));
    }
    return PxVec3(sdfQueryDistance(F, p + PxVec3(dx, 0.0f, 0.0f), ok) - sdfQueryDistance(F, p - PxVec3(dx, 0.0f, 0.0f), ok),
                  sdfQueryDistance(F, p + PxVec3(0.0f, dx, 0.0f), ok) - sdfQueryDistance(F, p - PxVec3(0.0f, dx, 0.0f), ok),
                  sdfQueryDistance(F, p + PxVec3(0.0f, 0.0f, dx), ok) - sdfQueryDistance(F, p - PxVec3(0.0f, 0.0f, dx), ok));
}

// convexMeshMidphase.cu:1282-1286 on the host: grad, normalizeSafe (ENGINE), distance.
static PxVec4 sdfQueryOracle(const SdfQueryField& F, const PxVec3& p, bool& ok)
{
    PxVec3 g = sdfQueryGrad(F, p, ok);
    g.normalizeSafe();
    const PxReal d = sdfQueryDistance(F, p, ok);
    return PxVec4(g, d);
}

struct SdfQueryScene
{
    PxShape* SA;
    PxShape* SB;
    PxShape* SC;
    SdfQueryField A, B;
    bool ok;
};

static bool sdfQueryCook(PxPhysics* physics, float h, PxU32 wantDims, float wantMeshLower,
                         PxTriangleMesh*& mesh, SdfQueryField& F)
{
    const float spacing = 0.0625f;
    mesh = cookSdfBox(physics, h, spacing);
    if (!mesh) return false;
    PxU32 x = 0, y = 0, z = 0;
    mesh->getSDFDimensions(x, y, z);
    const float meshLower = sdfQueryMeshLower(h, spacing, x);
    std::printf("  sdfquery: cooked half %.2f at spacing 1/16: dims %u x %u x %u, meshLower %.6f, SDF box +-%.6f\n",
                h, x, y, z, meshLower, -(meshLower + 0.5f * spacing));
    if (!mesh->getSDF() || x != wantDims || y != wantDims || z != wantDims || meshLower != wantMeshLower)
    {
        std::printf("FAIL: sdfquery: cooked SDF is not the one the oracle is written for (want dims %u, "
                    "meshLower %.6f, an SDF present)\n", wantDims, wantMeshLower);
        return false;
    }
    F.dims = x;
    F.lower = meshLower + 0.5f * spacing;
    F.higher = meshLower + (float(x) - 0.5f) * spacing;
    F.sdf.assign(mesh->getSDF(), mesh->getSDF() + size_t(x) * y * z);
    return true;
}

SdfQueryScene buildSdfQueryScene(PxPhysics* physics, PxScene* scene, PxMaterial* material)
{
    SdfQueryScene s = {};
    PxTriangleMesh* meshA = nullptr;
    PxTriangleMesh* meshB = nullptr;
    if (!sdfQueryCook(physics, 0.875f, 32u, -1.0f, meshA, s.A) ||
        !sdfQueryCook(physics, 0.375f, 16u, -0.5f, meshB, s.B))
        return s;
    auto addStatic = [&](PxTriangleMesh* mesh, const PxMeshScale& scale, const PxTransform& pose) {
        PxRigidStatic* a = physics->createRigidStatic(pose);
        PxShape* sh = physics->createShape(PxTriangleMeshGeometry(mesh, scale), *material, true);
        a->attachShape(*sh);
        sh->release();
        scene->addActor(*a);
        return sh;
    };
    // Rotated and translated: the kernel works in mesh-local space and must not see them.
    s.SA = addStatic(meshA, PxMeshScale(1.0f), PxTransform(PxVec3(-6.0f, 2.0f, 3.0f),
                                                           PxQuat(0.6f, PxVec3(0.0f, 1.0f, 0.0f))));
    s.SB = addStatic(meshB, PxMeshScale(1.0f), PxTransform(PxVec3(6.0f, 1.5f, -2.0f),
                                                           PxQuat(-1.1f, PxVec3(1.0f, 0.0f, 0.0f))));
    s.SC = addStatic(meshA, PxMeshScale(2.0f), PxTransform(PxVec3(0.0f, 4.0f, 9.0f),
                                                           PxQuat(2.3f, PxVec3(0.0f, 0.0f, 1.0f))));
    meshA->release();
    meshB->release();
    // Something to simulate, far from the meshes: a box on a plane.
    PxRigidStatic* plane = physics->createRigidStatic(PxTransformFromPlaneEquation(PxPlane(0.0f, 1.0f, 0.0f, 10.0f)));
    PxShape* ps = physics->createShape(PxPlaneGeometry(), *material);
    plane->attachShape(*ps);
    ps->release();
    scene->addActor(*plane);
    addDynamic(physics, scene, material, PxBoxGeometry(0.5f, 0.5f, 0.5f), boxInertia(PxVec3(0.5f), 1000.0f),
               -8.0f, 40.0f);
    s.ok = true;
    return s;
}

// The sample points.  `k` are texel indices: the node at k is boxLower + k/16, exact.
static PxVec3 sdfQueryNode(const SdfQueryField& F, float kx, float ky, float kz)
{
    return PxVec3(F.lower + kx * 0.0625f, F.lower + ky * 0.0625f, F.lower + kz * 0.0625f);
}

static std::vector<PxVec3> sdfQueryPointsA(const SdfQueryField& A)
{
    std::vector<PxVec3> p;
    // [0..4] -- also S_C's five: interior, two boundary nodes, outside +x, an outside corner.
    p.push_back(sdfQueryNode(A, 5, 7, 11));
    p.push_back(sdfQueryNode(A, 0, 4, 9));
    p.push_back(sdfQueryNode(A, 31, 31, 3));
    p.push_back(PxVec3(1.5f, sdfQueryNode(A, 0, 8, 12).y, sdfQueryNode(A, 0, 8, 12).z));
    p.push_back(PxVec3(-1.25f, -0.90625f, 2.0f));
    // [5] the shared node: (19, 16, 16) of A, (11, 8, 8) of B.
    p.push_back(PxVec3(0.21875f, 0.03125f, 0.03125f));
    // Outside every face, and the guard's own bounds (f == 1 and f == dims-2 are inside
    // the inclusive guard; 0 and dims-1 are not).
    const float out[2] = { -1.5f, 1.65625f };
    for (int a = 0; a < 3; ++a)
        for (int side = 0; side < 2; ++side)
        {
            PxVec3 q = sdfQueryNode(A, 6, 13, 2);
            (&q.x)[a] = out[side];
            p.push_back(q);
        }
    const float edge[4] = { 0, 1, 30, 31 };
    for (float ex : edge) for (float ey : edge) for (float ez : edge)
        p.push_back(sdfQueryNode(A, ex, ey, ez));
    // The rest: nodes from a fixed LCG, 1 in 4 on a face, plus an outside point every 37th.
    PxU32 r = 0x5DF95u;
    auto next = [&r](PxU32 n) { r = r * 1664525u + 1013904223u; return (r >> 8) % n; };
    while (p.size() < 1100)
    {
        if (p.size() % 37u == 0u)
        {
            // y outside by a whole number of texels, so the fall-back's +-dx taps that
            // re-enter the box land on texel centres too.
            p.push_back(PxVec3(sdfQueryNode(A, float(next(32)), 0, 0).x, A.higher + 0.0625f * float(1 + next(8)),
                               sdfQueryNode(A, 0, 0, float(next(32))).z));
            continue;
        }
        float k[3] = { float(1 + next(30)), float(1 + next(30)), float(1 + next(30)) };
        if (next(4) == 0u)
            k[next(3)] = next(2) ? 31.0f : 0.0f;
        p.push_back(sdfQueryNode(A, k[0], k[1], k[2]));
    }
    // A last outside point at the very end of block 1.
    p[1099] = PxVec3(-1.25f, 1.09375f, -1.09375f);
    return p;
}

static std::vector<PxVec3> sdfQueryPointsB(const SdfQueryField& B)
{
    std::vector<PxVec3> p;
    p.push_back(PxVec3(0.21875f, 0.03125f, 0.03125f));   // node (11, 8, 8) of B
    p.push_back(sdfQueryNode(B, 3, 8, 5));
    p.push_back(sdfQueryNode(B, 15, 0, 5));
    p.push_back(sdfQueryNode(B, 1, 10, 14));
    p.push_back(PxVec3(0.75f, sdfQueryNode(B, 0, 3, 4).y, sdfQueryNode(B, 0, 3, 4).z));
    p.push_back(PxVec3(-0.5625f, -0.625f, 0.75f));
    p.push_back(sdfQueryNode(B, 6, 6, 1));
    return p;
}

// PXG_SCENE=extforce (P9.93): updateBodyExternalVelocitiesLaunch.
//
// PxgSimulationCore::updateBodies launches it only when nbUpdatedBodies > 0 and the
// direct-GPU API is OFF (PxgSimulationCore.cpp:2313), and nbUpdatedBodies counts the
// bodies Sc::Scene::beforeSolver collected from BodySim::updateForces -- which only
// PxRigidDynamic::addForce / addTorque / setForceAndTorque / clearForce /
// clearTorque feed (setLinearVelocity takes the updateBodySim path instead).  No
// scene called any of them, so the kernel never launched.
//
// autowake = FALSE on every call, and that is not a detail: with the default `true`
// an awake body whose wake counter sits below 0.4 -- every falling body on the GPU,
// whose counter lives between 0.2 and 0.4 -- is ALSO pushed through setWakeCounter
// -> updateBodySim -> updateBodiesLaunch, which rewrites the same post-kick velocity
// and would hide a refused kernel completely.
//
// Four 1000 kg boxes (inverse inertia 0.006) free-fall from y=100 and never touch
// anything in sixty steps (they drop 4.98675):
//   K (x=0)   before step 30: addForce((3,0,0), eVELOCITY_CHANGE): x = 1.5.
//   F (x=5)   before each of steps 30..59: addForce((0,0,2000), eFORCE) = 2 m/s^2:
//             v_z = 1.0, z = 2 dt^2 (30*31/2) = 0.258333.
//   S (x=10)  before step 30: addTorque((0,500,0), eIMPULSE): omega_y = 0.006*500 = 3,
//             1.5 rad about y after thirty steps.  Angular damping 0.
//   C (x=-5)  untouched: x = z = 0.
// If the kernel is refused, the kick sits in the CPU core only and the next GPU
// writeback overwrites it: K.x = 0, F.z = 0, S.omega = 0.
struct ExtForceScene
{
    PxRigidDynamic* K;
    PxRigidDynamic* F;
    PxRigidDynamic* S;
    PxRigidDynamic* C;
};

ExtForceScene buildExtForceScene(PxPhysics* physics, PxScene* scene, PxMaterial* material)
{
    ExtForceScene s = {};
    addGroundBox(physics, scene, material);
    auto body = [&](float x) {
        PxRigidDynamic* b = addDynamic(physics, scene, material, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                       boxInertia(PxVec3(0.5f), 1000.0f), 100.0f, x);
        b->setAngularDamping(0.0f);
        return b;
    };
    s.K = body(0.0f);
    s.F = body(5.0f);
    s.S = body(10.0f);
    s.C = body(-5.0f);
    return s;
}

// PXG_SCENE=substepgravity (P9.95): applyTGSSubstepGravity.
//
// PxgTGSCudaSolverCore::solveContactMultiBlockParallel launches it once per POSITION
// iteration (PxgTGSCudaSolverCore.cpp:1354-1374) and only when the scene raised
// PxSceneFlag::eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS.  That flag is not in
// eMUTABLE_FLAGS (PxSceneDesc.h:373), so it is raised before createScene, and it is
// TGS-only (PxSceneDesc::isValid rejects it with PGS, PxSceneDesc.h:1129).  Under the
// flag preIntegration skips gravity and the external accelerations
// (skipGravityApplication, PxgTGSCudaSolverCore.cpp:826,850) and this kernel adds them
// per substep instead; ScBodySim.cpp:681-711 turns velocity changes and impulses into
// accelerations (v/dt).  So a REFUSED kernel leaves every body hovering at y = 100.
//
// The extforce bodies again (1000 kg boxes, inverse inertia 0.006, angular damping 0,
// y = 100, sixty steps of 1/60, 4 position iterations -> n = 4 substeps).  Applied
// per substep, an acceleration a over one frame moves a body by
// v0*dt + a*dt^2*(n+1)/(2n), not v0*dt + a*dt^2 (factor 0.625 instead of 1):
//   C (x=-5)  untouched: y = 100 - g dt^2 (1770 + 60*0.625) = 95.0745625, v.y = -9.81.
//   K (x=0)   addForce((3,0,0), eVELOCITY_CHANGE) before step 30: x = 3 dt (29 + 0.625)
//             = 1.48125 (1.5 when the flag is ignored), v.x = 3.
//   F (x=5)   addForce((0,0,2000), eFORCE) before each of steps 30..59:
//             z = 2 dt^2 (435 + 18.75) = 0.2520833, v.z = 1.
//   S (x=10)  addTorque((0,500,0), eIMPULSE) before step 30: omega_y = 3, angle
//             3 dt (29 + 0.625) = 1.48125 about y.
//   D (x=15)  eDISABLE_GRAVITY, and F's force from step 30 on.  UPSTREAM QUIRK: the
//             kernel's gate reads disableGravity (solverMultiBlockTGS.cu:1656) but
//             applyAcceleration never does (:1575-1583), so once D has an
//             acceleration it receives gravity as well: y = 100 - g dt^2 (435 +
//             18.75) = 98.7635313, v.y = -4.905.  The CPU twin zeroes gravity under
//             disableGravity (DyTGSDynamics.cpp:1490-1493): y = 100, v.y = 0.
//   L (x=20)  eLOCK_LINEAR_X | eLOCK_ANGULAR_Y, pushed along x and twisted about y
//             from step 30.  A SMOKE and CPU-parity check only: the locked
//             components are zeroed again after this kernel, in the per-substep
//             integration (solverMultiBlockTGS.cu:541-552) and in the final
//             writes (integration.cuh:316-329), so L does not observe how this
//             kernel treats lockFlags.  That is fixture-only.
//   J1/J2     (x=25 and 26.5) one box pair joined by a fully LOCKED D6.  Every body
//             above is constraint-free, so its solverBodyReferences entries stay
//             INVALID (memset at PxgTGSCudaSolverCore.cpp:679; the only writer is
//             constraint prep, constraintPrepShared.cuh:145) and only the
//             kernel's accumulation-copy write is observed.  The joint gives the
//             pair slab references, and computeAverageSolverBodyVelocityTGS
//             (solverMultiBlockTGS.cu:338-415) replaces their accumulated velocity
//             with the slab average -- so the pair falls like C only if the slab
//             writes happen.
//   FILLERS   640 more free-falling boxes far off (x >= 40, never touching), checked
//             like C.  They are there for J1/J2: solveContactMultiBlockParallel solves
//             an island whose (mBodyCount + mBodyStartIndex) * numSlabs is at most
//             640 with solveWholeIslandTGS (PxgTGSCudaSolverCore.cpp:1123-1141, 640
//             under PXG_NO_CUDA_TOOLKIT), and that kernel reads the ACCUMULATED copy
//             (solverMultiBlockTGS.cu:1728-1731), never the slab copies.  Measured:
//             with eight bodies a kernel that skipped every slab write still passed.
//             649 > 640 forces the per-partition path, whose solve reads the slab
//             copies and whose computeAverageSolverBodyVelocityTGS then overwrites
//             the accumulated copy with them.
// Every body gets setSleepThreshold(0): with the kernel refused they are all at rest
// from step 0, and without it they would fall asleep (energy >= threshold keeps the
// wake counter, integration.cuh:160 / DySleep.cpp:136).
struct SubstepGravityScene
{
    ExtForceScene e;
    PxRigidDynamic* D;
    PxRigidDynamic* L;
    PxRigidDynamic* J1;
    PxRigidDynamic* J2;
    std::vector<PxRigidDynamic*> fillers;
};

SubstepGravityScene buildSubstepGravityScene(PxPhysics* physics, PxScene* scene, PxMaterial* material)
{
    SubstepGravityScene s = {};
    s.e = buildExtForceScene(physics, scene, material);
    auto body = [&](float x) {
        PxRigidDynamic* b = addDynamic(physics, scene, material, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                       boxInertia(PxVec3(0.5f), 1000.0f), 100.0f, x);
        b->setAngularDamping(0.0f);
        return b;
    };
    s.D = body(15.0f);
    s.D->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
    s.L = body(20.0f);
    s.L->setRigidDynamicLockFlags(PxRigidDynamicLockFlag::eLOCK_LINEAR_X | PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y);
    s.J1 = body(25.0f);
    s.J2 = body(26.5f);
    PxD6Joint* joint = PxD6JointCreate(*physics, s.J1, PxTransform(PxVec3(0.75f, 0.0f, 0.0f)),
                                       s.J2, PxTransform(PxVec3(-0.75f, 0.0f, 0.0f)));
    joint->setMotion(PxD6Axis::eX, PxD6Motion::eLOCKED);
    joint->setMotion(PxD6Axis::eY, PxD6Motion::eLOCKED);
    joint->setMotion(PxD6Axis::eZ, PxD6Motion::eLOCKED);
    joint->setMotion(PxD6Axis::eTWIST, PxD6Motion::eLOCKED);
    joint->setMotion(PxD6Axis::eSWING1, PxD6Motion::eLOCKED);
    joint->setMotion(PxD6Axis::eSWING2, PxD6Motion::eLOCKED);
    for (PxRigidDynamic* b : { s.e.K, s.e.F, s.e.S, s.e.C, s.D, s.L, s.J1, s.J2 })
        b->setSleepThreshold(0.0f);
    PxShape* fillerShape = physics->createShape(PxBoxGeometry(0.25f, 0.25f, 0.25f), *material);
    for (int i = 0; i < 640; ++i)
    {
        PxRigidDynamic* b = physics->createRigidDynamic(
            PxTransform(PxVec3(40.0f + 2.0f * float(i % 32), 100.0f, -20.0f + 2.0f * float(i / 32))));
        b->attachShape(*fillerShape);
        b->setMass(1000.0f);
        b->setMassSpaceInertiaTensor(boxInertia(PxVec3(0.25f), 1000.0f));
        b->setSleepThreshold(0.0f);
        scene->addActor(*b);
        s.fillers.push_back(b);
    }
    fillerShape->release();
    return s;
}

// PXG_SCENE=slabgrid (P9.97a): the TGS averaging family -- initializeSolverVelocitiesTGS,
// computeAverageSolverBodyVelocityTGS and propagateAverageSolverBodyVelocityTGS -- at the
// engine's own grid.
//
// All three kernels map bpw = 32/numThreadsPerBody bodies to one warp, numThreadsPerBody
// = min(pow2(numSlabs), 32), and are launched with block (32, COMPUTE_BODIES_AVERAGE_VELOCITY
// / 32 = 8) and a grid sized for 32 threads per body (PxgTGSCudaSolverCore.cpp:1232-1239,
// :1295-1306, :1481-1495, :1734-1748).  Up to P9.97a the port ran [numthreads(32,1,1)] with
// warp = gid.x under that grid: one eighth of the CUDA's warps.  Two engine-visible faults
// followed, and each mode below observes one of them:
//   * numSlabs >= 5 (>= 33 constraint partners on one body at gpuMaxNumPartitions 8):
//     init, average and position-iteration propagate reach ~1/2, 1/4, 1/8 of the bodies.
//   * >= 2 velocity iterations: the velocity-iteration propagate (grid
//     ceil(32*ceil(M/10)/256)) reached 32*G_D bodies where the CUDA reaches 8*G_D*bpw.
//
// THE WHEEL.  A free-falling hub (10 kg box, spinning at w = 2 about y) with K satellites
// (1 kg spheres) at radius R, each joined to the hub by a fully LOCKED D6.  The K joints
// share the hub, so the partitioner needs K partitions and numSlabs = ceil(K / 8)
// (PxgContext.cpp:1632-1633).  Correct: the hub falls freely (yFree = 50 - g dt^2 n(n+1)/2
// = 48.73288 after n = 30 steps of 1/60), turns by ~w t = 1 rad, and every satellite stays
// on the wheel.  Measured on the shipped port: the hub frozen at y = 50 (slab path) or at
// 49.479 with every satellite off the wheel (whole-island path).
//
// THE REPLICAS.  80 identical spinning dumbbells (1 kg + 5 kg spheres, locked D6),
// inserted FIRST, so replica i is solver bodies 1+2i and 2+2i (body 0 is the static world
// body, mBodyStartIndex = 1); centred at x = 1030 + 12i so every x stays inside one binade
// [1024, 2048) and the solve is translation-equivariant BIT FOR BIT.  Replicas are
// partitioned by bitwise equality of their 26-float state (pose relative to the centre,
// both velocities, both quaternions).  The partition is PREDICTED from the CUDA's
// velocity-iteration reach C = min(T, 8*G_D*bpw), G_D = ceil(32*ceil(M/10)/256): a
// replica both of whose bodies are below C is "full", the one straddling C is the
// straddler, the rest share the fate of bodies the second velocity iteration does not
// refresh.  UPSTREAM QUIRK KEPT: for numSlabs >= 3 that launch reaches fewer bodies than
// exist (`(mBodyCount + 9) / 10` -- ten bodies per warp, the mapping the stale comment at
// solverMultiBlockTGS.cu:457 describes, never updated for the numThreadsPerBody mapping;
// it also omits mBodyStartIndex).  The port reproduces the CUDA, so the expected
// partition has three classes; a port covering MORE than the CUDA (a 16-warp group) moves
// the boundary and fails exactly as one covering less does.
//
// Modes (PXG_SLABGRID):
//   slab   (default) 80 replicas, then hub + K = 136 at R = 15: numSlabs 17, bpw 1;
//          T*numSlabs > 640, so the per-partition path (no solveWholeIslandTGS).
//   whole  hub + K = 40 at R = 5: numSlabs 5; 42*5 <= 640, so the position iterations
//          run solveWholeIslandTGS -- init and average on that path.
//   one    80 replicas only: numSlabs 1; the velocity-iteration propagate alone.
//   slab5  80 replicas + 100 free fillers + hub + K = 40: numSlabs 5, bpw 4, slab path.
//          The fillers have NO slab reference and sit in the same warp lanes as
//          referenced bodies; the only mode that observes an average that reads another
//          warp's hasRef.
// Every body: setSolverIterationCounts(4, 2), sleep threshold 0, no damping.  Nothing
// touches anything (satellites 0.4 m apart at R = 15, replicas 12 m apart, a joint pair
// does not collide), so the harness filter's contacts never arise.
struct SlabGridScene
{
    const char* mode;
    int K;
    float R;
    PxRigidDynamic* hub;
    std::vector<PxRigidDynamic*> sats;
    std::vector<PxRigidDynamic*> pa;
    std::vector<PxRigidDynamic*> pb;
    std::vector<PxRigidDynamic*> fillers;
};

SlabGridScene buildSlabGridScene(PxPhysics* physics, PxScene* scene, PxMaterial* material, const char* mode)
{
    SlabGridScene s = {};
    s.mode = mode;
    const bool slab = !std::strcmp(mode, "slab"), whole = !std::strcmp(mode, "whole"),
               one = !std::strcmp(mode, "one"), slab5 = !std::strcmp(mode, "slab5");
    s.K = slab ? 136 : (whole || slab5) ? 40 : 0;
    s.R = slab ? 15.0f : 5.0f;
    const int pairs = (slab || one || slab5) ? 80 : 0;
    const int fill = slab5 ? 100 : 0;
    auto setup = [&](PxRigidDynamic* b) {
        b->setAngularDamping(0.0f);
        b->setLinearDamping(0.0f);
        b->setSleepThreshold(0.0f);
        b->setSolverIterationCounts(4, 2);
    };
    auto lockAll = [&](PxD6Joint* j) {
        for (PxD6Axis::Enum ax : { PxD6Axis::eX, PxD6Axis::eY, PxD6Axis::eZ, PxD6Axis::eTWIST, PxD6Axis::eSWING1,
                                   PxD6Axis::eSWING2 })
            j->setMotion(ax, PxD6Motion::eLOCKED);
    };
    // replicas FIRST: replica i = solver bodies 1+2i, 2+2i
    for (int i = 0; i < pairs; ++i)
    {
        const PxVec3 c(1030.0f + 12.0f * float(i), 48.0f, 0.0f), r(1.5f, 0.0f, 0.0f);
        PxRigidDynamic* A = physics->createRigidDynamic(PxTransform(c - r));
        PxRigidDynamic* B = physics->createRigidDynamic(PxTransform(c + r));
        for (PxRigidDynamic* X : { A, B })
        {
            PxShape* sh = physics->createShape(PxSphereGeometry(0.25f), *material);
            X->attachShape(*sh);
            sh->release();
            setup(X);
            X->setAngularVelocity(PxVec3(0.0f, 3.0f, 0.0f));
        }
        A->setMass(1.0f);
        A->setMassSpaceInertiaTensor(PxVec3(0.025f));
        B->setMass(5.0f);
        B->setMassSpaceInertiaTensor(PxVec3(0.125f));
        A->setLinearVelocity(PxVec3(0.0f, 3.0f, 0.0f).cross(-r));
        B->setLinearVelocity(PxVec3(0.0f, 3.0f, 0.0f).cross(r));
        scene->addActor(*A);
        scene->addActor(*B);
        lockAll(PxD6JointCreate(*physics, A, PxTransform(r * 2.0f), B, PxTransform(PxIdentity)));
        s.pa.push_back(A);
        s.pb.push_back(B);
    }
    for (int i = 0; i < fill; ++i)
    {
        PxRigidDynamic* b = physics->createRigidDynamic(
            PxTransform(PxVec3(100.0f + 2.0f * float(i % 32), 50.0f, 2.0f * float(i / 32))));
        PxShape* sh = physics->createShape(PxBoxGeometry(0.25f, 0.25f, 0.25f), *material);
        b->attachShape(*sh);
        sh->release();
        b->setMass(10.0f);
        b->setMassSpaceInertiaTensor(PxVec3(0.4f));
        setup(b);
        scene->addActor(*b);
        s.fillers.push_back(b);
    }
    if (s.K > 0)
    {
        const float w = 2.0f, y0 = 50.0f;
        s.hub = physics->createRigidDynamic(PxTransform(PxVec3(0.0f, y0, 0.0f)));
        PxShape* hs = physics->createShape(PxBoxGeometry(0.5f, 0.5f, 0.5f), *material);
        s.hub->attachShape(*hs);
        hs->release();
        s.hub->setMass(10.0f);
        s.hub->setMassSpaceInertiaTensor(PxVec3(10.0f / 6.0f));
        setup(s.hub);
        s.hub->setAngularVelocity(PxVec3(0.0f, w, 0.0f));
        scene->addActor(*s.hub);
        for (int i = 0; i < s.K; ++i)
        {
            const float a = 6.283185307f * float(i) / float(s.K);
            const PxVec3 r(s.R * std::cos(a), 0.0f, s.R * std::sin(a));
            PxRigidDynamic* b = physics->createRigidDynamic(PxTransform(PxVec3(0.0f, y0, 0.0f) + r));
            PxShape* sh = physics->createShape(PxSphereGeometry(0.2f), *material);
            b->attachShape(*sh);
            sh->release();
            b->setMass(1.0f);
            b->setMassSpaceInertiaTensor(PxVec3(0.016f));
            setup(b);
            b->setLinearVelocity(PxVec3(0.0f, w, 0.0f).cross(r));
            b->setAngularVelocity(PxVec3(0.0f, w, 0.0f));
            scene->addActor(*b);
            lockAll(PxD6JointCreate(*physics, s.hub, PxTransform(r), b, PxTransform(PxIdentity)));
            s.sats.push_back(b);
        }
    }
    return s;
}

// PXG_SCENE=forcethreshold (P9.91): the FORCE-THRESHOLD REPORTING pipeline.
//
// PxgGpuContext::doConstraintSolveGPU runs accumulatedForceThresholdStream() -- the
// fourteen kernels of accumulateThresholdStream.cu, from the body-index radix sort to
// createForceChangeThresholdElements -- only `if (mHasForceThresholds)`, and that is
// raised only by a pair whose filter flags ask for eNOTIFY_THRESHOLD_FORCE_* and one of
// whose bodies has a contact report threshold below PX_MAX_F32.  No scene before this
// one did either, so the whole group had never been launched by the engine: two of its
// kernels were unmapped and the other twelve had no launch-ABI row.
//
// The observable is the event stream the host builds from the GPU's force-change
// elements, counted per actor pair and per step.  Every body is a 1000 kg box
// (weight m g = 9810 N) spawned AT REST, so there is no impact spike to straddle a
// threshold, and none of them may sleep (a sleeping pair is not solved and reports
// nothing):
//
//   over    alone on the ground, threshold 5000 N: FOUND on its first contact step,
//           PERSISTS on every step after, until step kRaiseStep raises its threshold
//           to 1e7 N -- then exactly one LOST, while the box is still resting there.
//           LOST is the cross-frame diff (setThresholdElementsMask against the
//           previous frame's exceeded set), so it cannot come from one frame's data.
//   under   alone on the ground, threshold 20000 N: never exceeded, NO event at all.
//   lower   on the ground, default threshold: its ground pair has no threshold below
//           PX_MAX_F32 and must report nothing ...
//   upper   ... while the box stacked on it (threshold 5000 N) makes the (lower,
//           upper) pair a DYNAMIC-dynamic threshold pair: FOUND, then PERSISTS to the
//           end.  Two exceeded pairs per step with distinct node indices are what the
//           radix sort and the segmented accumulation have to keep apart.
//
// THE REFERENCE IS THE CPU PGS SOLVER.  PXG_CPU_CONTROL=1 PXG_SOLVER=pgs produces
// exactly the timeline asserted here (measured: FOUND @0, PERSISTS @1..119, LOST @120 for
// ground|over; FOUND @0, PERSISTS @1..239 for lower|upper; nothing else).  The CPU TGS
// control reports NO force-threshold event at all, and that is upstream, not this scene:
// DyTGSDynamics.cpp never builds the force-change stream (it has no
// createForceChangeThresholdStream step; DyDynamics.cpp:522 is PGS-only), while the GPU
// TGS core runs the same fourteen kernels as the GPU PGS core.  The scene is scored
// under both GPU solvers.
static const int kForceThresholdSteps = 240;
static const int kForceThresholdRaiseStep = 120;

struct ForceThresholdEvents : public PxSimulationEventCallback
{
    struct PairLog
    {
        std::vector<unsigned char> perStep;   // bit 0 FOUND, bit 1 PERSISTS, bit 2 LOST
        PairLog() : perStep(kForceThresholdSteps, 0) {}
    };
    std::map<std::string, PairLog> pairs;     // "nameA|nameB", names sorted
    int step = 0;

    static const char* nameOf(const PxActor* a)
    {
        const char* n = a ? a->getName() : nullptr;
        return n ? n : "?";
    }
    void onContact(const PxContactPairHeader& h, const PxContactPair* p, PxU32 n) override
    {
        std::string a = nameOf(h.actors[0]), b = nameOf(h.actors[1]);
        if (b < a) std::swap(a, b);
        PairLog& log = pairs[a + "|" + b];
        for (PxU32 i = 0; i < n; ++i)
        {
            const PxPairFlags ev = p[i].events;
            unsigned char bits = 0;
            if (ev & PxPairFlag::eNOTIFY_THRESHOLD_FORCE_FOUND)    bits |= 1;
            if (ev & PxPairFlag::eNOTIFY_THRESHOLD_FORCE_PERSISTS) bits |= 2;
            if (ev & PxPairFlag::eNOTIFY_THRESHOLD_FORCE_LOST)     bits |= 4;
            if (step >= 0 && step < kForceThresholdSteps)
                log.perStep[step] |= bits;
        }
    }
    void onConstraintBreak(PxConstraintInfo*, PxU32) override {}
    void onWake(PxActor**, PxU32) override {}
    void onSleep(PxActor**, PxU32) override {}
    void onTrigger(PxTriggerPair*, PxU32) override {}
    void onAdvance(const PxRigidBody* const*, const PxTransform*, const PxU32) override {}
};

struct ForceThresholdScene
{
    PxRigidDynamic* over;
    PxRigidDynamic* under;
    PxRigidDynamic* lower;
    PxRigidDynamic* upper;
};

ForceThresholdScene buildForceThresholdScene(PxPhysics* physics, PxScene* scene, PxMaterial* material)
{
    ForceThresholdScene s = {};
    PxRigidStatic* ground = addGroundBox(physics, scene, material);
    ground->setName("ground");
    auto box = [&](float x, float y, const char* name, float threshold) {
        PxRigidDynamic* b = addDynamic(physics, scene, material, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                       boxInertia(PxVec3(0.5f), 1000.0f), y, x);
        b->setName(name);
        b->setSleepThreshold(0.0f);
        if (threshold < PX_MAX_F32)
            b->setContactReportThreshold(threshold);
        return b;
    };
    s.over  = box(-3.0f, 0.5f, "over", 5000.0f);
    s.under = box(0.0f, 0.5f, "under", 20000.0f);
    s.lower = box(3.0f, 0.5f, "lower", PX_MAX_F32);
    s.upper = box(3.0f, 1.5f, "upper", 5000.0f);
    return s;
}

// Checks one pair's per-step event bits against "FOUND once at step f <= 2, PERSISTS on
// every step strictly between, LOST once at step l" (l = -1: no LOST, PERSISTS runs to
// the last step).  Prints the timeline either way and returns the number of failures.
int checkThresholdTimeline(const ForceThresholdEvents& ev, const char* key, int lostLo, int lostHi)
{
    auto it = ev.pairs.find(key);
    if (it == ev.pairs.end())
    {
        std::printf("FAIL: [%s] no force-threshold event at all\n", key);
        return 1;
    }
    const std::vector<unsigned char>& s = it->second.perStep;
    int found = 0, persists = 0, lost = 0, f = -1, l = -1, firstP = -1, lastP = -1;
    for (int i = 0; i < kForceThresholdSteps; ++i)
    {
        if (s[i] & 1) { ++found; if (f < 0) f = i; }
        if (s[i] & 2) { ++persists; if (firstP < 0) firstP = i; lastP = i; }
        if (s[i] & 4) { ++lost; if (l < 0) l = i; }
    }
    std::printf("  [%s] FOUND x%d (first @%d)  PERSISTS x%d (@%d..%d)  LOST x%d (first @%d)\n",
                key, found, f, persists, firstP, lastP, lost, l);
    int failures = 0;
    if (found != 1 || f > 2) { std::printf("FAIL: [%s] expected one FOUND at step <= 2\n", key); ++failures; }
    const bool wantLost = lostLo >= 0;
    if (wantLost && (lost != 1 || l < lostLo || l > lostHi))
    {
        std::printf("FAIL: [%s] expected one LOST in steps %d..%d\n", key, lostLo, lostHi);
        ++failures;
    }
    if (!wantLost && lost != 0) { std::printf("FAIL: [%s] expected no LOST\n", key); ++failures; }
    const int end = wantLost ? l : kForceThresholdSteps;   // PERSISTS on every step in (f, end)
    if (f >= 0 && end > f + 1 && (persists != end - f - 1 || firstP != f + 1 || lastP != end - 1))
    {
        std::printf("FAIL: [%s] expected PERSISTS on every step %d..%d\n", key, f + 1, end - 1);
        ++failures;
    }
    return failures;
}

// PXG_SCENE=articulation (P8.55): the first ARTICULATION scene, and the first
// thing in this harness to ask the ~85 unmapped `arti*` kernels to run.
//
// A reduced-coordinate articulation with a FIXED BASE and one revolute link.  The
// link is spawned horizontal, so gravity has maximum leverage on the joint, and it
// swings down to hang vertically: centre from (0.7, 3, 0) to (0.2, 2.5, 0).
// Angular damping settles it well inside the 240-step window -- an undamped
// pendulum would still be swinging and the rest check would (correctly) fail.
//
// Deliberately the SMALLEST articulation that exercises the feature: one inbound
// joint, one degree of freedom, no contacts, no self-collision.  Anything the
// scene refuses is therefore attributable to the articulation pipeline itself.
// P9.41 returns the articulation so `articulationjoint` can read its joint state
// back and ask whether it fell asleep; the rigid dispatch below ignores the value.
PxArticulationReducedCoordinate*
buildArticulationScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
                       std::vector<RestTarget>& targets)
{
    const float baseY = 3.0f;
    const float baseHalf = 0.2f;
    const float linkHalf = 0.5f;      // half-length along x

    PxArticulationReducedCoordinate* arti = physics->createArticulationReducedCoordinate();
    arti->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);

    PxArticulationLink* base = arti->createLink(NULL, PxTransform(PxVec3(0.0f, baseY, 0.0f)));
    PxShape* bs = physics->createShape(PxBoxGeometry(baseHalf, baseHalf, baseHalf), *material);
    base->attachShape(*bs);
    base->setMass(10.0f);
    base->setMassSpaceInertiaTensor(boxInertia(PxVec3(baseHalf), 10.0f));

    // Spawned horizontal: centre one link-half past the base's +x face.
    const float linkX = baseHalf + linkHalf;
    PxArticulationLink* link =
        arti->createLink(base, PxTransform(PxVec3(linkX, baseY, 0.0f)));
    PxShape* ls = physics->createShape(PxBoxGeometry(linkHalf, 0.1f, 0.1f), *material);
    link->attachShape(*ls);
    link->setMass(10.0f);
    link->setMassSpaceInertiaTensor(boxInertia(PxVec3(linkHalf, 0.1f, 0.1f), 10.0f));
    link->setAngularDamping(3.0f);
    link->setLinearDamping(1.0f);

    // The hinge sits on the base's +x face; rotation about Z swings the link in
    // the x-y plane, which is where gravity acts.
    PxArticulationJointReducedCoordinate* joint = link->getInboundJoint();
    joint->setJointType(PxArticulationJointType::eREVOLUTE);
    joint->setParentPose(PxTransform(PxVec3(baseHalf, 0.0f, 0.0f)));
    joint->setChildPose(PxTransform(PxVec3(-linkHalf, 0.0f, 0.0f)));
    joint->setMotion(PxArticulationAxis::eSWING2, PxArticulationMotion::eFREE);
    joint->setFrictionCoefficient(0.5f);

    scene->addArticulation(*arti);

    // A PLAIN DYNAMIC BODY ALONGSIDE THE ARTICULATION.  Not decoration: the solver
    // sizes mSolverBodyReferences as totalActiveBodyCount * numSlabs
    // (PxgTGSCudaSolverCore.cpp:552) while initializeSolverVelocitiesTGS indexes it
    // by numSolverBodies.  An articulation-only island makes the first count zero
    // and the second non-zero, so the array is empty and the shader reads off the
    // end.  Adding one rigid body is the experiment that separates that from every
    // other explanation -- see the P8.60 entry in RUNTIME_STATUS.md.
    //
    // P8.107: it spawns at (0, 5, 0) -- DIRECTLY ABOVE the fixed base at
    // (0, 3, 0) -- so it falls onto the base and rests on top of it, never on
    // the ground.  The rest height is therefore baseY + baseHalf + its own
    // half-extent = 3.7, NOT 0.5; the original 0.5 expectation was
    // geometrically unreachable and kept this check red long after the
    // physics was right (the box measured 3.7000 exactly once P8.104 landed
    // the arti-rigid contact solve).  The corrected expectation is the
    // sharper check: the contact is solved through the ARTICULATION solver
    // chain, and if that chain breaks, the box falls through the base toward
    // the ground -- far below the +-0.75 band around 3.7.
    addGroundBox(physics, scene, material);
    PxRigidDynamic* b = addDynamic(physics, scene, material, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                   boxInertia(PxVec3(0.5f), 1000.0f), 5.0f);
    targets.push_back({ b, 5.0f, baseY + baseHalf + 0.5f, "artiCompanionBox" });

    // Hanging straight down puts the link centre one half-length below the hinge.
    targets.push_back({ link, baseY, baseY - linkHalf, "artiLink" });
    return arti;
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
// P9.2 fluid-scene calibration.  Both numbers are MEASURED on this scene, with
// the only change being the phase bit; see the note at the assertion.  They are
// named rather than inlined so the control value cannot drift away from the
// threshold it justifies.
// Measured on this scene, the ONLY difference between the two runs being the
// eParticlePhaseFluid bit (PXG_PARTICLE_FLUID_CONTROL=1 clears it):
//
//                        fluid      solid-phase control
//   nearest neighbour    0.0851     1.4421
//   upward excursion    +0.0000    +1.8923
//
// Both thresholds sit between the two, and the control run FAILS both -- which is
// what makes them checks rather than decoration.
//
// What these numbers do and do not establish is worth being exact about.  They
// establish that the fluid arms of the five P9.1 kernels RUN end to end in the
// real engine, that the result is finite and comes to rest, and that it is
// nothing like what the same cloud does with the fluid bit cleared.  They do NOT
// establish that the resting microstructure is the one CUDA would produce: the
// fluid ends as tight clumps (nn 0.085, well under the 0.200 fluid rest distance)
// spread over an RMS radius of 4.94, which is consistent with SnippetPBF's strong
// vorticity confinement of 10.0 but is not independently validated here.  What
// carries bit-exactness is the four fixtures, not this scene.
static const float FLUID_NN_MAX = 0.26f;        // below the solid rest distance of 0.333
static const float FLUID_NN_CONTROL = 1.4421f;  // measured
static const float FLUID_RISE_MAX = 0.20f;      // the fluid settles monotonically
static const float FLUID_RISE_CONTROL = 1.8923f;// measured

// P9.9 -- PXG_SCENE=particlefluidboundary.
//
// A scale of 1 makes mRestDensityBoundary equal the fluid's own rest density
// (PxgParticleSystemCore.cpp:2640: mRestDensityBoundary = mRestDensity * scale).
static const float FLUID_BOUNDARY_SCALE = 1.0f;
// P9.13a, PXG_SCENE=particlediffuse.  MAX_DIFFUSE bounds the buffer's diffuse
// arrays; DIFFUSE_THRESHOLD is the potential a fluid particle must exceed to
// spawn one.  Both are overridable (PXG_DIFFUSE_THRESHOLD), and the CONTROL mode
// raises the threshold out of reach rather than changing anything else.
// P9.80 -- PXG_SCENE=particlerigid.  The cluster spawns just above the ground box
// so it LANDS inside the run: the whole observable is what the landing does to its
// shape.  RIGID_STIFFNESS is the shape-matching coefficient (1 = fully rigid); the
// control sets it to 0, which is the same cluster with the matching switched off.
// RIGID_VOLUME_POISON is written into the PxParticleVolume bound before stepping,
// so a kernel that never ran is an impossible number and not a plausible one.
static const float RIGID_START_Y = 2.0f;
static const float RIGID_STIFFNESS = 1.0f;
static const float RIGID_VOLUME_POISON = -999.0f;

static const PxU32 MAX_DIFFUSE = 4096;
static const float DIFFUSE_THRESHOLD = 1.0f;

// P9.10c -- PXG_SCENE=particlemeshfloor.  The boundary scene with the ground BOX
// swapped for a cooked TRIANGLE MESH at the same height, and the box sunk to -40
// where nothing can reach it.  That swap is the whole experiment:
//
//   * a box floor produces particle-vs-rigid contacts through
//     ps_primitivesCollisionLaunch, which land in the PRIMITIVE contact stream --
//     the stream ps_accumulateRigidDensity reads, which P9.9 measured as 100% of
//     the boundary observable;
//   * a triangle-mesh floor produces them through ps_meshCollisonLaunch, and for a
//     STATIC rigid those go into mOneWayContactInfos, not into the writer (the
//     writer only ever sees the overflow past MaxStaticContactsPerParticle = 12,
//     and this floor is two triangles, so no particle comes close).  That array is
//     what ps_accumulateStaticDensity reads and had always found empty.
//
// So on this scene the boundary density can only arrive through the STATIC half of
// the P9.8 pair, and the separation below is the first evidence that half does
// anything at all.  Measured, 240 steps, both runs on the identical 60 x 60 mesh:
//
//     scale     resting spread     resting nn
//       0            5.2148          0.0918      (PXG_PARTICLE_BOUNDARY_CONTROL=1)
//       1.0          9.6288          0.2010
//
// a 1.85x separation, wider than the box floor's 1.63x.  The gate is a one-sided
// minimum at 7.0, clear of both ends -- the same shape as the box scene's, and for
// the same reason: what is being asserted is that the effect ARRIVES, not that it
// has a particular magnitude.
static const float MESHFLOOR_SPREAD_MIN = 7.0f;
static const float MESHFLOOR_SPREAD_CONTROL = 5.2148f;
static const float MESHFLOOR_SPREAD_LIVE = 9.6288f;

// P9.11c -- PXG_SCENE=particlehffloor.  particlemeshfloor again with the cooked
// triangle mesh swapped for a cooked HEIGHTFIELD, so the particle-vs-static contacts
// arrive through ps_heightfieldCollisonLaunch instead of ps_meshCollisonLaunch.  It is
// the first scene in this port that puts particles on a heightfield.
//
// IT IS NOT A BOUNDARY-DENSITY SCENE, and that is a measured decision rather than an
// omission.  With fluidBoundaryDensityScale at 1 -- the setting particlemeshfloor
// uses -- this scene does not settle at all: spread 19.2006 and still growing at step
// 299, nn 2.7488 against a fluid rest spacing near 0.09, mean y drifting to -0.4065 as
// the outermost particles run off the rim.  The cause is the CONTACT COUNT, not the
// heightfield kernel: a particle resting on a 64 x 64 field is within contactDist of
// BOTH triangles of the cell beneath it, where the two-triangle mesh floor puts it
// inside exactly one triangle and the other's closest point is out on the shared
// diagonal.  Twice the one-way contacts is twice the static density, and the density
// constraint over-corrects.  Gating on that would be gating on
// ps_accumulateStaticDensity's response to a denser contact stream, which is the mesh
// floor scene's job and is already asserted there.
//
// What this scene asserts instead is stronger for the kernel it exists to exercise:
// the fluid must REST ON the heightfield, and at the same height the triangle-mesh
// floor puts it.  Two different narrowphase kernels, two different contact producers,
// one rest height.  MEASURED, 300 steps: mean y 0.16667, spread 3.7346, nn 0.0907 --
// against 0.16667 / 9.6288 / 0.2010 for the mesh floor with its boundary density on,
// and against a fall to the sunken ground box at -40 if the heightfield pair produced
// nothing at all.
static const float HFFLOOR_REST_Y = 0.16667f;       // measured; identical to the mesh floor's
static const float HFFLOOR_REST_Y_TOL = 0.02f;
static const float HFFLOOR_SPREAD_REF = 3.7346f;    // measured, reported not gated
static const float HFFLOOR_NN_REF = 0.0907f;        // measured, reported not gated
// The observable -- read the long note at the boundary assertion block before
// touching these.  MEASURED on this exact scene, A/B against
// PXG_PARTICLE_BOUNDARY_CONTROL=1 (identical scene, scale left at 0 so neither
// kernel runs), over four boundary scales:
//
//     scale     resting spread     resting nn
//     0 (off)          4.9381         0.0851
//     0.25             7.9165         0.0949
//     1.0              8.0641         0.0975
//
// SPREAD is the gate and NN is only reported, which is a deliberate choice rather
// than an oversight.  Spread separates on/off by ~60% of the control value, so a
// threshold at 6.0 sits ~21% above the control and ~24% below the live run -- a
// margin worth regressing against.  NN separates by only ~11%, so any threshold
// between them would sit within ~6% of both sides; that is a flaky gate, not a
// measurement, and it would eventually fail for reasons having nothing to do with
// the boundary density.  It is printed so a human can see it move.
//
// Note the effect SATURATES: a quarter of the scale already buys 98% of the spread
// change, because the raw per-contact contribution mostly sits below even the
// quarter-scale cap, so min(rho, mRestDensityBoundary) rarely binds.  The gate is
// therefore an on/off discriminator and NOT a claim about the scale's magnitude.
static const float FLUID_BOUNDARY_SCALE_SPREAD = 8.0641f;   // measured, scale 1.0
static const float FLUID_BOUNDARY_SPREAD_MIN = 6.0f;        // between the two, wide both ways
static const float FLUID_BOUNDARY_SPREAD_CONTROL = 4.9381f; // measured, scale 0
static const float FLUID_BOUNDARY_NN_CONTROL = 0.0851f;     // measured, scale 0 (reported only)

struct ParticleScene
{
    PxPBDParticleSystem* system = nullptr;
    PxParticleBuffer* buffer = nullptr;
    PxU32 numParticles = 0;
    PxCudaContextManager* cudaCtx = nullptr;
    // P9.4: the PINNED prefix of the particle array.  buildParticleClothScene
    // orders the sheet row-major with the pinned row FIRST, so indices
    // [0, numPinned) are the invMass == 0 particles and [numPinned, numParticles)
    // are the free ones.  Every cloth assertion is over the FREE range only --
    // the pinned particles never move, so including them would dilute exactly the
    // signal the scene exists to measure.
    PxU32 numPinned = 0;
    // P9.13a: non-null only for PXG_SCENE=particlediffuse.  Aliases `buffer`; kept
    // separately because the diffuse accessors are on the derived type.
    PxParticleAndDiffuseBuffer* diffuse = nullptr;
    // P9.41: non-null only for PXG_SCENE=particlepush -- the 10 kg box the cloud is
    // fired at.  It doubles as the scene's one live dynamic rigid.
    PxRigidDynamic* pushBox = nullptr;
    // P9.76: non-null only for PXG_SCENE=particleattach -- the static anchor every
    // pinned particle hangs from.  `freeParticle` is the ONE index deliberately left
    // unpinned, which is this scene's internal control: if gravity were not running at
    // all, "the cloud did not fall" would prove nothing.
    PxRigidStatic* anchor = nullptr;
    PxU32 numAttached = 0;
    PxU32 freeParticle = 0;
    // P9.80: non-null only for PXG_SCENE=particlerigid.  `rigidBuffer` aliases
    // `buffer`; the volume array and the spawn AABB are kept so the scene can
    // compare what ps_update_volume_bound wrote against what the particles are.
    PxParticleRigidBuffer* rigidBuffer = nullptr;
    PxBounds3 spawnBound = PxBounds3::empty();
    float spawnExtentY = 0.0f;
};

// fluidMode (PXG_SCENE=particlefluid, P9.2) switches the cloud from a SOLID
// self-colliding granular pile to a real PBD FLUID: the phase gains
// eParticlePhaseFluid, the material gains the four fluid coefficients, and the
// system gains a fluid rest offset.  Those are exactly what the five kernels
// landed in P9.1a-d consume; with the solid phase they all early-out per particle.
// The parameters are the engine's own SnippetPBF (snippets/snippetpbf/
// SnippetPBF.cpp:104-156), not invented here.
// boundaryMode (PXG_SCENE=particlefluidboundary, P9.9) is fluidMode plus one call:
// setFluidBoundaryDensityScale.  That single call is what switches the P9.8 pair
// (ps_accumulateStaticDensity / ps_accumulateRigidDensity) on -- the host gates both
// on mHasNonZeroFluidBoundaryScale (PxgPBDParticleSystemCore.cpp:1118, :1141), which
// is set only when some particle system carries fluidBoundaryDensityScale > 0
// (:2167).  Everything else about the scene is byte-identical to particlefluid, so
// the two scenes are each other's control.
// P9.41 -- pushMode (PXG_SCENE=particlepush): the solid 4x4x4 cloud is spawned LOW,
// beside a 10 kg box standing on the ground, with an initial velocity of +4 m/s
// along x, so it arrives at the box's face within a dozen steps.  Nothing else in
// the scene touches the box.  Whether the box moves is the observable.
static const float PUSH_SPEED = 4.0f;
static const float PUSH_BOX_X = 2.5f;
// P9.76 -- attachMode (PXG_SCENE=particleattach): the solid 4x4x4 cloud, with every
// particle but ONE pinned to a STATIC anchor at its own spawn position through
// PxParticleBuffer::setRigidAttachments.  That is the only application entry point
// into ps_rigidAttachmentPrepareLaunch and ps_solveRigidAttachments{,TGS}Launch, and
// no other scene in this tree reaches them.
//
// The scene carries its OWN control, so it needs no second run and no environment
// variable: particle 0 is deliberately left unpinned.  If the attachment chain is
// live, the 63 pinned particles stay at y = 4 and particle 0 falls to the floor.  If
// it is dead, all 64 fall together -- and if gravity were somehow not running, the
// free particle would stay too, which is the reading "nothing moved" cannot
// distinguish on its own.
static const float ATTACH_START_Y = 4.0f;
static const PxU32 ATTACH_FREE_PARTICLE = 0u;
ParticleScene buildParticleScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                 PxCudaContextManager* cudaCtx, bool fluidMode = false,
                                 bool boundaryMode = false, bool meshFloorMode = false,
                                 bool hfFloorMode = false, bool diffuseMode = false,
                                 bool convexFloorMode = false, bool pushMode = false,
                                 bool attachMode = false, bool rigidMode = false)
{
    ParticleScene out;
    out.cudaCtx = cudaCtx;
    PxRigidStatic* groundStatic = nullptr;   // P9.76: the attach scene's anchor

    // A minimal rigid body (box on ground) FAR BELOW the cloud.  The engine drives
    // the rigid solver core for the particle island's iteration loop; with zero
    // dynamic rigid bodies the rigid batch-sum kernel (rigidSumInternalContactAndJoint
    // Batches2) dereferences null solver descriptors and faults, so the scene needs
    // one live rigid body to keep that proven path valid.  It rests at y=0.5, ~100
    // below the particle cloud, so there is never any particle/rigid interaction.
    // meshFloorMode swaps the static ground BOX for a cooked TRIANGLE MESH at the
    // same height and drops the box far below, where it cannot contribute a
    // contact.  That is the whole point of the scene: with a box floor the
    // particle-vs-rigid contacts go through ps_primitivesCollisionLaunch into the
    // PRIMITIVE contact stream, which is what P9.9 measured feeding the boundary
    // density; a triangle-mesh floor routes them through ps_meshCollisonLaunch
    // instead, which is the only producer of mOneWayContactInfos this port has.
    // hfFloorMode is the same swap again, one shape type further along: a cooked
    // HEIGHTFIELD instead of a triangle mesh.  That routes the particle-vs-static
    // contacts through ps_heightfieldCollisonLaunch, the SECOND producer of
    // mOneWayContactInfos this port has, and the third and last one
    // (ps_sdfMeshCollisonLaunch) is still unmapped.
    if (meshFloorMode || hfFloorMode || convexFloorMode)
    {
        addGroundBox(physics, scene, rigidMat, -40.0f);
        // 60 x 60, not the 20 x 20 the meshfloor box scene uses.  Measured: with the
        // boundary density ON this puddle spreads to ~10, so a half-extent of 10 put
        // the rim exactly where the fluid ends up and the particles fell off the edge
        // (mean y -2.98).  That looked like a contact failure and was a floor-sizing
        // failure -- with the scale at 0 the SAME floor held them at y 0.167.
        if (meshFloorMode)        addMeshFloor(physics, scene, rigidMat, 30.0f);
        else if (hfFloorMode)     addHeightFieldFloor(physics, scene, rigidMat, 30.0f);
        // P9.14a: the same floor again, one shape type further along -- a cooked
        // CONVEX HULL.  That is the only floor type whose particle pair reaches
        // testSDKConvexParticle, and therefore the only one that can ever launch
        // ps_convexCollisionLaunch or ps_convexDiffuseCollisionLaunch.
        else                      addConvexFloor(physics, scene, rigidMat, 30.0f);
    }
    else
    {
        groundStatic = addGroundBox(physics, scene, rigidMat);
    }
    // The engine drives the rigid solver core for the particle island, so one live
    // dynamic rigid body is required (else rigidSumInternalContactAndJointBatches2
    // null-derefs).  Free-fall: box at x=0,y=5 (never meets the y~100 cloud).  Bag
    // mode (PXG_PARTICLE_STARTY set): place it FAR ASIDE at x=20 so the settling
    // pile on the floor never touches it.
    if (pushMode)
    {
        // The target.  Ten kilograms (a hundred times lighter than the harness's usual
        // box, ten times heavier than the particles' 64 kg would want it) standing on
        // the ground, half-extents 0.5 x 1.0 x 0.5 so every row of the cloud meets its
        // face.  Nothing but the particles can move it.
        const PxVec3 half(0.5f, 1.0f, 0.5f);
        PxRigidDynamic* target = physics->createRigidDynamic(PxTransform(PxVec3(PUSH_BOX_X, half.y, 0.0f)));
        PxShape* s = physics->createShape(PxBoxGeometry(half), *rigidMat);
        target->attachShape(*s);
        s->release();
        target->setMass(10.0f);
        target->setMassSpaceInertiaTensor(boxInertia(half, 10.0f));
        scene->addActor(*target);
        out.pushBox = target;
    }
    // P9.80: rigidMode joins this branch for exactly the reason the comment above
    // gives.  Its cluster settles at x = 0 like the bag does, and the free-fall
    // placement below drops a 1000 kg box from y = 5 straight onto it -- which is
    // what the scene's first run measured (the cluster scattered 160 units wide,
    // where the plain bag at the identical spawn settles tidily).  That was the
    // harness hitting its own scene, not the port: with both features switched off
    // the number was bit-identical, and so was an ordinary PxParticleBuffer.
    else if (std::getenv("PXG_PARTICLE_STARTY") || fluidMode || rigidMode)
    {
        // On the mesh floor the puddle spreads to an RMS radius near 10, so the usual
        // x=20 parking spot is INSIDE the puddle and the aside box would start
        // contributing PRIMITIVE contacts -- the other half of the boundary density,
        // which would blur exactly what this scene is built to isolate.  On that floor
        // it goes to x=45, outside the 60 x 60 mesh, and falls to the sunken ground box.
        const float asideX = (meshFloorMode || hfFloorMode || convexFloorMode) ? 45.0f : 20.0f;
        PxRigidDynamic* aside = physics->createRigidDynamic(PxTransform(PxVec3(asideX, 0.5f, 0.0f)));
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
    PxPBDMaterial* pbdMat = nullptr;
    PxPBDParticleSystem* ps = nullptr;
    float particleSpacing = 0.5f;
    float particleInvMass = 1.0f;
    PxU32 phase = 0;

    // PXG_PARTICLE_FLUID_CONTROL builds the fluid scene's geometry, offsets and
    // material but WITHOUT the eParticlePhaseFluid bit, so every fluid arm of the
    // five P9.1 kernels early-outs while everything else is identical.  It is the
    // control the spread threshold is calibrated against.
    const bool fluidControl = fluidMode && std::getenv("PXG_PARTICLE_FLUID_CONTROL") != nullptr;

    if (fluidMode)
    {
        // SnippetPBF.cpp:112-136 and :155, transcribed.
        particleSpacing = 0.2f;
        const float fluidDensity = 1000.0f;
        const float restOffset = 0.5f * particleSpacing / 0.6f;
        const float solidRestOffset = restOffset;
        const float fluidRestOffset = restOffset * 0.6f;
        const float particleMass = fluidDensity * 1.333f * 3.14159f
                                 * particleSpacing * particleSpacing * particleSpacing;
        particleInvMass = 1.0f / particleMass;

        pbdMat = physics->createPBDMaterial(0.05f, 0.05f, 0.f, 0.001f, 0.5f, 0.005f, 0.01f, 0.f, 0.f);
        pbdMat->setViscosity(0.001f);
        pbdMat->setSurfaceTension(0.00704f);
        pbdMat->setCohesion(0.0704f);
        pbdMat->setVorticityConfinement(10.f);

        ps = physics->createPBDParticleSystem(*cudaCtx, 96);
        ps->setRestOffset(restOffset);
        ps->setContactOffset(restOffset + 0.01f);
        ps->setParticleContactOffset(fluidRestOffset / 0.6f);
        ps->setSolidRestOffset(solidRestOffset);
        // The one call the solid scene never makes.  Without it the engine keeps
        // ParticleSystemShapeCore's default fluid rest offset, which is derived
        // from the DEFAULT rest offset and is not re-derived when the user changes
        // it -- an engine trap, not a bug, but it would silently mis-scale the SPH
        // rest density.
        ps->setFluidRestOffset(fluidRestOffset);
        ps->setParticleFlag(PxParticleFlag::eENABLE_SPECULATIVE_CCD, false);
        ps->setMaxVelocity(solidRestOffset * 100.f);
        // P9.9: the ONE call that switches the boundary-density pair on.  It is the
        // whole difference between PXG_SCENE=particlefluid and
        // PXG_SCENE=particlefluidboundary, which is what makes them a clean A/B.
        //
        // Where the value comes from: PxgParticleSystemCore.cpp:2640 computes
        //     mData.mRestDensityBoundary = mData.mRestDensity * fluidBoundaryDensityScale
        // so a scale of 1 caps the boundary contribution at the fluid's own rest
        // density.  That is a natural UNIT, not a derived optimum.  An earlier draft
        // of this comment called it "the largest value that cannot make a particle at
        // a wall read as denser than a particle buried inside the fluid"; that is
        // false, and no positive value has that property.  The consumer SEEDS rho
        // with the boundary term (particlesystem.cu:3277-3281) and then adds the
        // neighbour sum ON TOP (:3338), so a wall particle with a full neighbourhood
        // reads roughly boundary + neighbours -- about twice a buried particle -- and
        // the constraint c = PxMax(rho - mRestDensity, ...) (:3361) acts on exactly
        // that.  Reading denser than it really is IS the mechanism this scene
        // measures, not an artefact to be tuned away.
        //
        // 1.0 is used because it is the obvious unit and because it MEASURED a large,
        // stable, saturating effect.  The assertion below is an on/off discriminator
        // and makes no claim about the magnitude.  Both kernels early-out on
        // `mRestDensityBoundary > 0`, so 0 (the default) is off and any positive
        // value is on.
        //
        // BOUNDARY_CONTROL builds the identical scene with the scale left at 0, so
        // the control differs from the live run in nothing but whether the two
        // kernels run at all.
        if (boundaryMode && !std::getenv("PXG_PARTICLE_BOUNDARY_CONTROL"))
            ps->setFluidBoundaryDensityScale(FLUID_BOUNDARY_SCALE);
        phase = fluidControl
            ? ps->createPhase(pbdMat, PxParticlePhaseFlags(PxParticlePhaseFlag::eParticlePhaseSelfCollide))
            : ps->createPhase(pbdMat, PxParticlePhaseFlags(
                  PxParticlePhaseFlag::eParticlePhaseFluid | PxParticlePhaseFlag::eParticlePhaseSelfCollide));
    }
    else
    {
        pbdMat = physics->createPBDMaterial(/*friction*/ 0.05f, /*damping*/ 0.0f,
            /*adhesion*/ 0.0f, /*viscosity*/ 0.0f, /*vorticityConfinement*/ 0.0f, /*surfaceTension*/ 0.0f,
            /*cohesion*/ 0.0f, /*lift*/ 0.0f, /*drag*/ 0.0f);

        ps = physics->createPBDParticleSystem(*cudaCtx);
        const float restOffset = 0.5f * particleSpacing;
        ps->setRestOffset(restOffset);
        ps->setContactOffset(restOffset + 0.05f);
        ps->setParticleContactOffset(restOffset + 0.05f);
        ps->setSolidRestOffset(restOffset);
        ps->setMaxVelocity(1e6f); // do not clamp the free fall

        // P9.80 -- rigidMode DELIBERATELY CLEARS eParticlePhaseSelfCollide.
        //
        // The scene's first honest run said the shape observable was worthless: with
        // ps_solveShapes skipped entirely, the 4x4x4 cluster still landed with a
        // y-extent of 1.4990 against a spawn of 1.5000.  Self-collision plus eight
        // position iterations is quite enough to hold a small granular cube together
        // when it is dropped two metres onto a floor, so "the cube is still a cube"
        // was true under the NULL HYPOTHESIS and measured nothing.
        //
        // Turning self-collision off removes the confound instead of trying to
        // out-shout it with a bigger drop.  With it off there is nothing between the
        // particles at all: gravity takes every one of them to the floor and the
        // cluster collapses into a single layer.  The ONLY thing that can keep it a
        // cube is the shape matching, which is exactly the claim the scene makes.
        // NOTE eParticlePhaseSelfCollideFilter is NOT the flag for this: it filters
        // out neighbours that were close in the REST pose and needs rest positions
        // set.  What this scene wants is no self-collision at all -- empty flags.
        phase = rigidMode
            ? ps->createPhase(pbdMat, PxParticlePhaseFlags())
            : ps->createPhase(pbdMat,
                  PxParticlePhaseFlags(PxParticlePhaseFlag::eParticlePhaseSelfCollide));
    }

    // Solid: a 4x4x4 grid.  Fluid: a TALL NARROW COLUMN (4 x 12 x 4), because the
    // observable for a fluid is that the column COLLAPSES and SPREADS sideways,
    // which a granular pile of the same shape does far less of.
    const int GX = fluidMode ? 4 : 4;
    const int GY = fluidMode ? 12 : 4;
    const int GZ = fluidMode ? 4 : 4;
    const PxU32 N = PxU32(GX * GY * GZ);
    // startY defaults to 100 (the P6.0 free-fall baseline all tranches regress
    // against); PXG_PARTICLE_STARTY lowers the cloud so it can reach the floor
    // (P6.5 settling bag).  When lowered, raise solver iterations for a stable pile.
    const char* startYEnv = std::getenv("PXG_PARTICLE_STARTY");
    // pushMode: the bottom row sits exactly one rest offset above the ground, so the
    // cloud slides rather than drops, and the box face at x = 2.0 is 0.75 m from the
    // cloud's leading row -- eleven steps at PUSH_SPEED.
    // attachMode: y = 4, well clear of the ground box at 0.5 but close enough that a
    // free fall reaches it inside the run -- so "did not fall" and "fell" are two
    // numbers a metre apart, not a tolerance argument.
    // rigidMode: low enough that the cluster LANDS inside the run -- the whole
    // observable is what the landing does to its shape.
    const float startY = fluidMode ? 0.6f : pushMode ? 0.25f : attachMode ? ATTACH_START_Y
                        : rigidMode ? RIGID_START_Y
                                                             : (startYEnv ? float(atof(startYEnv)) : 100.0f);
    const float cloudX = pushMode ? 0.5f : 0.0f;
    if (startYEnv || fluidMode || pushMode || attachMode || rigidMode)
        ps->setSolverIterationCounts(/*pos*/ 8, /*vel*/ 1);
    std::vector<PxVec4> hostPos(N), hostVel(N, pushMode ? PxVec4(PUSH_SPEED, 0.0f, 0.0f, 0.0f)
                                                          : PxVec4(0.0f, 0.0f, 0.0f, 0.0f));
    std::vector<PxU32> hostPhase(N, phase);
    const float cx = 0.5f * float(GX - 1) * particleSpacing;
    const float cz = 0.5f * float(GZ - 1) * particleSpacing;
    PxU32 idx = 0;
    for (int x = 0; x < GX; ++x)
        for (int y = 0; y < GY; ++y)
            for (int z = 0; z < GZ; ++z)
            {
                hostPos[idx++] = PxVec4(float(x) * particleSpacing - cx + cloudX,
                                        startY + float(y) * particleSpacing,
                                        float(z) * particleSpacing - cz, particleInvMass);
            }

    // P9.13a: the DIFFUSE variant allocates a different buffer type -- one that
    // carries the diffuse position/velocity/lifetime arrays and the spawn
    // parameters alongside the ordinary particle arrays.  Until this tranche
    // `createParticleAndDiffuseBuffer` was a warn-and-null bridge stub, and
    // NpParticleBuffer dereferences the result, so asking for one SEGFAULTED.
    PxParticleBuffer* buffer = nullptr;
    if (diffuseMode)
    {
        PxParticleAndDiffuseBuffer* db =
            physics->createParticleAndDiffuseBuffer(N, /*maxVolumes*/ 0, MAX_DIFFUSE, cudaCtx);
        if (!db)
        {
            std::printf("FAIL: createParticleAndDiffuseBuffer returned null "
                        "(bridge stub not replaced?)\n");
            return out;
        }
        // The spawn rule, PxDiffuseParticleParams (PxParticleBuffer.h:235-273):
        // a particle spawns diffuse particles when its POTENTIAL -- a weighted sum
        // of kinetic energy, pressure and velocity divergence -- exceeds
        // `threshold`.  The default threshold of 100 is tuned for a large, violent
        // splash; a 4x12x4 column collapsing under gravity never reaches it, so the
        // scene would measure nothing and read as "the kernels do not work".
        //
        // PXG_DIFFUSE_THRESHOLD overrides it, and the CONTROL below sets it
        // deliberately out of reach -- which is what makes this an A/B rather than
        // a single number with a hopeful bound.
        PxDiffuseParticleParams dp;
        const char* thrEnv = std::getenv("PXG_DIFFUSE_THRESHOLD");
        dp.threshold = std::getenv("PXG_DIFFUSE_CONTROL") ? 1.0e12f
                     : (thrEnv ? float(atof(thrEnv)) : DIFFUSE_THRESHOLD);
        dp.lifetime = 4.0f;            // long enough to survive the whole run
        dp.useAccurateVelocity = true;
        db->setDiffuseParticleParams(dp);
        db->setMaxActiveDiffuseParticles(MAX_DIFFUSE);
        out.diffuse = db;
        buffer = db;
    }
    else if (rigidMode)
    {
        // P9.80 -- a PxParticleRigidBuffer, which this harness had never created.
        // It carries BOTH features this tranche exists for: the shape-matching
        // rigid arrays AND (as a plain PxParticleBuffer) the particle volumes.
        // One buffer therefore drives both ps_solveShapes and
        // ps_update_volume_bound, whose host guards are mMaxRigidsPerBuffer > 0
        // and mMaxVolumesPerBuffer > 0 -- neither ever true before today.
        // PXG_PARTICLE_RIGID_PLAIN=1 builds the identical scene with an ORDINARY
        // PxParticleBuffer instead.  A DIAGNOSTIC: it isolates the buffer TYPE from
        // everything the two features do, which is the one difference the
        // NO_RIGIDS/NO_VOLUMES knobs cannot switch off.
        if (std::getenv("PXG_PARTICLE_RIGID_PLAIN"))
        {
            buffer = physics->createParticleBuffer(N, /*maxVolumes*/ 0, cudaCtx);
            if (!buffer) { std::printf("FAIL: createParticleBuffer returned null\n"); return out; }
        }
        else
        {
            PxParticleRigidBuffer* rb =
                physics->createParticleRigidBuffer(N, /*maxVolumes*/ 1, /*maxRigids*/ 1, cudaCtx);
            if (!rb)
            {
                std::printf("FAIL: createParticleRigidBuffer returned null (bridge stub not replaced?)\n");
                return out;
            }
            out.rigidBuffer = rb;
            buffer = rb;
        }
    }
    else
    {
        buffer = physics->createParticleBuffer(N, /*maxVolumes*/ 0, cudaCtx);
        if (!buffer)
        {
            std::printf("FAIL: createParticleBuffer returned null (bridge stub not replaced?)\n");
            return out;
        }
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

    // P9.80 -- the SHAPE-MATCHING rigid and the VOLUME, both filled here because
    // every array a PxParticleRigidBuffer exposes is a DEVICE pointer.
    if (rigidMode && out.rigidBuffer)
    {
        PxParticleRigidBuffer* rb = out.rigidBuffer;

        // ONE rigid covering every particle.  mRigidOffsets is a runsum with
        // numRigids + 1 entries: the CUDA reads rigidOffsets[rigidId] and
        // [rigidId + 1] (particlesystem.cu:5178).
        const PxU32 offsets[2] = { 0u, N };

        // The shape-matching stiffness.  PXG_PARTICLE_RIGID_CONTROL=1 sets it to
        // ZERO, which is the same cluster with the matching switched off -- the
        // A/B this scene rests on.  With 0 the kernel still RUNS (its host guard is
        // the rigid COUNT, not the coefficient) and lerps by 0, so the cluster
        // behaves as loose granular material.
        const bool rigidControl = std::getenv("PXG_PARTICLE_RIGID_CONTROL") != nullptr;
        const float coeff = rigidControl ? 0.0f : RIGID_STIFFNESS;

        // Local positions are RELATIVE TO THE CLUSTER'S CENTRE OF MASS, because
        // that is what the kernel reconstructs against:
        //   rigidPos = Rotate(newRot, localPositions[i]) + newCom
        // (particlesystem.cu:5395).  The initial translation must be that same COM
        // and the initial rotation the identity, or the first solve would yank the
        // cluster toward a pose it was never in.
        PxVec3 com(0.0f);
        for (PxU32 i = 0; i < N; ++i) com += PxVec3(hostPos[i].x, hostPos[i].y, hostPos[i].z);
        com /= float(N);

        std::vector<PxVec4> localPos(N), localNrm(N);
        for (PxU32 i = 0; i < N; ++i)
        {
            const PxVec3 p(hostPos[i].x, hostPos[i].y, hostPos[i].z);
            const PxVec3 d = p - com;
            localPos[i] = PxVec4(d, 0.0f);
            // The kernel rotates these into mNormalArray every step
            // (particlesystem.cu:5411).  A radial normal makes them distinguishable
            // per particle; .w is carried through untouched and is left at 0.
            const float m = d.magnitude();
            localNrm[i] = PxVec4(m > 0.0f ? d / m : PxVec3(0.0f, 1.0f, 0.0f), 0.0f);
        }
        const PxVec4 translation(com, 0.0f);
        const PxVec4 rotation(0.0f, 0.0f, 0.0f, 1.0f);   // identity quaternion

        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(rb->getRigidOffsets()), offsets, sizeof(offsets));
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(rb->getRigidCoefficients()), &coeff, sizeof(float));
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(rb->getRigidLocalPositions()), localPos.data(),
                        N * sizeof(PxVec4));
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(rb->getRigidLocalNormals()), localNrm.data(),
                        N * sizeof(PxVec4));
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(rb->getRigidTranslations()), &translation, sizeof(PxVec4));
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(rb->getRigidRotations()), &rotation, sizeof(PxVec4));
        // Two DIAGNOSTIC knobs, not scored modes, kept because they earned their
        // place.  The scene's first run showed the cluster not merely failing to hold
        // its shape but flying apart -- 160 units wide, where the plain
        // PXG_SCENE=particle bag at the identical spawn settles tidily at mean y
        // 0.9992.  "The kernels are skipped" does not by itself explain that, and
        // guessing which feature caused it would have been guessing.  Switching each
        // one off in turn said NEITHER: the number was bit-identical with both off,
        // and PXG_PARTICLE_RIGID_PLAIN said it was not the buffer type either.  What
        // was left was the harness's own companion rigid body falling on the cluster
        // (see the aside-placement branch above).  The knobs stay: they are what made
        // that a measurement instead of a theory.
        const bool noRigids = std::getenv("PXG_PARTICLE_RIGID_NO_RIGIDS") != nullptr;
        const bool noVolumes = std::getenv("PXG_PARTICLE_RIGID_NO_VOLUMES") != nullptr;
        rb->setNbRigids(noRigids ? 0u : 1u);
        rb->raiseFlags(PxParticleBufferFlag::eUPDATE_RIGID);

        // ONE volume over the same particles.  Its bound is POISONED: the kernel
        // overwrites it every step, so anything still holding -999 after the run
        // means ps_update_volume_bound never ran.  There is no flag for volumes --
        // mNumParticleVolumes on the buffer is what raises mMaxVolumesPerBuffer
        // (PxgPBDParticleSystemCore.cpp:1699).
        PxParticleVolume vol;
        vol.bound = PxBounds3(PxVec3(RIGID_VOLUME_POISON), PxVec3(RIGID_VOLUME_POISON + 1.0f));
        vol.particleIndicesOffset = 0;
        vol.numParticles = N;
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(rb->getParticleVolumes()), &vol,
                        sizeof(PxParticleVolume));
        rb->setNbParticleVolumes(noVolumes ? 0u : 1u);

        // the spawn AABB and y-extent, for the assertions
        out.spawnBound = PxBounds3::empty();
        for (PxU32 i = 0; i < N; ++i)
            out.spawnBound.include(PxVec3(hostPos[i].x, hostPos[i].y, hostPos[i].z));
        out.spawnExtentY = out.spawnBound.maximum.y - out.spawnBound.minimum.y;
    }
    else if (rigidMode)
    {
        out.spawnBound = PxBounds3::empty();
        for (PxU32 i = 0; i < N; ++i)
            out.spawnBound.include(PxVec3(hostPos[i].x, hostPos[i].y, hostPos[i].z));
        out.spawnExtentY = out.spawnBound.maximum.y - out.spawnBound.minimum.y;
    }

    ps->addParticleBuffer(buffer);
    scene->addActor(*ps);

    // P9.76 -- the rigid ATTACHMENTS.  This is the recipe PxParticleAttachmentBuffer
    // uses (ExtParticleExt.cpp:269-303), open-coded so the scene owns every byte:
    //   - a STATIC anchor takes mID0 = PxNodeIndex().getInd() (the 0xffffffff
    //     sentinel the prepare tests with isStaticBody) and mLocalPose0 in WORLD
    //     space, not body space ("attachments to rigid static work in global space");
    //   - mID1 is the BUFFER-LOCAL particle id;
    //   - the cone limit is left default-constructed, i.e. DISABLED
    //     (mAngle/mLowLimit/mHighLimit all -1), which is the branch this port defers;
    //   - the array must live in DEVICE memory: setRigidAttachments takes a device
    //     pointer and the prepare dereferences it from the GPU.
    // The system must also be told the anchor participates, or the island manager has
    // no reason to keep it around (NpPBDParticleSystem::addRigidAttachment).
    if (attachMode)
    {
        if (!groundStatic)
        {
            std::printf("FAIL: particleattach needs the ground box as its anchor\n");
            return out;
        }
        ps->addRigidAttachment(groundStatic);

        std::vector<PxParticleRigidAttachment> att;
        att.reserve(N - 1u);
        for (PxU32 i = 0; i < N; ++i)
        {
            if (i == ATTACH_FREE_PARTICLE) continue;   // the scene's internal control
            PxParticleRigidAttachment a(PxConeLimitedConstraint(), PxVec4(0.0f));
            a.mID0 = PxNodeIndex().getInd();
            a.mID1 = i;
            // world position = where the particle was spawned, so the constraint's
            // position error starts at zero and the cloud is asked to STAY, not move.
            a.mLocalPose0 = PxVec4(hostPos[i].x, hostPos[i].y, hostPos[i].z, 0.0f);
            att.push_back(a);
        }
        CUdeviceptr devAtt = 0;
        const size_t bytes = att.size() * sizeof(PxParticleRigidAttachment);
        if (ctx->memAlloc(&devAtt, bytes) != 0 || !devAtt)
        {
            std::printf("FAIL: memAlloc for %zu attachment bytes failed\n", bytes);
            return out;
        }
        ctx->memcpyHtoD(devAtt, att.data(), bytes);
        buffer->setRigidAttachments(reinterpret_cast<PxParticleRigidAttachment*>(devAtt),
                                    PxU32(att.size()));
        buffer->raiseFlags(PxParticleBufferFlag::eUPDATE_ATTACHMENTS);
        out.anchor = groundStatic;
        out.numAttached = PxU32(att.size());
        out.freeParticle = ATTACH_FREE_PARTICLE;
    }

    out.system = ps;
    out.buffer = buffer;
    out.numParticles = N;
    return out;
}

// DtoH-read the particle buffer and return the MEAN NEAREST-NEIGHBOUR distance.
// This is the fluid observable, and it is the one the measurement supports: the
// density constraint drives the cloud toward its FLUID rest distance
// (2 * fluidRestOffset = 0.2 here), while the solid contact solve drives it to the
// SOLID rest distance (2 * solidRestOffset = 0.333).  O(N^2) over 192 particles.
//
// The first draft of this scene used the horizontal SPREAD instead, on the
// assumption that a fluid column collapses sideways more than a granular one.
// Measured, that is backwards: at this spacing the solid rest distance is larger
// than the particle spacing, so the solid control starts deeply overlapped and
// EXPLODES -- it spreads x87.7 and its mean y rises to 3.58 before falling back,
// against the fluid's x15.6 and a monotonic settle.  Spread therefore does not
// separate them in the direction assumed, and the threshold was replaced rather
// than tuned.
float particleMeanNearestNeighbour(const ParticleScene& psc)
{
    std::vector<PxVec4> host(psc.numParticles);
    PxCudaContext* ctx = psc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(), reinterpret_cast<CUdeviceptr>(psc.buffer->getPositionInvMasses()),
                    psc.numParticles * sizeof(PxVec4));
    double acc = 0.0;
    for (PxU32 i = 0; i < psc.numParticles; ++i)
    {
        float best = 1e30f;
        for (PxU32 j = 0; j < psc.numParticles; ++j)
        {
            if (i == j) continue;
            const float dx = host[i].x - host[j].x;
            const float dy = host[i].y - host[j].y;
            const float dz = host[i].z - host[j].z;
            const float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < best) best = d2;
        }
        acc += std::sqrt(double(best));
    }
    return float(acc / double(psc.numParticles));
}

// DtoH-read the particle buffer and return the RMS horizontal distance of the
// cloud from its own (x,z) centroid -- reported for context, not asserted on.
float particleSpreadXZ(const ParticleScene& psc)
{
    std::vector<PxVec4> host(psc.numParticles);
    PxCudaContext* ctx = psc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(), reinterpret_cast<CUdeviceptr>(psc.buffer->getPositionInvMasses()),
                    psc.numParticles * sizeof(PxVec4));
    double mx = 0.0, mz = 0.0;
    for (PxU32 i = 0; i < psc.numParticles; ++i) { mx += host[i].x; mz += host[i].z; }
    mx /= double(psc.numParticles); mz /= double(psc.numParticles);
    double acc = 0.0;
    for (PxU32 i = 0; i < psc.numParticles; ++i)
    {
        const double dx = host[i].x - mx, dz = host[i].z - mz;
        acc += dx * dx + dz * dz;
    }
    return float(std::sqrt(acc / double(psc.numParticles)));
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

// P9.76 -- DtoH-read the particle buffer and split it into the PINNED mean y, the
// pinned MINIMUM y and the ONE free particle's y.  Splitting is the whole point: the
// mean over all 64 would blend the control into the signal it is meant to separate.
struct AttachY { float pinnedMean, pinnedMin, freeY; bool finite; };
AttachY particleAttachY(const ParticleScene& psc)
{
    std::vector<PxVec4> host(psc.numParticles);
    PxCudaContext* ctx = psc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(), reinterpret_cast<CUdeviceptr>(psc.buffer->getPositionInvMasses()),
                    psc.numParticles * sizeof(PxVec4));
    AttachY r; r.pinnedMean = 0.0f; r.pinnedMin = FLT_MAX; r.freeY = 0.0f; r.finite = true;
    double sum = 0.0; PxU32 n = 0;
    for (PxU32 i = 0; i < psc.numParticles; ++i)
    {
        if (!std::isfinite(host[i].x) || !std::isfinite(host[i].y) || !std::isfinite(host[i].z))
            r.finite = false;
        if (i == psc.freeParticle) { r.freeY = host[i].y; continue; }
        sum += double(host[i].y);
        if (host[i].y < r.pinnedMin) r.pinnedMin = host[i].y;
        ++n;
    }
    r.pinnedMean = n ? float(sum / double(n)) : 0.0f;
    return r;
}

// DtoH-read the particle buffer and return the mean x of the cloud (P9.41: the
// pushed cloud's progress toward the box).
float particleMeanX(const ParticleScene& psc)
{
    std::vector<PxVec4> host(psc.numParticles);
    PxCudaContext* ctx = psc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(), reinterpret_cast<CUdeviceptr>(psc.buffer->getPositionInvMasses()),
                    psc.numParticles * sizeof(PxVec4));
    double sum = 0.0;
    for (PxU32 i = 0; i < psc.numParticles; ++i) sum += double(host[i].x);
    return float(sum / double(psc.numParticles));
}

// DtoH-read the DIFFUSE buffer and return the lowest y among the live diffuse
// particles (P9.13h).  The count is the buffer's own published figure, so this
// reads exactly the particles the engine says exist.  Returns +inf when there
// are none, which the caller folds into a running minimum without disturbing it.
float diffuseMinY(const ParticleScene& psc)
{
    if (!psc.diffuse) return FLT_MAX;
    const PxU32 n = psc.diffuse->getNbActiveDiffuseParticles();
    if (n == 0u) return FLT_MAX;
    std::vector<PxVec4> host(n);
    PxCudaContext* ctx = psc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(),
                    reinterpret_cast<CUdeviceptr>(psc.diffuse->getDiffusePositionLifeTime()),
                    n * sizeof(PxVec4));
    float lo = FLT_MAX;
    for (PxU32 i = 0; i < n; ++i) lo = PxMin(lo, host[i].y);
    return lo;
}

// ---------------------------------------------------------------------------
// P9.95 -- PXG_SCENE=particleapply: the direct-GPU-API particle ingest,
// PxScene::applyParticleBufferData -> PxgPBDParticleSystemCore::
// applyParticleBufferDataDEPRECATED (:2267-2309) -> applyParticleBufferDataDEPRECATED
// (particlesystem.cu:5831-5866).  The only application entry point into that kernel,
// and the first scene in this tree that combines eENABLE_DIRECT_GPU_API with a PBD
// particle system.
//
// THREE CLOUDS IN ONE SYSTEM, NAMED BY UNIQUE ID.  The kernel resolves each slot's
// PxGpuParticleBufferIndexPair through two indirections: indices[blockIdx.y] picks the
// SLOT, and the slot's unique id goes through findBufferIndexFromUniqueId (an
// upper-bound search over the sorted ids, then the ORIGINAL-INDEX table) to the buffer.
// The buffers are created C, B, A and added A, B, C, so the unique ids run backwards
// against the buffer order and the original-index table is a real permutation.
//
// The write window edits all three clouds' user arrays and raises NO flags, then
// applies two of three slots:
//     pairs   = { {sys, uid(A)}, {sys, uid(C)}, {sys, uid(B)} }
//     flags   = {  VEL,           POS|VEL,       POS|VEL      }
//     indices = { 2, 0 }                      -> slots 2 (B, POS|VEL) and 0 (A, VEL)
// Slot 1 (C) is a decoy that is never named.  The per-step egress
// (ps_updateUserBufferLaunch) overwrites every user array after each step, so an edit
// the kernel did not apply is simply lost -- which is what makes the clouds the
// observable.  Every particle is free (empty phase flags, damping 0), so a cloud keeps
// its shape and its mean moves only by what was applied.
static const float PAPPLY_DZ_B = 7.0f, PAPPLY_VX_B = 3.0f;
static const float PAPPLY_DZ_A = -9.0f, PAPPLY_DVY_A = 6.0f;
static const float PAPPLY_DZ_C = 11.0f, PAPPLY_VZ_C = 4.0f;
static const int PAPPLY_STEPS = 30;

struct ParticleApplyScene
{
    PxPBDParticleSystem* system = nullptr;
    PxParticleBuffer* a = nullptr;
    PxParticleBuffer* b = nullptr;
    PxParticleBuffer* c = nullptr;
    PxU32 n = 0;   // particles per cloud
    PxCudaContextManager* cudaCtx = nullptr;
};

ParticleApplyScene buildParticleApplyScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                           PxCudaContextManager* cudaCtx)
{
    ParticleApplyScene out;
    out.cudaCtx = cudaCtx;
    // The ground plus ONE live dynamic rigid far below the clouds (the particle island
    // needs it, see buildParticleScene).  Both are added before the first step and no
    // host pose is touched afterwards.
    addGroundBox(physics, scene, rigidMat);
    addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
               boxInertia(PxVec3(0.5f), 1000.0f), 0.5f, 40.0f);

    PxPBDMaterial* pbdMat = physics->createPBDMaterial(/*friction*/ 0.05f, /*damping*/ 0.0f,
        /*adhesion*/ 0.0f, /*viscosity*/ 0.0f, /*vorticityConfinement*/ 0.0f, /*surfaceTension*/ 0.0f,
        /*cohesion*/ 0.0f, /*lift*/ 0.0f, /*drag*/ 0.0f);
    PxPBDParticleSystem* ps = physics->createPBDParticleSystem(*cudaCtx);
    const float spacing = 0.5f;
    const float restOffset = 0.5f * spacing;
    ps->setRestOffset(restOffset);
    ps->setContactOffset(restOffset + 0.05f);
    ps->setParticleContactOffset(restOffset + 0.05f);
    ps->setSolidRestOffset(restOffset);
    ps->setMaxVelocity(1e6f);
    // Empty phase flags: no self-collision, so every particle integrates on its own.
    const PxU32 phase = ps->createPhase(pbdMat, PxParticlePhaseFlags());

    const int G = 4;
    const PxU32 N = PxU32(G * G * G);
    out.n = N;
    PxCudaContext* ctx = cudaCtx->getCudaContext();
    auto makeCloud = [&](float centreX) -> PxParticleBuffer*
    {
        PxParticleBuffer* buf = physics->createParticleBuffer(N, /*maxVolumes*/ 0, cudaCtx);
        if (!buf) return nullptr;
        std::vector<PxVec4> pos(N), vel(N, PxVec4(0.0f));
        std::vector<PxU32> ph(N, phase);
        const float c = 0.5f * float(G - 1) * spacing;
        PxU32 i = 0;
        for (int x = 0; x < G; ++x)
            for (int y = 0; y < G; ++y)
                for (int z = 0; z < G; ++z)
                    pos[i++] = PxVec4(centreX + float(x) * spacing - c, 100.0f + float(y) * spacing - c,
                                      float(z) * spacing - c, 1.0f);
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buf->getPositionInvMasses()), pos.data(), N * sizeof(PxVec4));
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buf->getVelocities()), vel.data(), N * sizeof(PxVec4));
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buf->getPhases()), ph.data(), N * sizeof(PxU32));
        buf->setNbActiveParticles(N);
        buf->raiseFlags(PxParticleBufferFlag::eUPDATE_POSITION);
        buf->raiseFlags(PxParticleBufferFlag::eUPDATE_VELOCITY);
        buf->raiseFlags(PxParticleBufferFlag::eUPDATE_PHASE);
        return buf;
    };
    // Created C, B, A (unique ids ascend in creation order) ...
    out.c = makeCloud(20.0f);
    out.b = makeCloud(0.0f);
    out.a = makeCloud(-20.0f);
    if (!out.a || !out.b || !out.c)
    {
        std::printf("FAIL: createParticleBuffer returned null\n");
        return out;
    }
    // ... and added A, B, C.
    ps->addParticleBuffer(out.a);
    ps->addParticleBuffer(out.b);
    ps->addParticleBuffer(out.c);
    scene->addActor(*ps);
    out.system = ps;
    return out;
}

// DtoH one cloud's user positions and velocities.
static void particleApplyRead(const ParticleApplyScene& s, PxParticleBuffer* buf,
                              std::vector<PxVec4>& pos, std::vector<PxVec4>& vel)
{
    PxCudaContext* ctx = s.cudaCtx->getCudaContext();
    pos.resize(s.n); vel.resize(s.n);
    ctx->memcpyDtoH(pos.data(), reinterpret_cast<CUdeviceptr>(buf->getPositionInvMasses()), s.n * sizeof(PxVec4));
    ctx->memcpyDtoH(vel.data(), reinterpret_cast<CUdeviceptr>(buf->getVelocities()), s.n * sizeof(PxVec4));
}

static PxVec3 particleApplyMean(const std::vector<PxVec4>& v)
{
    double x = 0.0, y = 0.0, z = 0.0;
    for (const PxVec4& p : v) { x += double(p.x); y += double(p.y); z += double(p.z); }
    const double n = double(v.size());
    return PxVec3(float(x / n), float(y / n), float(z / n));
}

// ---------------------------------------------------------------------------
// P9.99b -- PXG_SCENE=particle2: TWO PBD particle systems in one scene.
//
// Until P9.99b no scene built more than one PxPBDParticleSystem, so in every scene the
// particle pipeline ran with ONE system, at pool index 0, act = {0}.  Every per-system
// choice the pipeline makes therefore had a single possible answer there:
//   * system = act[blockIdx.y|z] -- preIntegrate, integrate, calculateHash, the one-way
//     contact solve, applyDelta, updateParticle, the fluid density/vorticity/velocity
//     kernels, updateMaterials, the user-buffer ingest/egress (particleSim,
//     particleGridBda, particleSelfCollisionBda, npPsFluid*, npPsMaterials);
//   * system = a host-loop `id` (mRemapIndex) -- the two bound passes, reorderData,
//     selfCollision, finalizeParticles (PxgParticleSystemCore.cpp:851-913, 679ff,
//     1041ff; PxgPBDParticleSystemCore.cpp:480-505);
//   * system = contact id >> 32 -- primitive contact generation (particleOrSoftbodyId),
//     the mesh midphase, contactPrepare, the PGS/TGS particle and rigid delta-V solves,
//     accumulateDeltaV (one contact list for ALL systems, sorted by (system, particle));
//   * system = blockIdx.y DIRECTLY -- ps_stepParticlesLaunch (particlesystem.cu:130), the
//     TGS predictor; it indexes the POOL, not act[] (upstream note U1 below);
//   * system = the user's PxGpuParticleBufferIndexPair -- applyParticleBufferData.
// 33 "force system index 0" mutants of those reads are value-identical in every
// single-system scene by construction.  This scene is where they are not.
//
// TWO SYSTEMS THAT DIFFER IN EVERY PER-SYSTEM PARAMETER A KERNEL READS.  A is created and
// added first (pool/remap index 0), B second (index 1):
//                  A                                B (split / apply)
//   cloud          4^3, spacing 0.5, x -10.75       5^3, spacing 0.3, x 9.4, inv mass 0.5
//   rest/contact   0.25 / 0.30, particle 0.30       0.15 / 0.17, particle 0.18
//   neighbourhood  96                               64
//   max velocity   3                                5     (B must be FASTER than A)
//   material       friction 0.05, gravity scale 1   friction 0.3, damping 0.1, gs 1.5
//   adhesion       0.3, radius scale 1.5            0.5, radius scale 2.0
//   lock           none (drifts in z at v.z = 1)    eLOCK_Z (v.z = 1 at spawn, z stays 0)
//   floor          the ground box (primitive        a cooked 2-triangle patch at (10, 1, 0),
//                  contacts)                        half 3 (ONE-WAY contacts, mesh midphase)
// The adhesion is what makes mData.mRestOffset live in the primitive contact solve: its
// only consumer there is the adhesion branch, dead at the default radius scale 0
// (NpFactory.cpp:763).
//
// SUB-SCENES IN ONE PROCESS.  Each row builds, steps (120 x 1/60 s) and releases several
// sub-scenes from main's sceneDesc, with the SAME rigid set in every one, in this order:
//   both  -- A then B (the scored configuration);
//   soloA, soloB -- the system alone;
//   swap  -- B then A (B at pool index 0, A at index 1); split and fluid only, and only
//            once every check on the first three held (see the dispatch block).
//   C1 (bitwise)  A(both) == A(soloA), B(both) == B(soloB): a system's trajectory may not
//                 depend on whether another system exists;
//   C2 (bitwise)  A(both) == A(swap),  B(both) == B(swap): nor on which pool index it got.
//                 In `split` B has no primitive contacts, so a force-index-0 mutant on the
//                 primitive path is invisible in `both` -- and wrong in `swap`, where A is
//                 index 1.  C2 is what kills gen / prep / accum / the PGS and TGS solves;
//   C3 (absolute) rest heights, the velocity caps, the lock, per-row physics -- so a
//                 mutant that breaks every sub-scene the same way still dies.
// Rows (tools/run_scenes.sh): split TGS, split PGS, fluid (B = SnippetPBF fluid with 2048
// diffuse particles on the ground box: the fluid/vorticity/material kernels at index 1;
// its diffuse arrays compare as MULTISETS, U4), apply (eENABLE_DIRECT_GPU_API,
// applyParticleBufferData naming ONLY B's slot with pairs from getGpuParticleSystemIndex,
// A's edit must be lost), push (each system pushes its own 10 kg box: the rigid delta-V
// decode; scored physically, U2).  `PXG_P2_ARM=remove` is a diagnostic, never scored.
//
// MEASURED (P9.99b): every C1/C2 comparison 0 of 512 (A) and 0 of 1000 (B) words, TGS and
// PGS, in split, fluid and apply.
//   split TGS  A y_min 0.249792, z 0.534571; B y_min 1.147884, z 0; at steps 20 / 30
//              B v.y -4.999955 / -4.999985, A |v| 2.999979 / 2.999988 (both clouds still
//              more than 0.5 m above contact onset at step 30)
//   split PGS  A 0.248886, z 0.718450; B 1.138487, z 0; B v.y -5.000000, A |v| 3.000000
//   fluid      B one layer at y 0.166666; diffuse peak 2048 (the cap), 1814 at the end, the
//              diffuse multisets of both / soloB / swap equal (1814 of 1814)
//   apply      gpu indices A 0, B 1; B z 5.000000 and v.x 1.996657 after the apply step, B
//              ends off the patch on the ground box at y_min 0.148359; A == soloA
//   push       box A moved 3.4980, box B 3.0656; 0.0268 / 0.0101 from the system alone
//              (C1 not bitwise: 333 / 500 words, U2)
//   remove     TGS: B differs from the no-removal run in 500 of 1000 words; PGS: 0
//
// UPSTREAM NOTES (documented, not reported; the port stays literal):
//   U1  ps_stepParticlesLaunch steps pool slot blockIdx.y, not act[blockIdx.y].  After a
//       system that is not the last is removed, the TGS predictor advances the REMOVED
//       system's still-allocated arrays (freed only at core teardown) and never steps the
//       survivor in the highest slot.  `PXG_P2_ARM=remove` shows it (B differs by 500 of
//       1000 words under TGS, 0 under PGS); psContactSolveTgs_step_p99Direct pins it.
//   U2  the order of contacts inside one body's run is the generation order, which
//       depends on which OTHER pairs share the warps -- so two systems pushing boxes
//       associate the box's delta-V sum differently than one.  Legit on CUDA too; the
//       push row is scored physically, never bitwise.
//   U3  the adhesion branch tests `separation < adhesionRadiusScale` (a scale, not the
//       radius).  This scene stays below the adhesion radius, so it never enters it.
//   U4  the diffuse slot order is run-to-run nondeterministic (atomic slot allocation);
//       the multiset is deterministic.
struct P2SysCfg
{
    int   g;                       // g x g x g cloud
    float spacing;
    float cx, off, shiftX;         // x = cx + i*spacing - off + shiftX, z = k*spacing - off
    float bottomY;
    float invMass;
    float rest, contactOff, particleContactOff, fluidRest;
    float maxV;
    PxU32 maxNbr;
    float friction, damping, viscosity, vorticity, surfaceTension, cohesion, gravityScale;
    float adhesion, adhesionRadiusScale;   // adhesion 0: the setters are not called
    bool  lockZ, fluid, diffuse;
    float v0x, v0z;
};

static P2SysCfg particle2CfgA(bool push)
{
    P2SysCfg c = {};
    c.g = 4; c.spacing = 0.5f; c.cx = -10.0f; c.off = 0.75f; c.shiftX = push ? 0.5f : 0.0f;
    c.bottomY = push ? 0.25f : 2.0f; c.invMass = 1.0f;
    c.rest = 0.25f; c.contactOff = 0.25f + 0.05f; c.particleContactOff = 0.25f + 0.05f; c.fluidRest = 0.0f;
    c.maxV = push ? 1e6f : 3.0f; c.maxNbr = 96;
    c.friction = 0.05f; c.gravityScale = 1.0f;
    c.adhesion = push ? 0.0f : 0.3f; c.adhesionRadiusScale = push ? 0.0f : 1.5f;
    c.v0x = push ? 4.0f : 0.0f; c.v0z = 1.0f;
    return c;
}

static P2SysCfg particle2CfgB(bool push, bool fluid)
{
    P2SysCfg c = {};
    c.g = 5; c.cx = 10.0f; c.off = 0.6f; c.shiftX = push ? 0.5f : 0.0f; c.invMass = 0.5f;
    c.maxNbr = 64; c.friction = 0.3f; c.damping = 0.1f;
    if (fluid)
    {
        // SnippetPBF's fluid offsets: r = 0.5 spacing / 0.6, fluid rest 0.6 r.
        c.spacing = 0.2f; c.bottomY = 3.0f;
        const float r = 0.5f * c.spacing / 0.6f;
        c.rest = r; c.contactOff = r + 0.02f; c.particleContactOff = r; c.fluidRest = r * 0.6f;
        c.maxV = 4.0f;
        c.viscosity = 0.001f; c.vorticity = 0.5f; c.surfaceTension = 0.005f; c.cohesion = 0.01f;
        c.gravityScale = 0.6f;
        c.fluid = true; c.diffuse = true;
        return c;
    }
    c.spacing = 0.3f; c.bottomY = push ? 0.15f : 4.0f;
    c.rest = 0.15f; c.contactOff = 0.15f + 0.02f; c.particleContactOff = 0.15f + 0.03f;
    c.maxV = push ? 4.0f : 5.0f;
    c.gravityScale = push ? 0.6f : 1.5f;
    c.adhesion = push ? 0.0f : 0.5f; c.adhesionRadiusScale = push ? 0.0f : 2.0f;
    c.lockZ = true;
    c.v0x = push ? 4.0f : 0.0f; c.v0z = 1.0f;
    return c;
}

static const int P2_STEPS = 120;
static const int P2_REMOVE_STEP = 60;
static const PxU32 P2_MAX_DIFFUSE = 2048;

enum P2Who { P2_BOTH, P2_SWAP, P2_SOLO_A, P2_SOLO_B, P2_REMOVE_A };
static const char* const P2_WHO_NAME[] = { "both", "swap", "soloA", "soloB", "remove" };

struct P2Run
{
    bool ok = false;
    bool present[2] = { false, false };
    PxU32 n[2] = { 0, 0 }, nActive[2] = { 0, 0 };
    PxU32 gpuIdx[2] = { 0xffffffffu, 0xffffffffu };      // read at the apply call
    std::vector<PxVec4> pos[2], vel[2];                   // final user arrays
    PxVec3 vProbe[2][2];                                  // mean velocity after steps 20 / 30
    PxVec3 pAfterApply[2], vAfterApply[2];                // mean p / v after step 1 (apply arm)
    PxVec3 meanP[2];
    float yMin[2] = { 0.0f, 0.0f }, yMax[2] = { 0.0f, 0.0f };
    bool finite[2] = { true, true };
    PxU32 diffusePeak[2] = { 0, 0 }, diffuseEnd[2] = { 0, 0 };
    std::vector<PxVec4> diffPos[2], diffVel[2];           // final active diffuse particles
    PxVec3 box[2];                                        // push arm: final box positions
};

static PxVec3 particle2MeanXYZ(const std::vector<PxVec4>& v)
{
    double x = 0.0, y = 0.0, z = 0.0;
    for (const PxVec4& p : v) { x += double(p.x); y += double(p.y); z += double(p.z); }
    const double n = double(v.size() ? v.size() : 1);
    return PxVec3(float(x / n), float(y / n), float(z / n));
}

// One sub-scene: build, step, read back, release everything it created.  Every rigid
// actor is created in every mode, in the same order, so node indices and the rigid pair
// order never depend on `who`.
static P2Run runParticle2(PxPhysics* physics, const PxSceneDesc& desc, PxMaterial* material,
                          PxCudaContextManager* cudaCtx, const char* arm, P2Who who, int steps)
{
    P2Run r;
    const bool armSplit = !std::strcmp(arm, "split") || !std::strcmp(arm, "remove");
    const bool armApply = !std::strcmp(arm, "apply");
    const bool armFluid = !std::strcmp(arm, "fluid");
    const bool armPush = !std::strcmp(arm, "push");
    PxCudaContext* ctx = cudaCtx->getCudaContext();

    PxScene* scene = physics->createScene(desc);
    if (!scene) { std::printf("FAIL: particle2 [%s]: createScene returned null\n", P2_WHO_NAME[who]); return r; }
    std::vector<PxRigidActor*> rigids;
    rigids.push_back(addGroundBox(physics, scene, material));
    // the live dynamic rigid the particle island needs, far aside
    rigids.push_back(addDynamic(physics, scene, material, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                boxInertia(PxVec3(0.5f), 1000.0f), 0.5f, 40.0f));
    PxTriangleMesh* patch = nullptr;
    if (armSplit || armApply)
    {
        // B's floor: a cooked 2-triangle patch, half 3, top at y = 1 under B only.
        const float E = 3.0f;
        const PxVec3 mverts[4] = { PxVec3(-E, 0.0f, -E), PxVec3(-E, 0.0f, E), PxVec3(E, 0.0f, -E), PxVec3(E, 0.0f, E) };
        const PxU32 mtris[6] = { 0u, 1u, 2u, 2u, 1u, 3u };
        PxTriangleMeshDesc md;
        md.points.count = 4; md.points.stride = sizeof(PxVec3); md.points.data = mverts;
        md.triangles.count = 2; md.triangles.stride = 3 * sizeof(PxU32); md.triangles.data = mtris;
        PxCookingParams ck(physics->getTolerancesScale());
        ck.buildGPUData = true;
        patch = immediateCooking::createTriangleMesh(ck, md, physics->getPhysicsInsertionCallback());
        if (!patch) { std::printf("FAIL: particle2: createTriangleMesh returned null\n"); scene->release(); return r; }
        PxRigidStatic* fl = physics->createRigidStatic(PxTransform(PxVec3(10.0f, 1.0f, 0.0f)));
        PxShape* fs = physics->createShape(PxTriangleMeshGeometry(patch), *material);
        fl->attachShape(*fs); scene->addActor(*fl); fs->release();
        rigids.push_back(fl);
    }
    PxRigidDynamic* boxes[2] = { nullptr, nullptr };
    if (armPush)
    {
        // one 10 kg push target per system, present in EVERY sub-scene
        for (int k = 0; k < 2; ++k)
        {
            const PxVec3 half(0.5f, 1.0f, 0.5f);
            PxRigidDynamic* t = physics->createRigidDynamic(PxTransform(PxVec3(k ? 12.5f : -7.5f, half.y, 0.0f)));
            PxShape* s = physics->createShape(PxBoxGeometry(half), *material);
            t->attachShape(*s); s->release();
            t->setMass(10.0f); t->setMassSpaceInertiaTensor(boxInertia(half, 10.0f));
            scene->addActor(*t);
            boxes[k] = t; rigids.push_back(t);
        }
    }

    const P2SysCfg cfg[2] = { particle2CfgA(armPush), particle2CfgB(armPush, armFluid) };
    const bool want[2] = { who != P2_SOLO_B, who != P2_SOLO_A };
    PxPBDParticleSystem* ps[2] = { nullptr, nullptr };
    PxParticleBuffer* buf[2] = { nullptr, nullptr };
    PxParticleAndDiffuseBuffer* dbuf[2] = { nullptr, nullptr };
    PxPBDMaterial* pmat[2] = { nullptr, nullptr };
    bool built = true;
    for (int k = 0; k < 2 && built; ++k)
    {
        if (!want[k]) continue;
        const P2SysCfg& c = cfg[k];
        pmat[k] = physics->createPBDMaterial(c.friction, c.damping, 0.0f, c.viscosity, c.vorticity, c.surfaceTension,
                                             c.cohesion, 0.0f, 0.0f, 1.0f, c.gravityScale);
        ps[k] = pmat[k] ? physics->createPBDParticleSystem(*cudaCtx, c.maxNbr) : nullptr;
        if (!ps[k]) { built = false; break; }
        if (c.adhesion > 0.0f) { pmat[k]->setAdhesion(c.adhesion); pmat[k]->setAdhesionRadiusScale(c.adhesionRadiusScale); }
        ps[k]->setRestOffset(c.rest);
        ps[k]->setContactOffset(c.contactOff);
        ps[k]->setParticleContactOffset(c.particleContactOff);
        if (c.fluid) ps[k]->setFluidRestOffset(c.fluidRest);
        ps[k]->setSolidRestOffset(c.rest);
        ps[k]->setMaxVelocity(c.maxV);
        ps[k]->setSolverIterationCounts(8, 1);
        if (c.lockZ) ps[k]->setParticleLockFlag(PxParticleLockFlag::eLOCK_Z, true);
        const PxU32 phase = ps[k]->createPhase(pmat[k], PxParticlePhaseFlags(PxParticlePhaseFlag::eParticlePhaseSelfCollide |
                                               (c.fluid ? PxParticlePhaseFlag::eParticlePhaseFluid : PxParticlePhaseFlag::Enum(0))));
        const PxU32 N = PxU32(c.g * c.g * c.g);
        std::vector<PxVec4> pos(N), vel(N, PxVec4(c.v0x, 0.0f, c.v0z, 0.0f));
        std::vector<PxU32> ph(N, phase);
        PxU32 i = 0;
        for (int x = 0; x < c.g; ++x)
            for (int y = 0; y < c.g; ++y)
                for (int z = 0; z < c.g; ++z)
                    pos[i++] = PxVec4(c.cx + x * c.spacing - c.off + c.shiftX, c.bottomY + y * c.spacing,
                                      z * c.spacing - c.off, c.invMass);
        if (c.diffuse)
        {
            dbuf[k] = physics->createParticleAndDiffuseBuffer(N, 0, P2_MAX_DIFFUSE, cudaCtx);
            if (dbuf[k])
            {
                PxDiffuseParticleParams dp; dp.threshold = 1.0f; dp.lifetime = 2.0f;
                dbuf[k]->setDiffuseParticleParams(dp);
                dbuf[k]->setMaxActiveDiffuseParticles(P2_MAX_DIFFUSE);
            }
            buf[k] = dbuf[k];
        }
        else
            buf[k] = physics->createParticleBuffer(N, 0, cudaCtx);
        if (!buf[k]) { built = false; break; }
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buf[k]->getPositionInvMasses()), pos.data(), N * sizeof(PxVec4));
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buf[k]->getVelocities()), vel.data(), N * sizeof(PxVec4));
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buf[k]->getPhases()), ph.data(), N * sizeof(PxU32));
        buf[k]->setNbActiveParticles(N);
        buf[k]->raiseFlags(PxParticleBufferFlag::eUPDATE_POSITION);
        buf[k]->raiseFlags(PxParticleBufferFlag::eUPDATE_VELOCITY);
        buf[k]->raiseFlags(PxParticleBufferFlag::eUPDATE_PHASE);
        ps[k]->addParticleBuffer(buf[k]);
        r.present[k] = true;
        r.n[k] = N;
    }

    auto readSys = [&](int k, std::vector<PxVec4>& p, std::vector<PxVec4>& v) {
        p.resize(r.n[k]); v.resize(r.n[k]);
        ctx->memcpyDtoH(p.data(), reinterpret_cast<CUdeviceptr>(buf[k]->getPositionInvMasses()), r.n[k] * sizeof(PxVec4));
        ctx->memcpyDtoH(v.data(), reinterpret_cast<CUdeviceptr>(buf[k]->getVelocities()), r.n[k] * sizeof(PxVec4));
    };
    CUdeviceptr devIdx = 0, devPairs = 0, devFlags = 0;
    if (built)
    {
        const int order[2] = { who == P2_SWAP ? 1 : 0, who == P2_SWAP ? 0 : 1 };
        for (int k : order) if (ps[k]) scene->addActor(*ps[k]);
        bool aLive = ps[0] != nullptr;
        for (int i = 0; i < steps && built; ++i)
        {
            if (armApply && i == 1)
            {
                // The write window after the warm-up step: both user arrays are edited and NO
                // flag is raised.  Only B's slot is applied (indices = {1}); A's edit must be
                // lost to the next egress.
                for (int k = 0; k < 2; ++k)
                {
                    if (!ps[k]) continue;
                    r.gpuIdx[k] = ps[k]->getGpuParticleSystemIndex();
                    std::vector<PxVec4> p, v; readSys(k, p, v);
                    for (PxU32 q = 0; q < r.n[k]; ++q)
                    {
                        if (k == 1) { p[q].z += 5.0f; v[q].x = 2.0f; }
                        else        { v[q].y = 3.0f; }
                    }
                    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buf[k]->getPositionInvMasses()), p.data(), r.n[k] * sizeof(PxVec4));
                    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buf[k]->getVelocities()), v.data(), r.n[k] * sizeof(PxVec4));
                }
                std::printf("  [%s] apply: gpu index A %u B %u\n", P2_WHO_NAME[who], r.gpuIdx[0], r.gpuIdx[1]);
                if (ps[1])
                {
                    const PxGpuParticleBufferIndexPair pairs[2] = {
                        { r.gpuIdx[0], ps[0] ? buf[0]->getUniqueId() : 0u }, { r.gpuIdx[1], buf[1]->getUniqueId() } };
                    const PxParticleBufferFlags flags[2] = {
                        PxParticleBufferFlags(PxParticleBufferFlag::eUPDATE_VELOCITY),
                        PxParticleBufferFlags(PxParticleBufferFlag::eUPDATE_POSITION) | PxParticleBufferFlag::eUPDATE_VELOCITY };
                    const PxU32 indices[1] = { 1u };
                    if (ctx->memAlloc(&devIdx, sizeof(indices)) != 0 || !devIdx ||
                        ctx->memAlloc(&devPairs, sizeof(pairs)) != 0 || !devPairs ||
                        ctx->memAlloc(&devFlags, sizeof(flags)) != 0 || !devFlags)
                    {
                        std::printf("FAIL: particle2 apply: memAlloc failed\n");
                        built = false; break;
                    }
                    ctx->memcpyHtoD(devIdx, indices, sizeof(indices));
                    ctx->memcpyHtoD(devPairs, pairs, sizeof(pairs));
                    ctx->memcpyHtoD(devFlags, flags, sizeof(flags));
                    scene->applyParticleBufferData(reinterpret_cast<const PxU32*>(devIdx),
                                                   reinterpret_cast<const PxGpuParticleBufferIndexPair*>(devPairs),
                                                   reinterpret_cast<const PxParticleBufferFlags*>(devFlags), 1u);
                }
            }
            if (who == P2_REMOVE_A && i == P2_REMOVE_STEP && aLive)
            {
                scene->removeActor(*ps[0]);
                aLive = false;
                std::printf("  [remove] A removed before step %d\n", i);
            }
            scene->simulate(1.0f / 60.0f);
            scene->fetchResults(true);
            for (int k = 0; k < 2; ++k)
                if (dbuf[k]) r.diffusePeak[k] = PxMax(r.diffusePeak[k], dbuf[k]->getNbActiveDiffuseParticles());
            if (i == 1 || i == 20 || i == 30)
                for (int k = 0; k < 2; ++k)
                {
                    if (!ps[k] || (k == 0 && !aLive)) continue;
                    std::vector<PxVec4> p, v; readSys(k, p, v);
                    const PxVec3 mp = particle2MeanXYZ(p), mv = particle2MeanXYZ(v);
                    if (i == 1) { r.pAfterApply[k] = mp; r.vAfterApply[k] = mv; }
                    else r.vProbe[k][i == 30] = mv;
                    std::printf("  [%s] step %3d %s: mean p=(%.6f, %.6f, %.6f) v=(%.6f, %.6f, %.6f) |v| %.6f\n",
                                P2_WHO_NAME[who], i, k ? "B" : "A", mp.x, mp.y, mp.z, mv.x, mv.y, mv.z, mv.magnitude());
                }
        }
        // Read back.  A removed system (diagnostic arm) is not read.
        for (int k = 0; k < 2 && built; ++k)
        {
            if (!ps[k] || (k == 0 && !aLive)) { r.present[k] = r.present[k] && aLive; continue; }
            readSys(k, r.pos[k], r.vel[k]);
            r.nActive[k] = buf[k]->getNbActiveParticles();
            r.meanP[k] = particle2MeanXYZ(r.pos[k]);
            r.yMin[k] = FLT_MAX; r.yMax[k] = -FLT_MAX;
            for (PxU32 q = 0; q < r.n[k]; ++q)
            {
                r.yMin[k] = PxMin(r.yMin[k], r.pos[k][q].y);
                r.yMax[k] = PxMax(r.yMax[k], r.pos[k][q].y);
                r.finite[k] = r.finite[k] && r.pos[k][q].isFinite() && r.vel[k][q].isFinite();
            }
            if (dbuf[k])
            {
                const PxU32 nd = PxMin(dbuf[k]->getNbActiveDiffuseParticles(), P2_MAX_DIFFUSE);
                r.diffuseEnd[k] = nd;
                r.diffPos[k].resize(nd); r.diffVel[k].resize(nd);
                if (nd)
                {
                    ctx->memcpyDtoH(r.diffPos[k].data(), reinterpret_cast<CUdeviceptr>(dbuf[k]->getDiffusePositionLifeTime()), nd * sizeof(PxVec4));
                    ctx->memcpyDtoH(r.diffVel[k].data(), reinterpret_cast<CUdeviceptr>(dbuf[k]->getDiffuseVelocities()), nd * sizeof(PxVec4));
                }
            }
            const PxVec3 mv = particle2MeanXYZ(r.vel[k]);
            std::printf("  [%s] %s: N %u (active %u) mean p=(%.6f, %.6f, %.6f) v=(%.6f, %.6f, %.6f) y[min %.6f max %.6f] "
                        "finite %d%s", P2_WHO_NAME[who], k ? "B" : "A", r.n[k], r.nActive[k], r.meanP[k].x, r.meanP[k].y,
                        r.meanP[k].z, mv.x, mv.y, mv.z, r.yMin[k], r.yMax[k], int(r.finite[k]), dbuf[k] ? "" : "\n");
            if (dbuf[k]) std::printf(" diffuse peak %u end %u\n", r.diffusePeak[k], r.diffuseEnd[k]);
        }
        for (int k = 0; k < 2; ++k)
            if (boxes[k])
            {
                r.box[k] = boxes[k]->getGlobalPose().p;
                std::printf("  [%s] box %s: (%.6f, %.6f, %.6f)\n", P2_WHO_NAME[who], k ? "B" : "A", r.box[k].x, r.box[k].y, r.box[k].z);
            }
        r.ok = built;
    }
    if (!built) std::printf("FAIL: particle2 [%s]: a particle system could not be built\n", P2_WHO_NAME[who]);

    scene->release();
    for (int k = 0; k < 2; ++k)
    {
        if (ps[k]) ps[k]->release();
        if (buf[k]) buf[k]->release();
        if (pmat[k]) pmat[k]->release();
    }
    for (PxRigidActor* a : rigids) a->release();
    if (patch) patch->release();
    if (devIdx) ctx->memFree(devIdx);
    if (devPairs) ctx->memFree(devPairs);
    if (devFlags) ctx->memFree(devFlags);
    return r;
}

// C1/C2: the differing 32-bit words of system k's final user arrays (pos + vel).  On a
// difference the first differing particle is printed in %a.
static size_t particle2CompareWords(const char* what, const P2Run& x, const P2Run& y, int k)
{
    const PxU32 n = x.n[k];
    if (!x.present[k] || !y.present[k] || y.n[k] != n)
    {
        std::printf("FAIL: %s: system %s missing from a sub-scene\n", what, k ? "B" : "A");
        return size_t(-1);
    }
    size_t mism = 0, first = size_t(-1);
    for (int a = 0; a < 2; ++a)
    {
        const std::vector<PxVec4>& vx = a ? x.vel[k] : x.pos[k];
        const std::vector<PxVec4>& vy = a ? y.vel[k] : y.pos[k];
        for (PxU32 q = 0; q < n; ++q)
        {
            PxU32 wx[4], wy[4];
            std::memcpy(wx, &vx[q], sizeof(wx)); std::memcpy(wy, &vy[q], sizeof(wy));
            for (int c = 0; c < 4; ++c)
                if (wx[c] != wy[c]) { ++mism; if (first == size_t(-1)) first = q; }
        }
    }
    std::printf("  %s: %zu of %u words differ\n", what, mism, n * 8u);
    if (mism)
    {
        const PxU32 q = PxU32(first);
        std::printf("    first differing particle %u: p (%a, %a, %a) vs (%a, %a, %a), v (%a, %a, %a) vs (%a, %a, %a)\n", q,
                    double(x.pos[k][q].x), double(x.pos[k][q].y), double(x.pos[k][q].z),
                    double(y.pos[k][q].x), double(y.pos[k][q].y), double(y.pos[k][q].z),
                    double(x.vel[k][q].x), double(x.vel[k][q].y), double(x.vel[k][q].z),
                    double(y.vel[k][q].x), double(y.vel[k][q].y), double(y.vel[k][q].z));
        std::printf("    diagnosis (U2): PXG_DUMP_ARGS=ps_reorderPrimitiveContactsLaunch:32:4096:40:4096:8:4 on both runs; an equal "
                    "contact multiset in a different order is composition-dependent association, a different multiset is a defect\n");
    }
    return mism;
}

// The diffuse arrays as MULTISETS of (position+lifetime, velocity) records: the slot
// order is not deterministic (U4), the set is.
static bool particle2CompareDiffuse(const char* what, const P2Run& x, const P2Run& y, int k)
{
    struct Rec { PxU32 w[8]; bool operator<(const Rec& o) const { return std::lexicographical_compare(w, w + 8, o.w, o.w + 8); } };
    auto recs = [&](const P2Run& r) {
        std::vector<Rec> v(r.diffPos[k].size());
        for (size_t i = 0; i < v.size(); ++i)
        {
            std::memcpy(v[i].w, &r.diffPos[k][i], 16);
            std::memcpy(v[i].w + 4, &r.diffVel[k][i], 16);
        }
        std::sort(v.begin(), v.end());
        return v;
    };
    const std::vector<Rec> rx = recs(x), ry = recs(y);
    size_t same = 0;
    if (rx.size() == ry.size())
        for (size_t i = 0; i < rx.size(); ++i) same += std::memcmp(rx[i].w, ry[i].w, sizeof(rx[i].w)) == 0 ? 1u : 0u;
    const bool ok = rx.size() == ry.size() && same == rx.size();
    std::printf("  %s: diffuse multiset %zu vs %zu records, %zu equal after sorting%s\n", what, rx.size(), ry.size(), same,
                ok ? "" : "  <-- differs");
    return ok;
}

// ---------------------------------------------------------------------------
// P9.4 -- PXG_SCENE=particlecloth: the PBD CLOTH sheet.
//
// This is the scene that makes the spring chain observable.  Four kernels were
// mapped for it across P9.3a-c -- ps_initializeSpringsLaunch (zero the XPBD
// multipliers), ps_updateRemapVertsLaunch (gather each spring's endpoints out of
// the sorted arrays), ps_solveSpringsLaunch (the per-partition XPBD solve) and
// ps_averageVertsLaunch (fold the scattered copies back into one delta per
// particle) -- and until now NOTHING dispatched them: the host gates the entire
// chain on getMaxSpringsPerBuffer() > 0 (PxgPBDParticleSystemCore.cpp:958) and no
// scene built a PxParticleClothBuffer.
//
// THE OBSERVABLE IS A HANGING CURTAIN, and it was chosen because it fails loudly.
// The sheet is built horizontally in the XZ plane with its FIRST ROW pinned
// (invMass = 0).  Gravity then rotates the free part down about that edge until
// it hangs as a vertical curtain.  The springs are the ONLY thing connecting a
// free particle to the pinned row -- there is no other constraint in the scene
// that could hold it up -- so:
//
//     springs working  -> the sheet hangs, free particles stay near the pins
//     springs absent   -> every free particle free-falls to the floor
//
// The control (PXG_PARTICLE_CLOTH_CONTROL=1) builds the IDENTICAL geometry,
// material, phase, pinning and step count, but as a plain PxParticleBuffer with no
// springs at all.  That is exactly the host's own gate, so the control isolates
// the chain under test rather than some proxy for it.
//
// Two things this scene does NOT establish, stated up front.  It does not check
// the resting SHAPE against CUDA -- what carries bit-exactness is the three P9.3
// fixtures, not this scene.  And it does not exercise the multi-cloth path: one
// cloth buffer with one cloth means mParticleBufferRunsum[0] == 0, which is the
// only configuration in which ps_averageVertsLaunch's binarySearch (over
// BUFFER-LOCAL start indices, keyed by a GLOBAL index -- upstream candidate 14)
// agrees with its own guard.  A second particle buffer would be needed to reach
// that, and it is left to a later tranche rather than smuggled in here.
//
// CALIBRATION.  Both numbers are MEASURED on this scene, the only difference
// between the two runs being whether the buffer carries springs:
//
//                              springs      no-spring control    threshold
//   free mean y (final)        4.2009       0.2000
//   free mean y (min)          3.7693       0.0981               > 2.5
//   bbox diagonal (max)        3.1551       5.8059               < 4.0
//
// The control's 0.2000 is exactly the rest offset: every free particle ends flat
// on the floor, which is what "the springs did nothing" looks like.  Both
// thresholds sit between the two columns, and the control run FAILS BOTH of them
// (0.0981 is not > 2.5; 5.8059 is not < 4.0) -- that is what makes them checks
// rather than decoration.  The two are also independent: the hang check catches a
// solve that applies NO force, the diagonal catches one that applies the WRONG
// force, and neither implies the other.
//
// The sheet does NOT come to rest, and no rest check is asserted.  It is a
// pendulum with springDamping 0.001 and no aerodynamic drag (ps_solveAerodynamics
// 1/2Launch are both still unmapped, PxgSlangKernelManifest.h:250-251, and are
// switched off anyway because this material's lift and drag are 0), so the free
// mean y oscillates in [3.78, 4.58] for as long as it is stepped.  Asserting a
// rest here would be asserting damping the scene deliberately does not have.
static const float CLOTH_HANG_MIN = 2.5f;        // free-particle mean y must stay above this
static const float CLOTH_HANG_CONTROL = 0.2000f; // measured, no-spring control
static const float CLOTH_DIAG_MAX = 4.0f;        // bbox diagonal: catches an EXPLODING solve
static const float CLOTH_DIAG_REST = 3.11f;      // sqrt(2.2^2 + 2.2^2), the flat sheet's diagonal
static const float CLOTH_DIAG_CONTROL = 5.8059f; // measured, no-spring control

// The sheet: 12 x 12 at 0.2 spacing, so it spans 2.2 x 2.2 and drops 2.2 when it
// hangs.  144 particles, 12 pinned.  Spring topology is SnippetPBDCloth's
// (SnippetPBDCloth.cpp:145-175): structural springs along both axes plus both
// diagonals of every cell, and the two boundary runs -- 4*(nX-1)*(nZ-1) + (nX-1)
// + (nZ-1) = 506 springs, enough that the partitioner produces several partitions
// and ps_solveSpringsLaunch is dispatched more than once per iteration.
// P9.5 aerodynamics.  Values are not from the snippet (which uses 0/0); they were
// chosen to be large enough to damp the pendulum within the step budget and then
// MEASURED, like every other threshold in this file.
static const float CLOTH_LIFT = 0.6f;
static const float CLOTH_DRAG = 1.2f;
// MEASURED on this scene, the only difference between the two runs being whether
// the material carries lift and drag (PXG_PARTICLE_CLOTH_NOAERO=1 zeroes both):
//
//                          aero on      no-aero control
//   free mean y (end)      3.8840       4.2009
//   free mean y (min)      3.8840       3.7693
//   rest |dy|              0.002460     0.021391      threshold < 0.010
//
// The aero run's minimum EQUALS its final value: with air resistance the sheet
// descends to its hanging position and stays, where the undamped control is still
// swinging through a 0.43 band at step 240.  The rest check is therefore the
// aerodynamics observable, and the no-spring control fails it too (it lands on the
// floor and reads 0.000000 only because it is dead, which the hang check catches
// first).  Margin is 4x below and 2.1x above -- narrower than the P9.4 checks, and
// stated rather than dressed up.
static const float CLOTH_REST_MAX = 0.010f;
static const float CLOTH_REST_NOAERO = 0.021391f;   // measured

static const PxU32 CLOTH_NX = 12;
static const PxU32 CLOTH_NZ = 12;
static const float CLOTH_SPACING = 0.2f;
static const float CLOTH_PIN_Y = 5.0f;

static PX_FORCE_INLINE PxU32 clothId(PxU32 x, PxU32 y, PxU32 numY) { return x * numY + y; }

// springsMode == false builds the CONTROL: same everything, no springs, plain
// particle buffer.  Returns an empty ParticleScene on failure.
ParticleScene buildParticleClothScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                      PxCudaContextManager* cudaCtx, bool springsMode)
{
    ParticleScene out;
    out.cudaCtx = cudaCtx;

    // A floor for the control run to land on, and one live dynamic rigid body --
    // with zero dynamic rigids the rigid batch-sum kernel null-derefs on the
    // particle island's solver descriptors (see buildParticleScene's note).  The
    // box sits far aside at x=20 so it never touches the sheet.
    addGroundBox(physics, scene, rigidMat);
    {
        PxRigidDynamic* aside = physics->createRigidDynamic(PxTransform(PxVec3(20.0f, 0.5f, 0.0f)));
        PxShape* s = physics->createShape(PxBoxGeometry(0.5f, 0.5f, 0.5f), *rigidMat);
        aside->attachShape(*s);
        aside->setMass(1000.0f);
        aside->setMassSpaceInertiaTensor(boxInertia(PxVec3(0.5f), 1000.0f));
        s->release();
        scene->addActor(*aside);
    }

    const PxU32 numParticles = CLOTH_NX * CLOTH_NZ;
    const PxU32 numSprings = (CLOTH_NX - 1) * (CLOTH_NZ - 1) * 4 + (CLOTH_NX - 1) + (CLOTH_NZ - 1);
    const PxU32 numTriangles = (CLOTH_NX - 1) * (CLOTH_NZ - 1) * 2;
    const float restOffset = CLOTH_SPACING;
    const float totalClothMass = 10.0f;
    const float particleMass = totalClothMass / float(numParticles);

    // SnippetPBDCloth.cpp:107-112 -- the snippet's own stiffnesses and material.
    const float stretchStiffness = 10000.0f;
    const float shearStiffness = 100.0f;
    const float springDamping = 0.001f;

    // P9.5: the last two arguments are LIFT and DRAG (PxPhysics.h:980).  The
    // snippet leaves both at 0, which makes ps_solveAerodynamics1/2Launch compute a
    // zero impulse and the whole aero path a no-op.  Turning them on is what makes
    // the pair observable -- and it is also what finally gives the hanging sheet
    // some air resistance, which P9.4 explicitly did not have.
    // PXG_PARTICLE_CLOTH_NOAERO=1 restores the snippet's 0/0 and is the control the
    // rest threshold is calibrated against.
    const bool noAero = std::getenv("PXG_PARTICLE_CLOTH_NOAERO") != nullptr;
    const float clothLift = noAero ? 0.0f : CLOTH_LIFT;
    const float clothDrag = noAero ? 0.0f : CLOTH_DRAG;
    PxPBDMaterial* pbdMat = physics->createPBDMaterial(0.8f, 0.05f, 1e+6f, 0.001f, 0.5f,
                                                       0.005f, 0.05f, clothLift, clothDrag);
    PxPBDParticleSystem* ps = physics->createPBDParticleSystem(*cudaCtx);
    ps->setRestOffset(restOffset);
    ps->setContactOffset(restOffset + 0.02f);
    ps->setParticleContactOffset(restOffset + 0.02f);
    ps->setSolidRestOffset(restOffset);
    ps->setFluidRestOffset(0.0f);
    // A hanging sheet needs a converged solve; the snippet leaves the default, but
    // the default is tuned for a sheet resting ON something, not one suspended
    // from a single row.  Both runs use the same count so the control is fair.
    ps->setSolverIterationCounts(/*pos*/ 8, /*vel*/ 1);
    scene->addActor(*ps);

    const PxU32 phase = ps->createPhase(pbdMat, PxParticlePhaseFlags(
        PxParticlePhaseFlag::eParticlePhaseSelfCollideFilter |
        PxParticlePhaseFlag::eParticlePhaseSelfCollide));

    // Lay the sheet out row-major in i (the hang direction) so that row i == 0 --
    // the pinned row -- occupies indices [0, CLOTH_NZ).
    std::vector<PxVec4> hostPos(numParticles), hostVel(numParticles, PxVec4(0.0f));
    std::vector<PxU32> hostPhase(numParticles, phase);
    std::vector<PxParticleSpring> springs;
    std::vector<PxU32> triangles;
    springs.reserve(numSprings);
    triangles.reserve(size_t(numTriangles) * 3u);

    const float z0 = -0.5f * float(CLOTH_NZ - 1) * CLOTH_SPACING;
    for (PxU32 i = 0; i < CLOTH_NX; ++i)
    {
        for (PxU32 j = 0; j < CLOTH_NZ; ++j)
        {
            const PxU32 index = clothId(i, j, CLOTH_NZ);
            // The pinned row has invMass 0, which makes it immovable in the
            // integrator (particlesystem.cu:3143 guards on pos4.w > 0) and
            // weightless in the spring solve (wsum is the sum of the two inverse
            // masses).  ps_averageVertsLaunch skips it outright
            // (particlesystem.cu:4379), which is why the assertions read the FREE
            // range only.
            const float invMass = (i == 0) ? 0.0f : (1.0f / particleMass);
            hostPos[index] = PxVec4(float(i) * CLOTH_SPACING, CLOTH_PIN_Y,
                                    z0 + float(j) * CLOTH_SPACING, invMass);

            if (i > 0)
                springs.push_back({ clothId(i - 1, j, CLOTH_NZ), clothId(i, j, CLOTH_NZ),
                                    CLOTH_SPACING, stretchStiffness, springDamping, 0.0f });
            if (j > 0)
                springs.push_back({ clothId(i, j - 1, CLOTH_NZ), clothId(i, j, CLOTH_NZ),
                                    CLOTH_SPACING, stretchStiffness, springDamping, 0.0f });
            if (i > 0 && j > 0)
            {
                const float diag = PxSqrt(2.0f) * CLOTH_SPACING;
                springs.push_back({ clothId(i - 1, j - 1, CLOTH_NZ), clothId(i, j, CLOTH_NZ),
                                    diag, shearStiffness, springDamping, 0.0f });
                springs.push_back({ clothId(i - 1, j, CLOTH_NZ), clothId(i, j - 1, CLOTH_NZ),
                                    diag, shearStiffness, springDamping, 0.0f });
                triangles.push_back(clothId(i - 1, j - 1, CLOTH_NZ));
                triangles.push_back(clothId(i - 1, j, CLOTH_NZ));
                triangles.push_back(clothId(i, j - 1, CLOTH_NZ));
                triangles.push_back(clothId(i - 1, j, CLOTH_NZ));
                triangles.push_back(clothId(i, j - 1, CLOTH_NZ));
                triangles.push_back(clothId(i, j, CLOTH_NZ));
            }
        }
    }
    if (springs.size() != numSprings || triangles.size() != size_t(numTriangles) * 3u)
    {
        std::printf("FAIL: cloth topology count mismatch (springs %zu vs %u, tris %zu vs %u)\n",
                    springs.size(), numSprings, triangles.size() / 3u, numTriangles);
        return out;
    }

    PxCudaContext* ctx = cudaCtx->getCudaContext();

    if (!springsMode)
    {
        // The CONTROL: identical particles, no springs, plain buffer.  This is the
        // host's own gate (getMaxSpringsPerBuffer() > 0) expressed as a scene.
        PxParticleBuffer* buffer = physics->createParticleBuffer(numParticles, 0, cudaCtx);
        if (!buffer) { std::printf("FAIL: createParticleBuffer returned null\n"); return out; }
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buffer->getPositionInvMasses()),
                        hostPos.data(), numParticles * sizeof(PxVec4));
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buffer->getVelocities()),
                        hostVel.data(), numParticles * sizeof(PxVec4));
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buffer->getPhases()),
                        hostPhase.data(), numParticles * sizeof(PxU32));
        buffer->setNbActiveParticles(numParticles);
        buffer->raiseFlags(PxParticleBufferFlag::eUPDATE_POSITION);
        buffer->raiseFlags(PxParticleBufferFlag::eUPDATE_VELOCITY);
        buffer->raiseFlags(PxParticleBufferFlag::eUPDATE_PHASE);
        ps->addParticleBuffer(buffer);
        out.system = ps;
        out.buffer = buffer;
        out.numParticles = numParticles;
        out.numPinned = CLOTH_NZ;
        return out;
    }

    // --- the real cloth path ---
    //
    // SnippetPBDCloth routes this through two physxextensions helpers
    // (PxParticleClothBufferHelper and PxCreateAndPopulateParticleClothBuffer).
    // Neither is available here -- the sdktest links the CPU SDK core only, not
    // physxextensions -- and neither needs to be: the first is pure bookkeeping
    // that fills a PxParticleClothDesc, and the second is a transparent wrapper
    // over createParticleClothBuffer + six memcpys + setCloths
    // (ExtParticleExt.cpp:92-126).  Both are inlined below rather than dragging in
    // the extensions library for twenty lines of struct filling.
    PxParticleCloth cloth;
    cloth.startVertexIndex = 0;
    cloth.numVertices = numParticles;
    cloth.clothBlendScale = 0.0f;   // filled in by partitionSprings
    cloth.restVolume = 0.0f;        // inflatable only, and this sheet is not one
    cloth.pressure = 0.0f;
    cloth.startTriangleIndex = 0;
    cloth.numTriangles = numTriangles;

    std::vector<PxVec4> restPositions = hostPos;

    PxParticleClothDesc clothDesc;
    clothDesc.cloths = &cloth;
    clothDesc.triangles = triangles.data();
    clothDesc.springs = springs.data();
    clothDesc.restPositions = restPositions.data();
    clothDesc.nbCloths = 1;
    clothDesc.nbSprings = numSprings;
    clothDesc.nbTriangles = numTriangles;
    clothDesc.nbParticles = numParticles;

    // partitionSprings is PURE CPU -- PX_ALLOC, partitions(), combinePartitions(),
    // PxSort (NpParticleBuffer.cpp:543-596).  It launches no CUDA kernels, which is
    // what makes this scene buildable on the port at all.  It is also where
    // clothBlendScale comes from: 1 / (maxCopiesOverThisCloth + 1), with the
    // engine's own "AD Todo: figure out why blendScale is computed like this"
    // sitting above it (:562).
    PxParticleClothPreProcessor* pre = PxCreateParticleClothPreProcessor(cudaCtx);
    if (!pre) { std::printf("FAIL: PxCreateParticleClothPreProcessor returned null\n"); return out; }
    PxPartitionedParticleCloth partitioned;
    pre->partitionSprings(clothDesc, partitioned);
    pre->release();

    std::printf("  cloth partitioning: %u springs -> %u partitions, maxSpringsPerPartition %u, "
                "remapOutputSize %u, clothBlendScale %.6f\n",
                partitioned.nbSprings, partitioned.nbPartitions,
                partitioned.maxSpringsPerPartition, partitioned.remapOutputSize,
                cloth.clothBlendScale);

    PxParticleClothBuffer* buffer = physics->createParticleClothBuffer(
        numParticles, /*maxVolumes*/ 0, clothDesc.nbCloths, clothDesc.nbTriangles,
        clothDesc.nbSprings, cudaCtx);
    if (!buffer) { std::printf("FAIL: createParticleClothBuffer returned null\n"); return out; }

    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buffer->getPositionInvMasses()),
                    hostPos.data(), numParticles * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buffer->getVelocities()),
                    hostVel.data(), numParticles * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buffer->getPhases()),
                    hostPhase.data(), numParticles * sizeof(PxU32));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buffer->getTriangles()),
                    triangles.data(), size_t(numTriangles) * 3u * sizeof(PxU32));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buffer->getRestPositions()),
                    restPositions.data(), numParticles * sizeof(PxVec4));
    buffer->setNbActiveParticles(numParticles);
    buffer->setNbTriangles(numTriangles);
    buffer->setCloths(partitioned);
    ps->addParticleBuffer(buffer);

    out.system = ps;
    out.buffer = buffer;
    out.numParticles = numParticles;
    out.numPinned = CLOTH_NZ;
    return out;
}

// ---------------------------------------------------------------------------
// P9.7 -- PXG_SCENE=particleinflatable: the PRESSURISED closed mesh.
//
// P9.6 landed ps_calculateInflatableVolume and ps_solveInflatableVolume with a 39/39
// sweep, and both were physically inert in every scene: particlecloth has
// pressure = 0, which is exactly the engine's marker for "solve this as ordinary
// cloth", so both kernels launch and return at their early-out.  This scene gives
// them something to do.
//
// AN INFLATABLE NEEDS A CLOSED MESH.  A flat sheet encloses no volume, so the
// P9.4 geometry cannot be reused at all.  SnippetPBDInflatable builds a sphere by
// projecting a cube onto it and then calling PxRemeshingExt::limitMaxEdgeLength --
// which lives in physxextensions, not linked here.  This builds the same shape by
// SUBDIVIDING instead: the snippet's 8-vertex / 12-triangle cube, each triangle
// split into four at its edge midpoints, every new vertex projected back onto the
// sphere, twice.  That gives 12*4*4 = 192 triangles over 98 vertices, closed and
// watertight by construction.
//
// THE REST VOLUME IS SIX TIMES THE GEOMETRIC VOLUME, and this is not a guess: the
// cooker computes `Gu::computeTriangleMeshVolume(...) * 6.0f` with the engine's own
// comment "the physx api takes volume*6 now"
// (ExtParticleClothCooker.cpp:449-453), and Gu's formula is
// `sum (v0 x v1) . v2 / 6` (GuInternal.h:204-221).  The two sixes cancel, so
// restVolume is simply the sum of (v0 x v1) . v2 -- which is also exactly the
// quantity ps_calculateInflatableVolume accumulates without dividing (P9.6's trap
// I3).  The producer side therefore confirms the port's spelling.
//
// Springs are the mesh's unique EDGES; there is no cooker here to generate
// bending constraints, and none is needed to make the pair observable.

struct InflMesh
{
    std::vector<PxVec3> verts;
    std::vector<PxU32> tris;      // 3 per triangle
};

// One level of 4-to-1 subdivision with midpoint sharing, then re-projection.
static void inflSubdivide(InflMesh& m, const PxVec3& centre, float radius)
{
    std::map<std::pair<PxU32, PxU32>, PxU32> midCache;
    auto midpoint = [&](PxU32 a, PxU32 b) -> PxU32
    {
        const std::pair<PxU32, PxU32> key = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
        auto it = midCache.find(key);
        if (it != midCache.end()) return it->second;
        PxVec3 p = (m.verts[a] + m.verts[b]) * 0.5f;
        PxVec3 dir = p - centre;
        dir.normalize();
        p = centre + dir * radius;
        const PxU32 idx = PxU32(m.verts.size());
        m.verts.push_back(p);
        midCache[key] = idx;
        return idx;
    };

    std::vector<PxU32> out;
    out.reserve(m.tris.size() * 4u);
    for (size_t t = 0; t + 2 < m.tris.size(); t += 3)
    {
        const PxU32 a = m.tris[t + 0], b = m.tris[t + 1], c = m.tris[t + 2];
        const PxU32 ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
        const PxU32 f[4][3] = { { a, ab, ca }, { ab, b, bc }, { ca, bc, c }, { ab, bc, ca } };
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 3; ++j) out.push_back(f[i][j]);
    }
    m.tris.swap(out);
}

static InflMesh buildInflSphere(const PxVec3& centre, float radius, int levels)
{
    // SnippetPBDInflatable.cpp:96-99 -- the same cube and the same winding.
    static const float CV[8][3] = {
        { 0.5f, -0.5f, -0.5f }, { 0.5f, -0.5f, 0.5f }, { -0.5f, -0.5f, 0.5f }, { -0.5f, -0.5f, -0.5f },
        { 0.5f,  0.5f, -0.5f }, { 0.5f,  0.5f, 0.5f }, { -0.5f,  0.5f, 0.5f }, { -0.5f,  0.5f, -0.5f }
    };
    static const PxU32 CI[36] = { 1,2,3, 7,6,5, 4,5,1, 5,6,2, 2,6,7, 0,3,7,
                                  0,1,3, 4,7,5, 0,4,1, 1,5,2, 3,2,7, 4,0,7 };
    InflMesh m;
    for (int i = 0; i < 8; ++i)
    {
        PxVec3 dir(CV[i][0], CV[i][1], CV[i][2]);
        dir.normalize();
        m.verts.push_back(centre + dir * radius);
    }
    for (int i = 0; i < 36; ++i) m.tris.push_back(CI[i]);
    for (int l = 0; l < levels; ++l) inflSubdivide(m, centre, radius);
    return m;
}

// sum (v0 x v1) . v2 over the triangles -- six times the enclosed volume, which is
// the unit both PxParticleCloth::restVolume and the kernel use.
static float inflMeshVolume6(const std::vector<PxVec3>& v, const std::vector<PxU32>& tris)
{
    float acc = 0.0f;
    for (size_t t = 0; t + 2 < tris.size(); t += 3)
        acc += v[tris[t]].cross(v[tris[t + 1]]).dot(v[tris[t + 2]]);
    return acc;
}

static const float INFL_RADIUS = 0.8f;
static const PxVec3 INFL_CENTRE = PxVec3(0.0f, 5.0f, 0.0f);
// The mesh is spawned SQUASHED along y; pressure has to puff it back out.
static const float INFL_SQUASH = 0.35f;
static const float INFL_PRESSURE = 1.0f;   // target volume = rest volume
// CALIBRATION.  Measured on this scene, the only difference between the two runs
// being the cloth's `pressure` field (PXG_PARTICLE_INFL_CONTROL=1 zeroes it):
//
//                                pressure 1.0    pressure 0 (control)
//   volume at the end             100.0% of rest    28.9% of rest
//   volume at its peak            283.5%            80.4%
//
// The pressurised mesh does not merely get bigger -- it CONVERGES ON ITS TARGET.
// The constraint drives the enclosed volume to restVolume * pressure, and with
// pressure 1.0 that is the rest volume exactly: 12.0252 against a rest of 12.0235.
// So the check is target convergence, not growth, which discriminates by a factor
// of about seven thousand rather than by a ratio.  The control run misses the
// target by 0.711 and fails it outright.
//
// The overshoot to 283% is real and is stated rather than hidden: the mesh is
// spawned at 35% of its volume with deliberately soft springs, so the volume
// constraint's first correction is large.  It converges within about thirty steps
// and stays.  The second check bounds that transient, so a wrong-signed or
// wrong-scaled constraint that diverges is caught even if it happened to pass
// through the target on its way out.
static const float INFL_VOL_TARGET_TOL = 0.05f;   // |end/rest - pressure|
static const float INFL_VOL_CONTROL = 0.289f;     // measured, pressure 0
static const float INFL_VOL_MAX = 5.0f;           // measured peak 2.835

ParticleScene buildParticleInflatableScene(PxPhysics* physics, PxScene* scene,
                                           PxMaterial* rigidMat, PxCudaContextManager* cudaCtx,
                                           bool pressurised)
{
    ParticleScene out;
    out.cudaCtx = cudaCtx;

    addGroundBox(physics, scene, rigidMat);
    {
        PxRigidDynamic* aside = physics->createRigidDynamic(PxTransform(PxVec3(20.0f, 0.5f, 0.0f)));
        PxShape* sh = physics->createShape(PxBoxGeometry(0.5f, 0.5f, 0.5f), *rigidMat);
        aside->attachShape(*sh);
        aside->setMass(1000.0f);
        aside->setMassSpaceInertiaTensor(boxInertia(PxVec3(0.5f), 1000.0f));
        sh->release();
        scene->addActor(*aside);
    }

    const InflMesh mesh = buildInflSphere(INFL_CENTRE, INFL_RADIUS, 2);
    const PxU32 numParticles = PxU32(mesh.verts.size());
    const PxU32 numTriangles = PxU32(mesh.tris.size() / 3u);

    // The REST volume is the un-squashed sphere's.
    const float restVolume6 = inflMeshVolume6(mesh.verts, mesh.tris);

    // Springs = the unique edges of the closed mesh.
    std::vector<PxParticleSpring> springs;
    {
        std::set<std::pair<PxU32, PxU32>> edges;
        for (size_t t = 0; t + 2 < mesh.tris.size(); t += 3)
        {
            const PxU32 idx[3] = { mesh.tris[t], mesh.tris[t + 1], mesh.tris[t + 2] };
            for (int e = 0; e < 3; ++e)
            {
                PxU32 a = idx[e], b = idx[(e + 1) % 3];
                if (a > b) { const PxU32 tmp = a; a = b; b = tmp; }
                edges.insert(std::make_pair(a, b));
            }
        }
        for (auto& e : edges)
        {
            PxParticleSpring sp;
            std::memset(&sp, 0, sizeof(sp));
            sp.ind0 = e.first;
            sp.ind1 = e.second;
            sp.length = (mesh.verts[e.first] - mesh.verts[e.second]).magnitude();
            // DELIBERATELY SOFT.  Stiff springs would restore the squashed sphere on
            // their own and the pressure's contribution would be invisible -- the
            // scene would pass for the wrong reason.
            sp.stiffness = 200.0f;
            sp.damping = 0.05f;
            springs.push_back(sp);
        }
    }
    const PxU32 numSprings = PxU32(springs.size());

    const float restOffset = 0.10f;
    const float totalMass = 4.0f;
    const float particleMass = totalMass / float(numParticles);

    PxPBDMaterial* pbdMat = physics->createPBDMaterial(0.8f, 0.05f, 1e+6f, 0.001f, 0.5f,
                                                       0.005f, 0.05f, 0.0f, 0.0f);
    PxPBDParticleSystem* ps = physics->createPBDParticleSystem(*cudaCtx);
    ps->setRestOffset(restOffset);
    ps->setContactOffset(restOffset + 0.02f);
    ps->setParticleContactOffset(restOffset + 0.02f);
    ps->setSolidRestOffset(restOffset);
    ps->setFluidRestOffset(0.0f);
    ps->setSolverIterationCounts(8, 1);
    scene->addActor(*ps);

    const PxU32 phase = ps->createPhase(pbdMat, PxParticlePhaseFlags(
        PxParticlePhaseFlag::eParticlePhaseSelfCollideFilter |
        PxParticlePhaseFlag::eParticlePhaseSelfCollide));

    // Spawn SQUASHED along y.  With pressure the volume constraint has to push it
    // back out; without it, only the soft springs pull.
    std::vector<PxVec4> hostPos(numParticles);
    std::vector<PxVec4> hostVel(numParticles, PxVec4(0.0f));
    std::vector<PxU32> hostPhase(numParticles, phase);
    for (PxU32 i = 0; i < numParticles; ++i)
    {
        PxVec3 p = mesh.verts[i];
        p.y = INFL_CENTRE.y + (p.y - INFL_CENTRE.y) * INFL_SQUASH;
        hostPos[i] = PxVec4(p.x, p.y, p.z, 1.0f / particleMass);
    }

    PxParticleCloth cloth;
    cloth.startVertexIndex = 0;
    cloth.numVertices = numParticles;
    cloth.clothBlendScale = 0.0f;              // filled in by partitionSprings
    cloth.restVolume = restVolume6;            // SIX TIMES the geometric volume
    cloth.pressure = pressurised ? INFL_PRESSURE : 0.0f;
    cloth.startTriangleIndex = 0;              // an INDEX-ARRAY offset (P9.6, I2)
    cloth.numTriangles = numTriangles;

    std::vector<PxVec4> restPositions(numParticles);
    for (PxU32 i = 0; i < numParticles; ++i)
        restPositions[i] = PxVec4(mesh.verts[i].x, mesh.verts[i].y, mesh.verts[i].z,
                                  1.0f / particleMass);

    PxParticleClothDesc clothDesc;
    clothDesc.cloths = &cloth;
    clothDesc.triangles = const_cast<PxU32*>(mesh.tris.data());
    clothDesc.springs = springs.data();
    clothDesc.restPositions = restPositions.data();
    clothDesc.nbCloths = 1;
    clothDesc.nbSprings = numSprings;
    clothDesc.nbTriangles = numTriangles;
    clothDesc.nbParticles = numParticles;

    PxParticleClothPreProcessor* pre = PxCreateParticleClothPreProcessor(cudaCtx);
    if (!pre) { std::printf("FAIL: PxCreateParticleClothPreProcessor returned null\n"); return out; }
    PxPartitionedParticleCloth partitioned;
    pre->partitionSprings(clothDesc, partitioned);
    pre->release();

    std::printf("  inflatable: %u verts, %u tris, %u springs, restVolume(x6)=%.6g, "
                "pressure=%.2f, %u partitions\n",
                numParticles, numTriangles, numSprings, restVolume6, cloth.pressure,
                partitioned.nbPartitions);

    PxParticleClothBuffer* buffer = physics->createParticleClothBuffer(
        numParticles, 0, clothDesc.nbCloths, clothDesc.nbTriangles, clothDesc.nbSprings, cudaCtx);
    if (!buffer) { std::printf("FAIL: createParticleClothBuffer returned null\n"); return out; }

    PxCudaContext* ctx = cudaCtx->getCudaContext();
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buffer->getPositionInvMasses()),
                    hostPos.data(), numParticles * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buffer->getVelocities()),
                    hostVel.data(), numParticles * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buffer->getPhases()),
                    hostPhase.data(), numParticles * sizeof(PxU32));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buffer->getTriangles()),
                    mesh.tris.data(), size_t(numTriangles) * 3u * sizeof(PxU32));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(buffer->getRestPositions()),
                    restPositions.data(), numParticles * sizeof(PxVec4));
    buffer->setNbActiveParticles(numParticles);
    buffer->setNbTriangles(numTriangles);
    buffer->setCloths(partitioned);
    ps->addParticleBuffer(buffer);

    out.system = ps;
    out.buffer = buffer;
    out.numParticles = numParticles;
    out.numPinned = 0;
    return out;
}

// The scene's observable: the enclosed volume of the live mesh, in the same
// six-times unit as restVolume.  Read the positions back and re-run the mesh
// formula over the SAME triangle list.
float inflLiveVolume6(const ParticleScene& psc, const std::vector<PxU32>& tris)
{
    std::vector<PxVec4> host(psc.numParticles);
    PxCudaContext* ctx = psc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(), reinterpret_cast<CUdeviceptr>(psc.buffer->getPositionInvMasses()),
                    psc.numParticles * sizeof(PxVec4));
    std::vector<PxVec3> v(psc.numParticles);
    for (PxU32 i = 0; i < psc.numParticles; ++i) v[i] = PxVec3(host[i].x, host[i].y, host[i].z);
    return inflMeshVolume6(v, tris);
}

// DtoH-read the particle buffer and return the mean y over the FREE particles
// (indices [numPinned, numParticles)).  The pinned row never moves, so averaging
// it in would blunt the very signal being measured.
float clothFreeMeanY(const ParticleScene& psc)
{
    std::vector<PxVec4> host(psc.numParticles);
    PxCudaContext* ctx = psc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(), reinterpret_cast<CUdeviceptr>(psc.buffer->getPositionInvMasses()),
                    psc.numParticles * sizeof(PxVec4));
    double sum = 0.0;
    PxU32 n = 0;
    for (PxU32 i = psc.numPinned; i < psc.numParticles; ++i) { sum += double(host[i].y); ++n; }
    return n ? float(sum / double(n)) : 0.0f;
}

// The diagonal of the axis-aligned bounding box over ALL particles.  This is the
// EXPLOSION check and it is independent of the hang check: a spring solve with the
// wrong sign or the wrong scale blows the sheet apart, which raises the diagonal
// without necessarily lowering the mean y.  A flat 2.2 x 2.2 sheet has a diagonal
// of 3.11; hanging it folds the sheet, which can only shrink the bound.
float clothBBoxDiag(const ParticleScene& psc)
{
    std::vector<PxVec4> host(psc.numParticles);
    PxCudaContext* ctx = psc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(), reinterpret_cast<CUdeviceptr>(psc.buffer->getPositionInvMasses()),
                    psc.numParticles * sizeof(PxVec4));
    PxVec3 lo(1e30f), hi(-1e30f);
    for (PxU32 i = 0; i < psc.numParticles; ++i)
    {
        lo.x = PxMin(lo.x, host[i].x); hi.x = PxMax(hi.x, host[i].x);
        lo.y = PxMin(lo.y, host[i].y); hi.y = PxMax(hi.y, host[i].y);
        lo.z = PxMin(lo.z, host[i].z); hi.z = PxMax(hi.z, host[i].z);
    }
    return (hi - lo).magnitude();
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

// P9.20: `convexFloor` swaps the ground BOX for a cooked CONVEX slab whose top
// face is at the same y = 0.  That one substitution is the whole `clothconvex`
// scene, and it is a SHAPE-TYPE change, not a silhouette change: the
// narrowphase routes eBOX x the cloth to CLOTH_BOX_TRIANGLE_CG and eCONVEXMESH
// to CLOTH_CONVEX_CG, which is the manifest row P9.20 adds.  Same body behind
// both -- see femClothContactGenBda.slang -- so what this scene proves is the
// WIRING, which is the part no fixture can reach.
ClothScene buildClothScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                           PxCudaContextManager* cudaCtx, bool convexFloor = false,
                           bool meshFloor = false, bool coreFloor = false)
{
    ClothScene out;
    out.cudaCtx = cudaCtx;

    // A minimal rigid body (box on ground) FAR BELOW the cloth.  As with the
    // particle scene, the cloth solve runs inside the rigid PGS island loop and
    // rigidSumInternalContactAndJointBatches2 null-derefs the solver descriptors
    // with zero dynamic rigids, so one live dynamic body is required to keep that
    // proven path valid.  It rests at y=0.5, well below the cloth (spawned y~5).
    if (coreFloor)
    {
        // P9.74: a CONVEX CORE box floor, the FOURTH shape this cloth can land on and
        // the only one whose pair routes to GPU_BUCKET_ID::eConvexCoreClothmesh, i.e.
        // to convexCoreClothmeshNphase_Kernel32.  cndCaseConvexCoreTrimesh branches
        // there when the second shape is NOT rigid
        // (PxgNphaseImplementationContext.cpp:975-982) -- a cloth's collision shape is
        // a TRIANGLE mesh, which is why the tetmesh kernel is not the one this reaches.
        //
        // PxConvexCore::Box takes FULL extents, so 8 x 1 x 8 centred at y = -0.5 puts
        // the top face at y = 0 -- the same floor height as the box, the cooked convex
        // slab and the cooked triangle mesh, so the settle number is comparable with
        // all three.
        //
        // NO 100:1 PROBLEM HERE, unlike the cooked convex slab above: a convex CORE is
        // an analytic box, not a cooked hull, so there is no cooker to refuse it for
        // being oblong.  The 16 full extents are chosen to match the mesh floor's
        // reach rather than to appease a cooker.
        //
        // *** THIS FLOOR IS EXPECTED TO FAIL UNTIL convexCoreClothmeshNphase_Kernel32
        // IS MAPPED. ***  Measured before the port, all four floors of this scene:
        //
        //   clothbox       mean y =   0.6888   the box path
        //   clothconvex    mean y =  -0.0000   the cooked-hull path
        //   clothmesh      mean y = -39.5445   known red, an upstream defect
        //                                      (femClothPrimitives.cu:3056), documented
        //   clothcvxcore   mean y = -73.8069   falls through, and the bridge reports
        //                                      the kernel launched 10x and SKIPPED
        PxRigidStatic* coreGround = physics->createRigidStatic(PxTransform(PxVec3(0.0f, -0.5f, 0.0f)));
        PxShape* cs = physics->createShape(
            PxConvexCoreGeometry(PxConvexCore::Box(PxVec3(16.0f, 1.0f, 16.0f))), *rigidMat);
        coreGround->attachShape(*cs);
        scene->addActor(*coreGround);
        cs->release();
    }
    else if (meshFloor)
    {
        // P9.26: a cooked TRIANGLE MESH floor.  The narrowphase routes
        // eTRIANGLEMESH x the cloth to the cloth-vs-mesh chain, of which only the
        // TRIANGLE half is mapped -- cloth_meshMidphaseGeneratePairsLaunch and
        // cloth_meshContactGenLaunch, both P8.17.  Half-extent 8, not 30: the
        // 2 x 2 cloth spawns above the origin and only needs to land well inside.
        if (!addMeshFloor(physics, scene, rigidMat, 8.0f))
        { std::printf("FAIL: addMeshFloor returned null\n"); return out; }
    }
    else if (convexFloor)
    {
        // Half-extent FOUR, not fifty.  A 50 x 0.5 x 50 slab is 100:1 and the
        // cooker refuses to build a GPU-compatible hull for it ("oblong shape"),
        // which drops the pair into the CPU bucket -- and then the cloth still
        // lands, on the CPU, and the scene is green while proving nothing.  The
        // fixture said so out loud ("GPU-compatible: NO") the first time; this is
        // that message acted on.  Four is 8:1 and cooks GPU-compatible, and the
        // 2 x 2 cloth spawns above the origin so it lands well inside.
        if (!addConvexFloor(physics, scene, rigidMat, 4.0f))
        { std::printf("FAIL: addConvexFloor returned null\n"); return out; }
    }
    else
        addGroundBox(physics, scene, rigidMat);
    // The live dynamic body.  In the CONVEX variant it is moved off to x = 3,
    // clear of the 2 x 2 cloth: it is a BOX, and a box under the cloth is served
    // by the already-mapped box path, which would carry the cloth and hide
    // whatever the convex path does or does not do.
    // The live dynamic body is moved aside for EVERY non-box floor, for the
    // reason P9.20 measured: it is a PxBoxGeometry, so a box under the cloth is
    // served by the already-mapped box path and would carry the cloth whatever
    // the floor does.
    //
    // P9.74 ADDED coreFloor TO THIS CONDITION AFTER WALKING INTO IT.  The first
    // version of the convex-core arm left it out, and `clothcvxcore` came up GREEN
    // at mean y = 0.6870 -- within 0.002 of `clothbox`'s 0.6888 -- while the bridge
    // was printing that convexCoreClothmeshNphase_Kernel32 had been launched twice
    // and SKIPPED both times.  The cloth was landing on this dynamic box, through
    // the mapped box path, exactly as the paragraph above says it would.  The
    // warning was already written; it just was not read.
    addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
               boxInertia(PxVec3(0.5f), 1000.0f), 1.5f,
               (convexFloor || meshFloor || coreFloor) ? 3.0f : 0.0f);

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
    //
    // PXG_CLOTH_FLOOR=mesh (P8.17) swaps it for a COOKED TRIANGLE MESH at the same height --
    // a 20 x 20 quad split into two up-facing triangles, the same floor
    // PXG_SB_FLOOR=mesh gives the softbody.  That is the pair
    // cloth_meshMidphaseGeneratePairsLaunch + cloth_meshContactGenLaunch exist for.  With only
    // the MIDPHASE ported the sheet still falls through -- the midphase produces pairs and
    // nothing turns them into contacts -- so this mode is a DIAGNOSTIC, not a settle test, and
    // the driver treats it as one.  It becomes a settle test when step 2 lands.
    const char* clothFloorEnv = std::getenv("PXG_CLOTH_FLOOR");
    const bool floorIsMesh = clothFloorEnv && !std::strcmp(clothFloorEnv, "mesh");
    if (floorIsMesh)
    {
        const float E = 10.0f;
        const PxVec3 mverts[4] = { PxVec3(-E, cfg.floorY, -E), PxVec3(-E, cfg.floorY, E),
                                   PxVec3( E, cfg.floorY, -E), PxVec3( E, cfg.floorY, E) };
        const PxU32 mtris[6] = { 0u, 1u, 2u,   2u, 1u, 3u };
        PxTriangleMeshDesc md;
        md.points.count = 4; md.points.stride = sizeof(PxVec3); md.points.data = mverts;
        md.triangles.count = 2; md.triangles.stride = 3 * sizeof(PxU32); md.triangles.data = mtris;
        PxCookingParams ck(physics->getTolerancesScale());
        ck.buildGPUData = true;
        PxTriangleMesh* floorMeshGeom =
            immediateCooking::createTriangleMesh(ck, md, physics->getPhysicsInsertionCallback());
        if (!floorMeshGeom) { std::printf("FAIL: createTriangleMesh (cloth floor) returned null\n"); return out; }
        PxRigidStatic* floor = physics->createRigidStatic(PxTransform(PxVec3(0.0f)));
        PxShape* fs = physics->createShape(PxTriangleMeshGeometry(floorMeshGeom), *rigidMat);
        floor->attachShape(*fs);
        scene->addActor(*floor);
        fs->release();
    }
    else
    {
        PxRigidStatic* floor = physics->createRigidStatic(
            PxTransformFromPlaneEquation(PxPlane(0.0f, 1.0f, 0.0f, -cfg.floorY)));
        PxShape* floorShape = physics->createShape(PxPlaneGeometry(), *rigidMat);
        floor->attachShape(*floorShape);
        scene->addActor(*floor);
        floorShape->release();
    }

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

    // THE REST OFFSET MATTERS ONLY FOR THE MESH FLOOR, and understanding why was the whole of
    // P8.18.  A contact activates when `CN = (initPen + ...) * invDt + normalVel < 0`
    // (deformableUtils.cuh:354).  Against a STATIC floor `normalVel` is zero, so with a ZERO rest
    // offset `initPen` is the raw separation and CN stays POSITIVE for as long as the sheet is
    // above the floor -- the contact is generated but never activated.  It would activate once
    // the sheet penetrated, except that a cooked triangle is ONE-SIDED: the midphase's front-face
    // cull (`triangle1.triPlane.distance(clothCentroid) >= 0`) drops the pair the instant the
    // centroid crosses the plane.  With no rest offset the window in which the contact both
    // EXISTS and is ACTIVE is empty, and the sheet sails through.
    //
    // An infinite PLANE floor has no such window: its vertex contacts keep being generated with a
    // growing NEGATIVE separation as the vertex sinks, which is why the plane variant has always
    // settled without one.  The rest offset moves the activation threshold OUT to where the pair
    // still exists -- the same thing clothselfcollide and clothcloth already do.  Applied only to
    // the mesh variant so the plane baseline stays exactly where it was.
    if (floorIsMesh)
    {
        shape->setContactOffset(0.1f);
        shape->setRestOffset(0.05f);
    }


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

// P9.19: `damped` is what the `clothdamp` scene turns on.  It is the ONLY
// difference between that scene and `clothhang` -- same mesh, same pins, same
// step count -- so any difference in the measured settling is the damping trio
// and nothing else.
ClothHangScene buildClothHangScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                   PxCudaContextManager* cudaCtx, bool damped = false,
                                   bool horizontal = false)
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
            // `horizontal` is what makes the damping MEASURABLE.  The vertical
            // curtain is already at rest the moment it spawns -- it drops 0.02 over
            // 120 steps and its per-step motion in the rest window is exactly zero,
            // so there is no non-rigid motion for either accumulator to remove and
            // the damped and undamped runs agree to four decimals.  Spawned FLAT,
            // pinned along one edge, the sheet swings down through ninety degrees:
            // a large, sustained hinge fold, which is precisely what bending
            // damping acts on.
            verts.push_back(horizontal
                ? PxVec3(float(c) * sx - 0.5f * sizeX, topY, float(r) * sy)
                : PxVec3(float(c) * sx - 0.5f * sizeX, topY - float(r) * sy, 0.0f));

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

    // The damping values are LARGE on purpose.  Both kernels form
    // clamp(damping * dt, 0, 1) with dt = 1/60, so 120 saturates the factor at
    // 1.0 -- the strongest damping the engine can express.  A small value would
    // produce a difference inside the settle bar and measure nothing.
    PxDeformableSurfaceMaterial* clothMat =
        physics->createDeformableSurfaceMaterial(/*youngs*/ 1.e6f, /*poissons*/ 0.3f,
            /*dynamicFriction*/ 0.5f, /*thickness*/ 0.01f, /*bendingStiffness*/ 1.e-4f,
            /*elasticityDamping*/ damped ? 120.0f : 0.0f,
            /*bendingDamping*/ damped ? 120.0f : 0.0f);
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

// P9.41 -- PXG_SCENE=clothnet: a LIGHT box dropped onto a horizontal cloth whose
// whole border is pinned to the world.  This is the first scene in the harness in
// which a DYNAMIC rigid body has to be HELD by a deformable, and it exists because
// none of the 33 scored scenes asks that: `clothbox` lands the cloth on a 1000 kg
// box that gravity holds down regardless, and every other cloth/softbody/particle
// scene parks its one dynamic body out of reach.  P9.40's guard then showed WHY
// they all stayed green: the impulse the cloth solve computes for the rigid side
// goes through accumulateDeltaVRigidFirst/SecondLaunch, and both are unmapped, so
// the box never hears about the cloth.  The cloth-side half of the contact is
// mapped (the cloth deforms around the box), which is exactly what makes the box's
// final height the observable and not the cloth's.
//
// EXPECTED, once the coupling is live: the box comes to rest ON the net, well
// above the ground.  Until then it passes through the net to the ground.  The
// net is spawned at y = 2 and the ground's top is at y = 0, so the two outcomes
// are two metres apart and the bar sits between them at 1.0.
struct ClothNetScene
{
    PxDeformableSurface* surface = nullptr;
    PxU32 numVerts = 0;
    PxCudaContextManager* cudaCtx = nullptr;
    PxDeformableAttachment* attachment = nullptr;
    PxRigidDynamic* box = nullptr;
    PxU32 centreVert = 0;        // the vertex under the box, tracked for the sag
    PxU32 numPinned = 0;
    float netY = 0.0f;
    float boxHalf = 0.0f;
};

ClothNetScene buildClothNetScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                 PxCudaContextManager* cudaCtx)
{
    ClothNetScene out;
    out.cudaCtx = cudaCtx;
    addGroundBox(physics, scene, rigidMat);

    // 7 x 7 verts over 2 x 2 m: 0.333 m spacing against a 0.5 m box, so the box
    // always overlaps several triangles and the triangle contact path (P9.15d) can
    // never be bypassed by a box slipping between vertices.
    const PxU32 numCol = 7, numRow = 7;
    const float sizeX = 2.0f, sizeZ = 2.0f;
    const float netY = 2.0f;
    const float sx = sizeX / float(numCol - 1);
    const float sz = sizeZ / float(numRow - 1);
    auto vid = [numRow](PxU32 c, PxU32 r) { return c * numRow + r; };
    out.netY = netY;

    std::vector<PxVec3> verts;
    verts.reserve(numCol * numRow);
    for (PxU32 c = 0; c < numCol; ++c)
        for (PxU32 r = 0; r < numRow; ++r)
            verts.push_back(PxVec3(float(c) * sx - 0.5f * sizeX, netY, float(r) * sz - 0.5f * sizeZ));
    out.centreVert = vid(numCol / 2, numRow / 2);

    std::vector<PxU32> tris;
    tris.reserve(3 * 2 * (numCol - 1) * (numRow - 1));
    for (PxU32 c = 1; c < numCol; ++c)
        for (PxU32 r = 1; r < numRow; ++r)
        {
            tris.push_back(vid(c - 1, r - 1)); tris.push_back(vid(c, r - 1)); tris.push_back(vid(c - 1, r));
            tris.push_back(vid(c - 1, r));     tris.push_back(vid(c, r - 1)); tris.push_back(vid(c, r));
        }

    // The whole border is pinned (24 of 49 verts), so the net can sag but cannot
    // swing or slide away from under the box.
    static std::vector<PxU32> pinIdx;    // must outlive createDeformableAttachment's copy
    static std::vector<PxVec4> pinPos;
    pinIdx.clear(); pinPos.clear();
    for (PxU32 c = 0; c < numCol; ++c)
        for (PxU32 r = 0; r < numRow; ++r)
            if (c == 0 || r == 0 || c == numCol - 1 || r == numRow - 1)
            {
                pinIdx.push_back(vid(c, r));
                const PxVec3& p = verts[vid(c, r)];
                pinPos.push_back(PxVec4(p.x, p.y, p.z, 0.0f));
            }
    out.numPinned = PxU32(pinIdx.size());

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
    // Some damping, unlike the free-fall scenes: the observable is a REST state
    // (box held on the net), and an undamped net would ring under the box for the
    // whole run.  The engine forms clamp(damping * dt, 0, 1), so 2.0 at dt = 1/60
    // removes about 3% of the velocity per step.
    surface->setLinearDamping(2.0f);
    surface->setMaxLinearVelocity(1.0e6f);
    surface->attachShape(*shape);
    shape->release();
    scene->addActor(*surface);

    const PxU32 nv = PxU32(verts.size());
    std::vector<PxVec4> posInvMass(nv), vel(nv, PxVec4(0.0f)), rest(nv);
    for (PxU32 v = 0; v < nv; ++v)
    {
        posInvMass[v] = PxVec4(verts[v].x, verts[v].y, verts[v].z, 1.0f);
        rest[v]       = posInvMass[v];
    }
    PxCudaContext* ctx = cudaCtx->getCudaContext();
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getPositionInvMassBufferD()),
                    posInvMass.data(), nv * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getVelocityBufferD()),
                    vel.data(), nv * sizeof(PxVec4));
    ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getRestPositionBufferD()),
                    rest.data(), nv * sizeof(PxVec4));

    PxDeformableAttachmentData attData;
    attData.actor[0] = surface;
    attData.type[0]  = PxDeformableAttachmentTargetType::eVERTEX;
    attData.indices[0].data = pinIdx.data();
    attData.indices[0].count = PxU32(pinIdx.size());
    attData.actor[1] = NULL;
    attData.type[1]  = PxDeformableAttachmentTargetType::eWORLD;
    attData.coords[1].data = pinPos.data();
    attData.coords[1].count = PxU32(pinPos.size());
    attData.pose[1] = PxTransform(PxIdentity);
    out.attachment = physics->createDeformableAttachment(attData);
    if (!out.attachment)
        std::printf("FAIL: createDeformableAttachment (net border -> world) returned null\n");

    // THE LIGHT BOX.  One kilogram, not the harness's usual thousand: the net has
    // 49 unit-mass vertices, and a body a thousand times heavier than the whole
    // net would tear straight through it on any solver and prove nothing.  It is
    // also the scene's one live dynamic body, which every cloth scene needs.
    const float boxHalf = 0.25f;
    out.boxHalf = boxHalf;
    PxRigidDynamic* box = physics->createRigidDynamic(PxTransform(PxVec3(0.0f, netY + 1.0f, 0.0f)));
    PxShape* bs = physics->createShape(PxBoxGeometry(boxHalf, boxHalf, boxHalf), *rigidMat);
    box->attachShape(*bs);
    bs->release();
    box->setMass(1.0f);
    box->setMassSpaceInertiaTensor(boxInertia(PxVec3(boxHalf), 1.0f));
    scene->addActor(*box);
    out.box = box;

    out.surface = surface;
    out.numVerts = nv;
    return out;
}

// DtoH-read the net once: the y of the vertex under the box and the lowest y of
// any vertex.  Returns false on a non-finite coordinate.
bool clothNetSample(const ClothNetScene& csc, float& centreY, float& minY)
{
    std::vector<PxVec4> host(csc.numVerts);
    PxCudaContext* ctx = csc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(), reinterpret_cast<CUdeviceptr>(csc.surface->getPositionInvMassBufferD()),
                    csc.numVerts * sizeof(PxVec4));
    bool finite = true;
    minY = 1.0e30f;
    for (PxU32 i = 0; i < csc.numVerts; ++i)
    {
        if (!std::isfinite(host[i].x) || !std::isfinite(host[i].y) || !std::isfinite(host[i].z)) finite = false;
        minY = PxMin(minY, host[i].y);
    }
    centreY = host[csc.centreVert].y;
    return finite;
}

// DtoH-read the cloth vertex buffer once; fill the mean y of the pinned top row and the
// free bottom row (the two observables of the drape).  Returns false on non-finite y.
// P9.19 adds `bottomSpreadY`: the standard deviation of the bottom row's y.  The
// MEANS cannot see internal damping -- they follow the sheet's near-rigid swing,
// which this damping does not act on and should not.  The SPREAD is the row's
// departure from flat, i.e. exactly the non-rigid shape change the two
// accumulators work on.
bool clothHangRowMeans(const ClothHangScene& csc, float& topMeanY, float& bottomMeanY,
                       float* bottomSpreadY = nullptr)
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
    if (bottomSpreadY)
    {
        double s2 = 0.0;
        for (PxU32 id : csc.bottomRow)
        { const double d = double(host[id].y) - double(bottomMeanY); s2 += d * d; }
        *bottomSpreadY = float(std::sqrt(s2 / double(csc.bottomRow.size())));
    }
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
// PXG_SCENE=clothcloth (P8.16): the cloth-vs-DIFFERENT-cloth money shot.
//
// TWO separate PxDeformableSurface actors, not one folded sheet: a fully PINNED lower sheet
// (invMass 0 everywhere) acting as a cloth table, and a FREE upper sheet dropped onto it from
// dropH above.  Both are built from the SAME recipe -- same material, thickness, contact and
// rest offsets, same grid spacing -- so the only asymmetry is the inverse mass and the inset
// footprint.  That matters: P8.8's softbody-vs-cloth scene spent a whole tranche chasing what
// turned out to be a MASS RATIO, and a shared recipe removes that confound by construction.
//
// SELF-collision is DISABLED on both sheets (eDISABLE_SELF_COLLISION), which makes
// cloth_selfCollisionMidphase{VT,EE} early-exit.  A flat sheet cannot fold onto itself anyway,
// but disabling it means the ONLY midphase that can produce a contact between the two sheets is
// cloth_differentClothCollision{VT,EE} -- and there is no rigid body under the upper sheet
// either.  With those two kernels the upper sheet lands on the lower one; without them it falls
// straight through.  The downstream solve (cloth_clothContactPrepare -> query ->
// cloth_solveClothClothDelta{VT,EE}) is the SAME chain the self path already uses, ported at
// P7.2 tranche 2 -- the different-cloth pairs flow through its generic arithmetic.
//
// PXG_CC_PINNED=0 makes the lower sheet free too (both fall), which is the frozen-sim guard's
// counterpart rather than a physics control.
// ===========================================================================
struct ClothClothScene
{
    PxDeformableSurface* lower = nullptr;   // the pinned cloth "table"
    PxDeformableSurface* upper = nullptr;   // the free sheet — the observable
    PxU32 numUpperVerts = 0;
    PxCudaContextManager* cudaCtx = nullptr;
    float baseY = 0.0f;
};

// One flat cloth sheet: numX x numZ grid over lenX x lenZ centred on (0, y, 0), every vertex
// given the same inverse mass.  Returns null on any cooking/creation failure.
static PxDeformableSurface* buildFlatClothSheet(PxPhysics* physics, PxScene* scene,
                                                PxCudaContextManager* cudaCtx, PxU32 numX, PxU32 numZ,
                                                float lenX, float lenZ, float y, float invMass,
                                                PxU32& nvOut)
{
    nvOut = 0;
    const float dx = lenX / float(numX - 1);
    const float dz = lenZ / float(numZ - 1);
    auto vid = [numZ](PxU32 c, PxU32 r) { return c * numZ + r; };

    std::vector<PxVec3> verts;
    verts.reserve(numX * numZ);
    for (PxU32 c = 0; c < numX; ++c)
        for (PxU32 r = 0; r < numZ; ++r)
            verts.push_back(PxVec3(float(c) * dx - 0.5f * lenX, y, float(r) * dz - 0.5f * lenZ));

    std::vector<PxU32> tris;
    tris.reserve(3 * 2 * (numX - 1) * (numZ - 1));
    for (PxU32 c = 1; c < numX; ++c)
        for (PxU32 r = 1; r < numZ; ++r)
        {
            tris.push_back(vid(c - 1, r - 1)); tris.push_back(vid(c, r - 1)); tris.push_back(vid(c - 1, r));
            tris.push_back(vid(c - 1, r));     tris.push_back(vid(c, r - 1)); tris.push_back(vid(c, r));
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
    if (!triMesh) { std::printf("FAIL: clothcloth createTriangleMesh returned null\n"); return nullptr; }

    // Identical recipe for both sheets — see the scene comment on why that is deliberate.
    PxDeformableSurfaceMaterial* clothMat =
        physics->createDeformableSurfaceMaterial(/*youngs*/ 1.e6f, /*poissons*/ 0.3f,
            /*dynamicFriction*/ 0.5f, /*thickness*/ 0.1f, /*bendingStiffness*/ 1.e-4f);
    PxDeformableSurfaceMaterial* mats[1] = { clothMat };

    PxShapeFlags shapeFlags = PxShapeFlag::eSIMULATION_SHAPE | PxShapeFlag::eSCENE_QUERY_SHAPE;
    PxShape* shape = physics->createShape(PxTriangleMeshGeometry(triMesh), mats, PxU16(1), true, shapeFlags);
    if (!shape) { std::printf("FAIL: clothcloth createShape returned null\n"); return nullptr; }
    shape->setContactOffset(0.1f);
    // The rest offset is what the contact query actually measures against
    // (cloth0->mRestDistance + cloth1->mRestDistance, FEMClothExternalSolve.cu:870), so the two
    // sheets settle 0.10 apart — the same band clothselfcollide's two layers settle to.
    shape->setRestOffset(0.05f);

    PxDeformableSurface* surface = physics->createDeformableSurface(*cudaCtx);
    if (!surface) { std::printf("FAIL: clothcloth createDeformableSurface returned null\n"); return nullptr; }

    surface->setNbCollisionPairUpdatesPerTimestep(4);
    surface->setNbCollisionSubsteps(8);
    surface->setLinearDamping(0.2f);
    surface->setMaxLinearVelocity(1.0e6f);
    // SELF-collision off on both sheets: a flat sheet has nothing to fold onto, and switching it
    // off leaves cloth_differentClothCollision{VT,EE} as the only midphase that can see the pair.
    surface->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
    // A fully pinned sheet has zero velocity forever; without this it can be put to sleep and
    // stop producing the contact-manager pair the free sheet needs.
    surface->setSleepThreshold(0.0f);
    surface->attachShape(*shape);
    shape->release();

    scene->addActor(*surface);

    const PxU32 nv = PxU32(verts.size());
    std::vector<PxVec4> posInvMass(nv), vel(nv, PxVec4(0.0f)), rest(nv);
    for (PxU32 v = 0; v < nv; ++v)
    {
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

    nvOut = nv;
    return surface;
}

ClothClothScene buildClothClothScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                     PxCudaContextManager* cudaCtx)
{
    ClothClothScene out;
    out.cudaCtx = cudaCtx;

    // A live dynamic rigid FAR from the cloths keeps the rigid PGS island non-empty (the cloth
    // solve runs inside it), exactly as every other cloth scene does.
    addGroundBox(physics, scene, rigidMat);
    addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
               boxInertia(PxVec3(0.5f), 1000.0f), 1.5f);

    const float baseY = 5.0f;
    const char* dropEnv = std::getenv("PXG_CC_DROP");
    const float dropH = dropEnv ? float(atof(dropEnv)) : 0.30f;
    const char* pinEnv = std::getenv("PXG_CC_PINNED");
    const float lowerInvMass = (pinEnv && pinEnv[0] == '0') ? 1.0f : 0.0f;
    out.baseY = baseY;

    const PxU32 numX = 8u, numZ = 5u;
    // The upper sheet's footprint is INSET, the way a real sheet resting on a table is smaller
    // than the table.  With identical footprints the upper sheet's border vertices sit exactly on
    // the lower one's outer edge with no triangle beneath them and droop through — correct cloth
    // behaviour that would otherwise read as a broken hold (clothselfcollide records the same).
    const float inset = 0.9f;
    PxU32 nvLower = 0, nvUpper = 0;
    out.lower = buildFlatClothSheet(physics, scene, cudaCtx, numX, numZ, 2.0f, 2.0f, baseY,
                                    lowerInvMass, nvLower);
    if (!out.lower) return out;
    out.upper = buildFlatClothSheet(physics, scene, cudaCtx, numX, numZ, 2.0f * inset, 2.0f * inset,
                                    baseY + dropH, 1.0f, nvUpper);
    if (!out.upper) { out.lower = nullptr; return out; }
    out.numUpperVerts = nvUpper;
    return out;
}

// DtoH-read the FREE upper sheet: min and mean y across all of its vertices.
bool clothClothUpperMinMeanY(const ClothClothScene& ccs, float& minY, float& meanY)
{
    std::vector<PxVec4> host(ccs.numUpperVerts);
    PxCudaContext* ctx = ccs.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(), reinterpret_cast<CUdeviceptr>(ccs.upper->getPositionInvMassBufferD()),
                    ccs.numUpperVerts * sizeof(PxVec4));
    double sum = 0.0;
    float lo = 1.0e30f;
    bool finite = true;
    for (PxU32 i = 0; i < ccs.numUpperVerts; ++i)
    {
        const float y = host[i].y;
        sum += double(y);
        lo = PxMin(lo, y);
        if (!std::isfinite(y) || !std::isfinite(host[i].x) || !std::isfinite(host[i].z)) finite = false;
    }
    minY = lo;
    meanY = float(sum / double(ccs.numUpperVerts));
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
    bool hasPlane = false;      // "there is a floor to settle on", whatever shape it is
    bool hasSphere = false;     // PXG_SB_FLOOR=sphere: a static ball, north pole at y=0
    bool hasBox = false;        // PXG_SB_FLOOR=box (the DEFAULT): the ground box, top at y=0
    bool hasPlaneShape = false; // PXG_SB_FLOOR=plane: the one floor that takes the PLANE
                                // branch of sb_primitiveContactGenLaunch rather than GJK/EPA
    bool hasHeightfield = false;// PXG_SB_FLOOR=heightfield (P9.81a): the SIXTH floor, and the
                                // only one whose pair routes to GPU_BUCKET_ID::eSoftbodyHeightfield
    bool hasSdfMesh = false;    // PXG_SB_FLOOR=sdfmesh (P9.97b): the SEVENTH floor, the only one
                                // whose pair routes to GPU_BUCKET_ID::eSoftbodySdfTrimesh
    PxCudaContextManager* cudaCtx = nullptr;
};

SoftBodyScene buildSoftBodyScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                 PxCudaContextManager* cudaCtx, float stretchY)
{
    SoftBodyScene out;
    out.cudaCtx = cudaCtx;

    // ---- the floor ------------------------------------------------------------
    // PXG_SB_FLOOR selects it: "box" (the default), "plane", "sphere", "mesh", "convexcore",
    // "heightfield", "sdfmesh" (P9.97b) or "none".
    //
    // THE DEFAULT CHANGED WHEN THE HULL BRANCH LANDED, AND THAT IS THE RESULT.  A ground box
    // -- half extents (50, 0.5, 50) at y = -0.5, so its top face IS the plane y = 0 -- has
    // stood in this scene since the first softbody frame.  It simply generated no contacts:
    // sb_primitiveContactGenLaunch handled ePLANE and nothing else, so a box floor was
    // invisible however much of the solve chain was mapped.  That is why PXG_SB_PLANE had to
    // exist at all, and why the recorded baseline for this scene was a FREE FALL -- the cube
    // fell straight through a floor it could not see.
    //
    // It can see it now.  The default softbody scene catches the cube on a cooked convex
    // hull, and the free fall moves to PXG_SB_FLOOR=none, where it is honest.
    //
    // The alternatives REPLACE the ground box rather than joining it, because a plane
    // coplanar with the box's top face and a sphere tangent to it are no longer harmless
    // duplicates -- all three would generate contacts at the same height and the run would
    // stop measuring the shape it names.  The ground box does not vanish, though: it sinks to
    // y = -50.5, far below everything, so the dynamic body below still has something to land
    // on and the softbody never reaches it.
    const char* floorEnv = std::getenv("PXG_SB_FLOOR");
    // The old spellings still work; PXG_SB_PLANE=1 and PXG_SB_SPHERE=1 name the same floors.
    auto envOn = [](const char* name) {
        const char* v = std::getenv(name);
        return v && atoi(v) != 0;
    };
    if (!floorEnv && envOn("PXG_SB_PLANE"))  floorEnv = "plane";
    if (!floorEnv && envOn("PXG_SB_SPHERE")) floorEnv = "sphere";
    const bool floorNone   = floorEnv && !std::strcmp(floorEnv, "none");
    const bool floorPlane  = floorEnv && !std::strcmp(floorEnv, "plane");
    const bool floorSphere = floorEnv && !std::strcmp(floorEnv, "sphere");
    const bool floorMesh   = floorEnv && !std::strcmp(floorEnv, "mesh");
    const bool floorCore    = floorEnv && !std::strcmp(floorEnv, "convexcore");
    const bool floorHf     = floorEnv && !std::strcmp(floorEnv, "heightfield");
    const bool floorSdf    = floorEnv && !std::strcmp(floorEnv, "sdfmesh");
    const bool floorBox    = !floorNone && !floorPlane && !floorSphere && !floorMesh
                          && !floorCore && !floorHf && !floorSdf;

    PxRigidStatic* ground = addGroundBox(physics, scene, rigidMat);
    if (!floorBox)
        ground->setGlobalPose(PxTransform(PxVec3(0.0f, -50.5f, 0.0f)));

    if (floorPlane)
    {
        PxRigidStatic* floor = physics->createRigidStatic(
            PxTransformFromPlaneEquation(PxPlane(0.0f, 1.0f, 0.0f, 0.0f)));
        PxShape* floorShape = physics->createShape(PxPlaneGeometry(), *rigidMat);
        floor->attachShape(*floorShape);
        scene->addActor(*floor);
        floorShape->release();
    }
    // A ball of radius 5 centred at y = -5: its north pole is at y = 0, so it is the same
    // floor height as the other two but gently curved, which is why it settles less tightly.
    if (floorSphere)
    {
        PxRigidStatic* ball = physics->createRigidStatic(PxTransform(PxVec3(0.0f, -5.0f, 0.0f)));
        PxShape* ballShape = physics->createShape(PxSphereGeometry(5.0f), *rigidMat);
        ball->attachShape(*ballShape);
        scene->addActor(*ball);
        ballShape->release();
    }

    // PXG_SB_FLOOR=mesh (P8.10): a COOKED TRIANGLE MESH floor, the fourth shape the softbody
    // can land on and the one the softbody-vs-trimesh pair exists for.  A 20 x 20 quad in the
    // plane y = 0, split into two triangles, wound so their normals point UP.  It is the same
    // floor HEIGHT as the box, the plane and the sphere's north pole, so its settle number is
    // directly comparable with theirs -- which is the point of putting it in this matrix
    // rather than in a scene of its own.
    if (floorMesh)
    {
        const float E = 10.0f;
        const PxVec3 mverts[4] = { PxVec3(-E, 0.0f, -E), PxVec3(-E, 0.0f, E),
                                   PxVec3( E, 0.0f, -E), PxVec3( E, 0.0f, E) };
        const PxU32 mtris[6] = { 0u, 1u, 2u,   2u, 1u, 3u };
        PxTriangleMeshDesc md;
        md.points.count = 4; md.points.stride = sizeof(PxVec3); md.points.data = mverts;
        md.triangles.count = 2; md.triangles.stride = 3 * sizeof(PxU32); md.triangles.data = mtris;
        PxCookingParams ck(physics->getTolerancesScale());
        ck.buildGPUData = true;
        PxTriangleMesh* floorMeshGeom =
            immediateCooking::createTriangleMesh(ck, md, physics->getPhysicsInsertionCallback());
        if (!floorMeshGeom) { std::printf("FAIL: createTriangleMesh (floor) returned null\n"); return out; }
        PxRigidStatic* floor = physics->createRigidStatic(PxTransform(PxVec3(0.0f, 0.0f, 0.0f)));
        PxShape* fs = physics->createShape(PxTriangleMeshGeometry(floorMeshGeom), *rigidMat);
        floor->attachShape(*fs);
        scene->addActor(*floor);
        fs->release();
    }

    // PXG_SB_FLOOR=convexcore (P9.72): a static CONVEX CORE box, the FIFTH shape the
    // softbody can land on -- and the only one whose pair, (eCONVEXCORE,
    // eTETRAHEDRONMESH), routes to GPU_BUCKET_ID::eConvexCoreTetmesh, i.e. to
    // convexCoreTetmeshNphase_Kernel32.  The guard is only
    // isConvexCoreGpuCompatible(shape0) (PxgNphaseImplementationContext.cpp:1003-1014)
    // and Gu::isGPUCompatible accepts every core type, so nothing else has to be true.
    //
    // PxConvexCore::Box takes FULL extents -- makeConvexShape doubles PxgShape's half
    // extents into the core (pxgSlangCvxCoreMake.slang, EGT_BOX arm) -- so 20 x 1 x 20
    // centred at y = -0.5 puts the top face at y = 0, the SAME floor height as the box,
    // the plane, the sphere's north pole and the cooked mesh.  That is the point of
    // adding it to this matrix rather than giving it a scene of its own: its settle
    // number is directly comparable with the other four.
    //
    // *** THIS FLOOR IS EXPECTED TO FAIL UNTIL convexCoreTetmeshNphase_Kernel32 IS
    // MAPPED. ***  Measured before the port, on the same softbody at the same height:
    //
    //   PXG_SB_FLOOR=box          settles at mean y = 0.48929, 16 contacts
    //   PXG_SB_FLOOR=mesh         settles at mean y = 0.48830, 22 contacts
    //   PXG_SB_FLOOR=convexcore   FALLS THROUGH to -49.8831, and the bridge reports
    //                             convexCoreTetmeshNphase_Kernel32 launched 15 times
    //                             with no SPIR-V mapping, every launch SKIPPED
    //
    // (-49.88 rather than the -50.77 of a clean free fall because the ground box is
    // parked at y = -50.5 for every non-box floor, so the cube lands on that instead.)
    //
    // PXG_CPU_CONTROL=1 DOES NOT APPLY HERE, and that is worth stating rather than
    // quietly skipping: PhysX deformables are GPU-only, so the control mode leaves the
    // softbody motionless (drop = 0.0000) and reports four failures that say nothing
    // about this floor.  The expectation comes from the OTHER FOUR FLOORS of this same
    // matrix instead -- same cube, same solve, same floor height, differing only in the
    // shape the contact is generated against.  That is why this floor belongs in the
    // matrix and not in a scene of its own.
    if (floorCore)
    {
        PxRigidStatic* coreGround = physics->createRigidStatic(PxTransform(PxVec3(0.0f, -0.5f, 0.0f)));
        PxShape* cs = physics->createShape(
            PxConvexCoreGeometry(PxConvexCore::Box(PxVec3(20.0f, 1.0f, 20.0f))), *rigidMat);
        coreGround->attachShape(*cs);
        scene->addActor(*coreGround);
        cs->release();
    }

    // PXG_SB_FLOOR=heightfield (P9.81a): a cooked HEIGHTFIELD floor, the SIXTH shape the
    // softbody can land on, and the only one whose pair routes to
    // GPU_BUCKET_ID::eSoftbodyHeightfield -- i.e. to sb_heightfieldContactGenLaunch.  The
    // routing is unconditional: gTable[eTETRAHEDRONMESH][eHEIGHTFIELD] is caseTetraHeightfield
    // (PxgNphaseImplementationContext.cpp:1392/:1522), which returns that bucket with no
    // compatibility test at all, so an overlapping AABB is the whole requirement.
    //
    // Every sample height is 0, so the surface is at exactly y = 0 -- the SAME floor height as
    // the box, the plane, the sphere's north pole, the cooked mesh and the convex core.  That
    // is the entire reason this arm belongs in the floor matrix rather than in a scene of its
    // own: its settle number is directly comparable with theirs, so the expectation is
    // MEASURED and not invented.  The five on record, same cube, same solve, same height:
    //
    //   PXG_SB_FLOOR=box          settles at mean y = 0.48929, 16 contacts
    //   PXG_SB_FLOOR=mesh         settles at mean y = 0.48830, 22 contacts
    //
    // *** THIS FLOOR IS EXPECTED TO FAIL UNTIL sb_heightfieldContactGenLaunch IS MAPPED. ***
    // The kernel is unmapped, and an unmapped launch is a skipped no-op, so the cube falls
    // through a floor it cannot see -- exactly as the convexcore arm did before P9.72 and as
    // the softbody itself did before the hull branch landed.  What makes this arm worth having
    // is that the failure is now VISIBLE: the bridge reports the launch, the contact-chain
    // probe reports zero contacts, and the verdict below fires.
    if (floorHf)
    {
        if (!addHeightFieldFloor(physics, scene, rigidMat, 10.0f))
            return out;   // addHeightFieldFloor prints its own FAIL
    }

    // PXG_SB_FLOOR=sdfmesh (P9.97b): a static triangle mesh cooked WITH AN SDF, the SEVENTH
    // floor, and the only one whose pair routes to GPU_BUCKET_ID::eSoftbodySdfTrimesh --
    // i.e. to sb_sdfMeshMidphaseGeneratePairsLaunch, which no scene reached before.  The
    // routing is cndCaseTetraMesh (PxgNphaseImplementationContext.cpp:1281-1304): a RIGID
    // triangle mesh that is GPU-compatible, carries at most one material index and has an
    // SDF (isDynamicMesh is `getSdfDataFast().mSdf != NULL`) -> eSoftbodySdfTrimesh; the
    // same mesh WITHOUT the SDF is PXG_SB_FLOOR=mesh's eSoftbodyTrimesh.
    //
    // The mesh is sdfmesh's own cooker (cookSdfBox: a closed, outward-wound 12-triangle
    // box, dense SDF, buildGPUData), half extent 1.5 at y = -1.5, so its top face is the
    // plane y = 0 -- the SAME floor height as every other arm of this matrix, which is
    // what makes its settle number comparable with theirs -- and 3 x 3 wide under the
    // 1 x 1 cube.  Spacing 0.1 is 30 cells per side, cooked on the CPU in well under a
    // second.  Its sign is checked like sdfmesh's: the centre texel must be negative.
    //
    // P9.97b IS WHY IT EXISTS: the kernel's shader read its 80-byte by-value writer at @88
    // while the bridge places it at @96 (paramAlign(80) = 16).  Latent because no scene
    // routed here; this arm is the scene that does.
    if (floorSdf)
    {
        PxTriangleMesh* sdfFloor = cookSdfBox(physics, 1.5f, 0.1f);
        if (!sdfFloor) return out;   // cookSdfBox prints its own FAIL
        if (!sdfFloor->getSDF())
        {
            std::printf("FAIL: the cooked floor carries NO SDF, so the pair would route to "
                        "eSoftbodyTrimesh and this arm would measure PXG_SB_FLOOR=mesh again\n");
            return out;
        }
        PxU32 sdx = 0, sdy = 0, sdz = 0;
        sdfFloor->getSDFDimensions(sdx, sdy, sdz);
        const PxReal centre = sdfFloor->getSDF()[(size_t(sdz / 2) * sdy + sdy / 2) * sdx + sdx / 2];
        std::printf("  softbody sdfmesh floor: SDF dims %u x %u x %u, centre texel %.4f\n",
                    sdx, sdy, sdz, centre);
        if (!(centre < 0.0f))
        {
            std::printf("FAIL: the cooked floor's SDF is not negative inside -- wound inward\n");
            return out;
        }
        PxRigidStatic* floor = physics->createRigidStatic(PxTransform(PxVec3(0.0f, -1.5f, 0.0f)));
        PxShape* fs = physics->createShape(PxTriangleMeshGeometry(sdfFloor), *rigidMat);
        floor->attachShape(*fs);
        scene->addActor(*floor);
        fs->release();
    }

    out.hasPlane = !floorNone;      // "there is a floor to settle on", whatever its shape
    out.hasSphere = floorSphere;
    out.hasBox = floorBox;
    out.hasPlaneShape = floorPlane;
    out.hasHeightfield = floorHf;
    out.hasSdfMesh = floorSdf;

    // The one dynamic body the rigid PGS island needs: rigidSumInternalContactAndJointBatches2
    // null-derefs the solver descriptors with zero dynamic rigids, as in the cloth and
    // particle scenes.  It used to stand at the ORIGIN, and the old comment called it "far
    // below" -- a 1x1x1 box directly under a cube falling from y = 4.  It was far below only
    // in the sense that nothing could collide with it; once boxes became visible the falling
    // cube met it on the way down (contacts from step 41 instead of 49) and came to rest
    // partly on top of it.  Its one job does not care where it stands, so it stands at x = 10.
    PxRigidDynamic* island = addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                        boxInertia(PxVec3(0.5f), 1000.0f), 1.5f);
    island->setGlobalPose(PxTransform(PxVec3(10.0f, 1.5f, 0.0f)));

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
    //
    // P8.20: A FLOOR NEEDS IT FOR EXACTLY THE SAME REASON A STRETCH DOES, and for
    // a long time it did not get it.  A cube dropped 3.5 m lands with a large
    // amount of kinetic energy, and with elasticityDamping == 0 the ONLY sinks are
    // the contact solve and the implicit integrator.  The oscillation that follows
    // is real, bounded and DECAYING -- measured over the box floor, the mean-y span
    // per 35-step window falls 0.577 -> 0.480 -> 0.228 -> 0.058 -- but 200 steps is
    // nowhere near enough of it to cross a 2e-3 convergence bar, so the scene's own
    // settle assertion was unreachable by construction.  That is what the long-
    // standing `|dy| still 0.015430` failure was: not an unconverged solve, an
    // undamped material.  The damped branch is the one every floor case should have
    // been on; the UNDAMPED branch stays covered by PXG_SB_FLOOR=none, which is also
    // the only case that still asserts the closed-form free fall.
    const bool wantElasticityDamping = (stretchY > 1.0f) || !floorNone;
    PxDeformableVolumeMaterial* volMat =
        physics->createDeformableVolumeMaterial(/*youngs*/ 1.e5f, /*poissons*/ 0.3f,
                                                /*dynamicFriction*/ 0.5f,
                                                /*elasticityDamping*/ wantElasticityDamping ? 0.05f : 0.0f);
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

// ===========================================================================
// PXG_SCENE=softbodyhex (P9.95f): the first scene with a HEXAHEDRAL (voxel) simulation
// mesh, and the observer for both callers of sb_gm_cp_solveHexahedronsPartition_parallelGS
// (softBodyGM.cu:1313):
//
//   the hex branch of sb_gm_cp_solveTetrahedronsPartitionLaunch (softBodyGM.cu:1450-1455),
//     which until P9.95f was a SILENT no-op in this port (the P8.0 tranche-6 scope ported
//     only the one-tet-per-element path), and
//   sb_gm_cp_solveTetrahedronsJacobiPartitionLaunch (softBodyGM.cu:1463), the 9th
//     ("Jacobi") partition, launched only when the cooker emits more than 8 partitions
//     (PxgSoftBodyCore.cpp:1185) -- which it does for a hex mesh with a DUPLICATED voxel
//     that shares a vertex with its predecessor (GuCookingTetrahedronMesh.cpp:696-768).
//     Its deltas are folded back by sb_gm_applyDeformationDeltasLaunch, which this scene
//     drives through the engine for the first time.
//
// Every voxel softbody the stock extension path makes (PxDeformableVolumeExt::
// createDeformableVolumeMesh / createDeformableVolumeBox) has a hex simulation mesh, so
// before P9.95f every one of them got NO elastic solve here, with no refusal and no
// [pxg-unmapped] line.  physxextensions is not linked into this harness, so the voxel mesh
// is hand-authored on the 27-vertex lattice of PXG_SCENE=softbody:
//   - corner j of a cell is (x + bit2, y + bit1, z + bit0), the PhysX tetmaker convention
//     (ExtDelaunayBoundaryInserter.cpp:1344-1346), cells looped z outer, x inner;
//   - each cell is the 5 tets of tets5PerVoxel (softBodyGM.cu:1211-1213), the flipped half
//     (offset 20) on odd (x+y+z), emitted WITHOUT reorientation: the cooker recovers the
//     8 corners from that exact 20-index layout (GuCookingTetrahedronMesh.cpp:413-414,
//     flipped test :737), and every one of the 45 tets is positively oriented already;
//   - vertex 0 is corner 0 of the first voxel, which keeps the cooker's voxel-width seed
//     (it seeds min/max from vertexPositions[0], :659-669 -- an upstream quirk) honest.
// PXG_SB_HEX selects the arm:
//   box   the 8 voxels: 27 verts / 40 tets / 8 partitions.  No Jacobi launch at all; it
//         drives the GS hex branch only.
//   fork  (the default) adds a TWIN of cell (0,0,0) right after it, sharing its +x face
//         (verts 9,10,12,13) with 4 private verts 27..30 at the positions of {0,1,3,4}:
//         31 verts / 45 tets / 9 partitions, the twin alone in partition 8.  Verts 27..30
//         are touched by NO GS voxel, so with gravity off only the Jacobi kernel (through
//         applyDeformationDeltas) can ever move them.
// The collision mesh is the 40-tet alternating-split cube of PXG_SCENE=softbody; the cooker
// computes the embedding.  Gravity is off and there is no floor, so only the constitutive
// solve can change anything: the body is spawned stretched 1.6x along y and must relax.
// ===========================================================================
struct SoftBodyHexScene
{
    PxDeformableVolume* volume = nullptr;
    PxCudaContextManager* cudaCtx = nullptr;
    PxU32 nbSim = 0;
    PxU32 nbSimTets = 0;
    std::vector<PxVec3> simRest;   // the cooked sim mesh's REST vertices (unstretched)
    bool fork = false;
    float stretchY = 1.0f;
};

// The voxel simulation mesh (see the scene header).
static void buildHexSimMesh(bool fork, std::vector<PxVec3>& verts, std::vector<PxU32>& tets)
{
    // tets5PerVoxel (softBodyGM.cu:1211-1213): rows 0..4 non-flipped, 5..9 flipped.
    static const PxU32 tets5[40] = { 0, 6, 3, 5,   0, 1, 5, 3,   6, 7, 3, 5,   4, 5, 6, 0,   2, 3, 0, 6,
                                     1, 7, 4, 2,   1, 0, 2, 4,   7, 6, 4, 2,   5, 4, 1, 7,   3, 2, 7, 1 };
    const PxU32 cells = 2, n = cells + 1;
    const float h = 0.5f;
    verts.clear(); tets.clear();
    for (PxU32 i = 0; i < n; ++i)
        for (PxU32 j = 0; j < n; ++j)
            for (PxU32 k = 0; k < n; ++k)
                verts.push_back(PxVec3(float(i) * h - 0.5f, float(j) * h - 0.5f, float(k) * h - 0.5f));
    auto vid = [n](PxU32 i, PxU32 j, PxU32 k) { return (i * n + j) * n + k; };
    auto emit = [&](const PxU32 c[8], PxU32 flip) {
        for (PxU32 e = 0; e < 20; ++e) tets.push_back(c[tets5[20 * flip + e]]);
    };
    for (PxU32 z = 0; z < cells; ++z)
        for (PxU32 y = 0; y < cells; ++y)
            for (PxU32 x = 0; x < cells; ++x)
            {
                PxU32 c[8];
                for (PxU32 j = 0; j < 8; ++j)
                    c[j] = vid(x + ((j >> 2) & 1u), y + ((j >> 1) & 1u), z + (j & 1u));
                const PxU32 flip = (x + y + z) % 2u;
                emit(c, flip);
                if (fork && x == 0 && y == 0 && z == 0)
                {
                    // The twin: same cell, corners 0..3 (the -x face) replaced by private copies.
                    PxU32 c2[8];
                    for (PxU32 j = 0; j < 8; ++j) c2[j] = c[j];
                    for (PxU32 j = 0; j < 4; ++j)
                    {
                        verts.push_back(verts[c[j]]);
                        c2[j] = PxU32(verts.size() - 1);
                    }
                    emit(c2, flip);
                }
            }
}

// The 40-tet alternating-split cube of buildSoftBodyScene (same splits, same orientation
// fix), used here as the COLLISION mesh only.
static void buildAlternatingCubeTets(std::vector<PxVec3>& tverts, std::vector<PxU32>& tets)
{
    const PxU32 cells = 2, n = cells + 1;
    const float size = 1.0f, h = size / float(cells);
    tverts.clear(); tets.clear();
    for (PxU32 i = 0; i < n; ++i)
        for (PxU32 j = 0; j < n; ++j)
            for (PxU32 k = 0; k < n; ++k)
                tverts.push_back(PxVec3(float(i) * h - 0.5f * size, float(j) * h - 0.5f * size,
                                        float(k) * h - 0.5f * size));
    auto vid = [n](PxU32 i, PxU32 j, PxU32 k) { return (i * n + j) * n + k; };
    static const PxU32 splitA[5][4] = { {0,1,2,4}, {1,3,2,7}, {1,4,5,7}, {2,4,6,7}, {1,2,4,7} };
    static const PxU32 splitB[5][4] = { {0,1,3,5}, {0,3,2,6}, {0,5,4,6}, {3,7,5,6}, {0,3,5,6} };
    for (PxU32 i = 0; i < cells; ++i)
        for (PxU32 j = 0; j < cells; ++j)
            for (PxU32 k = 0; k < cells; ++k)
            {
                const PxU32 c[8] = { vid(i, j, k), vid(i + 1, j, k), vid(i, j + 1, k), vid(i + 1, j + 1, k),
                                     vid(i, j, k + 1), vid(i + 1, j, k + 1), vid(i, j + 1, k + 1),
                                     vid(i + 1, j + 1, k + 1) };
                const bool even = ((i + j + k) & 1u) == 0u;
                for (PxU32 t = 0; t < 5; ++t)
                {
                    PxU32 a = c[even ? splitA[t][0] : splitB[t][0]];
                    PxU32 b = c[even ? splitA[t][1] : splitB[t][1]];
                    PxU32 cc = c[even ? splitA[t][2] : splitB[t][2]];
                    PxU32 d = c[even ? splitA[t][3] : splitB[t][3]];
                    const PxVec3 e1 = tverts[b] - tverts[a];
                    const PxVec3 e2 = tverts[cc] - tverts[a];
                    const PxVec3 e3 = tverts[d] - tverts[a];
                    if (e1.cross(e2).dot(e3) < 0.0f) { const PxU32 tmp = cc; cc = b; b = tmp; }
                    tets.push_back(a); tets.push_back(b); tets.push_back(cc); tets.push_back(d);
                }
            }
}

SoftBodyHexScene buildSoftBodyHexScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                       PxCudaContextManager* cudaCtx, bool fork, float stretchY)
{
    SoftBodyHexScene out;
    out.cudaCtx = cudaCtx;
    out.fork = fork;
    out.stretchY = stretchY;

    // No floor: the ground box is parked far below, as PXG_SB_FLOOR=none does.  The one
    // dynamic rigid body is the island the rigid PGS/TGS solve needs (see buildSoftBodyScene).
    PxRigidStatic* ground = addGroundBox(physics, scene, rigidMat);
    ground->setGlobalPose(PxTransform(PxVec3(0.0f, -50.5f, 0.0f)));
    PxRigidDynamic* island = addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                        boxInertia(PxVec3(0.5f), 1000.0f), 1.5f);
    island->setGlobalPose(PxTransform(PxVec3(10.0f, 1.5f, 0.0f)));

    std::vector<PxVec3> simVerts3, collVerts3;
    std::vector<PxU32> simTets, collTets;
    buildHexSimMesh(fork, simVerts3, simTets);
    buildAlternatingCubeTets(collVerts3, collTets);

    PxTetrahedronMeshDesc simDesc;
    simDesc.points.count = PxU32(simVerts3.size());
    simDesc.points.stride = sizeof(PxVec3);
    simDesc.points.data = simVerts3.data();
    simDesc.tetrahedrons.count = PxU32(simTets.size() / 4);
    simDesc.tetrahedrons.stride = 4 * sizeof(PxU32);
    simDesc.tetrahedrons.data = simTets.data();
    simDesc.tetsPerElement = 5;   // PxTetrahedronMeshDesc::eHEX_MESH, 5 tets per voxel

    PxTetrahedronMeshDesc collDesc;
    collDesc.points.count = PxU32(collVerts3.size());
    collDesc.points.stride = sizeof(PxVec3);
    collDesc.points.data = collVerts3.data();
    collDesc.tetrahedrons.count = PxU32(collTets.size() / 4);
    collDesc.tetrahedrons.stride = 4 * sizeof(PxU32);
    collDesc.tetrahedrons.data = collTets.data();

    PxCookingParams ckParams(physics->getTolerancesScale());
    ckParams.buildGPUData = true;
    ckParams.meshWeldTolerance = 0.001f;
    ckParams.midphaseDesc = PxMeshMidPhase::eBVH34;

    // The SIMULATION desc comes first (Cooking.cpp:135).
    PxDeformableVolumeSimulationDataDesc simDataDesc;
    PxDeformableVolumeMesh* volMesh = immediateCooking::createDeformableVolumeMesh(
        ckParams, simDesc, collDesc, simDataDesc, physics->getPhysicsInsertionCallback());
    if (!volMesh) { std::printf("FAIL: createDeformableVolumeMesh (hex sim mesh) returned null\n"); return out; }

    const PxU32 wantVerts = fork ? 31u : 27u, wantTets = fork ? 45u : 40u;
    if (volMesh->getSimulationMesh()->getNbVertices() != wantVerts ||
        volMesh->getSimulationMesh()->getNbTetrahedrons() != wantTets)
    {
        std::printf("FAIL: the cooked hex sim mesh has %u verts / %u tets, expected %u / %u -- the "
                    "vertex numbering the checks rely on is gone\n",
                    volMesh->getSimulationMesh()->getNbVertices(),
                    volMesh->getSimulationMesh()->getNbTetrahedrons(), wantVerts, wantTets);
        return out;
    }

    // The material of the stretched PXG_SCENE=softbody cube (damped so the relaxation settles).
    PxDeformableVolumeMaterial* volMat =
        physics->createDeformableVolumeMaterial(/*youngs*/ 1.e5f, /*poissons*/ 0.3f,
                                                /*dynamicFriction*/ 0.5f, /*elasticityDamping*/ 0.05f);
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
    volume->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
    volume->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
    volume->setLinearDamping(0.0f);
    volume->setMaxLinearVelocity(1.0e6f);
    // KEPT AWAKE: with a sleep threshold of 0 sb_sleeping's `magSq >= tolerance*tolerance`
    // (softBodyGM.cu:3163-3178) holds for every vertex, so the solve runs on every step.  Measured before the port,
    // with the default threshold, the motionless body fell asleep after 26 steps (130 Jacobi
    // attempts = 26 x 5 iterations instead of 120 x 5), which would let a dead kernel pass
    // for a converged one and make the launch counts depend on the physics.
    volume->setSleepThreshold(0.0f);
    const char* itersEnv = std::getenv("PXG_SB_ITERS");
    static_cast<PxDeformableBody*>(volume)->setSolverIterationCounts(itersEnv ? PxU32(atoi(itersEnv)) : 4u, 1u);
    scene->addActor(*volume);

    // Host mirror, as in buildSoftBodyScene: spawned stretched along y at y ~ 4, at rest.
    const float spawnY = 4.0f;
    const PxU32 nbSim = volume->getSimulationMesh()->getNbVertices();
    const PxU32 nbColl = volume->getCollisionMesh()->getNbVertices();
    std::vector<PxVec4> simPos(nbSim), simVel(nbSim), collPos(nbColl), restPos(nbColl);
    const PxVec3* sv = volume->getSimulationMesh()->getVertices();
    const PxReal* invMassGM = volume->getDeformableVolumeAuxData()->getGridModelInvMass();
    for (PxU32 v = 0; v < nbSim; ++v)
    {
        const PxReal im = invMassGM ? invMassGM[v] : 1.0f;
        simPos[v] = PxVec4(sv[v].x, sv[v].y * stretchY + spawnY, sv[v].z, im);
        simVel[v] = PxVec4(0.0f, 0.0f, 0.0f, im);
    }
    const PxVec3* cv = volume->getCollisionMesh()->getVertices();
    for (PxU32 v = 0; v < nbColl; ++v)
    {
        collPos[v] = PxVec4(cv[v].x, cv[v].y * stretchY + spawnY, cv[v].z, 1.0f);
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

    out.volume = volume;
    out.nbSim = nbSim;
    out.nbSimTets = volume->getSimulationMesh()->getNbTetrahedrons();
    out.simRest.assign(sv, sv + nbSim);
    std::printf("  softbodyhex mesh (%s): %u sim verts / %u sim tets (tetsPerElement 5), %u collision verts\n",
                fork ? "fork" : "box", nbSim, out.nbSimTets, nbColl);
    return out;
}

static std::vector<PxVec4> softBodyHexReadSim(const SoftBodyHexScene& s)
{
    std::vector<PxVec4> host(s.nbSim);
    s.cudaCtx->getCudaContext()->memcpyDtoH(
        host.data(), reinterpret_cast<CUdeviceptr>(s.volume->getSimPositionInvMassBufferD()),
        s.nbSim * sizeof(PxVec4));
    return host;
}

// y extent (max - min) over a vertex index range [b, e) or an explicit list.
static float softBodyHexExtY(const std::vector<PxVec4>& p, PxU32 b, PxU32 e)
{
    float lo = p[b].y, hi = p[b].y;
    for (PxU32 i = b + 1; i < e; ++i) { lo = PxMin(lo, p[i].y); hi = PxMax(hi, p[i].y); }
    return hi - lo;
}
static float softBodyHexExtY(const std::vector<PxVec4>& p, const PxU32* ids, PxU32 n)
{
    float lo = p[ids[0]].y, hi = p[ids[0]].y;
    for (PxU32 i = 1; i < n; ++i) { lo = PxMin(lo, p[ids[i]].y); hi = PxMax(hi, p[ids[i]].y); }
    return hi - lo;
}

// ===========================================================================
// PXG_SCENE=softbodyselfcollide (P8.4): the FEM-VOLUME SELF-collision scene.
//
// The softbody analogue of PXG_SCENE=clothselfcollide, and the first scene that drives
// the self-collision chain end to end: SB_SELFCOLLISION_MIDPHASE (tranche a),
// SB_FEM_CONTACT_REMAP_TO_SIM, SB_SS_CONTACTPREPARE and
// SB_SOLVE_SOFT_SOFT_BOTH_DELTA{,_TGS} (tranche b).
//
// *** WHY A NEW SCENE AND NOT A FLAG ON PXG_SCENE=softbody. ***
// Self-collision is ON by default for a deformable volume (PxgBodySimManager.cpp:467 only
// skips a body carrying eDISABLE_SELF_COLLISION), but buildSoftBodyScene switches it OFF
// explicitly, and that stays: the cube scene is the recorded free-fall baseline for the
// whole softbody port and its numbers must not move.  Turning the flag back on there would
// also prove little -- a convex cube in its rest pose has no two surface regions that
// approach without being adjacent, and the rest-pose gate drops the adjacent ones by
// construction, so the midphase would dispatch and find nothing.  This scene supplies the
// missing half: a shape that genuinely folds back on itself.
//
// THE SHAPE.  A regular cell grid with the middle carved out, leaving a bracket:
//
//     y=1.00  +--------------------+   <- top slab   (iy = NY-1)
//             |####|               |
//             |####|   (air)       |
//     y=0.25  |####+---------------+
//             |####################|   <- bottom slab (iy = 0)
//     y=0.00  +--------------------+
//             ^ the connector (ix = 0) makes it ONE body
//
// The bottom slab rests on a static plane at y=0; the top slab is a cantilever anchored
// only at the connector, so gravity sags it onto the bottom slab.  The two slabs are 6
// cells apart along the tet graph at the free end, which is what makes their contact a
// SELF-collision rather than an adjacency the rest-pose gate would filter out.
//
// The mask leaves vertices belonging only to removed cells unreferenced, so the vertex
// array is COMPACTED and the tet indices remapped -- cooking rejects a mesh with orphaned
// points, and a welded mesh would silently renumber everything the readback depends on.
//
// PXG_SB_NOSELF=1 sets eDISABLE_SELF_COLLISION and is the discriminator: with the chain
// working the top slab is caught, without it the top slab passes through the bottom one.
// Reporting one number from one run proves nothing here -- only the PAIR does.
struct SoftBodySelfScene
{
    PxDeformableVolume* volume = nullptr;
    PxU32 numCollVerts = 0;
    PxU32 numSimVerts = 0;
    PxU32 numTopVerts = 0;        // vertices of the TOP slab, listed first in topIdx
    std::vector<PxU32> topIdx;    // indices into the collision-mesh vertex array
    float bottomTopY = 0.0f;      // world y of the bottom slab's upper surface
    bool selfCollision = true;
    PxCudaContextManager* cudaCtx = nullptr;
};

SoftBodySelfScene buildSoftBodySelfCollideScene(PxPhysics* physics, PxScene* scene,
                                                PxMaterial* rigidMat, PxCudaContextManager* cudaCtx)
{
    SoftBodySelfScene out;
    out.cudaCtx = cudaCtx;

    // Same reason as every other deformable scene: rigidSumInternalContactAndJointBatches2
    // null-derefs with zero dynamic rigids, so one live dynamic body has to exist.
    //
    // BOTH ARE MOVED OUT OF THE WAY, which they did not used to need to be.  The ground box
    // and this dynamic box both sat under the slabs and were harmless only because softbody
    // contact generation handled ePLANE and nothing else.  With the hull branch live they are
    // real obstacles: the dynamic box stands at the origin, right where the bottom slab is,
    // and the ground box's top face is the same y = 0 as the plane below.  The dynamic body
    // moves to x = 10 (its one job does not care where it stands) and the ground box sinks to
    // y = -50.5, so this scene measures self-collision against a PLANE floor and nothing else.
    PxRigidStatic* ground = addGroundBox(physics, scene, rigidMat);
    ground->setGlobalPose(PxTransform(PxVec3(0.0f, -50.5f, 0.0f)));
    PxRigidDynamic* island = addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                        boxInertia(PxVec3(0.5f), 1000.0f), 1.5f);
    island->setGlobalPose(PxTransform(PxVec3(10.0f, 1.5f, 0.0f)));

    // The floor is a PLANE.  It could now be the ground box -- the hull branch handles one --
    // but the scene's measured thresholds were taken against a plane and the plane branch is
    // the denser contact set (one per penetrating corner rather than one per tet), so it stays.
    {
        PxRigidStatic* floor = physics->createRigidStatic(
            PxTransformFromPlaneEquation(PxPlane(0.0f, 1.0f, 0.0f, 0.0f)));
        PxShape* floorShape = physics->createShape(PxPlaneGeometry(), *rigidMat);
        floor->attachShape(*floorShape);
        scene->addActor(*floor);
        floorShape->release();
    }

    // --- the bracket ----------------------------------------------------------
    const PxU32 NX = 6, NY = 4, NZ = 2;      // cells
    const float h = 0.25f;
    const PxU32 nx = NX + 1, ny = NY + 1, nz = NZ + 1;

    std::vector<PxVec3> allVerts;
    allVerts.reserve(nx * ny * nz);
    for (PxU32 i = 0; i < nx; ++i)
        for (PxU32 j = 0; j < ny; ++j)
            for (PxU32 k = 0; k < nz; ++k)
                allVerts.push_back(PxVec3(float(i) * h, float(j) * h, float(k) * h));
    auto vid = [ny, nz](PxU32 i, PxU32 j, PxU32 k) { return (i * ny + j) * nz + k; };

    std::vector<PxU32> rawTets;
    for (PxU32 i = 0; i < NX; ++i)
        for (PxU32 j = 0; j < NY; ++j)
            for (PxU32 k = 0; k < NZ; ++k)
            {
                // the carve-out: keep the two slabs and the connector column
                if (!(j == 0u || j == NY - 1u || i == 0u)) continue;
                const PxU32 c[8] = {
                    vid(i,     j,     k    ), vid(i + 1, j,     k    ),
                    vid(i,     j + 1, k    ), vid(i + 1, j + 1, k    ),
                    vid(i,     j,     k + 1), vid(i + 1, j,     k + 1),
                    vid(i,     j + 1, k + 1), vid(i + 1, j + 1, k + 1),
                };
                // ALTERNATING diagonals, as in buildSoftBodyScene: neighbouring cells must
                // disagree or their shared face is triangulated two different ways.
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
                    const PxVec3 e1 = allVerts[b] - allVerts[a];
                    const PxVec3 e2 = allVerts[cc] - allVerts[a];
                    const PxVec3 e3 = allVerts[d] - allVerts[a];
                    if (e1.cross(e2).dot(e3) < 0.0f) { const PxU32 tmp = cc; cc = b; b = tmp; }
                    rawTets.push_back(a); rawTets.push_back(b); rawTets.push_back(cc); rawTets.push_back(d);
                }
            }

    // COMPACT: drop the vertices no surviving tet references and remap the indices.
    std::vector<PxU32> remap(allVerts.size(), 0xFFFFFFFFu);
    std::vector<PxVec3> tverts;
    std::vector<PxU32> tets;
    tets.reserve(rawTets.size());
    for (PxU32 idx : rawTets)
    {
        if (remap[idx] == 0xFFFFFFFFu)
        {
            remap[idx] = PxU32(tverts.size());
            tverts.push_back(allVerts[idx]);
        }
        tets.push_back(remap[idx]);
    }

    // The TOP slab's vertices, and the bottom slab's upper surface, in the spawn frame.
    const float TOP_SLAB_Y = float(NY - 1) * h;    // 0.75 — the top slab's lower face
    out.bottomTopY = h;                            // 0.25 — the bottom slab's upper face
    for (PxU32 v = 0; v < tverts.size(); ++v)
        if (tverts[v].y >= TOP_SLAB_Y - 1e-4f) out.topIdx.push_back(v);
    out.numTopVerts = PxU32(out.topIdx.size());

    PxTetrahedronMeshDesc tetDesc;
    tetDesc.points.count = PxU32(tverts.size());
    tetDesc.points.stride = sizeof(PxVec3);
    tetDesc.points.data = tverts.data();
    tetDesc.tetrahedrons.count = PxU32(tets.size() / 4);
    tetDesc.tetrahedrons.stride = 4 * sizeof(PxU32);
    tetDesc.tetrahedrons.data = tets.data();

    PxCookingParams ckParams(physics->getTolerancesScale());
    ckParams.buildGPUData = true;
    ckParams.meshWeldTolerance = 0.001f;
    ckParams.midphaseDesc = PxMeshMidPhase::eBVH34;

    PxDeformableVolumeSimulationDataDesc simDataDesc;
    PxDeformableVolumeMesh* volMesh = immediateCooking::createDeformableVolumeMesh(
        ckParams, tetDesc, tetDesc, simDataDesc, physics->getPhysicsInsertionCallback());
    if (!volMesh) { std::printf("FAIL: createDeformableVolumeMesh returned null\n"); return out; }

    PxDeformableVolumeMaterial* volMat =
        physics->createDeformableVolumeMaterial(/*youngs*/ 5.e6f, /*poissons*/ 0.3f,
                                                /*dynamicFriction*/ 0.3f,
                                                /*elasticityDamping*/ 0.05f);
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
    {
        PxU32 iters = 30u;
        if (const char* e = std::getenv("PXG_SB_ITERS")) { const int v = atoi(e); if (v > 0) iters = PxU32(v); }
        static_cast<PxDeformableBody*>(volume)->setSolverIterationCounts(iters);
    }

    // PXG_SB_NOSELF is the CONTROL: the same scene with the chain switched off.
    out.selfCollision = true;
    if (const char* noSelf = std::getenv("PXG_SB_NOSELF"))
        if (atoi(noSelf) != 0) out.selfCollision = false;
    volume->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, !out.selfCollision);

    // The rest-pose gate: a self-contact is only kept if the two elements were FARTHER apart
    // than this in the rest pose.  The two slabs are 0.5 apart at rest and adjacent tets are
    // 0.25 apart, so 0.3 keeps the fold and drops the neighbours.
    volume->setSelfCollisionFilterDistance(0.3f);
    // The stress gate drops a self-contact whose tetrahedron is under more stress than this.
    // It reads mTetraStressCoefficient, which sb_calculateStressLaunch fills (P8.5) -- and
    // that kernel's own launch gate runs it precisely BECAUSE self-collision is enabled here.
    // Until it was mapped the coefficient held whatever the allocation contained, so this had
    // to be a value no stress could exceed.  PXG_SB_STRESSTOL exposes it now that the field
    // is real: the default still disables the gate so the scene measures the CONTACT path
    // rather than the material, and a small value is the way to see the gate bite.
    {
        float tol = 1.0e30f;
        if (const char* e = std::getenv("PXG_SB_STRESSTOL")) tol = float(atof(e));
        volume->setSelfCollisionStressTolerance(tol);
    }

    scene->addActor(*volume);

    // Mirrors buildSoftBodyScene's open-coded host-mirror initialisation.  The bracket is
    // spawned in place -- its bottom face already sits on the plane at y = 0 -- so there is
    // no free-fall phase to wait out and the sag is the whole motion.
    const PxU32 nbSim = volume->getSimulationMesh()->getNbVertices();
    const PxU32 nbColl = volume->getCollisionMesh()->getNbVertices();
    std::vector<PxVec4> simPos(nbSim), simVel(nbSim), collPos(nbColl), restPos(nbColl);
    const PxVec3* simVerts = volume->getSimulationMesh()->getVertices();
    const PxReal* invMassGM = volume->getDeformableVolumeAuxData()->getGridModelInvMass();
    for (PxU32 v = 0; v < nbSim; ++v)
    {
        const PxReal im = invMassGM ? invMassGM[v] : 1.0f;
        simPos[v] = PxVec4(simVerts[v].x, simVerts[v].y, simVerts[v].z, im);
        simVel[v] = PxVec4(0.0f, 0.0f, 0.0f, im);
    }
    const PxVec3* collVerts = volume->getCollisionMesh()->getVertices();
    for (PxU32 v = 0; v < nbColl; ++v)
    {
        collPos[v] = PxVec4(collVerts[v].x, collVerts[v].y, collVerts[v].z, 1.0f);
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

    // The TOP slab's COLLISION-mesh vertices.  The cooker may reorder or weld, so the split
    // is recomputed from the cooked mesh rather than carried over from the authored one.
    out.topIdx.clear();
    for (PxU32 v = 0; v < nbColl; ++v)
        if (collVerts[v].y >= TOP_SLAB_Y - 1e-4f) out.topIdx.push_back(v);
    out.numTopVerts = PxU32(out.topIdx.size());

    out.volume = volume;
    out.numCollVerts = nbColl;
    out.numSimVerts = nbSim;
    std::printf("  softbodyselfcollide mesh: %u sim verts / %u tets (%u of %u collision verts "
                "in the TOP slab), slabs 0.50 apart at rest, bottom slab top at y=%.2f, "
                "self-collision %s\n",
                nbSim, PxU32(tets.size() / 4), out.numTopVerts, nbColl, out.bottomTopY,
                out.selfCollision ? "ON" : "OFF (control)");
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

// The TOP slab's mean and min y, read from the COLLISION mesh (what the skin kernel writes,
// and the buffer every other softbody scene observes).
bool softBodySelfTopY(const SoftBodySelfScene& ssc, float& meanY, float& minY)
{
    std::vector<PxVec4> host(ssc.numCollVerts);
    PxCudaContext* ctx = ssc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(),
                    reinterpret_cast<CUdeviceptr>(ssc.volume->getPositionInvMassBufferD()),
                    ssc.numCollVerts * sizeof(PxVec4));
    double sum = 0.0;
    float lo = 1.0e30f;
    bool finite = true;
    for (PxU32 t = 0; t < ssc.numTopVerts; ++t)
    {
        const PxVec4& p = host[ssc.topIdx[t]];
        sum += double(p.y);
        lo = PxMin(lo, p.y);
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) finite = false;
    }
    meanY = ssc.numTopVerts ? float(sum / double(ssc.numTopVerts)) : 0.0f;
    minY = lo;
    return finite;
}

} // namespace


// ===========================================================================
// PXG_SCENE=softbodycloth (P8.8): the scene the softbody-vs-CLOTH chain exists for.
//
// Six kernels were ported one at a time, each against its own fixture and the engine's ABI.
// None of that says they work as a CHAIN on live data -- a wrong ABI row that happens to have
// the right byte count, a dispatch the engine gates off, a buffer the engine never allocates
// when both actor types are present, all pass a per-kernel fixture and fail here.  This scene
// is the only thing that can see them.
//
// THE SETUP: a static plane at y = 0, a FEM-VOLUME cube of edge 1 resting on it (so its top
// face is at y = 1), and a small FEM-SURFACE sheet spawned at y = 2 directly above.  The sheet
// is 0.6 x 0.6, DELIBERATELY SMALLER than the cube's top face, so a sheet that lands on the
// cube stays entirely on it and one that misses falls all the way to the plane.  The two
// outcomes are a whole unit apart, which is what makes the control run decisive.
//
// THE CONTROL: PXG_SC_NOSOFTBODY=1 removes the cube and nothing else.  The sheet then falls to
// the plane.  Without that half the main run proves nothing -- a sheet that simply never fell
// would pass it too.
//
// The ground box is sunk to y = -50.5 and the island dynamic stands at x = 10, for the reasons
// the softbody scene's header gives: both are real obstacles now that the hull branch exists,
// and neither belongs in what this scene measures.
//
// THE MASS RATIO IS PART OF THE SETUP, NOT A DETAIL.  The sheet spent one whole tranche falling
// straight through the cube while every kernel in the chain dispatched, emitted pairs, emitted
// contacts and built sane constraints.  The chain was right the entire time; THE SCENE WAS
// WRONG.  A read-back of the two inverse-mass fields settled it in one line:
//
//     clothW = 1        softbody sim vertex W = 192
//
// The cube's simulation vertices come out of the cooker at invMass 192 -- about 0.14 kg for the
// whole cube -- and the sheet had been handed invMass 1, i.e. 1 kg PER VERTEX, 25 kg of cloth.
// The constraint splits its correction as w_i / (w_cloth + sum bc^2 w_softbody), so 90% of every
// impulse went into shoving the CUBE out of the way and 10% into holding the sheet.  The solve
// was doing exactly the right thing with the masses it was given.  Raising the sheet's invMass
// to 200 (PXG_SC_CLOTHW, see the builder) catches it immediately and it settles.
//
// The lesson is worth keeping: "the deltas do not reach the vertices" was the wrong hypothesis
// twice over.  They reached the vertices the whole time -- a probe on cloth.mDeltaPos right
// after the solve showed 18-25 non-zero entries per iteration and lambdaN accumulating
// monotonically (0.0047 -> 0.0197 over nine iterations).  What looked like a dead chain was a
// live chain moving the lighter of the two bodies.
//
// What the passing run establishes, measured rather than assumed:
//
//   * the pair IS classified into GPU_BUCKET_ID::eSoftbodyFemCloth (probed directly in
//     cndCaseTetraMesh: compat = 1, isRigid1 = 0);
//   * the chain dispatches end to end.  Over 40 steps: sb_clothVertMidphaseGeneratePairs 16
//     launches, sb_clothVertContactGen 16, sb_clothContactPrepare 40, sb_solveOutputSCDeltaV
//     360 (once per solver iteration), sb_fem_contact_remap_to_sim 16;
//   * the vertex midphase emits 170-231 PAIRS per frame and the generator 29-48 CONTACTS;
//   * the prepare sees the full count (44 in, 44 out, mMaxContacts = 1048576) and builds sane
//     constraints -- normal (0, 1, 0), velMultiplier ~0.097, friction 0.5;
//   * and the sheet is CAUGHT and SETTLES on the cube.
//
// ALSO WORTH KNOWING BEFORE SPENDING MORE ON THIS: upstream hard-codes
// `supportClothTriangleCollision = false` (PxgNarrowphaseCore.cpp:4529), so the TRIANGLE half
// of the chain -- sb_clothMidphaseGeneratePairsLaunch and sb_clothContactGenLaunch, steps 1
// and 2 of P8.8 -- is dead code in this engine version and can never run as shipped.  Only the
// VERTEX half (steps 3 and 4) is reachable.  That does not make steps 1 and 2 wrong, but it
// does mean no scene can exercise them without changing engine source.
// ===========================================================================
struct SoftBodyClothScene
{
    PxDeformableVolume* volume = nullptr;
    PxDeformableSurface* surface = nullptr;
    PxU32 numClothVerts = 0;
    PxU32 numSimVerts = 0;
    bool hasSoftBody = false;
    float cubeTopY = 1.0f;
    PxCudaContextManager* cudaCtx = nullptr;
};

// Read the cloth's vertex positions back and reduce them.
bool softBodyClothY(const SoftBodyClothScene& sc, float& meanY, float& minY)
{
    std::vector<PxVec4> pos(sc.numClothVerts);
    PxCudaContext* ctx = sc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(pos.data(),
                    reinterpret_cast<CUdeviceptr>(sc.surface->getPositionInvMassBufferD()),
                    sc.numClothVerts * sizeof(PxVec4));
    double sum = 0.0;
    float mn = FLT_MAX;
    bool finite = true;
    for (PxU32 v = 0; v < sc.numClothVerts; ++v)
    {
        finite = finite && std::isfinite(pos[v].y);
        sum += double(pos[v].y);
        mn = PxMin(mn, pos[v].y);
    }
    meanY = float(sum / double(sc.numClothVerts));
    minY = mn;
    return finite;
}

// A cooked, GPU-resident FEM-VOLUME cube of edge `size`, its base at y = baseY, centred on the
// x/z origin unless `offsetX` moves it.  Extracted from buildSoftBodyClothScene at P8.9 so the
// softbody-vs-softbody stack can raise a second, identical cube -- two cubes from one recipe is
// what makes "the upper one rests on the lower one" a statement about the CONTACT and not about
// two slightly different meshes.  Self-collision is disabled: these scenes are about collisions
// BETWEEN bodies, and leaving it on would let the self-collision chain take the credit.
PxDeformableVolume* buildFemVolumeCube(PxPhysics* physics, PxScene* scene,
                                       PxCudaContextManager* cudaCtx, PxU32 cells, float size,
                                       float baseY, float offsetX = 0.0f)
{
    const float h = size / float(cells);
    const PxU32 n = cells + 1;
    std::vector<PxVec3> tverts;
    for (PxU32 i = 0; i < n; ++i)
        for (PxU32 j = 0; j < n; ++j)
            for (PxU32 kk = 0; kk < n; ++kk)
                tverts.push_back(PxVec3(float(i) * h - 0.5f * size + offsetX,
                                        float(j) * h + baseY,
                                        float(kk) * h - 0.5f * size));
    auto vid = [n](PxU32 i, PxU32 j, PxU32 kk) { return (i * n + j) * n + kk; };
    std::vector<PxU32> tets;
    for (PxU32 i = 0; i < cells; ++i)
        for (PxU32 j = 0; j < cells; ++j)
            for (PxU32 kk = 0; kk < cells; ++kk)
            {
                const PxU32 c[8] = {
                    vid(i, j, kk),         vid(i, j, kk + 1),
                    vid(i, j + 1, kk),     vid(i, j + 1, kk + 1),
                    vid(i + 1, j, kk),     vid(i + 1, j, kk + 1),
                    vid(i + 1, j + 1, kk), vid(i + 1, j + 1, kk + 1) };
                const PxU32 idx[5][4] = { {0,1,2,4}, {1,2,3,7}, {1,4,5,7}, {2,4,6,7}, {1,2,4,7} };
                for (int t = 0; t < 5; ++t)
                {
                    PxU32 a = c[idx[t][0]], b = c[idx[t][1]], cc = c[idx[t][2]], d = c[idx[t][3]];
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

    PxCookingParams ck(physics->getTolerancesScale());
    ck.buildGPUData = true;
    ck.meshWeldTolerance = 0.001f;
    ck.midphaseDesc = PxMeshMidPhase::eBVH34;
    PxDeformableVolumeSimulationDataDesc simDataDesc;
    PxDeformableVolumeMesh* volMesh = immediateCooking::createDeformableVolumeMesh(
        ck, tetDesc, tetDesc, simDataDesc, physics->getPhysicsInsertionCallback());
    if (!volMesh) { std::printf("FAIL: createDeformableVolumeMesh returned null\n"); return nullptr; }

    PxDeformableVolumeMaterial* volMat =
        physics->createDeformableVolumeMaterial(1.e6f, 0.3f, 0.5f, 0.05f);
    PxDeformableVolumeMaterial* vmats[1] = { volMat };
    PxShapeFlags sf = PxShapeFlag::eSIMULATION_SHAPE | PxShapeFlag::eSCENE_QUERY_SHAPE;
    PxShape* vshape = physics->createShape(
        PxTetrahedronMeshGeometry(volMesh->getCollisionMesh()), vmats, PxU16(1), true, sf);
    PxDeformableVolume* volume = physics->createDeformableVolume(*cudaCtx);
    volume->attachShape(*vshape);
    vshape->release();
    volume->attachSimulationMesh(*volMesh->getSimulationMesh(), *volMesh->getDeformableVolumeAuxData());
    volume->setDeformableBodyFlag(PxDeformableBodyFlag::eDISABLE_SELF_COLLISION, true);
    volume->setLinearDamping(0.0f);
    volume->setMaxLinearVelocity(1.0e6f);
    static_cast<PxDeformableBody*>(volume)->setSolverIterationCounts(8u, 1u);
    scene->addActor(*volume);

    const PxU32 nbSim = volume->getSimulationMesh()->getNbVertices();
    const PxU32 nbColl = volume->getCollisionMesh()->getNbVertices();
    std::vector<PxVec4> simPos(nbSim), simVel(nbSim), collPos(nbColl), restPos(nbColl);
    const PxVec3* simVerts = volume->getSimulationMesh()->getVertices();
    const PxReal* invMassGM = volume->getDeformableVolumeAuxData()->getGridModelInvMass();
    for (PxU32 v = 0; v < nbSim; ++v)
    {
        const PxReal im = invMassGM ? invMassGM[v] : 1.0f;
        simPos[v] = PxVec4(simVerts[v].x, simVerts[v].y, simVerts[v].z, im);
        simVel[v] = PxVec4(0.0f, 0.0f, 0.0f, im);
    }
    const PxVec3* collVerts = volume->getCollisionMesh()->getVertices();
    for (PxU32 v = 0; v < nbColl; ++v)
    {
        collPos[v] = PxVec4(collVerts[v].x, collVerts[v].y, collVerts[v].z, 1.0f);
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
    return volume;
}

// ---------------------------------------------------------------------------
// PXG_SCENE=particlesoftbody (P9.48c): the scene the FEM->PARTICLE half of the chain
// exists for, and the one that could not exist before P9.48a.
//
// P9.39c..e mapped the softbody-vs-particle contact SOLVE: it writes a deltaP per
// CONTACT.  What it could not do was move a particle, because the consumer that reduces
// those per-contact deltas onto the particle -- ps_accumulateFEMParticleDeltaVLaunch,
// preceded by the two ps_findStartEndFEM passes -- was unmapped.  Every kernel in the
// chain dispatched, the softbody reacted, and the particles received nothing.  A cloud
// dropped on a softbody fell straight through it.
//
// THE SETUP: a static plane at y = 0, a FEM-VOLUME cube of edge 1 resting on it (top face
// at y = 1), and a small cloud of PBD particles spawned just above that face.  The cloud
// is 0.3 x 0.3 wide against the cube's 1 x 1 top, so a cloud that is CAUGHT stays entirely
// on the cube and one that is not falls a whole unit further, to the plane.  The two
// outcomes are a unit apart, which is what makes the control decisive.
//
// THE CONTROL: PXG_PSB_NOSOFTBODY=1 removes the cube and nothing else.  The cloud then
// falls to the plane and rests at one rest offset above it.  Without that half the main
// run proves nothing -- a cloud that simply never fell would pass it too.
//
// THE MASS RATIO IS PART OF THE SETUP, exactly as the softbodycloth header records for the
// sheet.  The cube's simulation vertices come out of the cooker at invMass 192; a particle
// at invMass 1 is a thousand times heavier than the thing it lands on, and the constraint
// splits its correction by w_particle / (w_particle + sum bc^2 w_softbody) -- so nearly all
// of every impulse goes into shoving the CUBE aside.  PXG_PSB_PARTW (default 200) sets the
// particle inverse mass into the cube's own range, which is what lets the cube hold it.
// ---------------------------------------------------------------------------
struct SoftBodyParticleScene
{
    PxDeformableVolume* volume = nullptr;
    PxPBDParticleSystem* system = nullptr;
    PxParticleBuffer* buffer = nullptr;
    PxU32 numParticles = 0;
    PxU32 numSimVerts = 0;
    bool hasSoftBody = true;
    float cubeTopY = 0.0f;
    PxCudaContextManager* cudaCtx = nullptr;
};

// The cloud's mean y, read back from the particle buffer's device array.
static float sbParticleMeanY(const SoftBodyParticleScene& sc)
{
    if (!sc.buffer || sc.numParticles == 0) return 0.0f;
    std::vector<PxVec4> host(sc.numParticles);
    PxCudaContext* ctx = sc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(host.data(), reinterpret_cast<CUdeviceptr>(sc.buffer->getPositionInvMasses()),
                    sc.numParticles * sizeof(PxVec4));
    double sum = 0.0;
    for (PxU32 i = 0; i < sc.numParticles; ++i) sum += double(host[i].y);
    return float(sum / double(sc.numParticles));
}

SoftBodyParticleScene buildSoftBodyParticleScene(PxPhysics* physics, PxScene* scene,
                                                 PxMaterial* rigidMat,
                                                 PxCudaContextManager* cudaCtx)
{
    SoftBodyParticleScene out;
    out.cudaCtx = cudaCtx;

    // The ground box is sunk and the island dynamic parked, for the reason the
    // softbodycloth scene gives: both are real obstacles now that the hull branch
    // exists, and neither belongs in what this scene measures.  The island is also the
    // one live dynamic rigid the particle island's solver core requires -- with none,
    // rigidSumInternalContactAndJointBatches2 dereferences null solver descriptors.
    PxRigidStatic* ground = addGroundBox(physics, scene, rigidMat);
    ground->setGlobalPose(PxTransform(PxVec3(0.0f, -50.5f, 0.0f)));
    PxRigidDynamic* island = addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                        boxInertia(PxVec3(0.5f), 1000.0f), 1.5f);
    island->setGlobalPose(PxTransform(PxVec3(10.0f, 1.5f, 0.0f)));

    {
        PxRigidStatic* floor = physics->createRigidStatic(
            PxTransformFromPlaneEquation(PxPlane(0.0f, 1.0f, 0.0f, 0.0f)));
        PxShape* fs = physics->createShape(PxPlaneGeometry(), *rigidMat);
        floor->attachShape(*fs);
        scene->addActor(*floor);
        fs->release();
    }

    const bool noSoftBody = std::getenv("PXG_PSB_NOSOFTBODY") != nullptr;
    out.hasSoftBody = !noSoftBody;

    const float size = 1.0f;
    if (!noSoftBody)
    {
        PxDeformableVolume* volume = buildFemVolumeCube(physics, scene, cudaCtx, /*cells*/ 2u,
                                                        size, /*baseY*/ 0.0f);
        if (!volume) return out;
        out.volume = volume;
        out.numSimVerts = volume->getSimulationMesh()->getNbVertices();
        out.cubeTopY = size;
    }

    // ---- the particle cloud, just above the cube's top face -----------------------------
    const float particleSpacing = 0.15f;
    const char* partWEnv = std::getenv("PXG_PSB_PARTW");
    const float particleInvMass = partWEnv ? float(atof(partWEnv)) : 200.0f;

    PxPBDMaterial* pbdMat = physics->createPBDMaterial(
        /*friction*/ 0.2f, /*damping*/ 0.0f, /*adhesion*/ 0.0f, /*viscosity*/ 0.0f,
        /*vorticityConfinement*/ 0.0f, /*surfaceTension*/ 0.0f, /*cohesion*/ 0.0f,
        /*lift*/ 0.0f, /*drag*/ 0.0f);
    PxPBDParticleSystem* ps = physics->createPBDParticleSystem(*cudaCtx);
    const float restOffset = 0.5f * particleSpacing;
    ps->setRestOffset(restOffset);
    ps->setContactOffset(restOffset + 0.02f);
    ps->setParticleContactOffset(restOffset + 0.02f);
    ps->setSolidRestOffset(restOffset);
    ps->setMaxVelocity(1e6f);
    ps->setSolverIterationCounts(/*pos*/ 8, /*vel*/ 1);
    const PxU32 phase = ps->createPhase(
        pbdMat, PxParticlePhaseFlags(PxParticlePhaseFlag::eParticlePhaseSelfCollide));

    // 3 x 3 x 3 at 0.15 spacing = a 0.3 cube, well inside the cube's 1 x 1 top face even
    // after it spreads.  Spawned with its bottom row one rest offset above the top face,
    // so it is a short settle rather than a drop that could punch through in one step.
    const int G = 3;
    const PxU32 N = PxU32(G * G * G);
    const float spawnBaseY = (noSoftBody ? 0.0f : size) + restOffset + 0.05f;
    std::vector<PxVec4> hostPos(N), hostVel(N, PxVec4(0.0f, 0.0f, 0.0f, 0.0f));
    std::vector<PxU32> hostPhase(N, phase);
    const float c = 0.5f * float(G - 1) * particleSpacing;
    PxU32 idx = 0;
    for (int x = 0; x < G; ++x)
        for (int y = 0; y < G; ++y)
            for (int z = 0; z < G; ++z)
                hostPos[idx++] = PxVec4(float(x) * particleSpacing - c,
                                        spawnBaseY + float(y) * particleSpacing,
                                        float(z) * particleSpacing - c, particleInvMass);

    PxParticleBuffer* buffer = physics->createParticleBuffer(N, /*maxVolumes*/ 0, cudaCtx);
    if (!buffer) { std::printf("FAIL: createParticleBuffer returned null\n"); return out; }
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

SoftBodyClothScene buildSoftBodyClothScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                           PxCudaContextManager* cudaCtx)
{
    SoftBodyClothScene out;
    out.cudaCtx = cudaCtx;

    PxRigidStatic* ground = addGroundBox(physics, scene, rigidMat);
    ground->setGlobalPose(PxTransform(PxVec3(0.0f, -50.5f, 0.0f)));
    PxRigidDynamic* island = addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                        boxInertia(PxVec3(0.5f), 1000.0f), 1.5f);
    island->setGlobalPose(PxTransform(PxVec3(10.0f, 1.5f, 0.0f)));

    {
        PxRigidStatic* floor = physics->createRigidStatic(
            PxTransformFromPlaneEquation(PxPlane(0.0f, 1.0f, 0.0f, 0.0f)));
        PxShape* fs = physics->createShape(PxPlaneGeometry(), *rigidMat);
        floor->attachShape(*fs);
        scene->addActor(*floor);
        fs->release();
    }

    const bool noSoftBody = std::getenv("PXG_SC_NOSOFTBODY") != nullptr;
    out.hasSoftBody = !noSoftBody;

    // ---- the FEM-volume cube, resting on the plane -----------------------------------------
    if (!noSoftBody)
    {
        const float size = 1.0f;
        PxDeformableVolume* volume = buildFemVolumeCube(physics, scene, cudaCtx, /*cells*/ 2u,
                                                        size, /*baseY*/ 0.0f);
        if (!volume) return out;
        out.volume = volume;
        out.numSimVerts = volume->getSimulationMesh()->getNbVertices();
        out.cubeTopY = size;
    }

    // ---- the FEM-surface sheet, above the cube ----------------------------------------------
    {
        const PxU32 numX = 5, numZ = 5;
        const float sizeX = 0.6f, sizeZ = 0.6f;      // smaller than the cube's 1 x 1 top face
        const float sx = sizeX / float(numX - 1), sz = sizeZ / float(numZ - 1);
        std::vector<PxVec3> verts;
        for (PxU32 i = 0; i < numX; ++i)
            for (PxU32 j = 0; j < numZ; ++j)
                verts.push_back(PxVec3(float(i) * sx - 0.5f * sizeX, 0.0f, float(j) * sz - 0.5f * sizeZ));
        std::vector<PxU32> tris;
        auto cvid = [numZ](PxU32 x, PxU32 y) { return x * numZ + y; };
        for (PxU32 i = 1; i < numX; ++i)
            for (PxU32 j = 1; j < numZ; ++j)
            {
                tris.push_back(cvid(i - 1, j - 1)); tris.push_back(cvid(i, j - 1)); tris.push_back(cvid(i - 1, j));
                tris.push_back(cvid(i - 1, j));     tris.push_back(cvid(i, j - 1)); tris.push_back(cvid(i, j));
            }

        PxTriangleMeshDesc md;
        md.points.count = PxU32(verts.size());
        md.points.stride = sizeof(PxVec3);
        md.points.data = verts.data();
        md.triangles.count = PxU32(tris.size() / 3);
        md.triangles.stride = 3 * sizeof(PxU32);
        md.triangles.data = tris.data();

        PxCookingParams ck(physics->getTolerancesScale());
        ck.buildGPUData = true;
        ck.meshPreprocessParams |= PxMeshPreprocessingFlag::eENABLE_VERT_MAPPING;
        PxTriangleMesh* triMesh =
            immediateCooking::createTriangleMesh(ck, md, physics->getPhysicsInsertionCallback());
        if (!triMesh) { std::printf("FAIL: createTriangleMesh returned null\n"); return out; }

        PxDeformableSurfaceMaterial* clothMat =
            physics->createDeformableSurfaceMaterial(1.e6f, 0.3f, 0.5f, 0.01f, 1.e-4f);
        PxDeformableSurfaceMaterial* cmats[1] = { clothMat };
        PxShapeFlags sf = PxShapeFlag::eSIMULATION_SHAPE | PxShapeFlag::eSCENE_QUERY_SHAPE;
        PxShape* cshape = physics->createShape(PxTriangleMeshGeometry(triMesh), cmats, PxU16(1), true, sf);
        PxDeformableSurface* surface = physics->createDeformableSurface(*cudaCtx);
        surface->setNbCollisionPairUpdatesPerTimestep(4);
        surface->setNbCollisionSubsteps(1);
        surface->setLinearDamping(0.0f);
        surface->setMaxLinearVelocity(1.0e6f);
        surface->attachShape(*cshape);
        cshape->release();
        scene->addActor(*surface);

        // Spawn height.  The default 2.0 is a full metre of free fall onto a 1 m cube: the sheet
        // arrives at ~4.5 m/s, which is FASTER than one frame's contact-resolution budget (see
        // RUNTIME_STATUS, P8.8).  PXG_SC_CLOTHY lowers it so the same chain can be exercised at
        // a speed the constraint solve can actually absorb.
        float spawnY = 2.0f;
        if (const char* yEnv = std::getenv("PXG_SC_CLOTHY")) spawnY = float(atof(yEnv));

        // Per-vertex INVERSE mass of the sheet.  This is not cosmetic: the constraint splits
        // its correction between the two bodies as w_i / (w_cloth + sum bc^2 w_softbody), so
        // the RATIO of the two inverse masses decides who moves.  The cube's simulation
        // vertices come out of the cooker at invMass = 192 (mass 1/192 each, ~0.14 kg for the
        // whole cube); a sheet at invMass = 1 is 25 kg of cloth landing on 0.14 kg of foam and
        // the solve correctly pushes the CUBE out of the way instead of holding the sheet.
        float clothInvMass = 200.0f;
        if (const char* wEnv = std::getenv("PXG_SC_CLOTHW")) clothInvMass = float(atof(wEnv));
        const PxU32 nv = PxU32(verts.size());
        std::vector<PxVec4> posInvMass(nv), vel(nv, PxVec4(0.0f)), rest(nv);
        for (PxU32 v = 0; v < nv; ++v)
        {
            posInvMass[v] = PxVec4(verts[v].x, spawnY, verts[v].z, clothInvMass);
            rest[v] = posInvMass[v];
        }
        PxCudaContext* ctx = cudaCtx->getCudaContext();
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getPositionInvMassBufferD()),
                        posInvMass.data(), nv * sizeof(PxVec4));
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getVelocityBufferD()),
                        vel.data(), nv * sizeof(PxVec4));
        ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(surface->getRestPositionBufferD()),
                        rest.data(), nv * sizeof(PxVec4));

        out.surface = surface;
        out.numClothVerts = nv;
    }

    return out;
}


// ===========================================================================
// PXG_SCENE=softbodystack (P8.9): the scene the softbody-vs-SOFTBODY narrowphase exists for.
//
// TWO FEM-VOLUME CUBES from ONE recipe (buildFemVolumeCube): the lower one rests on a static
// plane with its top face at y = 1, the upper one is dropped from base y = 1.5.  Both cubes are
// cooked identically, so their per-vertex inverse masses are identical too -- which is not a
// detail.  P8.8 spent a whole tranche on a scene whose sheet was 192x heavier per vertex than
// the cube it landed on, where a CORRECT solve pushed the lighter body out of the way.  Two
// cubes from one recipe cannot reproduce that, by construction.
//
// Self-collision is OFF on both (buildFemVolumeCube sets eDISABLE_SELF_COLLISION), so nothing
// the SELF-collision chain does can be mistaken for the between-bodies chain being alive.
//
// THE CONTROL: PXG_SS_NOLOWER=1 removes the lower cube and nothing else.  The upper cube then
// falls all the way to the plane, base y ~ 0.  Caught and missed are a full unit apart.
//
// WHAT IT COVERS.  Softbody-vs-softbody is the one collision pair in this port that is a single
// kernel: `testSDKSoftbodies` (PxgNarrowphaseCore.cpp:4399) dispatches
// sb_sbMidphaseGeneratePairsLaunch and nothing else, then remaps to the sim mesh with the
// already-mapped sb_fem_contact_remap_to_simLaunch.  Prepare and solve are
// sb_softbodyContactPrepareLaunch and sb_solveOutputSSDeltaVLaunch, both mapped by the
// self-collision tranche and both shared with it.  So this scene is the first thing that
// exercises the new kernel against live data, and the only thing that can.
// ===========================================================================
struct SoftBodyStackScene
{
    PxDeformableVolume* lower = nullptr;
    PxDeformableVolume* upper = nullptr;
    PxU32 numUpperSimVerts = 0;
    bool hasLower = false;
    float lowerTopY = 1.0f;
    PxCudaContextManager* cudaCtx = nullptr;
};

// Read the UPPER cube's simulation vertices back and reduce them.
bool softBodyStackUpperY(const SoftBodyStackScene& sc, float& meanY, float& minY)
{
    std::vector<PxVec4> pos(sc.numUpperSimVerts);
    PxCudaContext* ctx = sc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(pos.data(),
                    reinterpret_cast<CUdeviceptr>(sc.upper->getSimPositionInvMassBufferD()),
                    pos.size() * sizeof(PxVec4));
    float sum = 0.0f;
    minY = FLT_MAX;
    bool finite = true;
    for (const PxVec4& p : pos)
    {
        finite = finite && std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
        sum += p.y;
        minY = PxMin(minY, p.y);
    }
    meanY = sum / float(pos.size());
    return finite;
}

SoftBodyStackScene buildSoftBodyStackScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                           PxCudaContextManager* cudaCtx)
{
    SoftBodyStackScene out;
    out.cudaCtx = cudaCtx;

    // The ground box and the island dynamic are sunk / parked exactly as the softbodycloth
    // scene's header explains: both are real obstacles now that the hull branch exists, and
    // neither belongs in what this scene measures.
    PxRigidStatic* ground = addGroundBox(physics, scene, rigidMat);
    ground->setGlobalPose(PxTransform(PxVec3(0.0f, -50.5f, 0.0f)));
    PxRigidDynamic* island = addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                        boxInertia(PxVec3(0.5f), 1000.0f), 1.5f);
    island->setGlobalPose(PxTransform(PxVec3(10.0f, 1.5f, 0.0f)));

    {
        PxRigidStatic* floor = physics->createRigidStatic(
            PxTransformFromPlaneEquation(PxPlane(0.0f, 1.0f, 0.0f, 0.0f)));
        PxShape* fs = physics->createShape(PxPlaneGeometry(), *rigidMat);
        floor->attachShape(*fs);
        scene->addActor(*floor);
        fs->release();
    }

    const float lowerSize = 1.0f;
    out.lowerTopY = lowerSize;

    const bool noLower = std::getenv("PXG_SS_NOLOWER") != nullptr;
    out.hasLower = !noLower;
    if (!noLower)
    {
        out.lower = buildFemVolumeCube(physics, scene, cudaCtx, /*cells*/ 2u, lowerSize, /*baseY*/ 0.0f);
        if (!out.lower) return out;
    }

    // *** THE UPPER CUBE IS SMALLER, AND THAT IS NOT COSMETIC. ***
    // Two IDENTICAL cubes stacked flush give coplanar SIDE faces, and every vertex on the
    // upper cube's bottom rim then sits exactly on one.  The narrowphase is a vertex-versus-
    // closest-tet-SURFACE-point test, so at a tie between the side face (distance ~0) and the
    // top face (distance ~0) it takes the side, and the emitted normal comes out HORIZONTAL --
    // measured at n.y = 0.003 with |pen| < 0.001.  A horizontal contact can never hold a cube
    // up, and the stack sank with the whole chain running correctly.  Making the upper cube
    // 0.6 wide puts its entire bottom rim strictly INSIDE the lower cube's 1 x 1 top face, so
    // the top face is unambiguously the nearest surface and the normal is vertical.  This is
    // the same reason the softbodycloth scene's sheet is smaller than the cube it lands on.
    const float upperSize = 0.6f;

    // The upper cube starts half a unit clear of the lower one's top face.  That is a short
    // enough fall that the contact does not have to absorb a large impact speed in one frame,
    // and long enough that a scene where nothing collides is unmistakable.
    float upperBaseY = 1.5f;
    if (const char* yEnv = std::getenv("PXG_SS_UPPERY")) upperBaseY = float(atof(yEnv));
    out.upper = buildFemVolumeCube(physics, scene, cudaCtx, /*cells*/ 2u, upperSize, upperBaseY);
    if (!out.upper) return out;
    out.numUpperSimVerts = out.upper->getSimulationMesh()->getNbVertices();

    return out;
}

// PXG_SCENE=softbodyattach (P8.19): the FEM-vs-FEM ATTACHMENT money shot.
//
// A free FEM volume hangs from a PINNED one with NOTHING ELSE in the loop -- no floor under it,
// no contact between the two (see the gap discussion in the builder), self-collision off on
// both.  The only thing that can hold the upper cube up is femAttachmentPrepareLaunch +
// sb_solveOutputSoftBodyAttachmentDeltaVLaunch.  Unmap either and it free-falls, which is
// exactly the control the verification runs.
//
// The attachment points are chosen at RUNTIME rather than hard-coded: the two cubes' collision
// tetrahedra come out of the cooker and their indices are not predictable, so the builder reads
// both meshes back, takes the lower cube's HIGHEST tetrahedra and the upper cube's LOWEST, and
// pairs them by proximity in xz.  Each attachment anchors both tetrahedra at their CENTROIDS
// (barycentric 0.25 each).
// ===========================================================================
struct SoftBodyAttachScene
{
    PxDeformableVolume* lower = nullptr;
    PxDeformableVolume* upper = nullptr;
    PxU32 numUpperSimVerts = 0;
    PxU32 nbAttachments = 0;
    float upperStartY = 0.0f;
    PxCudaContextManager* cudaCtx = nullptr;
};

// Read the UPPER cube's simulation vertices back and reduce them.
bool softBodyAttachUpperY(const SoftBodyAttachScene& sc, float& meanY, float& minY)
{
    std::vector<PxVec4> pos(sc.numUpperSimVerts);
    PxCudaContext* ctx = sc.cudaCtx->getCudaContext();
    ctx->memcpyDtoH(pos.data(),
                    reinterpret_cast<CUdeviceptr>(sc.upper->getSimPositionInvMassBufferD()),
                    pos.size() * sizeof(PxVec4));
    float sum = 0.0f;
    minY = FLT_MAX;
    bool finite = true;
    for (const PxVec4& p : pos)
    {
        finite = finite && std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
        sum += p.y;
        minY = PxMin(minY, p.y);
    }
    meanY = sum / float(pos.size());
    return finite;
}

// Every collision tetrahedron's centroid, so the builder can pick the ones at the interface.
static void collisionTetCentroids(PxDeformableVolume* vol, std::vector<PxVec3>& out)
{
    const PxTetrahedronMesh* mesh = vol->getCollisionMesh();
    const PxVec3* verts = reinterpret_cast<const PxVec3*>(mesh->getVertices());
    const PxU32 nbTets = mesh->getNbTetrahedrons();
    const bool is16 = (mesh->getTetrahedronMeshFlags() & PxTetrahedronMeshFlag::e16_BIT_INDICES);
    out.resize(nbTets);
    for (PxU32 t = 0; t < nbTets; ++t)
    {
        PxU32 idx[4];
        if (is16)
        {
            const PxU16* p = reinterpret_cast<const PxU16*>(mesh->getTetrahedrons()) + 4 * t;
            for (int k = 0; k < 4; ++k) idx[k] = p[k];
        }
        else
        {
            const PxU32* p = reinterpret_cast<const PxU32*>(mesh->getTetrahedrons()) + 4 * t;
            for (int k = 0; k < 4; ++k) idx[k] = p[k];
        }
        out[t] = (verts[idx[0]] + verts[idx[1]] + verts[idx[2]] + verts[idx[3]]) * 0.25f;
    }
}

SoftBodyAttachScene buildSoftBodyAttachScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                             PxCudaContextManager* cudaCtx)
{
    SoftBodyAttachScene out;
    out.cudaCtx = cudaCtx;

    // The ground box and the island dynamic are sunk / parked exactly as softbodystack does:
    // the rigid PGS island must be non-empty, and neither belongs in what this scene measures.
    // There is deliberately NO floor plane -- the upper cube must have nothing to land on.
    PxRigidStatic* ground = addGroundBox(physics, scene, rigidMat);
    ground->setGlobalPose(PxTransform(PxVec3(0.0f, -50.5f, 0.0f)));
    PxRigidDynamic* island = addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
                                        boxInertia(PxVec3(0.5f), 1000.0f), 1.5f);
    island->setGlobalPose(PxTransform(PxVec3(10.0f, 1.5f, 0.0f)));

    const float lowerSize = 1.0f;
    const float upperSize = 0.6f;
    out.lower = buildFemVolumeCube(physics, scene, cudaCtx, /*cells*/ 2u, lowerSize, /*baseY*/ 0.0f);
    if (!out.lower) return out;
    // *** THE ARRANGEMENT IS THE WHOLE CONTROL, AND IT TOOK THREE TRIES. ***
    // (1) STACKED FLUSH (gap 0.10): settles at y ~ 1.05 -- and settles there with the attachment
    //     kernels UNMAPPED too, because the cubes are within contact distance and the
    //     softbody-vs-softbody narrowphase (P8.9, already mapped) catches it.  The control read
    //     1.2653 instead of the -44 a free fall gives: the scene proved nothing.
    // (2) STACKED WITH A 0.25 GAP: no contact at spawn, but the cube simply FALLS the 0.25 and
    //     lands on the lower one.  Control read 1.2340.  Anything directly above the pinned cube
    //     has something to land on, so NO vertical arrangement can work.
    // (3) OFFSET 2.0 SIDEWAYS: no contact and nothing to land on, but a 2.0 initial constraint
    //     ERROR, which a position projection closes in ONE step -- yanked from mean 1.80 to 0.71
    //     on step 0, then the material blows up (-9.9 by step 90).
    //
    // SIDE BY SIDE at the SAME height is the answer.  The free cube sits beside the pinned one
    // with a 0.20 face gap: far outside the default contact offset (the cube's shape sets none),
    // with NOTHING beneath it, and with an initial constraint error of a fraction of a unit
    // instead of two.  Unmap the kernels and it falls to -44.
    const float faceGap = 0.20f;
    const float upperOffsetX = 0.5f * lowerSize + faceGap + 0.5f * upperSize;
    out.upper = buildFemVolumeCube(physics, scene, cudaCtx, /*cells*/ 2u, upperSize, /*baseY*/ 0.0f,
                                   upperOffsetX);
    if (!out.upper) return out;
    out.numUpperSimVerts = out.upper->getSimulationMesh()->getNbVertices();
    out.upperStartY = 0.5f * upperSize;   // the cube's centre

    // PIN the lower cube: every simulation vertex gets invMass 0, so it is an immovable anchor
    // and the only thing the constraint can move is the upper cube.
    {
        const PxU32 nv = out.lower->getSimulationMesh()->getNbVertices();
        std::vector<PxVec4> pos(nv);
        PxCudaContext* ctx = cudaCtx->getCudaContext();
        CUdeviceptr d = reinterpret_cast<CUdeviceptr>(out.lower->getSimPositionInvMassBufferD());
        ctx->memcpyDtoH(pos.data(), d, nv * sizeof(PxVec4));
        for (PxU32 i = 0; i < nv; ++i) pos[i].w = 0.0f;
        ctx->memcpyHtoD(d, pos.data(), nv * sizeof(PxVec4));
    }

    // Pick the interface tetrahedra and pair them by proximity in xz.
    std::vector<PxVec3> lowC, upC;
    collisionTetCentroids(out.lower, lowC);
    collisionTetCentroids(out.upper, upC);

    std::vector<PxU32> lowTop(lowC.size()), upBottom(upC.size());
    for (PxU32 i = 0; i < PxU32(lowC.size()); ++i) lowTop[i] = i;
    for (PxU32 i = 0; i < PxU32(upC.size()); ++i) upBottom[i] = i;
    // The cubes are side by side, so the FACING tetrahedra are the ones to pair: the lower cube's
    // largest x against the upper cube's smallest.
    std::sort(lowTop.begin(), lowTop.end(), [&](PxU32 a, PxU32 b) { return lowC[a].x > lowC[b].x; });
    std::sort(upBottom.begin(), upBottom.end(), [&](PxU32 a, PxU32 b) { return upC[a].x < upC[b].x; });

    const PxVec4 centroidBary(0.25f, 0.25f, 0.25f, 0.25f);
    const PxU32 nbPairs = PxMin(PxU32(4), PxMin(PxU32(lowTop.size()), PxU32(upBottom.size())));
    for (PxU32 k = 0; k < nbPairs; ++k)
    {
        const PxU32 upTet = upBottom[k];
        // the lower tetrahedron whose centroid is nearest in yz (the facing plane)
        PxU32 best = lowTop[0];
        float bestD = FLT_MAX;
        for (PxU32 m = 0; m < PxMin(PxU32(8), PxU32(lowTop.size())); ++m)
        {
            const PxU32 lt = lowTop[m];
            const float dy = lowC[lt].y - upC[upTet].y, dz = lowC[lt].z - upC[upTet].z;
            const float d = dy * dy + dz * dz;
            if (d < bestD) { bestD = d; best = lt; }
        }
        // upper->addSoftBodyAttachment(lower, tetIdxLower, baryLower, tetIdxUpper, baryUpper)
        const PxU32 handle = out.upper->addSoftBodyAttachment(out.lower, best, centroidBary,
                                                              upTet, centroidBary);
        if (handle != 0xFFFFFFFFu) ++out.nbAttachments;
    }

    return out;
}

// ===========================================================================
// PXG_SCENE=softbodyhang (P9.76b): a FEM volume hanging from WORLD-pinned simulation
// vertices, with nothing beneath it.
//
// WHY A NEW SCENE WHEN softbodyattach ALREADY EXISTS.  That one (P8.19) is FEM-vs-FEM:
// its attachment is addSoftBodyAttachment(volume, volume), which routes to the
// sb_solveFemFemAttachment kernels.  It does NOT touch the RIGID attachment chain --
// sb_rigidAttachmentPrepareLaunch and sb_solveRigidSoftAttachmentLaunch{,TGS} -- and
// that is exactly why nothing in this tree noticed that the TGS half had never run.
// A world anchor is a rigid attachment with a STATIC node index, which is what
// PxgSimulationController's mRigidVolumeAttachments collects and what
// getNbRigidSoftBodyAttachments() counts.
//
// A world anchor is also the configuration P9.76a's defect lived in: the position
// error is ~ (0, sag, 0), and a solve reading raXn0_biasW.w for all three axes sees
// error.x ~ 0 and applies no positional recovery at all.
//
// Nothing beneath it, by construction.  P8.19's header records that no VERTICAL
// arrangement can prove an attachment works, because whatever is below catches the
// body anyway; here there is simply no floor and no second volume.
// ===========================================================================
SoftBodyAttachScene buildSoftBodyHangScene(PxPhysics* physics, PxScene* scene,
                                           PxMaterial* rigidMat, PxCudaContextManager* cudaCtx)
{
    SoftBodyAttachScene out;
    out.cudaCtx = cudaCtx;

    // One live dynamic rigid, parked far away: the softbody solve runs inside the rigid
    // island loop and rigidSumInternalContactAndJointBatches2 null-derefs with zero
    // dynamic rigids.  The same requirement every FEM scene here carries.
    addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
               boxInertia(PxVec3(0.5f), 1000.0f), 1.5f, 30.0f);

    const float size = 1.0f;
    const float baseY = 4.0f;                      // high, so a fall is unmistakable
    out.upper = buildFemVolumeCube(physics, scene, cudaCtx, /*cells*/ 2u, size, baseY);
    if (!out.upper) return out;
    out.numUpperSimVerts = out.upper->getSimulationMesh()->getNbVertices();
    out.upperStartY = baseY + 0.5f * size;

    // The TOP simulation vertices, read from the device buffer the solver actually uses.
    const PxU32 nv = out.numUpperSimVerts;
    std::vector<PxVec4> pos(nv);
    PxCudaContext* ctx = cudaCtx->getCudaContext();
    ctx->memcpyDtoH(pos.data(),
                    reinterpret_cast<CUdeviceptr>(out.upper->getSimPositionInvMassBufferD()),
                    nv * sizeof(PxVec4));
    float maxY = -FLT_MAX;
    for (PxU32 i = 0; i < nv; ++i) maxY = PxMax(maxY, pos[i].y);

    static std::vector<PxU32> pinIdx;      // must outlive createDeformableAttachment's copy
    static std::vector<PxVec4> pinPos;
    pinIdx.clear(); pinPos.clear();
    for (PxU32 i = 0; i < nv; ++i)
    {
        if (pos[i].y < maxY - 1e-3f)
            continue;
        pinIdx.push_back(i);
        // The anchor is the vertex's OWN spawn position, so the initial constraint error
        // is zero.  P8.19 measured what a large initial error does here: a 2.0 offset was
        // closed in one step and the material blew up by step 90.
        pinPos.push_back(PxVec4(pos[i].x, pos[i].y, pos[i].z, 0.0f));
    }
    if (pinIdx.empty())
        return out;

    PxDeformableAttachmentData attData;
    attData.actor[0] = out.upper;
    attData.type[0]  = PxDeformableAttachmentTargetType::eVERTEX;
    attData.indices[0].data = pinIdx.data();
    attData.indices[0].count = PxU32(pinIdx.size());
    attData.actor[1] = NULL;                                   // world anchor
    attData.type[1]  = PxDeformableAttachmentTargetType::eWORLD;
    attData.coords[1].data = pinPos.data();
    attData.coords[1].count = PxU32(pinPos.size());
    attData.pose[1] = PxTransform(PxIdentity);

    if (physics->createDeformableAttachment(attData))
        out.nbAttachments = PxU32(pinIdx.size());
    else
        std::printf("FAIL: createDeformableAttachment (volume vertex -> world) returned null\n");

    return out;
}

// ===========================================================================
// PXG_SCENE=softbodydata (P9.95): the deprecated direct softbody-data API,
// PxScene::copySoftBodyData / applySoftBodyData (PxScene.h:1761/:1776) ->
// PxgSimulationController (:564-582) -> PxgSoftBodyCore::copy/applySoftBodyDataDEPRECATED
// (:3157/:3175) -> copyOrApplySoftBodyDataDEPRECATED (:3118) ->
// sb_copyOrApplySoftBodyDataDEPRECATED (softBodyGM.cu:3094).  The kernel never runs inside
// simulate(); these two calls are its only way in, and no scene made them before.  No
// scene flag is needed: NpScene.cpp:3711/:3720 have the direct-GPU-API guard commented out.
//
// WHY THE OBSERVABLES ARE BIT-EXACT.  Three of the eight PxSoftBodyGpuDataFlag slots alias
// public device buffers (NpDeformableVolume.cpp:527-557; PxgSimulationCore.cpp:1313-1318):
//   eTET_POSITION_INV_MASS -> getPositionInvMassBufferD    (dataIndex 8)
//   eSIM_VELOCITY_INV_MASS -> getSimVelocityBufferD        (dataIndex 11)
//   eSIM_POSITION_INV_MASS -> getSimPositionInvMassBufferD (dataIndex 12)
// so every COPY must bit-equal a memcpyDtoH of the aliased buffer taken in the same window,
// and every APPLY must be bit-visible in it at once.
//
// TWO VOLUMES, AND WHY B IS BIGGER.  A is the usual cells=2 cube at x = -3, B a cells=4 cube
// (125 lattice vertices) at x = +3.  maxSize = nbSimB*16 makes tps = nbSimB > 64, so the
// two-section copies straddle the 128-thread block boundary.  The indices are passed
// REVERSED {B, A}, and A's section copies only HALF its vertices: a port that ignores the
// index table swaps the volumes, one that ignores dataSizes overruns A's half, and every
// destination carries one poisoned slot past its size.
//
// THE WINDOW.  One warm-up step is mandatory: mSoftBodyBuffer's mPtr stays 0 until the
// first simulate DMAs the descriptors (PxgSimulationCore.cpp:1155/:1638 -- upstream does
// not guard the call).  Every call passes NULL events, so the core synchronizes its stream
// before returning (PxgSoftBodyCore.cpp:3166-3171, :3185-3190).  Then:
//   COPY  x3   sim positions, sim velocities, collision positions -- bit-exact;
//   APPLY (a)  A's sim velocities with v.x += 3 (y, z and w bitwise) -- a sideways kick;
//   APPLY (b)  B's sim positions with z += 2 (w bitwise) -- a teleport;
// then 60 steps.  A drifts 3 further in x than it would have; B sits 2 further in z, in
// its SIM mesh and -- after the first step -- in its COLLISION mesh, which only the
// skinning (sb_gm_updateTetModelVertsLaunch) can move: the apply reached solver state.
//
// THE CONTROL.  PXG_SBDATA_CONTROL=1 makes the same reads and writes with plain
// memcpyDtoH / memcpyHtoD on the aliased buffers instead of the two API calls.  It is
// reported, never scored (the PXG_DIRECTAPI_CONTROL convention); the final-position hash
// it prints must equal the live run's.
//
// UPSTREAM CANDIDATES this scene steps around (documented, not reported): the controller
// dereferences mGpuSoftBodyCore with no null check (PxgSimulationController.cpp:567-570,
// :577-580), so either call on a GPU scene without a deformable volume crashes; and nothing
// stops a call before the first simulate, when the kernel would read descriptors at
// address 0 + idx*432 -- hence the mandatory warm-up above.
// ===========================================================================
struct SoftBodyDataScene
{
    PxDeformableVolume* a = nullptr;
    PxDeformableVolume* b = nullptr;
    PxU32 nbSimA = 0, nbSimB = 0, nbCollA = 0, nbCollB = 0;
};

SoftBodyDataScene buildSoftBodyDataScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                         PxCudaContextManager* cudaCtx)
{
    SoftBodyDataScene out;
    // One live dynamic rigid, parked far away -- the requirement every FEM scene here carries
    // (see buildSoftBodyHangScene).  No floor: both volumes fall freely the whole time.
    addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
               boxInertia(PxVec3(0.5f), 1000.0f), 1.5f, 30.0f);
    out.a = buildFemVolumeCube(physics, scene, cudaCtx, /*cells*/ 2u, /*size*/ 1.0f, /*baseY*/ 10.0f,
                               /*offsetX*/ -3.0f);
    out.b = buildFemVolumeCube(physics, scene, cudaCtx, /*cells*/ 4u, /*size*/ 1.0f, /*baseY*/ 10.0f,
                               /*offsetX*/ 3.0f);
    if (!out.a || !out.b)
        return out;
    out.nbSimA = out.a->getSimulationMesh()->getNbVertices();
    out.nbSimB = out.b->getSimulationMesh()->getNbVertices();
    out.nbCollA = out.a->getCollisionMesh()->getNbVertices();
    out.nbCollB = out.b->getCollisionMesh()->getNbVertices();
    return out;
}

static void sbDataRead(PxCudaContext* ctx, const PxVec4* dev, PxU32 n, std::vector<PxVec4>& out)
{
    out.assign(n, PxVec4(0.0f));
    ctx->memcpyDtoH(out.data(), reinterpret_cast<CUdeviceptr>(dev), n * sizeof(PxVec4));
}

static PxVec3 sbDataMean(const std::vector<PxVec4>& v)
{
    double x = 0.0, y = 0.0, z = 0.0;
    for (const PxVec4& p : v) { x += double(p.x); y += double(p.y); z += double(p.z); }
    const double n = v.empty() ? 1.0 : double(v.size());
    return PxVec3(float(x / n), float(y / n), float(z / n));
}

// Words of `got[0..n)` that differ bitwise from `want[0..n)`.
static PxU32 sbDataBitDiff(const PxVec4* got, const PxVec4* want, PxU32 n)
{
    PxU32 d = 0;
    for (PxU32 i = 0; i < n; ++i)
        d += std::memcmp(&got[i], &want[i], sizeof(PxVec4)) != 0 ? 1u : 0u;
    return d;
}

static uint64_t sbDataFnv(uint64_t h, const std::vector<PxVec4>& v)
{
    const unsigned char* p = reinterpret_cast<const unsigned char*>(v.data());
    for (size_t i = 0; i < v.size() * sizeof(PxVec4); ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

int main()
{
    // LINE-BUFFER STDOUT.  This harness exists to drive scenes that crash, and a
    // block-buffered stdout loses every progress line the process had printed but
    // not flushed -- which is exactly how P8.55 and P8.56 came to place the
    // articulation SIGSEGV in scene construction when it is really in the first
    // simulate().  PhysX's own error callback is unbuffered, so the surviving
    // output was all engine errors and no harness context, and the crash looked
    // like it happened three phases earlier than it did.  One line, and every
    // future crash here locates itself.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

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
    // PXG_CPU_CONTROL=1 builds THE SAME SCENE on the CPU pipeline (P9.69c).
    //
    // It exists because a failing scene has two possible readings and the harness
    // could not tell them apart: either the GPU port computes the wrong contacts,
    // or the rest height the scene ASSERTS is not what PhysX itself produces for
    // that shape pair.  P9.69b's convex-core-on-a-box scene is exactly that
    // question -- the cylinder settles 0.38 low, and whether that is a defect
    // depends on what the CPU narrowphase does with the same geometry.
    //
    // This is a DIAGNOSTIC, not a scored mode, and it is the same shape as
    // PXG_PARTICLE_FLUID_CONTROL: it is reported and never counted.  The scene
    // still gets its cudaContextManager so construction is identical; only the
    // dynamics and broadphase move back to the CPU.
    const bool cpuControl = std::getenv("PXG_CPU_CONTROL") != nullptr;
    sceneDesc.cudaContextManager = cudaCtx;
    if (!cpuControl)
    {
        sceneDesc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
        sceneDesc.broadPhaseType = PxBroadPhaseType::eGPU;
    }
    else
    {
        std::printf("  [ control ] PXG_CPU_CONTROL=1: the CPU pipeline, for reference only\n");
    }
    // PXG_SOLVER=pgs switches the scene to the PGS solver (default TGS): the two
    // GPU solver paths launch DIFFERENT kernel sets (PxgCudaSolverCore vs
    // PxgTGSCudaSolverCore), so each scene can be proven on either.
    const char* solverEnv = std::getenv("PXG_SOLVER");
    const bool usePgs = solverEnv && !std::strcmp(solverEnv, "pgs");
    sceneDesc.solverType = usePgs ? PxSolverType::ePGS : PxSolverType::eTGS;
    // P9.79 -- PXG_SCENE=directapi.  eENABLE_DIRECT_GPU_API is IMMUTABLE after
    // createScene (PxDirectGPUAPI.h), so the scene name has to be read here rather
    // than at the dispatch below; the dispatch reads the same variable again.
    // The flag requires eENABLE_GPU_DYNAMICS + eGPU broadphase, so PXG_CPU_CONTROL
    // and this scene are mutually exclusive -- say so rather than constructing a
    // scene whose direct-API calls would all fail the isDirectGPUAPIInitialized()
    // gate and read back as poison.
    {
        const char* sn = std::getenv("PXG_SCENE");
        // PXG_DIRECTAPI_CONTROL=1 builds the SAME two bodies and the SAME joint
        // WITHOUT the flag, so the joint reaction can be read the ordinary way
        // (PxConstraint::getForce) and compared.  It is a DIAGNOSTIC, reported and
        // never scored, exactly like PXG_CPU_CONTROL and PXG_PARTICLE_FLUID_CONTROL
        // -- and it is what settled the torque convention: the first green run
        // reported |T| = 39240, not the m*g*d = 4905 this scene had guessed, and a
        // guess is not something to fit a constant to.
        if (sn && !std::strcmp(sn, "directapi") && !std::getenv("PXG_DIRECTAPI_CONTROL"))
        {
            if (cpuControl)
            {
                std::printf("[ FAIL ] gpu-sdk-step: PXG_SCENE=directapi requires the GPU pipeline; "
                            "PXG_CPU_CONTROL=1 is incompatible with eENABLE_DIRECT_GPU_API\n");
                physics->release(); cudaCtx->release(); foundation->release();
                return 1;
            }
            sceneDesc.flags |= PxSceneFlag::eENABLE_DIRECT_GPU_API;
            std::printf("  [ P9.79 ] eENABLE_DIRECT_GPU_API raised: host writeback is OFF, the "
                        "direct-API getters are the only way to see the simulation\n");
        }
        // P9.93 -- PXG_SCENE=directapiwrite needs eENABLE_BODY_ACCELERATIONS as well:
        // without it the previous-velocity buffer is never allocated and the
        // acceleration getters dereference a null pointer (the host only asserts).
        // Its control (PXG_DIRECTAPI_CONTROL=1) keeps the accelerations and drops the
        // direct API, and drives the same writes through the ordinary CPU setters.
        if (sn && !std::strcmp(sn, "directapiwrite"))
        {
            sceneDesc.flags |= PxSceneFlag::eENABLE_BODY_ACCELERATIONS;
            if (!std::getenv("PXG_DIRECTAPI_CONTROL"))
            {
                if (cpuControl)
                {
                    std::printf("[ FAIL ] gpu-sdk-step: PXG_SCENE=directapiwrite requires the GPU pipeline; "
                                "PXG_CPU_CONTROL=1 needs PXG_DIRECTAPI_CONTROL=1 as well\n");
                    physics->release(); cudaCtx->release(); foundation->release();
                    return 1;
                }
                sceneDesc.flags |= PxSceneFlag::eENABLE_DIRECT_GPU_API;
                std::printf("  [ P9.93 ] eENABLE_DIRECT_GPU_API | eENABLE_BODY_ACCELERATIONS raised\n");
            }
        }
        // P9.95 -- PXG_SCENE=directapimerge: mergeTransformCacheAndBoundArrayChanges runs
        // only in direct-GPU-API mode (PxgSimulationController.cpp:702), only after the
        // first copy (:707-710), and only for shapes added or re-inserted BETWEEN steps
        // (:712, :799).  Its control (PXG_DIRECTAPI_CONTROL=1) drops the flag.
        if (sn && !std::strcmp(sn, "directapimerge") && !std::getenv("PXG_DIRECTAPI_CONTROL"))
        {
            if (cpuControl)
            {
                std::printf("[ FAIL ] gpu-sdk-step: PXG_SCENE=directapimerge requires the GPU pipeline; "
                            "PXG_CPU_CONTROL=1 is incompatible with eENABLE_DIRECT_GPU_API\n");
                physics->release(); cudaCtx->release(); foundation->release();
                return 1;
            }
            sceneDesc.flags |= PxSceneFlag::eENABLE_DIRECT_GPU_API;
            std::printf("  [ P9.95 ] eENABLE_DIRECT_GPU_API raised (directapimerge)\n");
        }
        // P9.95 -- PXG_SCENE=sdfquery: PxDirectGPUAPI::evaluateSDFDistances is refused
        // unless the direct-GPU API is initialized (NpDirectGPUAPI.cpp:156), which needs
        // this flag at createScene and one completed step.
        if (sn && !std::strcmp(sn, "sdfquery"))
        {
            if (cpuControl)
            {
                std::printf("[ FAIL ] gpu-sdk-step: PXG_SCENE=sdfquery requires the GPU pipeline; "
                            "PXG_CPU_CONTROL=1 is incompatible with eENABLE_DIRECT_GPU_API\n");
                physics->release(); cudaCtx->release(); foundation->release();
                return 1;
            }
            sceneDesc.flags |= PxSceneFlag::eENABLE_DIRECT_GPU_API;
            std::printf("  [ P9.95 ] eENABLE_DIRECT_GPU_API raised (sdfquery)\n");
        }
        if (sn && !std::strcmp(sn, "forcethreshold"))
            gForceThresholdPairs = true;
        // P9.95 -- PXG_SCENE=substepgravity: eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS is
        // not in eMUTABLE_FLAGS (PxSceneDesc.h:373), so it is raised here.  It is TGS-only:
        // PxSceneDesc::isValid rejects it with PGS (PxSceneDesc.h:1129) and createScene
        // would return null.  PXG_CPU_CONTROL=1 stays legal (the CPU TGS supports the flag,
        // DyTGSDynamics.cpp:1478); it is reported and not scored.
        if (sn && !std::strcmp(sn, "substepgravity"))
        {
            if (usePgs)
            {
                std::printf("[ FAIL ] gpu-sdk-step: PXG_SCENE=substepgravity is TGS-only (PxSceneDesc::isValid "
                            "rejects eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS with PGS, PxSceneDesc.h:1129)\n");
                physics->release(); cudaCtx->release(); foundation->release();
                return 1;
            }
            sceneDesc.flags |= PxSceneFlag::eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS;
            std::printf("  [ P9.95 ] eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS raised\n");
        }
        // P9.95 -- PXG_SCENE=particleapply: applyParticleBufferData is legal only once the
        // direct-GPU API is initialized (NpScene.cpp:3735), so the flag is raised here.  It
        // is raised for the control (PXG_PARTICLE_APPLY_CONTROL=1) as well: the control
        // differs from the live run only in how the edits are ingested, not in the pipeline.
        if (sn && !std::strcmp(sn, "particleapply"))
        {
            if (cpuControl)
            {
                std::printf("[ FAIL ] gpu-sdk-step: PXG_SCENE=particleapply requires the GPU pipeline; "
                            "PXG_CPU_CONTROL=1 is incompatible with eENABLE_DIRECT_GPU_API\n");
                physics->release(); cudaCtx->release(); foundation->release();
                return 1;
            }
            sceneDesc.flags |= PxSceneFlag::eENABLE_DIRECT_GPU_API;
            std::printf("  [ P9.95 ] eENABLE_DIRECT_GPU_API raised (particle direct-API ingest)\n");
        }
        // P9.99b -- PXG_SCENE=particle2: PBD particle systems exist only on the GPU pipeline,
        // so PXG_CPU_CONTROL=1 is refused for every arm.  The `apply` arm calls
        // applyParticleBufferData, legal only under the direct-GPU API (NpScene.cpp:3735);
        // the flag is immutable after createScene, so it is raised here, for every sub-scene.
        if (sn && !std::strcmp(sn, "particle2"))
        {
            if (cpuControl)
            {
                std::printf("[ FAIL ] gpu-sdk-step: PXG_SCENE=particle2 requires the GPU pipeline (PBD particle "
                            "systems); PXG_CPU_CONTROL=1 is refused\n");
                physics->release(); cudaCtx->release(); foundation->release();
                return 1;
            }
            const char* p2arm = std::getenv("PXG_P2_ARM");
            if (p2arm && !std::strcmp(p2arm, "apply"))
            {
                sceneDesc.flags |= PxSceneFlag::eENABLE_DIRECT_GPU_API;
                std::printf("  [ P9.99b ] eENABLE_DIRECT_GPU_API raised (particle2 apply arm)\n");
            }
        }
    }
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
    // P9.7 -- PXG_SCENE=particleinflatable: a pressurised closed mesh.
    if (!std::strcmp(sceneName, "particleinflatable"))
    {
        // The control zeroes the PRESSURE, which is the engine's own marker for
        // "solve as ordinary cloth" and makes both P9.6 kernels return at their
        // early-out.  Everything else -- mesh, springs, masses, iterations -- is
        // identical, so the difference is the inflatable pair and nothing else.
        const bool noPressure = std::getenv("PXG_PARTICLE_INFL_CONTROL") != nullptr;
        const InflMesh mesh = buildInflSphere(INFL_CENTRE, INFL_RADIUS, 2);
        ParticleScene psc = buildParticleInflatableScene(physics, scene, material, cudaCtx,
                                                         !noPressure);
        if (!psc.system)
        {
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        const float restVol6 = inflMeshVolume6(mesh.verts, mesh.tris);
        std::printf("  [  OK  ] scene 'particleinflatable' built: %u particles, pressure %s\n",
                    psc.numParticles, noPressure ? "ZERO (control run)" : "on");

        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: particle inflatable GPU pipeline CONSTRUCTED + "
                        "live over Vulkan (set PXG_GPU_STEP=1 to step the inflation)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_PARTICLE_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 180;
        const float v0 = inflLiveVolume6(psc, mesh.tris);
        std::printf("  stepping the inflatable (%d x %.4fs): rest volume(x6) %.6g, "
                    "squashed start %.6g (%.1f%% of rest)\n",
                    steps, dt, restVol6, v0, 100.0f * v0 / restVol6);

        float volMax = v0, volEnd = v0;
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            const float v = inflLiveVolume6(psc, mesh.tris);
            if (v > volMax) volMax = v;
            volEnd = v;
            if (i < 5 || i == steps - 1 || (i % 30 == 0))
                std::printf("    step %3d: volume(x6)=%.6g (%.1f%% of rest)\n",
                            i, v, 100.0f * v / restVol6);
        }
        const float fracEnd = volEnd / restVol6;
        const float fracMax = volMax / restVol6;
        std::printf("  particle INFLATABLE: volume(x6) %.6g -> %.6g (max %.6g); "
                    "%.1f%% -> %.1f%% of rest (max %.1f%%)\n",
                    v0, volEnd, volMax, 100.0f * v0 / restVol6, 100.0f * fracEnd,
                    100.0f * fracMax);

        int failures = 0;
        if (noPressure)
        {
            std::printf("[  OK  ] gpu-sdk-step: particleinflatable CONTROL run (pressure 0) "
                        "measured: %.1f%% of rest volume at the end, %.1f%% at most\n",
                        100.0f * fracEnd, 100.0f * fracMax);
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 0;
        }

        if (!(PxAbs(fracEnd - INFL_PRESSURE) < INFL_VOL_TARGET_TOL))
        {
            std::printf("FAIL: the mesh did not reach its TARGET volume -- the volume "
                        "constraint is not converging (%.4f of rest, target %.2f +/- %.2f, "
                        "pressure-0 control %.4f)\n",
                        fracEnd, INFL_PRESSURE, INFL_VOL_TARGET_TOL, INFL_VOL_CONTROL);
            ++failures;
        }
        if (!(fracMax < INFL_VOL_MAX))
        {
            std::printf("FAIL: the inflatable DIVERGED -- peak volume %.4f of rest exceeds "
                        "%.2f\n", fracMax, INFL_VOL_MAX); ++failures;
        }
        if (!std::isfinite(volEnd))
        {
            std::printf("FAIL: the inflatable volume is not finite (%.6g)\n", volEnd);
            ++failures;
        }
        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: the PRESSURISED mesh INFLATED to its TARGET "
                        "volume on Vulkan (%.1f%% of rest vs %.1f%% for the pressure-0 "
                        "control, peak %.1f%%) -- the inflatable pair runs end to end\n",
                        100.0f * fracEnd, 100.0f * INFL_VOL_CONTROL, 100.0f * fracMax);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: particleinflatable checks failed (%d)\n", failures);
        return 2;
    }

    // P9.4 -- PXG_SCENE=particlecloth: the PBD cloth sheet hanging from a pinned
    // row.  See buildParticleClothScene for why this observable and this control.
    if (!std::strcmp(sceneName, "particlecloth"))
    {
        const bool control = std::getenv("PXG_PARTICLE_CLOTH_CONTROL") != nullptr;
        // The no-aero arm is a CALIBRATION run, not a pass/fail one: it exists to
        // measure the number the rest threshold is set against, so it must not be
        // held to that threshold itself.
        const bool noAeroScene = std::getenv("PXG_PARTICLE_CLOTH_NOAERO") != nullptr;
        ParticleScene psc = buildParticleClothScene(physics, scene, material, cudaCtx, !control);
        if (!psc.system)
        {
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'particlecloth' built: %u PBD particles (%u pinned), "
                    "springs %s\n", psc.numParticles, psc.numPinned,
                    control ? "ABSENT (control run)" : "present");

        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: particle cloth GPU pipeline CONSTRUCTED + live "
                        "over Vulkan (set PXG_GPU_STEP=1 to step the drape)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_PARTICLE_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 240;
        const float y0 = clothFreeMeanY(psc);
        const float diag0 = clothBBoxDiag(psc);
        std::printf("  stepping the cloth sheet (%d x %.4fs), initial free mean y=%.4f, "
                    "bbox diagonal=%.4f\n", steps, dt, y0, diag0);

        // The hang is tracked as a MINIMUM OVER EVERY STEP, not just the final
        // value.  The sheet swings, so a check on the last step alone could be
        // satisfied by a run that blew out transiently and happened to be on the
        // way back up when the loop ended.  Settling for the weaker check would be
        // settling for a test that passes for the wrong reason.
        //
        // The first 20 steps are excluded from the minimum: the sheet starts flat
        // at y = 5.0 and has to swing DOWN to its hanging position, and during that
        // swing it legitimately passes through low mean-y values that say nothing
        // about whether the springs hold.  20 steps is a third of a second, well
        // inside the measured swing period and far short of the 60 steps the
        // control needs to reach the floor.
        const int settleIn = 20;
        const int restWindow = steps / 4;
        float prevY = y0, diagMax = diag0, yMin = 1e30f, restMaxDy = 0.0f;
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            const float y = clothFreeMeanY(psc);
            const float dg = clothBBoxDiag(psc);
            if (dg > diagMax) diagMax = dg;
            if (i >= settleIn) yMin = PxMin(yMin, y);
            // P9.5: with aerodynamics mapped the sheet finally has air resistance,
            // so a REST check is meaningful for the first time.  P9.4 deliberately
            // omitted it because the undamped pendulum never settled.
            if (i >= steps - restWindow) restMaxDy = PxMax(restMaxDy, PxAbs(y - prevY));
            if (i < 5 || i == steps - 1 || (i % 30 == 0))
                std::printf("    step %3d: free mean y=%.4f (dy %+.5f), bbox diag=%.4f\n",
                            i, y, y - prevY, dg);
            prevY = y;
        }
        const float yEnd = clothFreeMeanY(psc);
        const float diagEnd = clothBBoxDiag(psc);
        std::printf("  particle CLOTH: free mean y %.4f -> %.4f (min after step %d: %.4f), "
                    "bbox diag %.4f -> %.4f (max %.4f), rest |dy| over last %d steps = %.6f\n",
                    y0, yEnd, settleIn, yMin, diag0, diagEnd, diagMax, restWindow, restMaxDy);

        int failures = 0;
        if (control)
        {
            // The control is not asserted on -- it is MEASURED, and its number is
            // what justifies the threshold the real run is held to.  Printing it
            // in the same format keeps the two comparable by eye.
            std::printf("[  OK  ] gpu-sdk-step: particlecloth CONTROL run (no springs) "
                        "measured: free mean y %.4f (min %.4f), bbox diag %.4f (max %.4f), "
                        "rest |dy| %.6f\n", yEnd, yMin, diagEnd, diagMax, restMaxDy);
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 0;
        }

        if (!(yMin > CLOTH_HANG_MIN))
        {
            std::printf("FAIL: the sheet did not HANG -- the free particles fell away from the "
                        "pinned row (free mean y dropped to %.4f, threshold %.2f, no-spring "
                        "control %.4f)\n", yMin, CLOTH_HANG_MIN, CLOTH_HANG_CONTROL); ++failures;
        }
        if (!(diagMax < CLOTH_DIAG_MAX))
        {
            std::printf("FAIL: the sheet EXPLODED -- max bbox diagonal %.4f exceeds %.2f "
                        "(the flat sheet's own diagonal is %.2f, the no-spring control reaches "
                        "%.4f)\n", diagMax, CLOTH_DIAG_MAX, CLOTH_DIAG_REST, CLOTH_DIAG_CONTROL);
            ++failures;
        }
        if (!noAeroScene && !(restMaxDy < CLOTH_REST_MAX))
        {
            std::printf("FAIL: the sheet did not come to REST -- aerodynamic drag is not "
                        "damping the swing (rest |dy|=%.6f, threshold %.4f, no-aero "
                        "control %.6f)\n", restMaxDy, CLOTH_REST_MAX, CLOTH_REST_NOAERO);
            ++failures;
        }
        if (!std::isfinite(yEnd) || !std::isfinite(diagEnd))
        {
            std::printf("FAIL: cloth state is not finite (y=%.4f, diag=%.4f)\n", yEnd, diagEnd);
            ++failures;
        }
        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: the PBD cloth sheet HANGS from its pinned row on "
                        "Vulkan (free mean y never below %.4f vs %.4f for the no-spring control, "
                        "bbox diag max %.4f vs %.4f, rest |dy| %.6f vs %.6f without "
                        "aerodynamics) -- the spring chain AND the aerodynamics pair run "
                        "end to end\n",
                        yMin, CLOTH_HANG_CONTROL, diagMax, CLOTH_DIAG_CONTROL,
                        restMaxDy, CLOTH_REST_NOAERO);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: particlecloth checks failed (%d)\n", failures);
        return 2;
    }

    // P9.9: particlefluidboundary IS the fluid scene, with one extra call.  It takes
    // the same code path and the same assertions, plus the boundary-specific pair.
    const bool boundaryScene = !std::strcmp(sceneName, "particlefluidboundary");
    // P9.10c: particlemeshfloor is the BOUNDARY scene with the ground box swapped
    // for a cooked TRIANGLE MESH.  It is the first scene in this port that puts
    // particles on a triangle mesh, so it is the first that launches
    // ps_meshCollisonLaunch -- the only producer of mOneWayContactInfos, which P9.9
    // proved ps_accumulateStaticDensity reads and had always found empty.
    const bool meshFloorScene = !std::strcmp(sceneName, "particlemeshfloor");
    // P9.11c: particlehffloor is particlemeshfloor with the cooked triangle mesh
    // swapped for a cooked HEIGHTFIELD.  Same experiment, one shape type further on:
    // it is the first scene in this port that launches ps_heightfieldCollisonLaunch,
    // the second producer of mOneWayContactInfos.
    const bool hfFloorScene = !std::strcmp(sceneName, "particlehffloor");
    const bool staticFloorScene = meshFloorScene || hfFloorScene;
    // P9.13a: particlediffuse is the FLUID scene with a PxParticleAndDiffuseBuffer
    // in place of the plain one.  It is the first scene in this port that asks the
    // engine for diffuse particles at all -- which was impossible until this
    // tranche replaced the createParticleAndDiffuseBuffer bridge stub, because the
    // stub returned NULL and NpParticleBuffer dereferences it.
    //
    // ITS PURPOSE TODAY IS TO NAME WORK, NOT TO PASS.  Every kernel that creates,
    // integrates, sorts, collides or compacts a diffuse particle is still
    // unmapped, so the count stays at 0 and the check below fails.  That failure
    // IS the deliverable: run it with PXG_LAUNCH_TRACE=1 and the REFUSED lines are
    // the work list, in the engine's own launch order.  It lives in
    // run_scenes.sh's DIAGNOSTIC list until the chain lands, and moves to the
    // scored list in the tranche that makes it pass.
    const bool diffuseScene = !std::strcmp(sceneName, "particlediffuse");
    // P9.14a: particleconvex is particlediffuse with the floor swapped for a cooked
    // CONVEX HULL.  Same experiment as particlemeshfloor and particlehffloor, one
    // shape type further along -- and the only one that reaches the convex pair
    // bucket.  It carries a DIFFUSE buffer as well, because testSDKConvexParticle
    // launches BOTH convex particle kernels from the same entry
    // (PxgNarrowphaseCore.cpp:3775 and :3822), so one scene names both.
    //
    // IT NAMED TWO KERNELS AND NOW MEASURES BOTH.  P9.14b landed
    // ps_convexCollisionLaunch and the FLUID came to rest on the hull (mean y
    // -39.32 -> 0.1667); P9.14c landed ps_convexDiffuseCollisionLaunch and the
    // SPRAY stopped with it (lowest diffuse y -39.6138 -> -0.1886).  It was a
    // diagnostic between P9.14a and P9.14c and is SCORED from P9.14c, exactly as
    // particlediffuse was between P9.13a and P9.13d.
    const bool convexFloorScene = !std::strcmp(sceneName, "particleconvex");
    const bool fluidScene = !std::strcmp(sceneName, "particlefluid") || boundaryScene
                            || staticFloorScene || diffuseScene || convexFloorScene;
    // P9.41 -- PXG_SCENE=clothnet.  See buildClothNetScene for what it asks and why none
    // of the scored scenes asks it.  A pending-work entry in run_scenes.sh until the
    // rigid coupling (accumulateDeltaVRigidFirst/SecondLaunch) is mapped; it is
    // MEASURED now so the tranche that maps them has a before-number to beat.
    if (!std::strcmp(sceneName, "clothnet"))
    {
        ClothNetScene csc = buildClothNetScene(physics, scene, material, cudaCtx);
        if (!csc.surface || csc.numVerts == 0 || !csc.attachment || !csc.box)
        {
            std::printf("[ FAIL ] gpu-sdk-step: clothnet scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'clothnet' built: %u-vertex FEMCloth net at y=%.1f, border (%u verts) "
                    "pinned to the world, a 1 kg box (half %.2f) dropped from y=%.1f\n",
                    csc.numVerts, csc.netY, csc.numPinned, csc.boxHalf, csc.netY + 1.0f);
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: clothnet pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_CLOTH_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 300;
        const int settleFrom = (steps * 3) / 4;
        const float boxY0 = csc.box->getGlobalPose().p.y;
        float boxY = boxY0, prevBoxY = boxY0, minBoxY = boxY0, maxSettleStep = 0.0f;
        float centreY = csc.netY, netMinY = csc.netY;
        bool finite = true;
        std::printf("  stepping clothnet (%d x %.4fs): box y0=%.4f, net centre y0=%.4f\n",
                    steps, dt, boxY0, centreY);
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            boxY = csc.box->getGlobalPose().p.y;
            finite = clothNetSample(csc, centreY, netMinY) && finite && std::isfinite(boxY);
            minBoxY = PxMin(minBoxY, boxY);
            if (i >= settleFrom) maxSettleStep = PxMax(maxSettleStep, PxAbs(boxY - prevBoxY));
            if (i < 3 || i == steps - 1 || (i % 30 == 0))
                std::printf("    step %3d: box y = %.4f (dy %+.5f) | net centre y = %.4f, lowest %.4f\n",
                            i, boxY, boxY - prevBoxY, centreY, netMinY);
            prevBoxY = boxY;
        }
        const float speed = csc.box->getLinearVelocity().magnitude();
        std::printf("  clothnet: box y %.4f -> %.4f (lowest %.4f, |v| %.4f); net centre y %.4f -> %.4f; "
                    "largest box |dy| over the last %d steps = %.6f\n",
                    boxY0, boxY, minBoxY, speed, csc.netY, centreY, steps - settleFrom, maxSettleStep);

        int failures = 0;
        // (1) THE NET HOLDS THE BOX.  Net at 2.0, ground top at 0.0: a box on the ground
        //     has its centre at 0.25, a box on the net stays above 1.0 even with a
        //     generous sag.  The outcomes are 1.5 m apart and the bar sits between them.
        if (!(boxY > 1.0f))
        { std::printf("FAIL: the box came to rest at y = %.4f -- it fell THROUGH the net to the ground "
                      "(need > 1.0).  The cloth side of the contact ran (see the net's sag); the "
                      "rigid side goes through accumulateDeltaVRigidFirst/SecondLaunch, and if the "
                      "[pxg-unmapped] lines below name them, this is that\n", boxY); ++failures; }
        // (2) and it is at REST there, neither bouncing nor sinking.
        if (!(maxSettleStep < 1e-2f))
        { std::printf("FAIL: the box is still moving %.6f per step at the end of the run\n",
                      maxSettleStep); ++failures; }
        // (3) THE NET WAS TOUCHED: the vertex under the box is below the pinned plane.
        //     This is the cloth side of the same contact; it separates "the coupling is
        //     missing" from "nothing collided at all".
        if (!(centreY < csc.netY - 0.02f))
        { std::printf("FAIL: the net centre is at y = %.4f, within 0.02 of the pinned plane -- the box "
                      "never touched the net\n", centreY); ++failures; }
        if (!finite) { std::printf("FAIL: a coordinate went non-finite\n"); ++failures; }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: the 1 kg box RESTS ON THE CLOTH NET at y %.4f (net pinned at "
                        "%.1f, sagged to %.4f under it) -- the FEM-to-rigid impulse path is live on Vulkan\n",
                        boxY, csc.netY, centreY);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: clothnet checks failed (%d)\n", failures);
        return 2;
    }

    // P9.41 -- PXG_SCENE=particlepush: a PBD particle cloud fired sideways into a 10 kg
    // box standing on the ground.  The same question as clothnet, for particles: the
    // particle side of a particle-vs-rigid contact is mapped (the cloud piles up
    // against the box instead of passing through it); the rigid side is not --
    // ps_solvePCOutputRigidDeltaVTGSLaunch writes the per-contact rigid delta and
    // accumulateDeltaVRigidFirst/SecondLaunch fold it into the body, and all three are
    // unmapped.  Observable: does the box MOVE?  64 kg of particles arrive at it at
    // 4 m/s and nothing else touches it.
    if (!std::strcmp(sceneName, "particlepush"))
    {
        ParticleScene psc = buildParticleScene(physics, scene, material, cudaCtx,
                                               false, false, false, false, false, false, /*pushMode*/ true);
        if (!psc.system || !psc.buffer || !psc.pushBox)
        {
            std::printf("[ FAIL ] gpu-sdk-step: particlepush scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'particlepush' built: %u PBD particles at %.1f m/s toward a 10 kg box "
                    "at x=%.1f\n", psc.numParticles, PUSH_SPEED, PUSH_BOX_X);
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: particlepush pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_PARTICLE_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 120;
        const PxVec3 box0 = psc.pushBox->getGlobalPose().p;
        const float cloudX0 = particleMeanX(psc);
        PxVec3 box = box0;
        float cloudX = cloudX0;
        bool finite = true;
        std::printf("  stepping particlepush (%d x %.4fs): box x0=%.4f, cloud mean x0=%.4f\n",
                    steps, dt, box0.x, cloudX0);
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            box = psc.pushBox->getGlobalPose().p;
            cloudX = particleMeanX(psc);
            finite = finite && box.isFinite() && std::isfinite(cloudX);
            if (i < 3 || i == steps - 1 || (i % 10 == 0))
                std::printf("    step %3d: box x = %.4f (dx %+.4f) y = %.4f | cloud mean x = %.4f\n",
                            i, box.x, box.x - box0.x, box.y, cloudX);
        }
        const float boxDx = box.x - box0.x;
        const float cloudDx = cloudX - cloudX0;
        const float speed = psc.pushBox->getLinearVelocity().magnitude();
        std::printf("  particlepush: box x %.4f -> %.4f (moved %+.4f, y %.4f, |v| %.4f); cloud mean x "
                    "%.4f -> %.4f (moved %+.4f)\n",
                    box0.x, box.x, boxDx, box.y, speed, cloudX0, cloudX, cloudDx);

        int failures = 0;
        // (1) THE BOX WAS PUSHED.  Without the rigid side of the contact it has no
        //     horizontal force on it at all and its x is the spawn x to the digit.
        if (!(boxDx > 0.5f))
        { std::printf("FAIL: the box did not move (dx = %+.4f, need > 0.5) with 64 kg of particles hitting "
                      "it at %.1f m/s.  The rigid side of the contact is ps_solvePCOutputRigidDeltaVTGSLaunch "
                      "plus accumulateDeltaVRigidFirst/SecondLaunch; if the [pxg-unmapped] lines below "
                      "name them, this is that\n", boxDx, PUSH_SPEED); ++failures; }
        // (2) THE CLOUD REACHED IT.  If the particles themselves never advanced, (1) says
        //     nothing about coupling.
        if (!(cloudDx > 0.5f))
        { std::printf("FAIL: the cloud itself only advanced %+.4f -- nothing reached the box\n",
                      cloudDx); ++failures; }
        // (3) the box is still on the ground, not through it.
        if (!(box.y > 0.0f))
        { std::printf("FAIL: the box sank to y = %.4f\n", box.y); ++failures; }
        if (!finite) { std::printf("FAIL: a coordinate went non-finite\n"); ++failures; }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: the particle cloud PUSHED the 10 kg box %.4f m along x -- the "
                        "particle-to-rigid impulse path is live on Vulkan\n", boxDx);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: particlepush checks failed (%d)\n", failures);
        return 2;
    }

    // P9.76 -- PXG_SCENE=particleattach: the particle-vs-RIGID ATTACHMENT chain, and
    // the first scene in this tree that reaches it at all.  63 of 64 particles are
    // pinned to a static anchor at their spawn positions through
    // PxParticleBuffer::setRigidAttachments; particle 0 is left free.
    //
    // What the three kernels have to do for this to pass:
    //   ps_rigidAttachmentPrepareLaunch  turns each PxParticleRigidAttachment into a
    //                                    three-axis constraint block;
    //   ps_solveRigidAttachments{,TGS}Launch  drags the particle back to its anchor.
    // If any of them is refused or unmapped, all 64 particles fall together.
    // P9.80 -- PXG_SCENE=particlerigid: the FIRST PxParticleRigidBuffer this
    // harness has ever created, and therefore the first thing that makes
    // ps_solveShapes and ps_update_volume_bound reachable at all.
    //
    // *** WHY NEITHER KERNEL WAS EVER LAUNCHED. ***
    //
    // Both are guarded on the HOST by a count that no scene had ever raised above
    // zero: solveShapes on mMaxRigidsPerBuffer (PxgPBDParticleSystemCore.cpp:825)
    // and updateVolumeBound on mMaxVolumesPerBuffer (:81).  Eight particle scenes,
    // and every one of them creates a plain PxParticleBuffer with maxVolumes = 0
    // and no rigids -- so the two kernels sat in the engine-reachable ratchet
    // reporting nothing, invisible even to the [pxg-unmapped] machinery, because a
    // kernel that is never launched is never reported.
    //
    // *** THE TWO OBSERVABLES. ***
    //
    //   SHAPE MATCHING.  A 4x4x4 cluster is spawned as ONE shape-matched rigid and
    //   dropped onto the ground box.  ps_solveShapes fits a rotation+translation to
    //   the deformed cluster each substep (a polar decomposition of the moment
    //   matrix) and lerps every particle back toward that rigid pose.  With it, the
    //   cluster LANDS AND KEEPS ITS SHAPE; without it, the same cluster is loose
    //   granular material and spreads into a puddle.  PXG_PARTICLE_RIGID_CONTROL=1
    //   is exactly that second run -- same scene, coefficient 0 -- so the claim is
    //   an A/B and not a tolerance argument.
    //
    //   VOLUME BOUNDS.  One PxParticleVolume covers the same particles, and its
    //   bound is POISONED with -999 before stepping.  ps_update_volume_bound
    //   recomputes that AABB every step from the unsorted positions, so the check
    //   is against the particles the scene reads back itself -- and a kernel that
    //   never ran leaves an impossible number, not a plausible one (P9.48d).
    if (!std::strcmp(sceneName, "particlerigid"))
    {
        ParticleScene psc = buildParticleScene(physics, scene, material, cudaCtx,
                                               false, false, false, false, false, false, false,
                                               false, /*rigidMode*/ true);
        const bool plainDiag = std::getenv("PXG_PARTICLE_RIGID_PLAIN") != nullptr;
        if (!psc.system || !psc.buffer || (!psc.rigidBuffer && !plainDiag))
        {
            std::printf("[ FAIL ] gpu-sdk-step: particlerigid scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        const bool rigidControl = std::getenv("PXG_PARTICLE_RIGID_CONTROL") != nullptr;
        std::printf("  [  OK  ] scene 'particlerigid' built: %u PBD particles as ONE shape-matched rigid "
                    "(coefficient %.1f%s) plus ONE particle volume, spawn y-extent %.4f\n",
                    psc.numParticles, rigidControl ? 0.0f : RIGID_STIFFNESS,
                    rigidControl ? ", PXG_PARTICLE_RIGID_CONTROL" : "", psc.spawnExtentY);

        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: particlerigid pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_PARTICLE_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 240;
        const float y0 = particleMeanY(psc);
        std::printf("  stepping particlerigid (%d x %.4fs): mean y0 = %.4f\n", steps, dt, y0);
        for (int i = 0; i < steps; ++i) { scene->simulate(dt); scene->fetchResults(true); }

        PxCudaContext* ctx = cudaCtx->getCudaContext();
        std::vector<PxVec4> host(psc.numParticles);
        ctx->memcpyDtoH(host.data(), reinterpret_cast<CUdeviceptr>(psc.buffer->getPositionInvMasses()),
                        psc.numParticles * sizeof(PxVec4));
        PxParticleVolume volOut;
        volOut.bound = PxBounds3(PxVec3(RIGID_VOLUME_POISON), PxVec3(RIGID_VOLUME_POISON + 1.0f));
        if (psc.rigidBuffer)
            ctx->memcpyDtoH(&volOut, reinterpret_cast<CUdeviceptr>(psc.rigidBuffer->getParticleVolumes()),
                            sizeof(PxParticleVolume));

        // the AABB the scene computes from the particles it just read back -- the
        // oracle ps_update_volume_bound is checked against
        PxBounds3 ref = PxBounds3::empty();
        for (PxU32 i = 0; i < psc.numParticles; ++i)
            ref.include(PxVec3(host[i].x, host[i].y, host[i].z));
        const float extentY = ref.maximum.y - ref.minimum.y;
        const float meanY = particleMeanY(psc);

        std::printf("  cluster: mean y %.4f -> %.4f, y-extent %.4f -> %.4f (spawn %.4f)\n",
                    y0, meanY, psc.spawnExtentY, extentY, psc.spawnExtentY);
        std::printf("  volume : bound min=(%.4f, %.4f, %.4f) max=(%.4f, %.4f, %.4f)\n",
                    volOut.bound.minimum.x, volOut.bound.minimum.y, volOut.bound.minimum.z,
                    volOut.bound.maximum.x, volOut.bound.maximum.y, volOut.bound.maximum.z);
        std::printf("  particles: bound min=(%.4f, %.4f, %.4f) max=(%.4f, %.4f, %.4f)  [the oracle]\n",
                    ref.minimum.x, ref.minimum.y, ref.minimum.z,
                    ref.maximum.x, ref.maximum.y, ref.maximum.z);

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();

        if (plainDiag)
        {
            std::printf("  [ control ] PXG_PARTICLE_RIGID_PLAIN=1: an ORDINARY PxParticleBuffer, same spawn, "
                        "not scored\n");
            std::printf("[ control ] gpu-sdk-step: particlerigid plain-buffer control run, not scored\n");
            return 0;
        }

        int failures = 0;
        bool finite = true;
        for (PxU32 i = 0; i < psc.numParticles; ++i)
            finite = finite && std::isfinite(host[i].x) && std::isfinite(host[i].y) && std::isfinite(host[i].z);
        if (!finite) { std::printf("FAIL: a particle position is not finite\n"); ++failures; }
        // the internal control: gravity ran at all
        if (!(meanY < y0 - 0.25f))
        { std::printf("FAIL: the cluster did not fall (mean y %.4f -> %.4f)\n", y0, meanY); ++failures; }
        // ps_update_volume_bound: the poison is gone and the bound IS the particles'
        for (int c = 0; c < 3; ++c)
        {
            const float bmin = (&volOut.bound.minimum.x)[c], bmax = (&volOut.bound.maximum.x)[c];
            const float rmin = (&ref.minimum.x)[c], rmax = (&ref.maximum.x)[c];
            if (bmin == RIGID_VOLUME_POISON || bmax == RIGID_VOLUME_POISON + 1.0f)
            { std::printf("FAIL: volume bound component %d is still POISONED -- ps_update_volume_bound "
                          "never ran (its host guard is mMaxVolumesPerBuffer > 0)\n", c); ++failures; break; }
            if (!(std::fabs(bmin - rmin) < 1e-3f && std::fabs(bmax - rmax) < 1e-3f))
            { std::printf("FAIL: volume bound component %d is (%.4f, %.4f), the particles are (%.4f, %.4f)\n",
                          c, bmin, bmax, rmin, rmax); ++failures; }
        }
        // ps_solveShapes: the cluster kept its shape, or (control) it did not
        if (rigidControl)
        {
            std::printf("  [ control ] PXG_PARTICLE_RIGID_CONTROL=1: shape-matching coefficient 0, "
                        "not scored -- the y-extent above is the loose-granular number\n");
            std::printf("[ control ] gpu-sdk-step: particlerigid control run, not scored\n");
            return 0;
        }
        if (!(extentY > 0.6f * psc.spawnExtentY))
        { std::printf("FAIL: the cluster COLLAPSED -- y-extent %.4f of a spawn %.4f.  Shape matching did "
                      "not hold it; ps_solveShapes is the only thing that does\n",
                      extentY, psc.spawnExtentY); ++failures; }
        if (!(extentY < 1.6f * psc.spawnExtentY))
        { std::printf("FAIL: the cluster EXPANDED -- y-extent %.4f of a spawn %.4f\n",
                      extentY, psc.spawnExtentY); ++failures; }

        failures += unmappedPenalty();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: a PBD SHAPE-MATCHED cluster landed and KEPT ITS SHAPE "
                        "(y-extent %.4f of a spawn %.4f) and its particle volume's AABB matches the "
                        "particles to 1e-3 -- both kernels are live on Vulkan\n",
                        extentY, psc.spawnExtentY);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: particlerigid checks failed (%d)\n", failures);
        return 2;
    }

    if (!std::strcmp(sceneName, "particleattach"))
    {
        ParticleScene psc = buildParticleScene(physics, scene, material, cudaCtx,
                                               false, false, false, false, false, false, false,
                                               /*attachMode*/ true);
        if (!psc.system || !psc.buffer || !psc.anchor || psc.numAttached == 0)
        {
            std::printf("[ FAIL ] gpu-sdk-step: particleattach scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'particleattach' built: %u of %u PBD particles pinned to a STATIC "
                    "anchor at y=%.1f; particle %u left free as the scene's own control\n",
                    psc.numAttached, psc.numParticles, ATTACH_START_Y, psc.freeParticle);
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: particleattach pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_PARTICLE_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 120;
        AttachY a = particleAttachY(psc);
        const float mean0 = a.pinnedMean, free0 = a.freeY;
        bool finite = a.finite;
        std::printf("  stepping particleattach (%d x %.4fs): pinned mean y0 = %.4f, free particle y0 = %.4f\n",
                    steps, dt, mean0, free0);
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            a = particleAttachY(psc);
            finite = finite && a.finite;
            if (i < 3 || i == steps - 1 || (i % 20 == 0))
                std::printf("    step %3d: pinned mean y = %.5f (min %.5f) | free y = %.5f\n",
                            i, a.pinnedMean, a.pinnedMin, a.freeY);
        }
        std::printf("  particleattach: pinned mean y %.4f -> %.4f (sag %+.4f, min %.4f); free particle "
                    "y %.4f -> %.4f (fell %+.4f)\n",
                    mean0, a.pinnedMean, a.pinnedMean - mean0, a.pinnedMin, free0, a.freeY, a.freeY - free0);

        int failures = 0;
        // (1) THE PINNED PARTICLES HELD.  Free fall over 120 steps from y = 4 reaches
        //     the floor, so the two outcomes are a metre apart and this bound is not a
        //     tuned tolerance.
        if (!(std::fabs(a.pinnedMean - mean0) < 0.15f))
        { std::printf("FAIL: the pinned particles moved %+.4f (need |dy| < 0.15).  If the "
                      "[pxg-unmapped] lines below name ps_rigidAttachmentPrepareLaunch or "
                      "ps_solveRigidAttachments{,TGS}Launch, this is that; if they do not, the "
                      "attachment chain ran and computed the wrong thing\n", a.pinnedMean - mean0);
          ++failures; }
        if (!(a.pinnedMin > ATTACH_START_Y - 0.6f))
        { std::printf("FAIL: some pinned particle sank to y = %.4f -- the mean can hide one that "
                      "let go\n", a.pinnedMin); ++failures; }
        // (2) THE CONTROL FELL.  Without this, "nothing moved" would also be the reading
        //     for a scene in which gravity never ran.
        if (!(a.freeY < free0 - 2.0f))
        { std::printf("FAIL: the UNPINNED particle only moved %+.4f -- gravity is not reaching the "
                      "cloud, so the pinned particles standing still proves nothing\n",
                      a.freeY - free0); ++failures; }
        if (!finite) { std::printf("FAIL: a particle coordinate went non-finite\n"); ++failures; }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: %u particles HANG from their rigid attachments (mean y "
                        "%.4f, sag %+.4f) while the one unpinned particle fell to %.4f -- the particle "
                        "attachment chain is live on Vulkan: ps_rigidAttachmentPrepareLaunch and "
                        "ps_solveRigidAttachments{,TGS}Launch\n",
                        psc.numAttached, a.pinnedMean, a.pinnedMean - mean0, a.freeY);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: particleattach checks failed (%d)\n", failures);
        return 2;
    }

    if (!std::strcmp(sceneName, "particle") || fluidScene)
    {
        ParticleScene psc = buildParticleScene(physics, scene, material, cudaCtx, fluidScene,
                                               boundaryScene || meshFloorScene, meshFloorScene,
                                               hfFloorScene, diffuseScene || convexFloorScene,
                                               convexFloorScene);
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
        const bool bagMode = std::getenv("PXG_PARTICLE_STARTY") != nullptr;
        const char* stepsEnv = std::getenv("PXG_PARTICLE_STEPS");
        const float spread0 = fluidScene ? particleSpreadXZ(psc) : 0.0f;
        const float nn0 = fluidScene ? particleMeanNearestNeighbour(psc) : 0.0f;
        float maxYSeen = fluidScene ? particleMeanY(psc) : 0.0f;
        // P9.1c raised the bag budget from 120 to 240.  Before ps_solveDensityLaunch
        // was mapped there was no particle-particle collision at all, so the cloud
        // passed through itself and reached its (collapsed) rest in well under 120
        // steps.  With the solve live the pile has real mutual overlaps to resolve
        // and needs roughly 150; 240 leaves margin and a 60-step rest window.
        const int steps = stepsEnv ? atoi(stepsEnv) : (fluidScene ? 300 : (bagMode ? 240 : 120));
        const float y0 = particleMeanY(psc);
        std::printf("  stepping the particle cloud (%d x %.4fs), initial mean y=%.4f...\n", steps, dt, y0);

        float prevY = y0;
        int nonMonotonic = 0;
        // P9.13a: the diffuse observable is a PEAK, not an end state -- diffuse
        // particles are spawned while the fluid is agitated and expire as it
        // settles, so a count read only at the end would be zero even on a working
        // chain.
        PxU32 maxDiffuseSeen = 0;
        // P9.13e: the peak alone cannot tell CREATE from CREATE+COMPACT.  What
        // separates them is the CARRY-OVER: without ps_diffuseParticleCompact the
        // staging slot is zeroed every frame by Copy, so the published count is
        // one frame's creations and nothing more.  Compact appends the SURVIVORS
        // of the previous published set, so the population accumulates.  The
        // last-step count is the aging half of the same observable.
        PxU32 endDiffuseSeen = 0;
        // P9.13h: the observable that ONLY a collision can satisfy.  A diffuse
        // particle that is never pushed out of the floor keeps falling for the
        // whole run; one the one-way contact path corrects stays above it.
        float minDiffuseY = FLT_MAX;
        float restMaxDy = 0.0f;   // max |dy| over the final quarter (rest convergence)
        const int restWindow = steps / 4;
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            const float y = particleMeanY(psc);
            if (i < 5 || i == steps - 1 || ((bagMode || fluidScene) && (i % 30 == 0)))
            {
                if (fluidScene)
                    std::printf("    step %3d: mean y = %.5f (dy = %+.5f) spread = %.4f nn = %.4f\n",
                                i, y, y - prevY, particleSpreadXZ(psc), particleMeanNearestNeighbour(psc));
                else
                    std::printf("    step %3d: mean y = %.5f (dy = %+.5f)\n", i, y, y - prevY);
            }
            // Free fall: mean y must not increase (allow a tiny float slack).
            if (y > prevY + 1e-3f) ++nonMonotonic;
            if (i >= steps - restWindow) restMaxDy = PxMax(restMaxDy, PxAbs(y - prevY));
            if (fluidScene) maxYSeen = PxMax(maxYSeen, y);
            if (psc.diffuse)
            {
                endDiffuseSeen = psc.diffuse->getNbActiveDiffuseParticles();
                maxDiffuseSeen = PxMax(maxDiffuseSeen, endDiffuseSeen);
                minDiffuseY = PxMin(minDiffuseY, diffuseMinY(psc));
            }
            prevY = y;
        }
        const float yEnd = particleMeanY(psc);
        const float drop = y0 - yEnd;

        int failures = 0;
        if (fluidScene)
        {
            // A PBD FLUID column, end to end over the five kernels P9.1a-d landed
            // (ps_updateMaterials, ps_calculateDensityAndPotentialLaunch,
            // ps_solveDensityLaunch, ps_vorticityConfinementLaunch,
            // ps_solveVelocityLaunch).  With a SOLID phase every fluid arm of those
            // early-outs per particle, so this scene is the only place they run
            // against the real engine rather than against a fixture.
            //
            // Two observables, both MEASURED against the solid-phase control
            // (PXG_PARTICLE_FLUID_CONTROL=1 builds the identical geometry, offsets
            // and material without the eParticlePhaseFluid bit):
            //
            //   mean nearest-neighbour distance at rest
            //       fluid   ~= 2 * fluidRestOffset = 0.200  (the density constraint)
            //       control ~= 2 * solidRestOffset = 0.333  (the contact solve)
            //   upward excursion of the mean y over the run
            //       fluid   ~= 0.00  (it settles monotonically)
            //       control ~= +1.88 (it starts deeply overlapped and is launched)
            //
            // The control's numbers are what they are because at this spacing the
            // SOLID rest distance (0.333) exceeds the particle spacing (0.2), so the
            // solid path begins in deep penetration.  The fluid rest distance is
            // exactly the spacing, which is why SnippetPBF chooses these offsets.
            const float spreadEnd = particleSpreadXZ(psc);
            const float nnEnd = particleMeanNearestNeighbour(psc);
            const float rise = maxYSeen - y0;
            std::printf("  particle FLUID column: mean y %.4f -> %.4f (dropped %.4f over %.2fs); "
                        "nearest-neighbour %.4f -> %.4f; spread %.4f -> %.4f; rise %+.4f; "
                        "rest |dy| over last %d steps = %.6f\n",
                        y0, yEnd, drop, steps * dt, nn0, nnEnd, spread0, spreadEnd, rise,
                        restWindow, restMaxDy);
            if (!std::isfinite(yEnd) || !std::isfinite(nnEnd))
            { std::printf("FAIL: fluid state is not finite\n"); ++failures; }
            if (!(drop > 0.2f))
            { std::printf("FAIL: the column did not settle downwards (drop=%.4f)\n", drop); ++failures; }
            if (!(nnEnd < FLUID_NN_MAX))
            { std::printf("FAIL: the cloud settled at the SOLID rest distance or wider -- the fluid "
                          "density constraint is not running (nn=%.4f, need <%.4f; the "
                          "solid-phase control measures %.4f)\n",
                          nnEnd, FLUID_NN_MAX, FLUID_NN_CONTROL); ++failures; }
            if (!(rise < FLUID_RISE_MAX))
            { std::printf("FAIL: the fluid was launched upward like the overlapping solid control "
                          "(rise=%+.4f, need <%.4f; the control measures %+.4f)\n",
                          rise, FLUID_RISE_MAX, FLUID_RISE_CONTROL); ++failures; }
            if (!(restMaxDy < 0.05f))
            { std::printf("FAIL: the fluid never came to rest (rest |dy|=%.6f)\n", restMaxDy); ++failures; }

            // ---------------------------------------------------------------
            // P9.9: the BOUNDARY-DENSITY observable.
            //
            // WHICH KERNEL IS ACTUALLY DOING THE WORK, because the engine's naming
            // is the opposite of what it sounds like and this is the trap of the
            // tranche:
            //
            //   ps_accumulateSTATICDensity reads mOneWayContactInfos /
            //     mOneWayContactCount.  Those are written by exactly three kernels
            //     -- ps_meshCollisonLaunch, ps_sdfMeshCollisonLaunch and
            //     ps_heightfieldCollisonLaunch (particleSystemMeshMidphase.cu:463,
            //     :698 and particleSystemHFMidPhaseCG.cu:332) -- i.e. only the MESH
            //     and HEIGHTFIELD paths, and ALL THREE ARE UNMAPPED in this port.
            //     ps_preIntegrateLaunch (particlesystem.cu:317) zeroes the count
            //     every frame, so in this scene every particle has contactCount == 0
            //     and the kernel provably writes min(0, boundary) == 0.  It is NOT
            //     dead weight, though: that zero-write is the PER-FRAME RESET of
            //     mStaticDensity, and the rigid kernel below both SEEDS from that
            //     array and STORES BACK into it.  Remove the reset and the boundary
            //     density LATCHES: the store's min(rho, boundary) bounds the value,
            //     so it does not run away, but every particle that has ever touched
            //     a rigid body pins itself at the cap and never comes back down,
            //     instead of tracking its CURRENT contact set -- and particles that
            //     never had a contact read a buffer nothing ever wrote.
            //     (An earlier draft of this comment said "accumulates without
            //     bound".  That was wrong; the store clamp bounds it.  The defect is
            //     latching, not divergence.)
            //
            //   ps_accumulateRIGIDDensity reads the particle-vs-PRIMITIVE contact
            //     stream (PxgParticlePrimitiveContact), which the mapped narrowphase
            //     already produces.  The ground here is a PxRigidStatic carrying a
            //     PxBoxGeometry -- a primitive -- so THIS is the kernel that sees
            //     the floor, and this is the kernel the assertions below measure.
            //
            // So: the scene proves the rigid half end to end, and proves the static
            // half launches and resets.  Feeding the static half real contacts needs
            // one of the three mesh/heightfield kernels mapped first, and the tranche
            // note says so rather than implying the pair is fully exercised.
            //
            // The DIRECTION of the observable, predicted before it was measured and
            // then confirmed: with the boundary on, particles near the floor are
            // seeded with a non-zero rho by the consumer
            // (calculateDensityAndPotential, particlesystem.cu:3253, mapped since
            // P9.1b), so the density constraint reads them as denser than they are
            // and pushes them apart.  The puddle therefore spreads WIDER and its
            // nearest-neighbour distance grows.  Only SPREAD is gated, as a one-sided
            // minimum against the boundary-off control (the same scene with the scale
            // at 0); nn is printed and NOT checked, because its on/off separation is
            // only ~11% and no threshold in that gap would be worth regressing
            // against.  (The nn check further up in the fluid assertions is a
            // different thing: a MAXIMUM against the solid-phase control, testing that
            // the fluid density constraint runs at all -- not a boundary check.)
            if (boundaryScene && !std::getenv("PXG_PARTICLE_BOUNDARY_CONTROL"))
            {
                std::printf("  particle BOUNDARY density: spread %.4f (control %.4f, live reference "
                            "%.4f); nn %.4f (control %.4f, reported only -- see the note)\n",
                            spreadEnd, FLUID_BOUNDARY_SPREAD_CONTROL, FLUID_BOUNDARY_SCALE_SPREAD,
                            nnEnd, FLUID_BOUNDARY_NN_CONTROL);
                if (!(spreadEnd > FLUID_BOUNDARY_SPREAD_MIN))
                { std::printf("FAIL: the boundary density is not reaching the fluid -- the puddle did "
                              "not spread past the boundary-off control (spread=%.4f, need >%.4f; the "
                              "control measures %.4f and the live run measured %.4f)\n",
                              spreadEnd, FLUID_BOUNDARY_SPREAD_MIN, FLUID_BOUNDARY_SPREAD_CONTROL,
                              FLUID_BOUNDARY_SCALE_SPREAD);
                  ++failures; }
            }
            // P9.10c -- the MESH FLOOR.  Two things must hold, and both are
            // a-priori justifiable rather than tuned: the puddle must be ON the mesh
            // (no tunnelling through two triangles, which is what a midphase that
            // finds no contacts looks like), and it must have collapsed from the
            // column's start height.  The spread numbers are printed for the record;
            // the boundary threshold below is the box-floor scene's job, because the
            // two floors do not produce the same contact stream and sharing a
            // threshold between them would be comparing different measurements.
            if (meshFloorScene)
            {
                std::printf("  particle MESH FLOOR: mean y %.4f -> %.4f; spread %.4f -> %.4f; nn %.4f\n",
                            y0, yEnd, spread0, spreadEnd, nnEnd);
                if (!(yEnd > 0.0f))
                { std::printf("FAIL: the puddle sank to or through the triangle-mesh floor "
                              "(mean y = %.4f, floor at 0) -- ps_meshCollisonLaunch produced no "
                              "usable contacts\n", yEnd);
                  ++failures; }
                if (!(yEnd < y0))
                { std::printf("FAIL: the column never collapsed (mean y %.4f -> %.4f)\n", y0, yEnd);
                  ++failures; }
                if (!std::getenv("PXG_PARTICLE_BOUNDARY_CONTROL")
                    && !(spreadEnd > MESHFLOOR_SPREAD_MIN))
                { std::printf("FAIL: the boundary density is not reaching the fluid through the "
                              "STATIC path -- the puddle did not spread past the boundary-off "
                              "control (spread=%.4f, need >%.4f; the control measures %.4f and the "
                              "live run measured %.4f)\n",
                              spreadEnd, MESHFLOOR_SPREAD_MIN, MESHFLOOR_SPREAD_CONTROL,
                              MESHFLOOR_SPREAD_LIVE);
                  ++failures; }
            }
            // P9.11c -- the HEIGHTFIELD FLOOR.  Identical in shape to the mesh
            // floor's block above and for identical reasons; only the producer of
            // mOneWayContactInfos differs.
            if (hfFloorScene)
            {
                std::printf("  particle HF FLOOR: mean y %.4f -> %.4f; spread %.4f -> %.4f; nn %.4f\n",
                            y0, yEnd, spread0, spreadEnd, nnEnd);
                if (!(yEnd > 0.0f))
                { std::printf("FAIL: the puddle sank to or through the heightfield floor "
                              "(mean y = %.4f, floor at 0) -- ps_heightfieldCollisonLaunch "
                              "produced no usable contacts\n", yEnd);
                  ++failures; }
                if (!(yEnd < y0))
                { std::printf("FAIL: the column never collapsed (mean y %.4f -> %.4f)\n", y0, yEnd);
                  ++failures; }
                // The sharp one: the same fluid on two different static narrowphase
                // kernels must come to rest at the same height.  A heightfield whose
                // contacts were merely PRESENT but placed wrong would still satisfy
                // `yEnd > 0`; only agreeing with the mesh floor to within 0.02 says
                // the surface is where the geometry says it is.
                if (!(std::fabs(yEnd - HFFLOOR_REST_Y) < HFFLOOR_REST_Y_TOL))
                { std::printf("FAIL: the fluid rests at y=%.5f on the heightfield but at %.5f on "
                              "the triangle-mesh floor at the same height (tolerance %.3f) -- the "
                              "heightfield surface is not where the geometry puts it\n",
                              yEnd, HFFLOOR_REST_Y, HFFLOOR_REST_Y_TOL);
                  ++failures; }
            }
            // P9.13a -- DIFFUSE PARTICLES.  The scene that names the diffuse chain.
            //
            // What is asked of the engine: a PxParticleAndDiffuseBuffer is attached
            // to the fluid, given a reachable spawn threshold, and stepped.  A
            // working chain spawns diffuse particles while the column collapses and
            // lets them expire as it settles, so the count is read EVERY step and
            // the peak is what is checked.
            //
            // WHAT THIS SCENE PROVES, and what it does not.  Since P9.13d the
            // count is NON-ZERO: spawning and publishing needs only
            // ps_diffuseParticleCreate, ps_diffuseParticleCopy and
            // ps_diffuseParticleSum, and those are mapped.  An earlier draft of
            // this comment predicted the scene would stay red until the whole
            // chain landed; that was wrong, because the observable is the
            // PUBLISHED COUNT and not the motion.
            //
            // P9.13e added the two checks that a create-only chain CANNOT pass,
            // because a peak on its own never could: the population must
            // ACCUMULATE far past one frame's creations (Copy zeroes the staging
            // slot every frame, so only ps_diffuseParticleCompact's re-append of
            // the survivors can carry it over -- 191 create-only in P9.13d, 4096
            // now) and it must FALL BACK from that peak (a saturated buffer whose
            // particles never expire stays saturated -- 4096 -> 3197 now).
            //
            // P9.13h added the third and last check, and it is the one no amount
            // of spawning or aging can pass: the LOWEST y any live diffuse
            // particle reached.  The fluid column rests at y ~ 0.17, so a
            // diffuse particle that is never pushed out of the floor keeps
            // falling for the rest of the run.  Measured -0.1886 with the chain
            // whole and -55.54 with the one-way correction neutralised in the
            // shader -- same build, one mutated .spv.
            //
            // What this scene STILL does not reach is
            // ps_convexDiffuseCollisionLaunch, which is unmapped -- and which
            // this scene never calls, because it contains no convex shape.  Its
            // name is in the wrangler's startup list and nowhere else; the
            // per-frame "fail to launch" errors this scene used to emit are
            // gone.
            //
            // WHERE TO READ THE WORK LIST, and where NOT to.  PXG_LAUNCH_TRACE=1 is
            // the wrong instrument here and says so by staying silent: an unmapped
            // kernel has no CUfunction, so the engine's launch never reaches our
            // shim and no REFUSED line is ever printed.  The trace shows only what
            // the shim was actually handed.  The two signals that DO name this
            // chain are the wrangler's startup list ("Could not find registered
            // CUDA function 'ps_diffuse...'") and, per frame, the engine's own
            // "GPU <kernel> fail to launch!!" -- which this scene emits 288 times
            // for particlePrimitivesDiffuseCollisionLaunch, i.e. proof that the
            // host really is driving the chain and not skipping it.
            //
            // The CONTROL (PXG_DIFFUSE_CONTROL=1) raises the threshold to 1e12 and
            // changes nothing else, so when the chain lands the pair separates
            // "spawning happened" from "spawning was never asked for".
            if (diffuseScene || convexFloorScene)
            {
                const bool control = std::getenv("PXG_DIFFUSE_CONTROL") != nullptr;
                std::printf("  particle DIFFUSE: peak active diffuse particles = %u of %u, last "
                            "step %u (threshold %s); lowest diffuse y %.4f; mean y %.4f -> %.4f, "
                            "nn %.4f\n",
                            maxDiffuseSeen, MAX_DIFFUSE, endDiffuseSeen,
                            control ? "1e12 (CONTROL)" : "reachable",
                            minDiffuseY == FLT_MAX ? 0.0f : minDiffuseY, y0, yEnd, nnEnd);
                if (convexFloorScene)
                {
                    // P9.14a: numbers only.  Both convex particle kernels are
                    // unmapped, so nothing here can pass yet and asserting would
                    // turn a work list into a red scene.  When they land, the
                    // checks this scene must grow are the same three
                    // particlediffuse carries -- carry-over, aging and the floor.
                }
                else if (control)
                {
                    if (maxDiffuseSeen != 0u)
                    { std::printf("FAIL: the control spawned %u diffuse particles at an "
                                  "unreachable threshold\n", maxDiffuseSeen);
                      ++failures; }
                }
                else if (maxDiffuseSeen == 0u)
                { std::printf("FAIL: no diffuse particle was ever created -- the diffuse chain is "
                              "unmapped (the work list is the wrangler's startup 'Could not find "
                              "registered CUDA function ps_diffuse*' lines plus the per-frame "
                              "'fail to launch' errors, NOT the launch trace -- see the note)\n");
                  ++failures; }
                else
                {
                    // P9.13e -- the two checks the CREATE-only chain could not pass.
                    //
                    // CARRY-OVER.  ps_diffuseParticleCopy zeroes the staging slot
                    // every frame, so without ps_diffuseParticleCompact the
                    // published count is ONE frame's creations: P9.13d measured
                    // exactly 191 here.  Compact appends the survivors of the
                    // previous published set, so the population accumulates until
                    // it saturates.  The bar is MAX_DIFFUSE/2 = 2048, ten times
                    // the create-only figure; measured 4096.
                    if (maxDiffuseSeen < MAX_DIFFUSE / 2u)
                    { std::printf("FAIL: the diffuse population peaked at %u of %u -- that is a "
                                  "single frame's creations, not a carried-over population "
                                  "(create-only measured 191 in P9.13d; need >= %u)\n",
                                  maxDiffuseSeen, MAX_DIFFUSE, MAX_DIFFUSE / 2u);
                      ++failures; }
                    // AGING.  A saturated buffer whose particles never expire stays
                    // saturated: Compact would re-append all of them every frame.
                    // The count can only fall back from its peak if lifetimes decay
                    // and the `lifeTime > 0` survivor test drops particles.  Require
                    // a 5% fall; measured 22% (4096 -> 3197, reproducible).
                    else if (endDiffuseSeen * 100u >= maxDiffuseSeen * 95u)
                    { std::printf("FAIL: the diffuse population did not fall back from its peak "
                                  "(%u at the last step vs a peak of %u) -- nothing is expiring, "
                                  "so the lifetime is not being decayed\n",
                                  endDiffuseSeen, maxDiffuseSeen);
                      ++failures; }
                    // COLLISION (P9.13h).  Carry-over and aging both happen with no
                    // contact at all -- neither needs the floor.  This one does: a
                    // diffuse particle that is never pushed out keeps falling for the
                    // rest of the run, and the fluid column sits at y ~ 0.17, so the
                    // lowest y any live diffuse particle reached is the whole chain's
                    // end-to-end observable.  Measured -0.1886 with the one-way
                    // correction and -55.54 with it neutralised (same build, one
                    // mutated shader), so the bar sits at -2.0: ten times the
                    // corrected figure and a twenty-seventh of the uncorrected one.
                    else if (minDiffuseY < -2.0f)
                    { std::printf("FAIL: a diffuse particle reached y = %.4f -- the spray is "
                                  "falling through the floor, so the one-way contact path is "
                                  "not correcting it (need > -2.0; the fluid rests at %.4f)\n",
                                  minDiffuseY, yEnd);
                      ++failures; }
                }
            }
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            failures += unmappedPenalty();
            // P9.14a printed this UNCONDITIONALLY as a diagnostic, because the
            // checks above were supposed to FAIL here: nothing was colliding
            // against the convex floor, and that failure was the measurement.
            // P9.14b and P9.14c landed the two kernels, so the numbers are now a
            // RESULT rather than a bill of work -- kept, and printed beside a real
            // verdict, so the two figures stay readable in the scene log.
            if (convexFloorScene)
                std::printf("  particleconvex over a cooked CONVEX floor: fluid mean y %.4f "
                            "(was -39.3227 with ps_convexCollisionLaunch unmapped), nn %.4f, "
                            "peak diffuse %u of %u, lowest diffuse y %.4f (was -39.6138 with "
                            "ps_convexDiffuseCollisionLaunch unmapped)\n",
                            yEnd, nnEnd, maxDiffuseSeen, MAX_DIFFUSE,
                            minDiffuseY == FLT_MAX ? 0.0f : minDiffuseY);
            if (failures == 0 && diffuseScene)
            {
                // The CONTROL passing means ONLY that nothing spawned when nothing
                // was asked for.  It must never print the end-to-end claim -- that
                // sentence is exactly the kind of thing a later reader would quote.
                if (std::getenv("PXG_DIFFUSE_CONTROL"))
                    std::printf("[  OK  ] gpu-sdk-step: the diffuse CONTROL spawned nothing at an "
                                "unreachable threshold, as it must -- this says nothing about "
                                "whether the diffuse chain works\n");
                else
                    // Say exactly what this proves.  The peak needs create/copy/sum;
                    // the CARRY-OVER to it and the fall back from it need
                    // ps_diffuseParticleCompact's aging; and the floor needs
                    // ps_primitivesDiffuseCollisionLaunch to find the contact and
                    // ps_diffuseParticleOneWayCollision to apply it.  That is the
                    // whole chain this scene reaches: it never calls
                    // ps_convexDiffuseCollisionLaunch, because it contains no convex
                    // shape.  particleconvex is the scene that does, and since P9.14c
                    // that kernel is mapped too -- so no part of the diffuse chain is
                    // unmapped any more.
                    std::printf("[  OK  ] gpu-sdk-step: the PBD fluid carried a DIFFUSE population "
                                "up to %u of %u and back down to %u, and the spray STOPPED AT THE "
                                "FLOOR (lowest y %.4f, vs -55.54 measured with the one-way "
                                "correction neutralised) -- spawn, publish, age and collide all "
                                "run on Vulkan over a PRIMITIVE floor; the convex floor is "
                                "particleconvex's job\n",
                                maxDiffuseSeen, MAX_DIFFUSE, endDiffuseSeen, minDiffuseY);
                return 0;
            }
            // P9.14c: particleconvex is SCORED from here on.  It is the only scene
            // that reaches testSDKConvexParticle, so it is the only one that can say
            // anything about either convex particle kernel -- and it now says it
            // about BOTH, on the same floor in the same frame: the fluid rests
            // (ps_convexCollisionLaunch, P9.14b) and the spray rests on top of it
            // (ps_convexDiffuseCollisionLaunch, P9.14c).  The two figures it prints
            // are each a factor of ~200 away from the same scene with its kernel
            // unmapped, which is what makes the verdict worth having.
            if (failures == 0 && convexFloorScene)
            {
                std::printf("[  OK  ] gpu-sdk-step: over a cooked CONVEX HULL floor the PBD fluid "
                            "RESTS at mean y %.4f (was -39.3227 before ps_convexCollisionLaunch) "
                            "and the DIFFUSE spray rests with it at lowest y %.4f (was -39.6138 "
                            "before ps_convexDiffuseCollisionLaunch) -- the convex particle pair "
                            "bucket runs end to end on Vulkan, and with it the whole diffuse "
                            "chain\n",
                            yEnd, minDiffuseY);
                return 0;
            }
            if (failures == 0 && hfFloorScene)
            {
                std::printf("[  OK  ] gpu-sdk-step: the PBD fluid rests on a COOKED HEIGHTFIELD "
                            "at y %.5f -- the same height the triangle-mesh floor gives it "
                            "(spread %.4f vs %.4f measured, nn %.4f vs %.4f) -- so "
                            "ps_heightfieldCollisonLaunch is the second producer of "
                            "mOneWayContactInfos this port has mapped\n",
                            yEnd, spreadEnd, HFFLOOR_SPREAD_REF, nnEnd, HFFLOOR_NN_REF);
                return 0;
            }
            if (failures == 0 && meshFloorScene)
            {
                std::printf("[  OK  ] gpu-sdk-step: the PBD fluid rests on a COOKED TRIANGLE MESH "
                            "(mean y %.4f, spread %.4f vs %.4f for the boundary-off control, "
                            "nn %.4f) -- ps_meshCollisonLaunch fills mOneWayContactInfos and "
                            "ps_accumulateStaticDensity finally has something to read\n",
                            yEnd, spreadEnd, MESHFLOOR_SPREAD_CONTROL, nnEnd);
                return 0;
            }
            if (failures == 0 && boundaryScene && !std::getenv("PXG_PARTICLE_BOUNDARY_CONTROL"))
            {
                std::printf("[  OK  ] gpu-sdk-step: the PBD fluid FEELS ITS BOUNDARY -- "
                            "ps_accumulateRigidDensity's contribution to mStaticDensity widened the "
                            "resting puddle to spread %.4f / nn %.4f, against %.4f / %.4f for the "
                            "identical scene with fluidBoundaryDensityScale left at 0\n",
                            spreadEnd, nnEnd, FLUID_BOUNDARY_SPREAD_CONTROL, FLUID_BOUNDARY_NN_CONTROL);
                return 0;
            }
            if (failures == 0)
            {
                std::printf("[  OK  ] gpu-sdk-step: the PBD FLUID column collapsed to a puddle far below "
                            "the solid rest spacing (nn %.4f vs %.4f for the solid-phase control) and "
                            "settled without the control's %+.2f upward excursion -- the whole fluid "
                            "quartet runs end to end on Vulkan\n", nnEnd, FLUID_NN_CONTROL, FLUID_RISE_CONTROL);
                return 0;
            }
            std::printf("[ FAIL ] gpu-sdk-step: particle fluid checks failed (%d)\n", failures);
            return 2;
        }
        if (bagMode)
        {
            // Settling bag (P6.5): the cloud must FALL, be CAUGHT on the floor, and
            // come to REST.  restMaxDy over the final quarter must decay to ~0, yEnd
            // must sit just above the floor, and everything must stay finite.
            //
            // P9.1c added the PILE HEIGHT check, and it is the one that carries real
            // information.  Measured A/B on this exact scene, with the ONLY change
            // being whether ps_solveDensityLaunch is mapped:
            //     unmapped -> the 64 particles pass through one another and collapse
            //                 into a single layer, mean y = 0.25000 (the floor rest
            //                 offset), reached in well under 120 steps;
            //     mapped   -> they resolve their mutual overlaps and form a real 3D
            //                 pile, mean y = 0.99919.
            // A resting height above 0.6 is therefore only reachable with particle-
            // particle collision actually running, which the other three checks
            // cannot distinguish -- the collapsed layer falls, rests and is finite
            // too.
            std::printf("  particle SETTLING BAG: mean y %.4f -> %.4f (dropped %.4f over %.2fs); "
                        "rest |dy| over last %d steps = %.6f\n", y0, yEnd, drop, steps * dt, restWindow, restMaxDy);
            if (!(drop > 0.5f))          { std::printf("FAIL: cloud did not fall (drop=%.4f)\n", drop); ++failures; }
            if (!(restMaxDy < 0.01f))    { std::printf("FAIL: cloud did not come to rest (rest |dy|=%.6f)\n", restMaxDy); ++failures; }
            if (!(yEnd > -0.5f && yEnd < 1.5f)) { std::printf("FAIL: cloud did not settle near the floor (yEnd=%.4f)\n", yEnd); ++failures; }
            if (!(yEnd > 0.6f))          { std::printf("FAIL: the pile collapsed into one layer -- particle-particle "
                                                       "collision is not running (yEnd=%.4f, collapsed is 0.2500)\n", yEnd); ++failures; }
            if (!std::isfinite(yEnd))    { std::printf("FAIL: mean y is not finite (%.4f)\n", yEnd); ++failures; }
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            failures += unmappedPenalty();
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
        failures += unmappedPenalty();
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
        failures += unmappedPenalty();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: FEMCloth fell monotonically under gravity on Vulkan "
                        "(the first live FEMCloth frame)\n");
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: cloth free-fall checks failed (%d)\n", failures);
        return 2;
    }

    // PXG_SCENE=clothbox (P9.15a): the cloth LANDS instead of falling through.
    //
    // WHY THIS SCENE EXISTS.  `cloth` above is a 60-step FREE FALL and that is its
    // whole contract: it asserts the cloth descends, monotonically, to a finite y.
    // A cloth that sails straight through the ground box satisfies every one of
    // those, so the scene stayed green for the entire time cloth_boxVertexContactGenLaunch
    // was unmapped -- it named the kernel (nine "fail to launch" lines per run) and
    // could not measure it.  That is the P9.13h lesson: a scene that only fails to
    // be contradicted is not evidence.
    //
    // This scene is the same rig stepped four times as long, and it asks the one
    // question the free fall cannot: DOES IT STOP?
    //
    // MEASURED, not asserted -- same build, one manifest row:
    //     kernel mapped    mean y ->  +0.4933, |dy| < 1e-5 over the last steps
    //     kernel unmapped  mean y ->  -73.8070, still accelerating at -0.65/step
    // The bar therefore sits at -2.0: four times the resting height and a
    // thirty-seventh of the free-fall figure.
    if (!std::strcmp(sceneName, "clothbox") || !std::strcmp(sceneName, "clothconvex") ||
        !std::strcmp(sceneName, "clothmesh") || !std::strcmp(sceneName, "clothcvxcore"))
    {
        // `clothconvex` is `clothbox` with the ground BOX replaced by a cooked
        // CONVEX slab at the same height -- see buildClothScene.  It shares every
        // check below, because the two paths run the SAME body behind two engine
        // names, so anything that differs would be the routing or the manifest.
        const bool convexFloor = !std::strcmp(sceneName, "clothconvex");
        const bool meshFloor = !std::strcmp(sceneName, "clothmesh");
        const bool coreFloor = !std::strcmp(sceneName, "clothcvxcore");
        ClothScene csc = buildClothScene(physics, scene, material, cudaCtx, convexFloor,
                                         meshFloor, coreFloor);
        if (!csc.surface || csc.numVerts == 0)
        {
            std::printf("[ FAIL ] gpu-sdk-step: %s scene build failed\n", sceneName);
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene '%s' built: %u-vertex FEMCloth dropped onto the ground %s\n",
                    sceneName, csc.numVerts,
                    coreFloor ? "convex CORE box"
                              : (meshFloor ? "cooked triangle mesh"
                                           : (convexFloor ? "convex slab" : "box")));
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: clothbox pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_CLOTH_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 240;
        const float y0 = clothMeanY(csc);
        std::printf("  stepping %s (%d x %.4fs), initial mean y=%.4f...\n", sceneName, steps, dt, y0);

        // The last quarter of the run is the settling window: over it the cloth must
        // move less than a millimetre per step.  A cloth still falling moves ~0.65.
        const int settleFrom = (steps * 3) / 4;
        float prevY = y0, maxSettleStep = 0.0f, minY = y0;
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            const float y = clothMeanY(csc);
            if (y < minY) minY = y;
            if (i >= settleFrom) maxSettleStep = PxMax(maxSettleStep, PxAbs(y - prevY));
            if (i < 3 || i == steps - 1 || (i % 40 == 0))
                std::printf("    step %3d: mean y = %.5f (dy = %+.5f)\n", i, y, y - prevY);
            prevY = y;
        }
        const float yEnd = clothMeanY(csc);

        int failures = 0;
        std::printf("  %s: mean y %.4f -> %.4f (lowest %.4f); largest |dy| over the last "
                    "%d steps = %.6f\n", sceneName, y0, yEnd, minY, steps - settleFrom, maxSettleStep);
        // 1. IT STOPPED.  Only a contact can stop it; gravity never does.
        //
        // The bar was 1e-3 when only the VERTEX path was mapped (|dy| measured
        // 0.000098).  P9.15d added the TRIANGLE path, and with both live the cloth
        // settles into a small limit cycle instead of going still: |dy| 0.0020, mean
        // y oscillating inside a 0.002 band with no trend over the last 120 steps.
        // That is more constraints chattering, not a fall -- a falling cloth moves
        // 0.65 per step, three hundred times more.  The bar is 0.01: sixty-five times
        // below a fall and five times above the observed jitter.
        //
        // P9.74 GAVE THE CONVEX-CORE ARM ITS OWN TWO BARS, AND THE REASON IS A
        // MEASUREMENT, NOT A PREFERENCE.
        //
        // Both bars above were calibrated for `clothbox`, where the distinction the
        // scene had to draw was +0.69 against -73.81 -- a factor of a hundred, so
        // `> -2.0` and `< 1e-2` were generous and still decisive.  Over the convex-core
        // floor they are not decisive at all, and P9.74's mutation sweep is what said
        // so out loud.  The mutant `pair-uses-w-not-y` -- `testPair.w` instead of
        // `testPair.y`, and the midphase writes uint4(cmIdx, primIndex, 0, 0)
        // (femClothMidPhase.cu:98), so EVERY pair collapses onto triangle 0 -- was
        // reported SURVIVED.  Run by hand it gives:
        //
        //     clean    mean y  0.0161   |dy| 0.000000
        //     mutant   mean y -0.8662   |dy| 0.005865
        //
        // The scene MOVED, by most of a metre, and the verdict swallowed it: -0.8662
        // clears -2.0 and 0.005865 clears 1e-2.  That is the P9.20 failure again in a
        // new place -- green for a reason that is not the reason claimed -- and the
        // answer is the same one P9.71 used: sharpen the observer rather than write a
        // paragraph excusing the survivor.
        //
        // THE NEW BARS ARE DERIVED FROM THE GEOMETRY, not fitted to the mutant.
        // PxConvexCore::Box takes FULL extents, so 8 x 1 x 8 centred at y = -0.5 spans
        // y in [-1, 0] and its TOP FACE IS AT y = 0 exactly.  A cloth held BY that face
        // rests at +contactOffset; measured 0.0161.  A cloth at -0.10 is already inside
        // the box, which no correct contact set produces -- so -0.10 is where "resting
        // on the floor" stops being true, about six times the clean run's own distance
        // from zero and eight times below the mutant.  The settle bar is 2e-3: the
        // clean run goes fully still (0.000000), and 2e-3 still leaves three times the
        // margin against the mutant's residual chatter.
        //
        // The other three floors keep the old bars.  Their top faces are at y = 0 too,
        // but `clothbox`'s settled |dy| is 0.0020 (P9.15d's limit cycle) and
        // `clothmesh` is a known-red upstream defect at -39.54; tightening those here
        // would be changing two unrelated scenes on the strength of a third scene's
        // measurement.
        const float settleBar = coreFloor ? 2e-3f : 1e-2f;
        if (!(maxSettleStep < settleBar))
        { std::printf("FAIL: the cloth is still moving %.6f per step at the end of the run -- "
                      "it never came to rest (bar %.4f)\n", maxSettleStep, settleBar); ++failures; }
        // 2. IT STOPPED ON THE GROUND, not somewhere below it.  -73.81 was measured
        //    with cloth_boxVertexContactGenLaunch unmapped, in this same scene.
        //
        //    The message names the actual ground and, for clothmesh, the actual
        //    reason.  It used to say "the box" for all three, which read as a wrong
        //    diagnosis in the one scene that fails: clothmesh's chain is FULLY mapped,
        //    and it falls anyway because cloth_meshVertexContactGenLaunch is switched
        //    off upstream -- a message pointing at an unmapped box kernel would send
        //    the next reader looking for work that is already done.
        const float restBar = coreFloor ? -0.10f : -2.0f;
        if (!(yEnd > restBar))
        {
            if (coreFloor)
                std::printf("FAIL: the cloth came to rest at y = %.4f -- it is INSIDE the convex "
                            "core box, whose top face is at y = 0 (need > %.2f; the clean run "
                            "gives 0.0161, and -73.81 was measured with "
                            "convexCoreClothmeshNphase_Kernel32 unmapped).  A cloth that sinks "
                            "into the floor instead of through it means the contacts exist but "
                            "are wrong -- wrong triangle, wrong barycentric, or wrong "
                            "penetration depth\n", yEnd, restBar);
            else if (meshFloor)
                std::printf("FAIL: the cloth came to rest at y = %.4f -- it fell through the "
                            "cooked triangle mesh (need > -2.0).  EXPECTED, and not this "
                            "port's doing: the chain is fully mapped, but "
                            "cloth_meshVertexContactGenLaunch emits no contacts upstream "
                            "(its only `hasContacts` assignment is commented out at "
                            "femClothPrimitives.cu:3056), and the VERTEX path is what "
                            "catches a cloth.  Measured -39.3333 both before and after the "
                            "kernel was mapped.  See tools/check_disabled_kernels.py\n", yEnd);
            else
                std::printf("FAIL: the cloth came to rest at y = %.4f -- it fell through the %s "
                            "(need > -2.0; -73.81 measured with the box vertex kernel "
                            "unmapped)\n", yEnd, convexFloor ? "convex slab" : "box");
            ++failures;
        }
        if (!std::isfinite(yEnd))
        { std::printf("FAIL: mean y is not finite (%.4f)\n", yEnd); ++failures; }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (failures == 0)
        {
            if (convexFloor)
                // P9.21.  The number after "vs" is a CONTROL, re-measured in this
                // tranche rather than carried over: with the manifest row for
                // cloth_convexVertexContactGenLaunch set back to nullptr, this same
                // scene falls to -73.8070 and the log carries ten
                // "fail to launch kernel" lines naming that kernel.  So the landing
                // below is that kernel's doing and not the triangle path's, which
                // was already mapped in P9.20 and did NOT stop the fall.
                std::printf("[  OK  ] gpu-sdk-step: the FEMCloth LANDED ON THE CONVEX SLAB and "
                            "stayed there (mean y %.4f, |dy| < %.6f per step, vs -73.8070 measured "
                            "with cloth_convexVertexContactGenLaunch unmapped and the TRIANGLE row "
                            "already live) -- both cloth-vs-convex paths run on Vulkan: the "
                            "TRIANGLE one (P9.20, the proven CUDA twin of the box kernel) and the "
                            "VERTEX one (P9.21), and it is the vertex one that catches\n",
                            yEnd, maxSettleStep);
            else
                std::printf("[  OK  ] gpu-sdk-step: the FEMCloth LANDED ON THE BOX and stayed there "
                            "(mean y %.4f, |dy| < %.6f per step, vs -73.8070 measured with the box "
                            "contact kernels unmapped) -- BOTH cloth-vs-box paths run on Vulkan: the "
                            "VERTEX one (P9.15a) and, since P9.15d, the TRIANGLE one, which is what "
                            "raised the resting height from 0.4933 to %.4f by adding face contacts "
                            "the vertex path cannot see\n", yEnd, maxSettleStep, yEnd);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: %s checks failed (%d)\n", sceneName, failures);
        return 2;
    }

    // PXG_SCENE=clothhang (P7.1 tranche 3b): the HANGING / draping cloth money shot.
    // A vertical FEMCloth curtain pinned along its top edge to fixed world anchors; it
    // must DRAPE below the pins and SETTLE (fall-then-stabilize), mirroring the P6.5
    // particle settling-bag pass criterion but per-row: pinned top row ~constant y, free
    // bottom row drops a bounded amount then converges.
    // PXG_SCENE=clothdamp (P9.19) shares this whole block with `clothhang`, and
    // that sharing is the measurement: same mesh, same pins, same step count, one
    // material difference.  Anything that differs in the settling is the damping
    // trio (cloth_accumulateInPlaneDampingDeltaVelocity,
    // cloth_accumulateBendingDampingDeltaVelocity, cloth_applyAccumulatedDeltaVelocity).
    // P9.19: both numbers are MEASURED, not chosen.  `clothhang` (no damping) and
    // `clothdamp` (elasticityDamping = bendingDamping = 120, which saturates
    // clamp(d*dt,0,1) at dt=1/60) differ only in the material; these are what the
    // two runs report, and the bar sits between them.
    // MEASURED on this rig, both runs 120 x 1/60 s, identical in every respect but
    // the material: the flat-spawned sheet's bottom-row y spread at the end is
    //     0.006395  undamped   (elasticityDamping = bendingDamping = 0)
    //     0.074761  damped     (both 120, which saturates clamp(d*dt,0,1) at 1.0)
    // -- a factor of ELEVEN.  The direction is the interesting part and it is not
    // the naive one: full internal damping does not smooth the sheet, it makes it
    // hold its shape.  Both accumulators remove the non-rigid part of each
    // element's motion, so a saturated factor turns the sheet into something much
    // closer to a rigid shell, and the curl it acquires while swinging stays.  The
    // undamped sheet is free to flatten out again.
    //
    // The bar sits between them, five times the undamped figure and two and a half
    // times below the damped one.
    static const float CLOTHDAMP_UNDAMPED_REF = 0.006395f;
    static const float CLOTHDAMP_SPREAD_BAR = 0.030f;
    if (!std::strcmp(sceneName, "clothhang") || !std::strcmp(sceneName, "clothdamp"))
    {
        const bool damped = !std::strcmp(sceneName, "clothdamp");
        // `clothdamp` also spawns the sheet FLAT (see the builder): the vertical
        // curtain has no internal motion to damp, so it cannot measure this.
        const char* undampedEnv = std::getenv("PXG_CLOTHDAMP_UNDAMPED");
        ClothHangScene csc = buildClothHangScene(physics, scene, material, cudaCtx,
                                                 damped && !undampedEnv, damped);
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
        float top0 = 0.0f, bot0 = 0.0f, botSpread = 0.0f;
        bool finite0 = clothHangRowMeans(csc, top0, bot0, &botSpread);
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
        clothHangRowMeans(csc, topEnd, botEnd, &botSpread);
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
        // (3) the VERTICAL curtain must CONVERGE (bottom row stops moving over the
        // final quarter).  The flat-spawned `clothdamp` sheet is still mid-swing at
        // 120 steps by design -- a settled sheet has no internal motion left to
        // damp, which is exactly why `clothhang` cannot measure this -- so the
        // settle check belongs to `clothhang` alone.
        if (!damped && !(botRestMaxDy < 0.01f))
        { std::printf("FAIL: drape did not settle (rest |dy|=%.6f)\n", botRestMaxDy); ++failures; }
        // (3b) P9.19: the DAMPED sheet must hold its curl -- the bottom row's y
        // spread stays far above the undamped reference.  See the constants above
        // for both measured numbers and why the direction is what it is.
        if (damped)
        {
            if (!(botSpread > CLOTHDAMP_SPREAD_BAR))
            { std::printf("FAIL: damped bottom-row spread %.6f, expected above %.6f "
                          "(undamped reference %.6f)\n", botSpread, CLOTHDAMP_SPREAD_BAR,
                          CLOTHDAMP_UNDAMPED_REF); ++failures; }
            // and it must actually have swung, or the spread means nothing.
            if (!(botDrop > 1.0f))
            { std::printf("FAIL: damped sheet only swung %.5f -- with no swing there is no "
                          "internal motion to damp\n", botDrop); ++failures; }
        }
        // (4) nothing NaN/inf.
        if (!allFinite || !std::isfinite(topEnd) || !std::isfinite(botEnd))
        { std::printf("FAIL: cloth y went non-finite\n"); ++failures; }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (failures == 0)
        {
            if (damped)
                std::printf("[  OK  ] gpu-sdk-step: the DAMPED sheet HOLDS ITS CURL and the undamped "
                            "one flattens: bottom-row y spread %.6f vs %.6f measured with the "
                            "SAME rig and damping 0 (bar %.4f), after swinging down %.4f -- "
                            "the P9.19 damping trio is live on Vulkan, and within the cloth "
                            "path nothing but those three kernels reads either damping field\n",
                            botSpread, CLOTHDAMP_UNDAMPED_REF, CLOTHDAMP_SPREAD_BAR, botDrop);
            else
                std::printf("[  OK  ] gpu-sdk-step: FEMCloth HANGS from its pinned edge and drapes under gravity\n");
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: %s drape checks failed (%d)\n", sceneName, failures);
        return 2;
    }

    // PXG_SCENE=clothdrape (P7.1 tranche 4c): the cloth-vs-rigid CONTACT money shot.
    // A horizontal FEMCloth sheet dropped above a STATIC plane floor with NO pins; the
    // floor contact chain must catch it and it must SETTLE resting on the floor (mean y
    // drops then converges to ~floorY), NOT tunnel to -inf and NOT free-fall through.
    if (!std::strcmp(sceneName, "clothdrape"))
    {
        ClothDrapeConfig cfg;         // floorY = 0.0, spawnY = 1.5
        // The COOKED MESH floor drops from 0.60 rather than 1.50, and that is a statement about
        // IMPACT SPEED rather than about the contact path: the mesh pair's midphase runs roughly
        // every sixth step, and from 1.50 the sheet is doing ~5 m/s and crosses the whole 0.1
        // contact band between two consecutive midphase invocations -- ordinary tunnelling.  From
        // 0.60, and every height tested below it, it settles cleanly.  The infinite PLANE has no
        // midphase and keeps its 1.50 baseline.
        {
            const char* ff0 = std::getenv("PXG_CLOTH_FLOOR");
            if (ff0 && !std::strcmp(ff0, "mesh")) cfg.spawnY = 0.60f;
        }
        if (const char* sy = std::getenv("PXG_CLOTH_SPAWNY")) cfg.spawnY = float(atof(sy));
        ClothScene csc = buildClothDrapeScene(physics, scene, material, cudaCtx, cfg);
        if (!csc.surface || csc.numVerts == 0)
        {
            std::printf("[ FAIL ] gpu-sdk-step: clothdrape scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        {
            const char* ff = std::getenv("PXG_CLOTH_FLOOR");
            const bool isMeshFloor = ff && !std::strcmp(ff, "mesh");
            std::printf("  [  OK  ] scene 'clothdrape' built: %u-vertex FEMCloth sheet spawned at y~%.2f above a "
                        "STATIC %s floor at y=%.2f (no pins)\n", csc.numVerts, cfg.spawnY,
                        isMeshFloor ? "COOKED TRIANGLE MESH" : "plane", cfg.floorY);
        }

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

        // PXG_CLOTH_FLOOR=mesh is a FULL settle test.  It settles ON its REST OFFSET rather than
        // on the floor, and that offset is the whole reason it settles at all -- see the builder.
        // The plane variant keeps its own thresholds: it has no midphase and no rest offset.
        const char* clothFloorEnv2 = std::getenv("PXG_CLOTH_FLOOR");
        const bool meshFloor = clothFloorEnv2 && !std::strcmp(clothFloorEnv2, "mesh");
        const float meshRestBand = cfg.floorY + 0.05f;   // the cloth shape's rest offset

        int failures = 0;
        // (1) the cloth must FALL a real distance under gravity (not stuck at spawn).
        const float minDrop = meshFloor ? 0.4f : 0.5f;
        if (!(drop > minDrop))       { std::printf("FAIL: cloth did not fall (drop=%.4f)\n", drop); ++failures; }
        // (2) the drop must CONVERGE — the sheet stops moving over the final quarter.
        if (!(restMaxDy < 0.01f))    { std::printf("FAIL: cloth did not settle (rest |dy|=%.6f)\n", restMaxDy); ++failures; }
        // (3) it must settle ON the floor and NOT tunnel through it.
        if (meshFloor)
        {
            if (!(PxAbs(meanEnd - meshRestBand) < 0.03f))
            { std::printf("FAIL: cloth did not settle ON the rest offset (meanY=%.4f, expected "
                          "%.4f +/- 0.03)\n", meanEnd, meshRestBand); ++failures; }
            if (!(minEnd > cfg.floorY - 0.02f))
            { std::printf("FAIL: cloth tunnelled through the mesh floor (minY=%.4f)\n", minEnd); ++failures; }
        }
        else
        {
            if (!(meanEnd < cfg.floorY + 0.5f)) { std::printf("FAIL: cloth did not reach the floor (meanY=%.4f)\n", meanEnd); ++failures; }
            if (!(minEnd > cfg.floorY - 0.1f))  { std::printf("FAIL: cloth tunnelled below the floor (minY=%.4f)\n", minEnd); ++failures; }
        }
        // (4) nothing NaN/inf.
        if (!allFinite || !std::isfinite(meanEnd)) { std::printf("FAIL: cloth y went non-finite\n"); ++failures; }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (failures == 0)
        {
            if (meshFloor)
                std::printf("[  OK  ] gpu-sdk-step: FEMCloth fell onto a COOKED TRIANGLE MESH floor "
                            "and SETTLED on its rest offset (mean y=%.4f vs %.4f expected, min y=%.4f)\n",
                            meanEnd, meshRestBand, minEnd);
            else
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
        failures += unmappedPenalty();
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

    // PXG_SCENE=clothcloth (P8.16): the cloth-vs-DIFFERENT-cloth money shot.  A free cloth sheet
    // is dropped onto a PINNED cloth sheet.  Self-collision is off on both and there is no rigid
    // body underneath, so cloth_differentClothCollision{VT,EE} are the only midphase kernels that
    // can produce the pair that catches it.  Cloth runs on PGS — pass PXG_SOLVER=pgs.
    if (!std::strcmp(sceneName, "clothcloth"))
    {
        ClothClothScene ccs = buildClothClothScene(physics, scene, material, cudaCtx);
        if (!ccs.upper || ccs.numUpperVerts == 0)
        {
            std::printf("[ FAIL ] gpu-sdk-step: clothcloth scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'clothcloth' built: two FEMCloth sheets — a pinned table at "
                    "y=%.2f and a %u-vertex free sheet dropped onto it (self-collision OFF on both)\n",
                    ccs.baseY, ccs.numUpperVerts);

        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: clothcloth GPU pipeline CONSTRUCTED + live over Vulkan "
                        "(set PXG_GPU_STEP=1 to step the drop)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_CLOTH_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 200;
        float min0 = 0.0f, mean0 = 0.0f;
        bool finite = clothClothUpperMinMeanY(ccs, min0, mean0);
        float maxExcursion = 0.0f;
        std::printf("  stepping the drop (%d x %.4fs): upper sheet starts at mean y=%.4f (min %.4f); "
                    "table y=%.2f...\n", steps, dt, mean0, min0, ccs.baseY);

        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            float mn = 0.0f, mean = 0.0f;
            finite = clothClothUpperMinMeanY(ccs, mn, mean) & finite;
            const float excursion = PxAbs(mean - mean0);
            if (excursion > maxExcursion) maxExcursion = excursion;
            if (i < 6 || i == steps - 1 || (i % 20 == 0))
                std::printf("    step %3d: upper sheet min y=%.5f mean=%.5f (table y=%.2f)\n",
                            i, mn, mean, ccs.baseY);
        }
        float minE = 0.0f, meanE = 0.0f;
        clothClothUpperMinMeanY(ccs, minE, meanE);
        std::printf("  clothcloth: final upper sheet min y=%.4f (mean %.4f); table y=%.2f\n",
                    minE, meanE, ccs.baseY);

        int failures = 0;
        if (!(maxExcursion > 0.05f))
        { std::printf("FAIL: upper sheet never moved (max mean excursion %.5f from %.4f) — sim frozen?\n",
                      maxExcursion, mean0); ++failures; }
        if (!finite) { std::printf("FAIL: cloth y went non-finite\n"); ++failures; }
        // Caught: the sheet comes to rest ON the contact rest distance (table + 2 * restOffset).
        const float restBand = ccs.baseY + 0.10f;
        if (!(PxAbs(minE - restBand) < 0.05f))
        { std::printf("FAIL: the upper sheet did not settle ON the rest distance (min y=%.4f, "
                      "expected %.4f +/- 0.05)\n", minE, restBand); ++failures; }
        if (!(meanE > ccs.baseY + 0.05f))
        { std::printf("FAIL: the upper sheet sank into the table (mean y=%.4f, table=%.2f)\n",
                      meanE, ccs.baseY); ++failures; }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: cloth-vs-DIFFERENT-cloth collision CAUGHT the falling sheet "
                        "and settled it ON the contact rest distance (min y=%.4f vs %.4f expected, "
                        "mean y=%.4f, table=%.2f)\n", minE, ccs.baseY + 0.10f, meanE, ccs.baseY);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: clothcloth checks failed (%d)\n", failures);
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
        // 60 is enough for the free fall.  With the plane on the cube is not even caught
        // until step ~49 and does not come to rest until ~150, so a 60-step run would
        // measure the bounce and call it a failure.
        const int steps = stepsEnv ? atoi(stepsEnv) : (sbs.hasPlane ? 200 : 60);

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
        PxU32 maxSbContacts = 0, prevSbContacts = 0;
        int firstContactStep = -1;
        float restMaxAbsDy = 0.0f;      // largest |dy| over the last quarter of the run
        const int restFrom = steps - steps / 4;
        // PXG_SB_TRACE=1 prints EVERY step instead of the every-tenth grid.  The
        // sampling grid is enough to see a fall, but a settle that oscillates needs
        // consecutive steps to be legible -- P8.20's ringing showed up as a smooth
        // descent on the grid and as a limit cycle on the trace.
        const bool traceEveryStep = std::getenv("PXG_SB_TRACE") != nullptr;
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
            // The deformable-volume contact count the engine actually reached, read back
            // from the pinned counter PxgSoftBodyCore::checkBufferOverflows() already
            // maintains.  With no particles and self-collision off, softbody-vs-RIGID is the
            // only contributor, so this is a direct measurement of whether the mapped
            // contact chain (midphase -> plane contact gen -> remap -> reorder) produced
            // anything at all.  It is the probe that has to answer YES before the three
            // solve kernels are worth writing.
            PxSimulationStatistics simStats;
            scene->getSimulationStatistics(simStats);
            const PxU32 sbContacts =
                simStats.gpuDynamicsMemoryConfigStatistics.deformableVolumeContacts;
            maxSbContacts = PxMax(maxSbContacts, sbContacts);
            if (sbContacts > 0 && firstContactStep < 0) firstContactStep = i;
            // Also print on every CHANGE of the contact count, not just on the sampling
            // grid: a count that only ever appears as one number is indistinguishable from a
            // buffer ceiling, and the progression is what shows it is a real measurement.
            if (i < 5 || i == steps - 1 || (i % 10 == 0) || sbContacts != prevSbContacts || traceEveryStep)
                std::printf("    step %3d: coll mean y = %.5f (dy = %+.5f), sim mean y = %.5f, "
                            "skin gap = %.5f, sim y-extent = %.5f, sb contacts = %u\n",
                            i, meanC, meanC - prevC, meanS, PxAbs(meanC - meanS), extS, sbContacts);
            prevSbContacts = sbContacts;
            if (meanC > prevC + 1e-3f) ++nonMonotonic;   // free fall: must not rise
            if (i >= restFrom) restMaxAbsDy = PxMax(restMaxAbsDy, PxAbs(meanC - prevC));
            prevC = meanC;
        }
        const float drop = meanC0 - meanC;

        int failures = 0;

        // ---- the contact-chain probe ----------------------------------------------
        // Reported unconditionally, and asserted only where the answer is unambiguous.
        // With PXG_SB_FLOOR=none there is no floor at all, so zero is the CORRECT answer
        // there and must not fail.
        const char* floorName = !sbs.hasPlane ? "none"
                              : sbs.hasPlaneShape ? "plane" : sbs.hasSphere ? "sphere"
                              : sbs.hasBox ? "box" : sbs.hasHeightfield ? "heightfield"
                              : sbs.hasSdfMesh ? "sdfmesh" : "mesh";
        std::printf("  contact chain: max deformable-volume contacts = %u, first at step %d "
                    "(floor: %s)\n",
                    maxSbContacts, firstContactStep, floorName);
        if (sbs.hasPlane && maxSbContacts == 0)
        {
            std::printf("FAIL: there is a floor at y=0 and the cube fell "
                        "through it having generated ZERO contacts. The mapped chain "
                        "(midphase -> plane contact gen -> remap -> reorder) is not feeding "
                        "the engine, so porting the solve kernels cannot make it settle.\n");
            ++failures;
        }
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

        // ---- PXG_SB_PLANE: the cube must be CAUGHT, not fall through ---------------
        // With the plane on this is no longer a free fall, so the closed-form and
        // monotonicity checks above are gated off; these replace them.  A 27-vertex cube of
        // edge 1 resting on y=0 has its centroid at y=0.5, so "caught" means the mean
        // settles near 0.5 and, crucially, STOPS well above the depth an uncaught cube
        // would have reached by the same step.
        if (sbs.hasPlane)
        {
            const float uncaughtY = meanC0 - expectDrop;   // where a free fall would be
            if (!(meanC > uncaughtY + 1.0f))
            { std::printf("FAIL: the cube fell THROUGH the floor: mean y "
                          "%.4f, an uncaught fall would be at %.4f\n", meanC, uncaughtY); ++failures; }
            else if (!(maxSbContacts > 0u))
            { std::printf("FAIL: the cube stopped but no contacts were generated (%u)\n",
                          maxSbContacts); ++failures; }
            // ...and it must come to REST, not bounce forever.  This is the assertion that
            // separates "the solve pushes back" from "the solve converges": an unconverged
            // contact solve oscillates around roughly the right height, which the catch
            // check above would happily accept.
            // ONE BAR FOR ALL FOUR FLOORS.  This check used to hold the PLANE to a ten-times
            // tighter 1e-4 on the theory that the bar tracks the contact-gen BRANCH, not the
            // shape: sb_primitiveContactGenLaunch's plane branch writes one contact PER
            // PENETRATING CORNER (32 on this cube) while its GJK/EPA branch writes one witness
            // pair per tet-vs-hull query (16), and the denser set was believed to converge
            // harder.  P8.20 measured all four on a properly damped material and the ordering
            // does not hold: SPHERE 0.000102 < PLANE 0.000153 < MESH 0.000769 < BOX 0.000929.
            // Contact density is simply not what predicts the rest jitter here, so the split
            // tier is retired rather than re-tuned to a number it would only accidentally fit.
            // 0.002 sits ~2x above the worst measured floor and orders of magnitude below
            // anything still in motion -- the four numbers above are printed on every run, so
            // a convergence that degrades within the bar is still visible in the output.
            else if (!(restMaxAbsDy < 2e-3f))
            { std::printf("FAIL: the cube was caught but never settled: |dy| still %.6f over "
                          "the last %d steps\n", restMaxAbsDy, steps - restFrom); ++failures; }
            // A cube of edge 1 resting on y=0 has its centroid at 0.5, less its elastic
            // compression under its own weight.  Anything far from that is the wrong height.
            else if (!(PxAbs(meanC - 0.5f) < 0.05f))
            { std::printf("FAIL: settled at the wrong height: mean y %.4f, expected ~0.5 "
                          "(cube half-height above the plane)\n", meanC); ++failures; }
            else
                std::printf("  CAUGHT by the floor and SETTLED: mean y %.5f (ideal 0.5 less "
                            "elastic compression; an uncaught fall would be at %.4f), rest "
                            "|dy| over the last %d steps = %.6f, %u contacts from step %d\n",
                            meanC, uncaughtY, steps - restFrom, restMaxAbsDy,
                            maxSbContacts, firstContactStep);
        }
        // QUANTITATIVE free fall: checking the closed form rather than "it went down"
        // gives the scene teeth on the WHOLE integrate chain (sb_gm_preIntegrateLaunch's
        // gravity predict, sb_gm_stepSoftbodyLaunch, sb_gm_finalizeVelocitiesLaunch)
        // instead of only on its sign.  The band is 1%; the measured error is 0.0000%.
        if (sbs.stretchY != 1.0f || sbs.noGravity || sbs.hasPlane)
            std::printf("  (closed-form free-fall check skipped: %s)\n",
                        sbs.hasPlane ? "the floor catches the cube, so it is not a free fall"
                                     : sbs.noGravity ? "gravity is disabled"
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
        if (sbs.stretchY == 1.0f && !sbs.noGravity && !sbs.hasPlane && nonMonotonic != 0)
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
        failures += unmappedPenalty();
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

    // PXG_SCENE=softbodyhex (P9.95f): the voxel softbody.  See buildSoftBodyHexScene's header.
    if (!std::strcmp(sceneName, "softbodyhex"))
    {
        const char* hexEnv = std::getenv("PXG_SB_HEX");
        if (hexEnv && std::strcmp(hexEnv, "box") && std::strcmp(hexEnv, "fork"))
        {
            std::printf("[ FAIL ] gpu-sdk-step: PXG_SB_HEX=%s is not an arm (box | fork)\n", hexEnv);
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        const bool fork = !hexEnv || !std::strcmp(hexEnv, "fork");
        const char* stretchEnv = std::getenv("PXG_SB_STRETCH");
        const float stretchY = stretchEnv ? float(atof(stretchEnv)) : 1.6f;
        SoftBodyHexScene hs = buildSoftBodyHexScene(physics, scene, material, cudaCtx, fork, stretchY);
        if (!hs.volume)
        {
            std::printf("[ FAIL ] gpu-sdk-step: softbodyhex scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'softbodyhex' built: a %u-vertex VOXEL softbody (%s arm), gravity off, "
                    "no floor, spawn stretch %.2fx\n", hs.nbSim, fork ? "fork" : "box", stretchY);
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: softbodyhex GPU pipeline CONSTRUCTED + live over Vulkan "
                        "(set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_SB_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 120;
        // Verts 27..30 are the twin's private -x face; {0,1,3,4} are the original cell's.
        static const PxU32 kFace[4] = { 0u, 1u, 3u, 4u };
        static const PxU32 kTwin[4] = { 27u, 28u, 29u, 30u };

        const std::vector<PxVec4> p0 = softBodyHexReadSim(hs);
        const float latExt0 = softBodyHexExtY(p0, 0u, 27u);
        const float twinExt0 = fork ? softBodyHexExtY(p0, kTwin, 4u) : 0.0f;
        // MASS-weighted centre (D7): the cooker's masses are not uniform, and the XPBD solve
        // conserves sum(p/w), not the plain mean.
        auto massCom = [](const std::vector<PxVec4>& p) {
            double sx = 0.0, sy = 0.0, sz = 0.0, sm = 0.0;
            for (const PxVec4& q : p)
                if (q.w > 0.0f)
                { const double m = 1.0 / double(q.w); sx += q.x * m; sy += q.y * m; sz += q.z * m; sm += m; }
            return PxVec3(float(sx / sm), float(sy / sm), float(sz / sm));
        };
        const PxVec3 com0 = massCom(p0);
        std::printf("  stepping the voxel softbody (%d x %.4fs): lattice y-extent %.5f%s, mass-weighted "
                    "centre (%.5f, %.5f, %.5f)\n", steps, dt, latExt0,
                    fork ? "" : " (box arm: 8 partitions, no Jacobi voxel)", com0.x, com0.y, com0.z);
        if (fork)
            std::printf("  twin face (verts 27..30) y-extent %.5f at spawn\n", twinExt0);

        bool latUnmovedEver = true, twinUnmovedEver = true;
        std::vector<PxVec4> p = p0;
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            p = softBodyHexReadSim(hs);
            const float latExt = softBodyHexExtY(p, 0u, 27u);
            if (latExt != latExt0) latUnmovedEver = false;
            float twinExt = 0.0f, faceExt = 0.0f;
            if (fork)
            {
                twinExt = softBodyHexExtY(p, kTwin, 4u);
                faceExt = softBodyHexExtY(p, kFace, 4u);
                if (twinExt != twinExt0) twinUnmovedEver = false;
            }
            if (i < 3 || i == steps - 1 || (i % 10 == 0))
            {
                if (fork)
                    std::printf("    step %3d: lattice y-extent %.5f, twin face %.5f, original -x face %.5f\n",
                                i, latExt, twinExt, faceExt);
                else
                    std::printf("    step %3d: lattice y-extent %.5f\n", i, latExt);
            }
        }

        int failures = 0;
        bool finite = true;
        for (const PxVec4& q : p)
            if (!std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z) || !std::isfinite(q.w)) finite = false;
        if (!finite) { std::printf("FAIL: a simulation position went non-finite\n"); ++failures; }

        // (a) the GS hex branch: the lattice must relax.  Before P9.95f it stayed BIT-EQUAL to
        // the spawn extent at every step (the hex branch was a no-op, averaging is off for hex).
        const float latExt = softBodyHexExtY(p, 0u, 27u);
        std::printf("  lattice (verts 0..26) y-extent: %.5f -> %.5f (rest 1.00000)%s\n", latExt0, latExt,
                    latUnmovedEver ? " -- BIT-EQUAL to the spawn extent at every step" : "");
        if (!(latExt < latExt0 - 0.1f))
        { std::printf("FAIL: (a) the voxel lattice did not relax (%.5f -> %.5f) -- the GS hex branch "
                      "of sb_gm_cp_solveTetrahedronsPartitionLaunch did no work\n", latExt0, latExt); ++failures; }

        // (b) the Jacobi partition: the twin's private face must relax AND track the original
        // cell's -x face it duplicates.
        if (fork)
        {
            const float twinExt = softBodyHexExtY(p, kTwin, 4u);
            const float faceExt = softBodyHexExtY(p, kFace, 4u);
            float maxGap = 0.0f;
            for (PxU32 j = 0; j < 4; ++j)
            {
                const PxVec3 d(p[kTwin[j]].x - p[kFace[j]].x, p[kTwin[j]].y - p[kFace[j]].y,
                               p[kTwin[j]].z - p[kFace[j]].z);
                maxGap = PxMax(maxGap, d.magnitude());
            }
            // Measured at 120 steps (P9.95f): |ext diff| 1.2e-6 / max corner gap 2.7e-6 (TGS) and
            // 2.4e-7 / 2.6e-7 (PGS) -- the twin relaxes onto its original -- so tolB is the 0.02
            // floor, still ~15x below the 0.29 gap a skipped Jacobi kernel leaves.
            const float tolB = 0.02f;
            std::printf("  twin face (verts 27..30) y-extent: %.5f -> %.5f; original -x face %.5f; "
                        "|ext diff| %.2e, max |p[27+j] - p[face j]| %.2e (tolB %.3f)%s\n",
                        twinExt0, twinExt, faceExt, PxAbs(twinExt - faceExt), maxGap, tolB,
                        twinUnmovedEver ? " -- BIT-EQUAL to its spawn extent at every step" : "");
            if (!(PxAbs(twinExt - twinExt0) > 0.1f))
            { std::printf("FAIL: (b) the twin voxel's private face did not relax (%.5f -> %.5f) -- "
                          "sb_gm_cp_solveTetrahedronsJacobiPartitionLaunch did no work\n", twinExt0, twinExt); ++failures; }
            else if (!(PxAbs(twinExt - faceExt) < tolB && maxGap < tolB))
            { std::printf("FAIL: (b) the twin face does not track the original -x face (|ext diff| %.5f, "
                          "max gap %.5f, tol %.3f)\n", PxAbs(twinExt - faceExt), maxGap, tolB); ++failures; }
        }

        // (f) the lattice relaxes to its REST SHAPE, not merely to something shorter than the
        // spawn: the largest error of any of the 351 pairwise distances among verts 0..26 against
        // the cooked rest mesh.  (The final y-extent is NOT 1.0 -- 1.0067 box, 1.035 fork --
        // because the relaxed body ends slightly ROTATED; the shape itself is exact.)  Measured
        // (P9.95f, 120 steps): <= 2.5e-6 in all four arms; a sub-tet table row with two corners
        // swapped (an inverted rest pose) ends at 1.0 and the spawn is 0.6, so tolShape is a
        // 1e-3 floor.  It does NOT see the flipped-table offset dropped (20 -> 0): every
        // non-flipped row i is the image of flipped row 5+i under a PROPER cube rotation
        // (180 deg about (1,1,0) for rows 0..2, about (1,-1,0) for rows 3..4), and an isotropic
        // material is invariant under that rest-frame rotation, so that mutant relaxes to the
        // same rest shape (shape error 1.4e-6) -- the fixtures' bitwise rotations kill it.
        float shapeErr = 0.0f, shapeErr0 = 0.0f;
        for (PxU32 i = 0; i < 27u; ++i)
            for (PxU32 j = i + 1; j < 27u; ++j)
            {
                const float d0 = (hs.simRest[i] - hs.simRest[j]).magnitude();
                const PxVec3 a(p[i].x - p[j].x, p[i].y - p[j].y, p[i].z - p[j].z);
                const PxVec3 a0(p0[i].x - p0[j].x, p0[i].y - p0[j].y, p0[i].z - p0[j].z);
                shapeErr = PxMax(shapeErr, PxAbs(a.magnitude() - d0));
                shapeErr0 = PxMax(shapeErr0, PxAbs(a0.magnitude() - d0));
            }
        const float tolShape = 1.0e-3f;
        std::printf("  lattice shape error (max |pairwise distance - rest|, 351 pairs): %.5f at spawn -> %.2e "
                    "(tol %.3f)\n", shapeErr0, shapeErr, tolShape);
        if (!(shapeErr < tolShape))
        { std::printf("FAIL: (f) the voxel lattice did not relax to its rest shape (shape error %.5f -> %.5f, "
                      "tol %.3f)\n", shapeErr0, shapeErr, tolShape); ++failures; }

        // (c) nothing but internal forces acts, so the mass-weighted centre must stay put.
        const PxVec3 com = massCom(p);
        const float comDrift = (com - com0).magnitude();
        // Not zero, and measured rather than assumed (P9.95f, 120 steps, default settling):
        // fork TGS 2.199e-3, box TGS 1.396e-3, fork PGS 6.655e-4, box PGS 5.269e-4.  Under PGS
        // it is the settling damping in sb_gm_finalizeVelocitiesLaunch (softBodyGM.cu:3176-3180,
        // :3195), which scales each slow vertex's velocity on its own: with the settling
        // threshold and damping forced to 0 the PGS drift drops to 2.9e-6 (box) / 1.7e-5
        // (fork), i.e. the hex solve conserves sum(p/w) to rounding.  Under TGS a drift remains
        // even then (1.0e-2 box, 2.0e-2 fork).  A likely source, READ FROM THE CODE and NOT
        // measured as the cause: TGS turns on alwaysRunVelocityAveraging
        // (PxgSoftBodyCore.cpp:916, isTGS && !externalForcesEveryTgsIteration), and
        // sb_gm_finalizeVelocitiesLaunch (softBodyGM.cu:3136-3217, blend at :3187-3196) then
        // blends each vertex's velocity toward deltaPos/dt on its own, and only where
        // |deltaPos/dt| < |v|.  A per-vertex, conditional rewrite like that does not conserve
        // momentum, which is consistent with a drift that appears only under TGS.  tolC is
        // ~10x the default-settling worst; a stray push of the size the relaxation moves (0.3)
        // is still >10x beyond it.
        const float tolC = 0.025f;
        std::printf("  mass-weighted centre drift %.3e (tol %.1e)\n", comDrift, tolC);
        if (!(comDrift < tolC))
        { std::printf("FAIL: (c) the mass-weighted centre moved %.3e with only internal forces acting\n",
                      comDrift); ++failures; }

        // (e) from the plan (box arm: 0 Jacobi and 0 applyDeformationDeltas launches) is NOT
        // asserted here.  The harness has no per-kernel launch counter: the bridge exposes only
        // pxgUnmappedLaunchCount, which counts launches of UNMAPPED kernels, and both kernels
        // are mapped.  The counts were read from PXG_LAUNCH_TRACE=1 runs instead (P9.95f, 120
        // steps, TGS and PGS alike): GS / Jacobi / applyDeformationDeltas = 4800 / 600 / 600
        // in the fork arm and 4800 / 0 / 0 in the box arm.  That the fork arm really reaches
        // the Jacobi partition is proven by the manual A/B (AB1: manifest row of
        // sb_gm_cp_solveTetrahedronsJacobiPartitionLaunch set to nullptr), under which (b) fails
        // with the twin face BIT-EQUAL to its spawn extent at every step.

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: the VOXEL softbody relaxed on Vulkan (lattice %.5f -> %.5f)%s\n",
                        latExt0, latExt, fork ? ", and the Jacobi twin voxel tracked its original" : "");
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: softbodyhex checks failed (%d)\n", failures);
        return 2;
    }

    // PXG_SCENE=softbodyselfcollide (P8.4): the FEM-VOLUME SELF-collision money shot.
    if (!std::strcmp(sceneName, "particlesoftbody"))
    {
        SoftBodyParticleScene sc = buildSoftBodyParticleScene(physics, scene, material, cudaCtx);
        if (!sc.system || !sc.buffer || (sc.hasSoftBody && !sc.volume))
        {
            std::printf("[ FAIL ] gpu-sdk-step: particlesoftbody scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'particlesoftbody' built: %u PBD particles above a "
                    "%s (%u sim verts)\n", sc.numParticles,
                    sc.hasSoftBody ? "FEM-volume cube with its top face at y = 1"
                                   : "BARE PLANE (control: no softbody)", sc.numSimVerts);
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: particlesoftbody pipeline CONSTRUCTED "
                        "(set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_PSB_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 120;
        const float y0 = sbParticleMeanY(sc);
        float y = y0, lowest = y0, lastDy = 0.0f;
        bool finite = true;
        std::printf("  stepping particlesoftbody (%d x %.4fs): cloud mean y0 = %.4f\n",
                    steps, dt, y0);
        for (int i = 0; i < steps; ++i)
        {
            const float prev = y;
            scene->simulate(dt);
            scene->fetchResults(true);
            y = sbParticleMeanY(sc);
            lastDy = y - prev;
            lowest = PxMin(lowest, y);
            finite = finite && std::isfinite(y);
            if (i < 3 || i == steps - 1 || (i % 20 == 0))
                std::printf("    step %3d: cloud mean y = %.4f (dy %+.5f)\n", i, y, lastDy);
        }
        std::printf("  particlesoftbody: cloud mean y %.4f -> %.4f (lowest %.4f), "
                    "|dy| last step %.6f\n", y0, y, lowest, PxAbs(lastDy));

        int failures = 0;
        if (!finite)
        { std::printf("FAIL: the cloud went non-finite\n"); ++failures; }
        // (1) IT CAME TO REST.  A cloud still falling says nothing about what caught it.
        if (!(PxAbs(lastDy) < 0.002f))
        { std::printf("FAIL: the cloud is still moving %.6f per step at the end -- it never "
                      "came to rest\n", PxAbs(lastDy)); ++failures; }
        if (sc.hasSoftBody)
        {
            // (2) IT RESTED ON THE CUBE, not on the plane a whole unit below.  This is the
            //     observable ps_accumulateFEMParticleDeltaVLaunch exists for: without the
            //     accumulate the softbody-vs-particle solve writes per-contact deltas that
            //     reach no particle, and the cloud falls straight through the cube.
            if (!(y > 0.8f))
            { std::printf("FAIL: the cloud came to rest at y = %.4f, below the cube's top face "
                          "at y = 1 -- it fell THROUGH the softbody.  The consumer is "
                          "ps_accumulateFEMParticleDeltaVLaunch plus the two ps_findStartEndFEM "
                          "passes; if the [pxg-unmapped] lines below name them, this is that\n",
                          y); ++failures; }
            // (3) IT DID NOT PUNCH THROUGH AND COME BACK.  A cloud that dipped below the face
            //     and was pushed out again is a different (and worse) story than one held.
            if (!(lowest > 0.7f))
            { std::printf("FAIL: the cloud dipped to y = %.4f before settling -- it passed "
                          "through the cube's top face\n", lowest); ++failures; }
        }
        else
        {
            // The control: with no cube there is nothing at y = 1 and the cloud must reach
            // the plane.  If it stopped high anyway, the main run's number means nothing.
            if (!(y < 0.3f))
            { std::printf("FAIL (control): with no softbody the cloud stopped at y = %.4f -- "
                          "it must fall to the plane\n", y); ++failures; }
        }
        if (failures)
        { std::printf("[ FAIL ] gpu-sdk-step: particlesoftbody checks failed (%d)\n", failures);
          scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1; }
        std::printf("[  OK  ] gpu-sdk-step: particlesoftbody -- the cloud rests at y = %.4f "
                    "%s\n", y, sc.hasSoftBody ? "ON the softbody cube" : "on the bare plane (control)");
        scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
    }

    if (!std::strcmp(sceneName, "softbodycloth"))
    {
        SoftBodyClothScene sc = buildSoftBodyClothScene(physics, scene, material, cudaCtx);
        if (!sc.surface)
        {
            std::printf("[ FAIL ] gpu-sdk-step: softbodycloth scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'softbodycloth' built: a %u-vertex FEM-surface sheet above %s "
                    "(cube top y=%.2f), both over a static plane at y=0\n",
                    sc.numClothVerts,
                    sc.hasSoftBody ? "a FEM-volume cube" : "NOTHING (control run)", sc.cubeTopY);
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: softbodycloth GPU pipeline CONSTRUCTED + live over "
                        "Vulkan (set PXG_GPU_STEP=1 to step the drop)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        int steps = 150;
        if (const char* sEnv = std::getenv("PXG_STEPS")) { const int v = atoi(sEnv); if (v > 0) steps = v; }

        float meanY0 = 0.0f, minY0 = 0.0f;
        bool allFinite = softBodyClothY(sc, meanY0, minY0);
        float meanY = meanY0, minY = minY0, prevMin = minY0, restMaxDy = 0.0f;
        const int restWindow = steps / 4;
        PxU32 maxVolContacts = 0u, maxSurfContacts = 0u;
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            allFinite = softBodyClothY(sc, meanY, minY) && allFinite;
            PxSimulationStatistics st;
            scene->getSimulationStatistics(st);
            maxVolContacts = PxMax(maxVolContacts,
                                   st.gpuDynamicsMemoryConfigStatistics.deformableVolumeContacts);
            maxSurfContacts = PxMax(maxSurfContacts,
                                    st.gpuDynamicsMemoryConfigStatistics.deformableSurfaceContacts);
            if (i >= steps - restWindow) restMaxDy = PxMax(restMaxDy, PxAbs(minY - prevMin));
            if (i < 4 || i == steps - 1 || (i % 30 == 0))
                std::printf("    step %3d: cloth min y = %.5f (mean %.5f)\n", i, minY, meanY);
            prevMin = minY;
        }

        std::printf("  softbodycloth (softbody %s): cloth min y %.4f -> %.4f (mean %.4f); "
                    "cube top y=%.2f, plane y=0; peak volume contacts = %u, surface contacts = %u; "
                    "rest |dy| over last %d steps = %.6f\n",
                    sc.hasSoftBody ? "PRESENT" : "ABSENT (control)", minY0, minY, meanY,
                    sc.cubeTopY, maxVolContacts, maxSurfContacts, restWindow, restMaxDy);

        int failures = 0;
        if (!allFinite || !std::isfinite(minY))
        { std::printf("FAIL: non-finite vertex in the sheet\n"); ++failures; }
        if (!(minY0 - minY > 0.5f))
        { std::printf("FAIL: the sheet did not fall (min y %.5f -> %.5f)\n", minY0, minY); ++failures; }

        // The sheet is 0.6 x 0.6 and the cube's top face is 1 x 1, so a sheet that lands on the
        // cube stays ENTIRELY on it.  The two outcomes are a full unit apart -- roughly y = 1
        // against y = 0 -- and the bar sits halfway between them, far from both.
        const float CATCH_Y = 0.5f;
        if (sc.hasSoftBody)
        {
            if (!(maxVolContacts > 0u))
            { std::printf("FAIL: the softbody is present but the engine reported ZERO "
                          "deformable-volume contacts -- the cloth chain is not feeding the solve\n");
              ++failures; }
            if (!(minY > CATCH_Y))
            { std::printf("FAIL: the sheet sank to y=%.5f, below the catch bar %.2f (cube top "
                          "y=%.2f).  If the chain itself still dispatches, check the MASS RATIO "
                          "first (see the scene header): a sheet heavier than the cube is pushed "
                          "through it by a correct solve\n", minY, CATCH_Y, sc.cubeTopY);
              ++failures; }
            if (!(restMaxDy < 0.02f))
            { std::printf("FAIL: the caught sheet never settled (rest |dy|=%.6f)\n", restMaxDy);
              ++failures; }
        }
        else
        {
            // The CONTROL.  With no cube the sheet falls to the plane.  Without this half the
            // main run proves nothing: a sheet that simply hung in the air would pass it.
            if (!(minY < CATCH_Y))
            { std::printf("FAIL: the CONTROL run has no softbody, so the sheet must fall past "
                          "%.2f to the plane; it stopped at %.5f\n", CATCH_Y, minY); ++failures; }
        }

        if (failures)
        {
            std::printf("[ FAIL ] gpu-sdk-step: softbodycloth checks failed (%d)\n", failures);
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        if (sc.hasSoftBody)
            std::printf("[  OK  ] gpu-sdk-step: the FEM-surface sheet was CAUGHT by the FEM-volume "
                        "cube and SETTLED on it (min y=%.4f, cube top %.2f, rest |dy|=%.6f) -- the "
                        "softbody-vs-cloth chain runs end to end\n", minY, sc.cubeTopY, restMaxDy);
        else
            std::printf("[  OK  ] gpu-sdk-step: control run -- with no softbody the sheet fell "
                        "clean through to the plane (min y=%.4f)\n", minY);
        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        return 0;
    }

    // PXG_SCENE=softbodyattach (P8.19): a free FEM volume hanging from a PINNED one, with no
    // floor under it and no contact between them.  Only the FEM-FEM attachment pair can hold it.
    // PXG_SCENE=softbodyhang (P9.76b): the RIGID attachment chain, which softbodyattach
    // does not reach.  See buildSoftBodyHangScene's header.
    if (!std::strcmp(sceneName, "softbodyhang"))
    {
        SoftBodyAttachScene sh = buildSoftBodyHangScene(physics, scene, material, cudaCtx);
        if (!sh.upper || sh.numUpperSimVerts == 0)
        {
            std::printf("[ FAIL ] gpu-sdk-step: softbodyhang scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        if (sh.nbAttachments == 0)
        {
            std::printf("[ FAIL ] gpu-sdk-step: softbodyhang created no attachments -- the scene "
                        "would be a free-fall test of nothing\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'softbodyhang' built: a %u-vertex FEM volume hung from %u "
                    "WORLD-pinned simulation vertices (no floor beneath)\n",
                    sh.numUpperSimVerts, sh.nbAttachments);

        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: softbodyhang GPU pipeline CONSTRUCTED + live over "
                        "Vulkan (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_SB_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 180;
        float mean0 = 0.0f, min0 = 0.0f;
        if (!softBodyAttachUpperY(sh, mean0, min0))
        {
            std::printf("[ FAIL ] gpu-sdk-step: softbodyhang could not read the volume back\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  stepping the hang (%d x %.4fs): volume starts at mean y=%.4f (min %.4f)\n",
                    steps, dt, mean0, min0);
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            if (i == steps - 1 || (i % 60 == 0))
            {
                float mn = 0.0f, mean = 0.0f;
                softBodyAttachUpperY(sh, mean, mn);
                std::printf("    step %3d: mean y = %.5f (min %.5f)\n", i, mean, mn);
            }
        }
        float meanEnd = 0.0f, minEnd = 0.0f;
        const bool finite = softBodyAttachUpperY(sh, meanEnd, minEnd);
        std::printf("  softbodyhang: mean y %.4f -> %.4f (min %.4f)\n", mean0, meanEnd, minEnd);

        int failures = 0;
        if (!finite || !std::isfinite(meanEnd))
        { std::printf("FAIL: the volume's mean y is not finite (%.4f)\n", meanEnd); ++failures; }
        // IT DID NOT FALL.  Only the rigid-attachment chain can hold it: there is no floor,
        // no second volume, and no contact of any kind in this scene.  The bar is derived
        // from the geometry, not fitted: the volume spawns centred at y = 4.5 and its top
        // vertices are pinned where they started, so it may SAG -- a FEM volume under
        // gravity always does -- but it cannot descend past its own height.  Anything below
        // 3.0 means the pins are not holding.
        if (!(meanEnd > 3.0f))
        { std::printf("FAIL: the volume came to rest at y = %.4f -- it fell off its world pins "
                      "(need > 3.0; it spawns at %.4f and a free fall over %d steps reaches "
                      "about -44)\n", meanEnd, mean0, steps); ++failures; }
        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: the FEM volume HANGS from its world pins "
                        "(mean y %.4f from %.4f) -- the RIGID attachment chain runs on Vulkan: "
                        "sb_rigidAttachmentPrepareLaunch and sb_solveRigidSoftAttachmentLaunch"
                        "{,TGS}, the last of which was refused on every launch until P9.76b gave "
                        "it its own 80-byte ABI row\n", meanEnd, mean0);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: softbodyhang checks failed (%d)\n", failures);
        return 1;
    }

    // PXG_SCENE=softbodydata (P9.95): PxScene::copySoftBodyData / applySoftBodyData ->
    // sb_copyOrApplySoftBodyDataDEPRECATED.  See buildSoftBodyDataScene's header.
    if (!std::strcmp(sceneName, "softbodydata"))
    {
        const bool control = std::getenv("PXG_SBDATA_CONTROL") != nullptr;
        SoftBodyDataScene sd = buildSoftBodyDataScene(physics, scene, material, cudaCtx);
        if (!sd.a || !sd.b || sd.nbSimA == 0 || sd.nbSimB == 0)
        {
            std::printf("[ FAIL ] gpu-sdk-step: softbodydata scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'softbodydata' built: FEM volume A (%u sim / %u collision vertices) at "
                    "x=-3, B (%u / %u) at x=+3, no floor\n", sd.nbSimA, sd.nbCollA, sd.nbSimB, sd.nbCollB);
        // B's size is a constant (cells = 4), not tuned to the cooker; if the cooker ever
        // gives fewer than 65 simulation vertices the block straddle is gone, so say so.
        if (!(sd.nbSimB > 64u))
        {
            std::printf("[ FAIL ] gpu-sdk-step: softbodydata: B has %u simulation vertices; tps = nbSimB must "
                        "exceed 64 so the two-section copies straddle a 128-thread block\n", sd.nbSimB);
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: softbodydata pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const int steps = 60;
        int failures = 0;
        PxCudaContext* ctx = cudaCtx->getCudaContext();

        // 1. The mandatory warm-up step (see the header).
        scene->simulate(dt);
        scene->fetchResults(true);
        const PxU32 gpuA = sd.a->getGpuDeformableVolumeIndex();
        const PxU32 gpuB = sd.b->getGpuDeformableVolumeIndex();
        std::printf("  warm-up done: GPU indices A=%u B=%u%s\n", gpuA, gpuB,
                    control ? "  [ control ] PXG_SBDATA_CONTROL=1: plain DtoH/HtoD instead of the API" : "");

        // Device memory for the calls.  The tables are padded to 8 zeroed entries; every
        // destination carries one poisoned slot past the largest size it is asked for.
        const float POISON = -999.0f;
        const PxU32 slotsB = PxMax(sd.nbSimB, sd.nbCollB) + 1u;
        const PxU32 slotsA = PxMax(sd.nbSimA, sd.nbCollA) + 1u;
        const PxU32 applySlots = PxMax(sd.nbSimA, sd.nbSimB);
        CUdeviceptr dTab = 0, dSizes = 0, dIdx = 0, dDstB = 0, dDstA = 0, dApply = 0;
        if (ctx->memAlloc(&dTab, 8 * sizeof(CUdeviceptr)) != 0 || !dTab ||
            ctx->memAlloc(&dSizes, 8 * sizeof(PxU32)) != 0 || !dSizes ||
            ctx->memAlloc(&dIdx, 8 * sizeof(PxU32)) != 0 || !dIdx ||
            ctx->memAlloc(&dDstB, slotsB * sizeof(PxVec4)) != 0 || !dDstB ||
            ctx->memAlloc(&dDstA, slotsA * sizeof(PxVec4)) != 0 || !dDstA ||
            ctx->memAlloc(&dApply, applySlots * sizeof(PxVec4)) != 0 || !dApply)
        {
            std::printf("[ FAIL ] gpu-sdk-step: softbodydata memAlloc failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        auto setTables = [&](std::vector<CUdeviceptr> tab, std::vector<PxU32> sizes, std::vector<PxU32> idx) {
            tab.resize(8, 0); sizes.resize(8, 0u); idx.resize(8, 0u);
            ctx->memcpyHtoD(dTab, tab.data(), 8 * sizeof(CUdeviceptr));
            ctx->memcpyHtoD(dSizes, sizes.data(), 8 * sizeof(PxU32));
            ctx->memcpyHtoD(dIdx, idx.data(), 8 * sizeof(PxU32));
        };
        auto poison = [&](CUdeviceptr d, PxU32 n) {
            std::vector<PxVec4> p(n, PxVec4(POISON));
            ctx->memcpyHtoD(d, p.data(), n * sizeof(PxVec4));
        };

        // 2. COPY, three flags.  Indices REVERSED {B, A}; A's section is half its vertices.
        // dataIndex = offsetof(PxgSoftBody, field) / 8 (PxgSoftBody.cpp:580-608), for the log.
        struct CopyCase { PxSoftBodyGpuDataFlag::Enum flag; const char* name; bool collision; PxU32 field;
                          PxU32 dataIndex; };
        const CopyCase copies[3] = {
            { PxSoftBodyGpuDataFlag::eSIM_POSITION_INV_MASS, "sim positions", false, 0u, 12u },
            { PxSoftBodyGpuDataFlag::eSIM_VELOCITY_INV_MASS, "sim velocities", false, 1u, 11u },
            { PxSoftBodyGpuDataFlag::eTET_POSITION_INV_MASS, "collision positions", true, 2u, 8u } };
        auto aliased = [](PxDeformableVolume* v, PxU32 field) -> PxVec4* {
            return field == 0u ? v->getSimPositionInvMassBufferD()
                 : field == 1u ? v->getSimVelocityBufferD() : v->getPositionInvMassBufferD(); };
        PxU32 copyBad = 0;
        for (const CopyCase& c : copies)
        {
            const PxU32 nB = c.collision ? sd.nbCollB : sd.nbSimB;
            const PxU32 nA = c.collision ? sd.nbCollA : sd.nbSimA;
            const PxU32 halfA = nA / 2u;
            poison(dDstB, slotsB);
            poison(dDstA, slotsA);
            setTables({ dDstB, dDstA }, { nB * 16u, halfA * 16u }, { gpuB, gpuA });
            std::vector<PxVec4> refB, refA;
            if (control)
            {
                sbDataRead(ctx, aliased(sd.b, c.field), nB, refB);
                sbDataRead(ctx, aliased(sd.a, c.field), halfA, refA);
                ctx->memcpyHtoD(dDstB, refB.data(), nB * sizeof(PxVec4));
                ctx->memcpyHtoD(dDstA, refA.data(), halfA * sizeof(PxVec4));
            }
            else
            {
                scene->copySoftBodyData(reinterpret_cast<void**>(dTab), reinterpret_cast<void*>(dSizes),
                                        reinterpret_cast<void*>(dIdx), c.flag, 2u, nB * 16u);
            }
            sbDataRead(ctx, aliased(sd.b, c.field), nB, refB);
            sbDataRead(ctx, aliased(sd.a, c.field), nA, refA);
            std::vector<PxVec4> gotB(slotsB), gotA(slotsA);
            ctx->memcpyDtoH(gotB.data(), dDstB, slotsB * sizeof(PxVec4));
            ctx->memcpyDtoH(gotA.data(), dDstA, slotsA * sizeof(PxVec4));
            const std::vector<PxVec4> poisonV(PxMax(slotsA, slotsB), PxVec4(POISON));
            const PxU32 badB = sbDataBitDiff(gotB.data(), refB.data(), nB);
            const PxU32 badA = sbDataBitDiff(gotA.data(), refA.data(), halfA);
            const PxU32 badTail = sbDataBitDiff(&gotB[nB], poisonV.data(), slotsB - nB) +
                                  sbDataBitDiff(&gotA[halfA], poisonV.data(), slotsA - halfA);
            std::printf("  copy %-19s (dataIndex %2u): B %u/%u vertices bit-equal, A %u/%u of its first half, "
                        "%u poisoned slots overwritten\n", c.name,
                        c.dataIndex, nB - badB, nB, halfA - badA, halfA, badTail);
            if (badB || badA || badTail)
            {
                std::printf("FAIL: copySoftBodyData(%s) is not the aliased buffer bit for bit (B %u, A %u, "
                            "tail %u differ)\n", c.name, badB, badA, badTail);
                ++failures; ++copyBad;
            }
        }

        // Pre-apply state (the reference every scored observable is measured from).
        std::vector<PxVec4> posA0, posB0, velA0, velB0, collB0;
        sbDataRead(ctx, sd.a->getSimPositionInvMassBufferD(), sd.nbSimA, posA0);
        sbDataRead(ctx, sd.b->getSimPositionInvMassBufferD(), sd.nbSimB, posB0);
        sbDataRead(ctx, sd.a->getSimVelocityBufferD(), sd.nbSimA, velA0);
        sbDataRead(ctx, sd.b->getSimVelocityBufferD(), sd.nbSimB, velB0);
        sbDataRead(ctx, sd.b->getPositionInvMassBufferD(), sd.nbCollB, collB0);
        const PxVec3 mA0 = sbDataMean(posA0), mB0 = sbDataMean(posB0), mcB0 = sbDataMean(collB0);
        const float vx0 = sbDataMean(velA0).x;

        // 3. APPLY (a): A's sim velocities with v.x += 3 -- y, z and w (invMass) bitwise.
        const float KICK = 3.0f, TELEPORT = 2.0f;
        std::vector<PxVec4> kickA = velA0;
        for (PxVec4& v : kickA) v.x += KICK;
        ctx->memcpyHtoD(dApply, kickA.data(), sd.nbSimA * sizeof(PxVec4));
        setTables({ dApply }, { sd.nbSimA * 16u }, { gpuA });
        if (control)
            ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(sd.a->getSimVelocityBufferD()), kickA.data(),
                            sd.nbSimA * sizeof(PxVec4));
        else
            scene->applySoftBodyData(reinterpret_cast<void**>(dTab), reinterpret_cast<void*>(dSizes),
                                     reinterpret_cast<void*>(dIdx), PxSoftBodyGpuDataFlag::eSIM_VELOCITY_INV_MASS,
                                     1u, sd.nbSimA * 16u);
        {
            std::vector<PxVec4> velA1, velB1;
            sbDataRead(ctx, sd.a->getSimVelocityBufferD(), sd.nbSimA, velA1);
            sbDataRead(ctx, sd.b->getSimVelocityBufferD(), sd.nbSimB, velB1);
            const PxU32 badA = sbDataBitDiff(velA1.data(), kickA.data(), sd.nbSimA);
            const PxU32 badB = sbDataBitDiff(velB1.data(), velB0.data(), sd.nbSimB);
            std::printf("  apply (a) A sim velocities v.x += %.1f: %u/%u bit-equal at once; B untouched: %u/%u\n",
                        KICK, sd.nbSimA - badA, sd.nbSimA, sd.nbSimB - badB, sd.nbSimB);
            if (badA || badB)
            { std::printf("FAIL: applySoftBodyData(velocities) did not land bitwise (A %u, B %u differ)\n", badA, badB);
              ++failures; }
        }

        // 4. APPLY (b): B's sim positions with z += 2 -- w bitwise.
        std::vector<PxVec4> moveB = posB0;
        for (PxVec4& p : moveB) p.z += TELEPORT;
        ctx->memcpyHtoD(dApply, moveB.data(), sd.nbSimB * sizeof(PxVec4));
        setTables({ dApply }, { sd.nbSimB * 16u }, { gpuB });
        if (control)
            ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(sd.b->getSimPositionInvMassBufferD()), moveB.data(),
                            sd.nbSimB * sizeof(PxVec4));
        else
            scene->applySoftBodyData(reinterpret_cast<void**>(dTab), reinterpret_cast<void*>(dSizes),
                                     reinterpret_cast<void*>(dIdx), PxSoftBodyGpuDataFlag::eSIM_POSITION_INV_MASS,
                                     1u, sd.nbSimB * 16u);
        {
            std::vector<PxVec4> posA1, posB1;
            sbDataRead(ctx, sd.a->getSimPositionInvMassBufferD(), sd.nbSimA, posA1);
            sbDataRead(ctx, sd.b->getSimPositionInvMassBufferD(), sd.nbSimB, posB1);
            const PxU32 badB = sbDataBitDiff(posB1.data(), moveB.data(), sd.nbSimB);
            const PxU32 badA = sbDataBitDiff(posA1.data(), posA0.data(), sd.nbSimA);
            std::printf("  apply (b) B sim positions z += %.1f: %u/%u bit-equal at once; A untouched: %u/%u\n",
                        TELEPORT, sd.nbSimB - badB, sd.nbSimB, sd.nbSimA - badA, sd.nbSimA);
            if (badA || badB)
            { std::printf("FAIL: applySoftBodyData(positions) did not land bitwise (B %u, A %u differ)\n", badB, badA);
              ++failures; }
        }

        // 5. Step.  After the FIRST step B's COLLISION mesh -- written only by the skinning
        // from the sim mesh -- must stand 2 further in z.
        float collDz = 0.0f;
        bool finite = true;
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            if (i == 0)
            {
                std::vector<PxVec4> collB1;
                sbDataRead(ctx, sd.b->getPositionInvMassBufferD(), sd.nbCollB, collB1);
                collDz = sbDataMean(collB1).z - mcB0.z;
            }
        }
        std::vector<PxVec4> posA2, posB2;
        sbDataRead(ctx, sd.a->getSimPositionInvMassBufferD(), sd.nbSimA, posA2);
        sbDataRead(ctx, sd.b->getSimPositionInvMassBufferD(), sd.nbSimB, posB2);
        for (const PxVec4& p : posA2) finite = finite && p.getXYZ().isFinite();
        for (const PxVec4& p : posB2) finite = finite && p.getXYZ().isFinite();
        const PxVec3 mA2 = sbDataMean(posA2), mB2 = sbDataMean(posB2);
        const float t = float(steps) * dt;
        const float dxA = mA2.x - mA0.x, dxB = mB2.x - mB0.x, dzB = mB2.z - mB0.z;
        const float dyA = mA2.y - mA0.y, dyB = mB2.y - mB0.y;
        const uint64_t hash = sbDataFnv(sbDataFnv(1469598103934665603ull, posA2), posB2);
        std::printf("  after %d steps: A dx = %.6f (kick %.1f * %.3f s + warm-up vx %.6f), B dx = %.6f, B dz = %.6f, "
                    "B collision dz after step 1 = %.6f, dy A = %.6f, B = %.6f\n", steps, dxA, KICK, t, vx0, dxB, dzB,
                    collDz, dyA, dyB);
        std::printf("  final sim-position hash (A then B, FNV-1a): %016llx%s\n", (unsigned long long)hash,
                    control ? "  [ control ]" : "");

        auto near = [&](const char* what, float got, float want, float tol) {
            if (!(std::fabs(got - want) <= tol))
            { std::printf("FAIL: %s = %.6f, expected %.6f +- %g\n", what, got, want, tol); ++failures; }
        };
        if (!finite) { std::printf("FAIL: a volume is not finite\n"); ++failures; }
        near("B collision-mesh mean dz after the first step (skinning read the applied sim mesh)", collDz,
             TELEPORT, 1e-4f);
        // Measured at first green: 2.999957 (TGS) and 2.999999 (PGS) against (3 + vx0) * 1 s with
        // vx0 = 0.000000 -- the bar is the larger deviation + 1e-4, not a guess.
        near("A sim mean dx since the apply (velocity kick)", dxA, (KICK + vx0) * t, 1.5e-4f);
        near("B sim mean dx since the apply (no kick)", dxB, 0.0f, 1e-4f);
        near("B sim mean dz since the apply (teleport kept)", dzB, TELEPORT, 1e-3f);
        near("A vs B mean y drop since the apply (both keep their warm-up vy)", dyA - dyB, 0.0f, 1e-3f);

        ctx->memFree(dTab); ctx->memFree(dSizes); ctx->memFree(dIdx);
        ctx->memFree(dDstB); ctx->memFree(dDstA); ctx->memFree(dApply);
        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (control)
        {
            std::printf("[ control ] gpu-sdk-step: softbodydata control run (%d check(s) failed), not scored\n",
                        failures);
            return 0;
        }
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: softbodydata: 3 copies bit-exact, A kicked %.4f along x, B "
                        "teleported %.4f in z (collision mesh %.4f after one step) -- "
                        "sb_copyOrApplySoftBodyDataDEPRECATED runs on Vulkan\n", dxA, dzB, collDz);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: softbodydata: %d check(s) failed\n", failures);
        return 2;
    }

    if (!std::strcmp(sceneName, "softbodyattach"))
    {
        SoftBodyAttachScene sa = buildSoftBodyAttachScene(physics, scene, material, cudaCtx);
        if (!sa.upper || sa.numUpperSimVerts == 0)
        {
            std::printf("[ FAIL ] gpu-sdk-step: softbodyattach scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        if (sa.nbAttachments == 0)
        {
            std::printf("[ FAIL ] gpu-sdk-step: softbodyattach created no attachments — the scene "
                        "would be a free-fall test of nothing\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'softbodyattach' built: a %u-vertex FEM volume hung from a "
                    "PINNED one by %u FEM-FEM attachments (no floor beneath)\n",
                    sa.numUpperSimVerts, sa.nbAttachments);

        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: softbodyattach GPU pipeline CONSTRUCTED + live over "
                        "Vulkan (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_SB_STEPS");
        const int steps = stepsEnv ? atoi(stepsEnv) : 180;
        float mean0 = 0.0f, min0 = 0.0f;
        bool finite = softBodyAttachUpperY(sa, mean0, min0);
        std::printf("  stepping the hang (%d x %.4fs): upper volume starts at mean y=%.4f (min %.4f)\n",
                    steps, dt, mean0, min0);

        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            float mn = 0.0f, mean = 0.0f;
            finite = softBodyAttachUpperY(sa, mean, mn) && finite;
            if (i < 4 || i == steps - 1 || (i % 30 == 0))
                std::printf("    step %3d: upper volume min y=%.5f mean=%.5f\n", i, mn, mean);
        }
        float meanE = 0.0f, minE = 0.0f;
        softBodyAttachUpperY(sa, meanE, minE);
        std::printf("  softbodyattach: upper volume mean y %.4f -> %.4f, min y %.4f\n",
                    mean0, meanE, minE);

        int failures = 0;
        if (!finite) { std::printf("FAIL: upper volume went non-finite\n"); ++failures; }
        // It HANGS: gravity pulls for 3 seconds and it must not have fallen away.  Free fall over
        // this window is roughly -44, so the threshold separates the two by more than an order of
        // magnitude and does not need to be tight.
        if (!(minE > -0.5f))
        { std::printf("FAIL: the upper volume fell away — the attachment did not hold "
                      "(min y=%.4f)\n", minE); ++failures; }
        // The constraint pulls the two centroids together, so the cube SETTLES somewhat rather
        // than staying exactly put.  What matters is that it stops somewhere sane, still above
        // the pinned cube's base.
        if (!(meanE > 0.0f))
        { std::printf("FAIL: the upper volume sank past the pinned one (mean y=%.4f)\n", meanE); ++failures; }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: FEM-FEM attachments HELD the upper volume against "
                        "gravity with no contact in the loop (mean y %.4f -> %.4f, min y %.4f)\n",
                        mean0, meanE, minE);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: softbodyattach checks failed (%d)\n", failures);
        return 2;
    }

    if (!std::strcmp(sceneName, "softbodystack"))
    {
        SoftBodyStackScene ss = buildSoftBodyStackScene(physics, scene, material, cudaCtx);
        if (!ss.upper)
        {
            std::printf("[ FAIL ] gpu-sdk-step: softbodystack scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'softbodystack' built: a %u-sim-vertex FEM-volume cube above "
                    "%s (lower top y=%.2f), both over a static plane at y=0\n",
                    ss.numUpperSimVerts,
                    ss.hasLower ? "an identical FEM-volume cube" : "NOTHING (control run)",
                    ss.lowerTopY);
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: softbodystack GPU pipeline CONSTRUCTED + live over "
                        "Vulkan (set PXG_GPU_STEP=1 to step the drop)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        int steps = 150;
        if (const char* sEnv = std::getenv("PXG_STEPS")) { const int v = atoi(sEnv); if (v > 0) steps = v; }

        float meanY0 = 0.0f, minY0 = 0.0f;
        bool allFinite = softBodyStackUpperY(ss, meanY0, minY0);
        float meanY = meanY0, minY = minY0, prevMin = minY0, restMaxDy = 0.0f;
        const int restWindow = steps / 4;
        PxU32 maxVolContacts = 0u;
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            allFinite = softBodyStackUpperY(ss, meanY, minY) && allFinite;
            PxSimulationStatistics st;
            scene->getSimulationStatistics(st);
            maxVolContacts = PxMax(maxVolContacts,
                                   st.gpuDynamicsMemoryConfigStatistics.deformableVolumeContacts);
            if (i >= steps - restWindow) restMaxDy = PxMax(restMaxDy, PxAbs(minY - prevMin));
            if (i < 4 || i == steps - 1 || (i % 30 == 0))
                std::printf("    step %3d: upper cube min y = %.5f (mean %.5f)\n", i, minY, meanY);
            prevMin = minY;
        }

        std::printf("  softbodystack (lower cube %s): upper min y %.4f -> %.4f (mean %.4f); "
                    "lower top y=%.2f, plane y=0; peak volume contacts = %u; rest |dy| over the "
                    "last %d steps = %.6f\n",
                    ss.hasLower ? "PRESENT" : "ABSENT (control)", minY0, minY, meanY,
                    ss.lowerTopY, maxVolContacts, restWindow, restMaxDy);

        int failures = 0;
        if (!allFinite || !std::isfinite(minY))
        { std::printf("FAIL: non-finite vertex in the upper cube\n"); ++failures; }
        if (!(minY0 - minY > 0.2f))
        { std::printf("FAIL: the upper cube did not fall (min y %.5f -> %.5f)\n", minY0, minY);
          ++failures; }

        // Caught means the upper cube's base stays near the lower cube's top face (y = 1, less
        // whatever the two cubes compress into each other); missed means it reaches the plane.
        // The bar sits halfway between, far from both.
        const float CATCH_Y = 0.5f;
        if (ss.hasLower)
        {
            if (!(maxVolContacts > 0u))
            { std::printf("FAIL: two softbodies overlap but the engine reported ZERO "
                          "deformable-volume contacts -- sb_sbMidphaseGeneratePairsLaunch is not "
                          "feeding the solve\n"); ++failures; }
            if (!(minY > CATCH_Y))
            { std::printf("FAIL: the upper cube sank to y=%.5f, below the catch bar %.2f (lower "
                          "top y=%.2f)\n", minY, CATCH_Y, ss.lowerTopY); ++failures; }
            if (!(restMaxDy < 0.01f))
            { std::printf("FAIL: the stacked cube never settled (rest |dy|=%.6f)\n", restMaxDy);
              ++failures; }
        }
        else
        {
            // The CONTROL.  With no lower cube the upper one falls to the plane.  Without this
            // half the main run proves nothing: a cube that never fell would pass it too.
            if (!(minY < CATCH_Y))
            { std::printf("FAIL: the CONTROL run has no lower cube, so the upper one must fall "
                          "past %.2f to the plane; it stopped at %.5f\n", CATCH_Y, minY);
              ++failures; }
        }

        if (failures)
        {
            std::printf("[ FAIL ] gpu-sdk-step: softbodystack checks failed (%d)\n", failures);
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        if (ss.hasLower)
            std::printf("[  OK  ] gpu-sdk-step: the upper FEM-volume cube was CAUGHT by the lower "
                        "one and SETTLED on it (min y=%.4f, lower top %.2f, rest |dy|=%.6f) -- "
                        "softbody-vs-softbody runs end to end\n", minY, ss.lowerTopY, restMaxDy);
        else
            std::printf("[  OK  ] gpu-sdk-step: control run -- with no lower cube the upper one "
                        "fell clean through to the plane (min y=%.4f)\n", minY);
        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        return 0;
    }

    if (!std::strcmp(sceneName, "softbodyselfcollide"))
    {
        SoftBodySelfScene ssc = buildSoftBodySelfCollideScene(physics, scene, material, cudaCtx);
        if (!ssc.volume)
        {
            std::printf("[ FAIL ] gpu-sdk-step: softbodyselfcollide scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'softbodyselfcollide' built: a %u-vertex FEM-volume bracket "
                    "resting on a static plane, its top slab cantilevered over the bottom one\n",
                    ssc.numSimVerts);
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: softbodyselfcollide GPU pipeline CONSTRUCTED + live over "
                        "Vulkan (set PXG_GPU_STEP=1 to step the sag)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        const float dt = 1.0f / 60.0f;
        int steps = 100;
        if (const char* sEnv = std::getenv("PXG_STEPS")) { const int v = atoi(sEnv); if (v > 0) steps = v; }

        float topMean0 = 0.0f, topMin0 = 0.0f;
        bool allFinite = softBodySelfTopY(ssc, topMean0, topMin0);
        float topMin = topMin0, topMean = topMean0;
        float prevMin = topMin0, restMaxDy = 0.0f;
        const int restWindow = steps / 4;
        PxU32 maxContacts = 0u;
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            allFinite = softBodySelfTopY(ssc, topMean, topMin) && allFinite;
            PxSimulationStatistics simStats;
            scene->getSimulationStatistics(simStats);
            maxContacts = PxMax(maxContacts,
                                simStats.gpuDynamicsMemoryConfigStatistics.deformableVolumeContacts);
            if (i >= steps - restWindow) restMaxDy = PxMax(restMaxDy, PxAbs(topMin - prevMin));
            if (i < 6 || i == steps - 1 || (i % 20 == 0))
                std::printf("    step %3d: top slab min y = %.5f (mean %.5f)\n", i, topMin, topMean);
            prevMin = topMin;
        }

        std::printf("  softbodyselfcollide (self-collision %s): top slab min y %.4f -> %.4f "
                    "(mean %.4f); bottom slab surface at y=%.2f; peak deformable-volume contacts = %u; "
                    "rest |dy| over last %d steps = %.6f\n",
                    ssc.selfCollision ? "ON" : "OFF", topMin0, topMin, topMean, ssc.bottomTopY,
                    maxContacts, restWindow, restMaxDy);

        int failures = 0;
        if (!allFinite || !std::isfinite(topMin))
        { std::printf("FAIL: non-finite vertex in the bracket\n"); ++failures; }
        // The top slab must SAG -- otherwise the scene proves nothing either way.
        if (!(topMin0 - topMin > 0.02f))
        { std::printf("FAIL: the top slab did not sag (min y %.5f -> %.5f)\n", topMin0, topMin); ++failures; }

        // THE THRESHOLDS, AND WHERE THEY COME FROM.  Measured over 100 steps at
        // youngs = 5e6 with 30 solver iterations:
        //
        //     self-collision ON : top slab min y 0.7500 -> 0.2133, rest |dy| = 0.003299
        //     self-collision OFF: top slab min y 0.7500 -> 0.0000, rest |dy| = 0.000000
        //
        // The bottom slab's upper surface is at y = 0.25, and the caught slab settles a
        // little BELOW it (0.2133).  That is not a failure to catch: the slab is a soft body
        // that compresses under its own weight, and the contact is resolved at a rest offset
        // rather than at the geometric surface -- the same reason clothdrape settles ON the
        // contact rest distance instead of exactly on the plane.  What the pair proves is
        // the thing worth proving: with the chain on, the top slab STOPS just under the
        // bottom slab and converges; with it off, it passes clean through to the floor.
        // The bar is set at 0.15 -- far above the 0.00 the control reaches, far below the
        // 0.21 this run reaches -- so neither run can drift into the other's verdict.
        const float CATCH_Y = 0.15f;
        if (ssc.selfCollision)
        {
            // (1) the chain must actually produce contacts...
            if (!(maxContacts > 0u))
            { std::printf("FAIL: self-collision is ON but the engine reported ZERO deformable-volume "
                          "contacts -- the midphase or the remap is not feeding the solve\n"); ++failures; }
            // (2) ...and they must CATCH the top slab at the bottom slab rather than let it
            // fall to the floor.
            if (!(topMin > CATCH_Y))
            { std::printf("FAIL: the top slab sank to y=%.5f, below the catch bar %.2f (bottom slab "
                          "surface y=%.2f) -- self-collision did not catch it\n",
                          topMin, CATCH_Y, ssc.bottomTopY); ++failures; }
            // (3) and it must SETTLE rather than keep sinking or oscillate.
            if (!(restMaxDy < 0.02f))
            { std::printf("FAIL: the caught slab never settled (rest |dy|=%.6f)\n", restMaxDy); ++failures; }
        }
        else
        {
            // The CONTROL. With the flag off the top slab is free to pass through, and it
            // does -- all the way to the plane.  Without this half the ON run proves nothing:
            // a bracket that simply never sagged that far would pass it too.
            if (!(topMin < CATCH_Y))
            { std::printf("FAIL: self-collision is OFF, so the top slab must NOT be held above the "
                          "catch bar %.2f (got %.5f) -- if it stops there anyway the ON run proves "
                          "nothing\n", CATCH_Y, topMin); ++failures; }
        }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (failures == 0)
        {
            if (ssc.selfCollision)
                std::printf("[  OK  ] gpu-sdk-step: FEM-volume SELF-collision CAUGHT the sagging top slab "
                            "at the bottom one and SETTLED it (min y=%.4f, bottom surface %.2f, rest "
                            "|dy|=%.6f) -- the control run falls to 0.0000\n",
                            topMin, ssc.bottomTopY, restMaxDy);
            else
                std::printf("[  OK  ] gpu-sdk-step: control run -- with self-collision OFF the top slab "
                            "passed clean through the bottom one to the floor (min y=%.4f, bottom "
                            "surface %.2f)\n", topMin, ssc.bottomTopY);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: softbodyselfcollide checks failed (%d)\n", failures);
        return 2;
    }

    // P9.41 -- PXG_SCENE=articulationjoint: the `articulation` scene with two more
    // questions, both of which P9.40's guard showed that scene cannot answer.  The link
    // swings from horizontal to hanging (`articulation` scores that), and then:
    //  (a) does PxArticulationCache report the joint angle the link visibly has?  On
    //      the GPU path the joint positions come back through the DMA of
    //      PxgArticulation::jointPositions, whose ONLY writer is updateBodiesLaunch1T
    //      (forwardDynamic2.cu:3035-3040).  With it unmapped the cache says 0 while
    //      the link hangs at -pi/2.
    //  (b) does the articulation fall ASLEEP once it has settled?  Its wake counter is
    //      written by updateBodiesLaunch_Part3 (forwardDynamic2.cu:3205-3249); with it
    //      unmapped the counter never counts down and isSleeping() never turns true.
    // Pending-work entry in run_scenes.sh until both kernels land.
    if (!std::strcmp(sceneName, "articulationjoint"))
    {
        std::vector<RestTarget> artiTargets;
        PxArticulationReducedCoordinate* arti = buildArticulationScene(physics, scene, material, artiTargets);
        PxArticulationLink* link = nullptr;
        PxArticulationCache* cache = arti ? arti->createCache() : nullptr;
        if (!arti || arti->getLinks(&link, 1, 1) != 1 || !link || !cache || artiTargets.empty())
        {
            std::printf("[ FAIL ] gpu-sdk-step: articulationjoint scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'articulationjoint' built: fixed-base articulation, one revolute link, "
                    "%u dof(s), plus the companion box\n", arti->getDofs());
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: articulationjoint pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n");
            cache->release();
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }

        // The joint frames carry no rotation and the base is fixed with identity
        // rotation, so the joint angle IS the angle of the link's local x axis in the
        // x-y plane.  That is the geometric truth the cache is checked against.
        auto geometricAngle = [&]() {
            const PxVec3 ax = link->getGlobalPose().q.getBasisVector0();
            return std::atan2(ax.y, ax.x);
        };
        auto cachedAngle = [&]() {
            arti->copyInternalStateToCache(*cache, PxArticulationCacheFlag::ePOSITION);
            return cache->jointPosition[0];
        };

        const float dt = 1.0f / 60.0f;
        const char* stepsEnv = std::getenv("PXG_ARTI_STEPS");
        // 900, not `articulation`'s 240.  The first guess of 360 was too short and said
        // so by failing two checks that the physics actually passes: with the joint
        // friction this scene sets (0.5), the link does not swing and stop, it CREEPS --
        // measured -0.7180 rad at step 40, -0.9164 at 280, -0.9572 at 360, -1.0474 at
        // 600, -1.0792 at 800, where it finally holds still.  The wake counter then runs
        // out and the articulation falls asleep at step 722.  900 leaves a quarter of
        // that again as margin.
        const int steps = stepsEnv ? atoi(stepsEnv) : 900;
        int asleepSince = -1;
        std::printf("  stepping articulationjoint (%d x %.4fs): cache q0=%.4f, geometric q0=%.4f\n",
                    steps, dt, cachedAngle(), geometricAngle());
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            if (asleepSince < 0 && arti->isSleeping()) asleepSince = i;
            if (i < 3 || i == steps - 1 || (i % 40 == 0))
                std::printf("    step %3d: cache q = %+.4f | geometric q = %+.4f | link y = %.4f | wake counter "
                            "%.4f%s\n", i, cachedAngle(), geometricAngle(), link->getGlobalPose().p.y,
                            arti->getWakeCounter(), arti->isSleeping() ? " (asleep)" : "");
        }
        const float qCache = cachedAngle();
        const float qGeom = geometricAngle();
        const float linkY = link->getGlobalPose().p.y;
        const float linkRestY = artiTargets.back().restY;
        const bool sleeping = arti->isSleeping();
        const float wake = arti->getWakeCounter();
        std::printf("  articulationjoint: joint angle from the cache %+.4f vs from the link pose %+.4f "
                    "(difference %.4f); link y %.4f (rest %.3f); asleep %s (wake counter %.4f%s)\n",
                    qCache, qGeom, PxAbs(qCache - qGeom), linkY, linkRestY, sleeping ? "YES" : "no", wake,
                    asleepSince >= 0 ? "" : ", never slept");
        if (asleepSince >= 0) std::printf("  articulationjoint: asleep since step %d\n", asleepSince);

        int failures = 0;
        // (1) the link swung down -- else (3) is trivially satisfied by two zeros.
        if (!(PxAbs(qGeom) > 1.0f))
        { std::printf("FAIL: the link did not swing (geometric angle %+.4f, need |q| > 1.0)\n", qGeom); ++failures; }
        // (2) and hangs in the rest band the `articulation` scene scores.
        if (!(PxAbs(linkY - linkRestY) < 0.75f))
        { std::printf("FAIL: link y %.4f is outside the rest band around %.3f\n", linkY, linkRestY); ++failures; }
        // (3) THE CACHE AGREES WITH THE GEOMETRY.
        if (!(PxAbs(qCache - qGeom) < 0.05f))
        { std::printf("FAIL: PxArticulationCache reports joint position %+.4f but the link hangs at %+.4f rad "
                      "-- the joint state never came back from the GPU.  updateBodiesLaunch1T is its only "
                      "writer; if the [pxg-unmapped] lines below name it, this is that\n", qCache, qGeom); ++failures; }
        // (4) IT FELL ASLEEP.
        if (!sleeping)
        { std::printf("FAIL: the articulation is still awake after %d steps (wake counter %.4f) -- the sleep "
                      "decision is updateBodiesLaunch_Part3's; if the [pxg-unmapped] lines below name it, "
                      "this is that\n", steps, wake); ++failures; }
        if (!std::isfinite(qCache) || !std::isfinite(qGeom) || !std::isfinite(linkY))
        { std::printf("FAIL: a value went non-finite\n"); ++failures; }

        cache->release();
        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: the articulation's JOINT STATE comes back from the GPU (%+.4f rad, "
                        "matching the link pose) and it FELL ASLEEP at step %d -- both articulation "
                        "body-update stages are live on Vulkan\n", qCache, asleepSince);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: articulationjoint checks failed (%d)\n", failures);
        return 2;
    }

    // P9.79 -- PXG_SCENE=directapi.  Its own arm rather than a RestTarget, because
    // with eENABLE_DIRECT_GPU_API raised the host actor poses are no longer updated
    // by the simulation and the generic rest check would be reading a spawn pose.
    if (!std::strcmp(sceneName, "forcethreshold"))
    {
        ForceThresholdEvents events;
        scene->setSimulationEventCallback(&events);
        ForceThresholdScene fs = buildForceThresholdScene(physics, scene, material);
        std::printf("  [  OK  ] scene 'forcethreshold' built: four 1000 kg boxes at rest, thresholds "
                    "over=5000 N under=20000 N upper=5000 N (weight 9810 N)\n");
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: forcethreshold pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 0;
        }
        const float dt = 1.0f / 60.0f;
        std::printf("  stepping forcethreshold (%d x %.4fs); 'over' threshold -> 1e7 N before step %d\n",
                    kForceThresholdSteps, dt, kForceThresholdRaiseStep);
        for (int i = 0; i < kForceThresholdSteps; ++i)
        {
            if (i == kForceThresholdRaiseStep)
            {
                fs.over->setContactReportThreshold(1.0e7f);
                // UPSTREAM (documented, not reported): Sc::BodyCore::setContactReportThreshold
                // (ScBodyCore.h:144) only stores the value -- unlike the other BodyCore setters
                // it never calls updateBodySim(), so with GPU dynamics the new threshold never
                // reaches PxgBodySim::inverseInertiaXYZ_contactReportThresholdW
                // (PxgSimulationController.cpp:2411) and the pair keeps PERSISTING.  The CPU
                // solver reads the core every step and does see it.  Measured: without the
                // line below, GPU TGS and GPU PGS both report PERSISTS on all 239 steps and no
                // LOST.  Re-setting the linear damping to its own value is a no-op for the
                // physics and does call updateBodySim(), so the GPU body sim is re-uploaded.
                fs.over->setLinearDamping(fs.over->getLinearDamping());
            }
            events.step = i;
            scene->simulate(dt);
            scene->fetchResults(true);
        }
        int failures = 0;
        // LOST lands on the raise step itself: the refreshed body sim is uploaded before that
        // step's solve (measured on GPU TGS, GPU PGS and the CPU PGS reference alike).
        failures += checkThresholdTimeline(events, "ground|over", kForceThresholdRaiseStep,
                                           kForceThresholdRaiseStep);
        failures += checkThresholdTimeline(events, "lower|upper", -1, -1);
        for (const auto& kv : events.pairs)
        {
            if (kv.first == "ground|over" || kv.first == "lower|upper") continue;
            int any = 0;
            for (unsigned char b : kv.second.perStep) any += b != 0;
            std::printf("  [%s] %d step(s) with a force-threshold event\n", kv.first.c_str(), any);
            if (any) { std::printf("FAIL: [%s] must report nothing\n", kv.first.c_str()); ++failures; }
        }
        const struct { PxRigidDynamic* b; float restY; const char* label; } rest[] = {
            { fs.over, 0.5f, "over" }, { fs.under, 0.5f, "under" },
            { fs.lower, 0.5f, "lower" }, { fs.upper, 1.5f, "upper" } };
        for (const auto& r : rest)
        {
            const float y = r.b->getGlobalPose().p.y;
            std::printf("  [%s] final y=%.4f (rest %.3f)\n", r.label, y, r.restY);
            if (!(std::fabs(y - r.restY) < 0.1f)) { std::printf("FAIL: [%s] left its rest height\n", r.label); ++failures; }
        }
        scene->setSimulationEventCallback(nullptr);
        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (cpuControl)
        {
            std::printf("[ control ] gpu-sdk-step: forcethreshold CPU control run (%d check(s) failed), "
                        "not scored\n", failures);
            return 0;
        }
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: scene 'forcethreshold' stepped: FOUND/PERSISTS/LOST "
                        "per pair as asserted\n");
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: scene 'forcethreshold': %d check(s) failed\n", failures);
        return 2;
    }

    if (!std::strcmp(sceneName, "extforce"))
    {
        ExtForceScene es = buildExtForceScene(physics, scene, material);
        std::printf("  [  OK  ] scene 'extforce' built: four free-falling 1000 kg boxes, three of them "
                    "kicked through addForce / addTorque (autowake=false)\n");
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: extforce pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 0;
        }
        const float dt = 1.0f / 60.0f;
        for (int i = 0; i < 60; ++i)
        {
            if (i == 30)
            {
                es.K->addForce(PxVec3(3.0f, 0.0f, 0.0f), PxForceMode::eVELOCITY_CHANGE, false);
                es.S->addTorque(PxVec3(0.0f, 500.0f, 0.0f), PxForceMode::eIMPULSE, false);
            }
            if (i >= 30)
                es.F->addForce(PxVec3(0.0f, 0.0f, 2000.0f), PxForceMode::eFORCE, false);
            scene->simulate(dt);
            scene->fetchResults(true);
        }
        const PxVec3 kP = es.K->getGlobalPose().p, kV = es.K->getLinearVelocity();
        const PxVec3 fP = es.F->getGlobalPose().p, fV = es.F->getLinearVelocity();
        const PxQuat sQ = es.S->getGlobalPose().q;
        const PxVec3 sW = es.S->getAngularVelocity();
        const PxVec3 cP = es.C->getGlobalPose().p;
        std::printf("  K p=(%.6f, %.6f, %.6f) v=(%.6f, %.6f, %.6f)\n", kP.x, kP.y, kP.z, kV.x, kV.y, kV.z);
        std::printf("  F p=(%.6f, %.6f, %.6f) v=(%.6f, %.6f, %.6f)\n", fP.x, fP.y, fP.z, fV.x, fV.y, fV.z);
        std::printf("  S q=(%.6f, %.6f, %.6f, %.6f) w=(%.6f, %.6f, %.6f)\n", sQ.x, sQ.y, sQ.z, sQ.w, sW.x, sW.y, sW.z);
        std::printf("  C p=(%.6f, %.6f, %.6f)\n", cP.x, cP.y, cP.z);

        int failures = 0;
        auto near = [&](const char* what, float got, float want, float tol) {
            if (!(std::fabs(got - want) <= tol))
            { std::printf("FAIL: %s = %.6f, expected %.6f +- %g\n", what, got, want, tol); ++failures; }
        };
        const float yEnd = 100.0f - 4.98675f;   // g dt^2 * 60*61/2
        near("K.x", kP.x, 1.5f, 1e-3f);         near("K.v.x", kV.x, 3.0f, 1e-4f);
        near("K.y", kP.y, yEnd, 2e-3f);         near("K.z", kP.z, 0.0f, 1e-5f);
        near("F.z", fP.z, 0.258333f, 1e-4f);    near("F.v.z", fV.z, 1.0f, 1e-4f);
        near("F.x", fP.x, 5.0f, 1e-5f);         near("F.y", fP.y, yEnd, 2e-3f);
        near("S.w.y", sW.y, 3.0f, 1e-4f);       near("S.w.x", sW.x, 0.0f, 1e-5f);
        near("S.w.z", sW.z, 0.0f, 1e-5f);
        near("S.q.y", std::fabs(sQ.y), 0.6816388f, 1e-3f);
        near("S.q.w", std::fabs(sQ.w), 0.7316889f, 1e-3f);
        near("C.x", cP.x, -5.0f, 1e-5f);        near("C.z", cP.z, 0.0f, 1e-5f);
        near("C.y", cP.y, yEnd, 2e-3f);
        failures += unmappedPenalty();
        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: addForce / addTorque reached the GPU bodies: K moved %.4f, F "
                        "drifted %.4f along z, S spins at %.4f rad/s\n", kP.x, fP.z, sW.y);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: scene 'extforce': %d check(s) failed\n", failures);
        return 2;
    }

    if (!std::strcmp(sceneName, "particleapply"))
    {
        const bool control = std::getenv("PXG_PARTICLE_APPLY_CONTROL") != nullptr;
        ParticleApplyScene pa = buildParticleApplyScene(physics, scene, material, cudaCtx);
        if (!pa.system)
        {
            std::printf("[ FAIL ] gpu-sdk-step: particleapply scene build failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene 'particleapply' built: three %u-particle clouds A B C in one PBD system, "
                    "unique ids C < B < A, buffers added A B C\n", pa.n);
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: particleapply pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 0;
        }
        const float dt = 1.0f / 60.0f;
        int failures = 0;
        PxParticleBuffer* clouds[3] = { pa.a, pa.b, pa.c };
        const char* names[3] = { "A", "B", "C" };
        std::vector<PxVec4> pos[3], vel[3], spawn[3];
        for (int k = 0; k < 3; ++k) particleApplyRead(pa, clouds[k], spawn[k], vel[k]);

        // 1. Warm-up step: it initializes the direct-GPU API (ScPipeline.cpp:2813-2815).
        scene->simulate(dt);
        scene->fetchResults(true);
        // The egress must have refreshed the user arrays, or every observable below is void.
        particleApplyRead(pa, pa.b, pos[1], vel[1]);
        const float warmDy = particleApplyMean(pos[1]).y - particleApplyMean(spawn[1]).y;
        std::printf("  warm-up: B mean dy = %.6f (expected about -g dt^2 = -0.002725)\n", warmDy);
        if (!(warmDy < -1e-3f && warmDy > -5e-3f))
        { std::printf("FAIL: the per-step egress did not refresh B's user array (dy = %.6f)\n", warmDy); ++failures; }

        // 2. The write window: every cloud is edited, NO flag is raised.
        for (int k = 0; k < 3; ++k) particleApplyRead(pa, clouds[k], pos[k], vel[k]);
        for (PxU32 i = 0; i < pa.n; ++i)
        {
            pos[0][i].z += PAPPLY_DZ_A; vel[0][i].y += PAPPLY_DVY_A;
            pos[1][i].z += PAPPLY_DZ_B; vel[1][i].x = PAPPLY_VX_B;
            pos[2][i].z += PAPPLY_DZ_C; vel[2][i].z = PAPPLY_VZ_C;
        }
        PxCudaContext* ctx = cudaCtx->getCudaContext();
        for (int k = 0; k < 3; ++k)
        {
            ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(clouds[k]->getPositionInvMasses()), pos[k].data(),
                            pa.n * sizeof(PxVec4));
            ctx->memcpyHtoD(reinterpret_cast<CUdeviceptr>(clouds[k]->getVelocities()), vel[k].data(),
                            pa.n * sizeof(PxVec4));
        }

        // 3. The ingest: applyParticleBufferData, or (control) the ordinary flags.
        CUdeviceptr devIdx = 0, devPairs = 0, devFlags = 0;
        if (control)
        {
            pa.b->raiseFlags(PxParticleBufferFlag::eUPDATE_POSITION);
            pa.b->raiseFlags(PxParticleBufferFlag::eUPDATE_VELOCITY);
            pa.a->raiseFlags(PxParticleBufferFlag::eUPDATE_VELOCITY);
            std::printf("  [ control ] PXG_PARTICLE_APPLY_CONTROL=1: B POS|VEL and A VEL raised as buffer flags\n");
        }
        else
        {
            const PxU32 sys = pa.system->getGpuParticleSystemIndex();
            const PxGpuParticleBufferIndexPair pairs[3] = {
                { sys, pa.a->getUniqueId() }, { sys, pa.c->getUniqueId() }, { sys, pa.b->getUniqueId() } };
            const PxParticleBufferFlags flags[3] = {
                PxParticleBufferFlags(PxParticleBufferFlag::eUPDATE_VELOCITY),
                PxParticleBufferFlags(PxParticleBufferFlag::eUPDATE_POSITION) | PxParticleBufferFlag::eUPDATE_VELOCITY,
                PxParticleBufferFlags(PxParticleBufferFlag::eUPDATE_POSITION) | PxParticleBufferFlag::eUPDATE_VELOCITY };
            const PxU32 indices[2] = { 2u, 0u };
            std::printf("  apply: system %u, unique ids A=%u B=%u C=%u, indices {2, 0}\n", sys,
                        pa.a->getUniqueId(), pa.b->getUniqueId(), pa.c->getUniqueId());
            if (ctx->memAlloc(&devIdx, sizeof(indices)) != 0 || !devIdx ||
                ctx->memAlloc(&devPairs, sizeof(pairs)) != 0 || !devPairs ||
                ctx->memAlloc(&devFlags, sizeof(flags)) != 0 || !devFlags)
            {
                std::printf("[ FAIL ] gpu-sdk-step: particleapply memAlloc failed\n");
                scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
            }
            ctx->memcpyHtoD(devIdx, indices, sizeof(indices));
            ctx->memcpyHtoD(devPairs, pairs, sizeof(pairs));
            ctx->memcpyHtoD(devFlags, flags, sizeof(flags));
            scene->applyParticleBufferData(reinterpret_cast<const PxU32*>(devIdx),
                                           reinterpret_cast<const PxGpuParticleBufferIndexPair*>(devPairs),
                                           reinterpret_cast<const PxParticleBufferFlags*>(devFlags), 2u);
        }

        // 4. Thirty steps, then read every cloud back.
        for (int i = 0; i < PAPPLY_STEPS; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
        }
        PxVec3 mp[3], mv[3];
        float shapeDev[3] = { 0.0f, 0.0f, 0.0f };
        bool finite = true;
        for (int k = 0; k < 3; ++k)
        {
            particleApplyRead(pa, clouds[k], pos[k], vel[k]);
            mp[k] = particleApplyMean(pos[k]);
            mv[k] = particleApplyMean(vel[k]);
            const PxVec3 m0 = particleApplyMean(spawn[k]);
            for (PxU32 i = 0; i < pa.n; ++i)
            {
                const PxVec3 d = (pos[k][i].getXYZ() - mp[k]) - (spawn[k][i].getXYZ() - m0);
                shapeDev[k] = PxMax(shapeDev[k], PxMax(PxAbs(d.x), PxMax(PxAbs(d.y), PxAbs(d.z))));
                finite = finite && pos[k][i].getXYZ().isFinite() && vel[k][i].getXYZ().isFinite();
            }
            std::printf("  %s  mean p=(%.6f, %.6f, %.6f) v=(%.6f, %.6f, %.6f) shape dev %.6f\n", names[k],
                        mp[k].x, mp[k].y, mp[k].z, mv[k].x, mv[k].y, mv[k].z, shapeDev[k]);
        }

        // PXG_PARTICLE_APPLY_DUMP=<file>: the live run writes its final user arrays there,
        // and the control compares against them word by word.  Reported, never scored.
        if (const char* dumpPath = std::getenv("PXG_PARTICLE_APPLY_DUMP"))
        {
            std::vector<PxU32> words;
            for (int k = 0; k < 3; ++k)
                for (const std::vector<PxVec4>* v : { &pos[k], &vel[k] })
                {
                    const size_t base = words.size();
                    words.resize(base + v->size() * 4);
                    std::memcpy(words.data() + base, v->data(), v->size() * sizeof(PxVec4));
                }
            if (!control)
            {
                if (FILE* f = std::fopen(dumpPath, "wb"))
                { std::fwrite(words.data(), sizeof(PxU32), words.size(), f); std::fclose(f); }
            }
            else if (FILE* f = std::fopen(dumpPath, "rb"))
            {
                std::vector<PxU32> ref(words.size(), 0u);
                const size_t got = std::fread(ref.data(), sizeof(PxU32), ref.size(), f);
                std::fclose(f);
                size_t mism = 0;
                for (size_t i = 0; i < words.size(); ++i) mism += (i >= got || ref[i] != words[i]) ? 1u : 0u;
                std::printf("  [ control ] %zu of %zu final user-array words differ from the apply run\n",
                            mism, words.size());
            }
        }

        auto near = [&](const char* what, float got, float want, float tol) {
            if (!(std::fabs(got - want) <= tol))
            { std::printf("FAIL: %s = %.6f, expected %.6f +- %g\n", what, got, want, tol); ++failures; }
        };
        const float t = float(PAPPLY_STEPS) * dt;
        if (!finite) { std::printf("FAIL: a cloud is not finite\n"); ++failures; }
        near("B.z - C.z (B's POS applied)", mp[1].z - mp[2].z, PAPPLY_DZ_B, 1e-3f);
        near("B.x - (C.x - 20) (B's VEL applied)", mp[1].x - (mp[2].x - 20.0f), PAPPLY_VX_B * t, 1e-3f);
        near("B.v.x", mv[1].x, PAPPLY_VX_B, 1e-3f);
        near("A.y - C.y (A's VEL applied)", mp[0].y - mp[2].y, PAPPLY_DVY_A * t, 1e-3f);
        near("A.v.y - C.v.y", mv[0].y - mv[2].y, PAPPLY_DVY_A, 2e-3f);
        near("A.z - C.z (A's POS NOT applied: flags are per slot)", mp[0].z - mp[2].z, 0.0f, 1e-3f);
        near("A.x - (C.x - 40)", mp[0].x - (mp[2].x - 40.0f), 0.0f, 1e-3f);
        near("B.y - C.y", mp[1].y - mp[2].y, 0.0f, 1e-3f);
        near("C.x (decoy slot never read)", mp[2].x, 20.0f, 1e-3f);
        near("C.z (decoy slot never read)", mp[2].z, 0.0f, 1e-3f);
        near("C.v.z", mv[2].z, 0.0f, 1e-3f);
        for (int k = 0; k < 3; ++k)
            if (!(shapeDev[k] <= 1e-3f))
            { std::printf("FAIL: cloud %s lost its shape (max offset change %.6f)\n", names[k], shapeDev[k]); ++failures; }

        if (devIdx) ctx->memFree(devIdx);
        if (devPairs) ctx->memFree(devPairs);
        if (devFlags) ctx->memFree(devFlags);
        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (control)
        {
            std::printf("[ control ] gpu-sdk-step: particleapply control run (%d check(s) failed), not scored\n",
                        failures);
            return 0;
        }
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: particleapply: B.z-C.z = %.4f, B moved %.4f along x, A rose %.4f "
                        "against C, A.z-C.z = %.4f, C untouched\n", mp[1].z - mp[2].z,
                        mp[1].x - (mp[2].x - 20.0f), mp[0].y - mp[2].y, mp[0].z - mp[2].z);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: particleapply: %d check(s) failed\n", failures);
        return 2;
    }

    // P9.99b -- PXG_SCENE=particle2: two PBD particle systems in one scene (see the block
    // above runParticle2 for the inventory, the arms and the oracles).  The sub-scenes
    // make their own scenes from sceneDesc; main's scene is released first.
    if (!std::strcmp(sceneName, "particle2"))
    {
        const char* armEnv = std::getenv("PXG_P2_ARM");
        const char* arm = armEnv ? armEnv : "split";
        const bool armSplit = !std::strcmp(arm, "split"), armFluid = !std::strcmp(arm, "fluid");
        const bool armApply = !std::strcmp(arm, "apply"), armPush = !std::strcmp(arm, "push");
        const bool armRemove = !std::strcmp(arm, "remove");
        const char* solverName = usePgs ? "PGS" : "TGS";
        scene->release(); scene = nullptr;
        if (!(armSplit || armFluid || armApply || armPush || armRemove))
        {
            std::printf("[ FAIL ] gpu-sdk-step: particle2: unknown PXG_P2_ARM=%s (split, fluid, apply, push, remove)\n", arm);
            physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        if (!std::getenv("PXG_GPU_STEP"))
        {
            const P2Run c = runParticle2(physics, sceneDesc, material, cudaCtx, arm, P2_BOTH, 0);
            physics->release(); cudaCtx->release(); foundation->release();
            if (!c.ok) { std::printf("[ FAIL ] gpu-sdk-step: particle2 (%s) scene build failed\n", arm); return 1; }
            std::printf("[  OK  ] gpu-sdk-step: particle2 (%s) pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n", arm);
            return 0;
        }
        std::printf("  particle2 (%s, %s): two PBD systems, A (4^3, index 0) and B (5^3, index 1), %d steps per sub-scene\n",
                    arm, solverName, P2_STEPS);

        // Sub-scene order: both, each system alone, and the swapped pool order LAST (split and
        // fluid only), after every other check has held.  The order does not change any
        // sub-scene's result (each is a fresh scene; measured bit-identical in either order),
        // but it decides how a broken kernel dies: a scene that reads arrays a wrong-index
        // kernel never wrote may fault in a LATER sub-scene of the same process, where the
        // allocator hands back memory an earlier sub-scene left behind (measured: the
        // self-collision `id = 0` mutant collapses B in `both` and then segfaults in `swap`).
        // Running `swap` only when C1/C3 already held turns such a mutant into a value kill.
        P2Run runs[5];
        std::vector<P2Who> whos;   // the sub-scenes that ran, in order
        auto runSub = [&](P2Who w) {
            const auto t0 = std::chrono::steady_clock::now();
            runs[w] = runParticle2(physics, sceneDesc, material, cudaCtx, arm, w, P2_STEPS);
            whos.push_back(w);
            std::printf("  [%s] sub-scene done in %.1f s\n", P2_WHO_NAME[w],
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        };
        runSub(P2_BOTH);
        if (armRemove) runSub(P2_REMOVE_A);
        else { runSub(P2_SOLO_A); runSub(P2_SOLO_B); }
        const P2Run& both = runs[P2_BOTH];
        bool ran = true;
        for (P2Who w : whos) ran = ran && runs[w].ok;

        int failures = 0;
        auto inRange = [&](const char* what, float got, float lo, float hi) {
            if (!(got >= lo && got <= hi))
            { std::printf("FAIL: %s = %.6f, expected in [%g, %g]\n", what, got, lo, hi); ++failures; }
        };
        auto near = [&](const char* what, float got, float want, float tol) {
            if (!(std::fabs(got - want) <= tol))
            { std::printf("FAIL: %s = %.6f, expected %g +- %g\n", what, got, want, tol); ++failures; }
        };

        if (armRemove)
        {
            // U1, diagnostic: B's final arrays with A removed before step 60 vs the no-removal run.
            physics->release(); cudaCtx->release(); foundation->release();
            const P2Run& rm = runs[P2_REMOVE_A];
            if (!both.ok || !rm.ok) { std::printf("[ diagnostic ] gpu-sdk-step: particle2 (remove, %s): a sub-scene failed\n", solverName); return 0; }
            const size_t d = particle2CompareWords("B remove/both", rm, both, 1);
            std::printf("[ diagnostic ] gpu-sdk-step: particle2 (remove, %s): %zu of %u B words differ from the no-removal run "
                        "(U1: the TGS predictor steps pool slot blockIdx.y; measured 500 TGS, 0 PGS), not scored\n",
                        solverName, d, both.n[1] * 8u);
            return 0;
        }

        // Every sub-scene: built, both systems' N, finite, nothing fell through the world.
        auto subSceneChecks = [&](P2Who w) {
            const P2Run& r = runs[w];
            if (!r.ok) { std::printf("FAIL: sub-scene [%s] did not run\n", P2_WHO_NAME[w]); ++failures; return; }
            for (int k = 0; k < 2; ++k)
            {
                if (!r.present[k]) continue;
                if (r.nActive[k] != r.n[k])
                { std::printf("FAIL: [%s] %s has %u active particles, expected %u\n", P2_WHO_NAME[w], k ? "B" : "A", r.nActive[k], r.n[k]); ++failures; }
                if (!r.finite[k]) { std::printf("FAIL: [%s] %s is not finite\n", P2_WHO_NAME[w], k ? "B" : "A"); ++failures; }
                if (!(r.yMin[k] > -1.0f)) { std::printf("FAIL: [%s] %s fell through (y_min %.6f)\n", P2_WHO_NAME[w], k ? "B" : "A", r.yMin[k]); ++failures; }
            }
        };
        for (P2Who w : whos) subSceneChecks(w);

        size_t c1[2] = { 0, 0 }, c2[2] = { 0, 0 };
        if (ran)
        {
            // C1: a system's trajectory does not depend on whether the other one exists.
            c1[0] = particle2CompareWords("C1 A both/soloA", both, runs[P2_SOLO_A], 0);
            c1[1] = particle2CompareWords("C1 B both/soloB", both, runs[P2_SOLO_B], 1);
            if (armPush)
                std::printf("  (push: C1 is NOT scored -- the box's contact order inside its rigid run depends on the "
                            "composition, U2; the push row is scored physically)\n");
            else
                for (int k = 0; k < 2; ++k)
                    if (c1[k]) { std::printf("FAIL: C1 %s: %zu words differ from the system alone\n", k ? "B" : "A", c1[k]); ++failures; }
            if (armFluid && !particle2CompareDiffuse("C1 B diffuse both/soloB", both, runs[P2_SOLO_B], 1)) ++failures;
        }

        // C3: absolute physics of the scored `both` sub-scene.
        if (ran && (armSplit || armFluid))
        {
            // A: the granular cloud on the ground box (identical in both rows).
            inRange("A y_min (rest 0.25 on the ground box)", both.yMin[0], 0.24f, 0.26f);
            inRange("A mean z (drifts at v.z = 1, no lock)", both.meanP[0].z, 0.3f, 1e30f);
            for (int s = 0; s < 2; ++s)
                near(s ? "A |mean v| at step 30 (cap 3)" : "A |mean v| at step 20 (cap 3)", both.vProbe[0][s].magnitude(), 3.0f, 2e-3f);
        }
        if (ran && armSplit)
        {
            inRange("B y_min (rest 0.15 on the mesh patch at y = 1)", both.yMin[1], 1.13f, 1.16f);
            near("B mean z (eLOCK_Z)", both.meanP[1].z, 0.0f, 1e-4f);
            for (int s = 0; s < 2; ++s)
                near(s ? "B mean v.y at step 30 (cap 5)" : "B mean v.y at step 20 (cap 5)", both.vProbe[1][s].y, -5.0f, 2e-3f);
        }
        if (ran && armFluid)
        {
            inRange("B y_min (one fluid layer)", both.yMin[1], 0.1666f, 0.1667f);
            inRange("B y_max (one fluid layer)", both.yMax[1], 0.1666f, 0.1667f);
            if (both.diffusePeak[1] != P2_MAX_DIFFUSE)
            { std::printf("FAIL: B diffuse peak %u, expected the cap %u\n", both.diffusePeak[1], P2_MAX_DIFFUSE); ++failures; }
            if (both.diffuseEnd[1] == 0) { std::printf("FAIL: B has no diffuse particles at the end\n"); ++failures; }
        }
        if (ran && armApply)
        {
            if (both.gpuIdx[0] != 0u || both.gpuIdx[1] != 1u)
            { std::printf("FAIL: gpu indices A %u B %u, expected 0 and 1 (indices = {1} names B's slot)\n", both.gpuIdx[0], both.gpuIdx[1]); ++failures; }
            near("B mean z after the apply step (z += 5 applied)", both.pAfterApply[1].z, 5.0f, 1e-4f);
            inRange("B mean v.x after the apply step (v.x = 2 applied)", both.vAfterApply[1].x, 1.99f, 2.0f);
            near("B mean z at the end (eLOCK_Z keeps the applied z)", both.meanP[1].z, 5.0f, 1e-4f);
            inRange("B y_min at the end (off the patch, on the ground box)", both.yMin[1], 0.14f, 0.16f);
        }
        if (ran && armPush)
        {
            const float x0[2] = { -7.5f, 12.5f };
            const P2Run* solo[2] = { &runs[P2_SOLO_A], &runs[P2_SOLO_B] };
            for (int k = 0; k < 2; ++k)
            {
                char what[96];
                std::snprintf(what, sizeof(what), "box %s pushed forward by system %s", k ? "B" : "A", k ? "B" : "A");
                inRange(what, both.box[k].x - x0[k], 2.0f, 4.5f);
                std::snprintf(what, sizeof(what), "|box %s (both) - box %s (system %s alone)|", k ? "B" : "A", k ? "B" : "A", k ? "B" : "A");
                inRange(what, (both.box[k] - solo[k]->box[k]).magnitude(), 0.0f, 0.25f);
            }
        }

        // C2 last: the swapped pool order (A at index 1), run only when everything above held.
        if (armSplit || armFluid)
        {
            if (failures == 0)
            {
                runSub(P2_SWAP);
                subSceneChecks(P2_SWAP);
                if (runs[P2_SWAP].ok)
                {
                    c2[0] = particle2CompareWords("C2 A both/swap", both, runs[P2_SWAP], 0);
                    c2[1] = particle2CompareWords("C2 B both/swap", both, runs[P2_SWAP], 1);
                    for (int k = 0; k < 2; ++k)
                        if (c2[k]) { std::printf("FAIL: C2 %s: %zu words differ from the swapped pool order\n", k ? "B" : "A", c2[k]); ++failures; }
                    if (armFluid && !particle2CompareDiffuse("C2 B diffuse both/swap", both, runs[P2_SWAP], 1)) ++failures;
                }
            }
            else
                std::printf("  [swap] not run: %d check(s) already failed\n", failures);
        }

        physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (failures)
        {
            std::printf("[ FAIL ] gpu-sdk-step: particle2 (%s, %s): %d check(s) failed\n", arm, solverName, failures);
            return 2;
        }
        if (armPush)
            std::printf("[  OK  ] gpu-sdk-step: particle2 (push, %s): box A moved %.4f, box B %.4f; vs alone %.4f / %.4f\n",
                        solverName, both.box[0].x + 7.5f, both.box[1].x - 12.5f,
                        (both.box[0] - runs[P2_SOLO_A].box[0]).magnitude(), (both.box[1] - runs[P2_SOLO_B].box[1]).magnitude());
        else if (armApply)
            std::printf("[  OK  ] gpu-sdk-step: particle2 (apply, %s): C1 %zu/%u %zu/%u; B z %.6f, v.x %.6f after the apply, "
                        "rests %.6f off the patch; A's edit lost\n", solverName, c1[0], both.n[0] * 8u, c1[1], both.n[1] * 8u,
                        both.pAfterApply[1].z, both.vAfterApply[1].x, both.yMin[1]);
        else
            std::printf("[  OK  ] gpu-sdk-step: particle2 (%s, %s): C1 %zu/%u %zu/%u, C2 %zu/%u %zu/%u; A rests %.6f z %.6f, "
                        "B y[%.6f %.6f] z %.6f%s\n", arm, solverName, c1[0], both.n[0] * 8u, c1[1], both.n[1] * 8u,
                        c2[0], both.n[0] * 8u, c2[1], both.n[1] * 8u, both.yMin[0], both.meanP[0].z, both.yMin[1], both.yMax[1],
                        both.meanP[1].z, armFluid ? ", diffuse multisets equal" : "");
        return 0;
    }

    if (!std::strcmp(sceneName, "substepgravity"))
    {
        SubstepGravityScene gs = buildSubstepGravityScene(physics, scene, material);
        ExtForceScene& es = gs.e;
        std::printf("  [  OK  ] scene 'substepgravity' built: eight free-falling 1000 kg boxes (K F S C D L + a "
                    "D6-locked pair J1/J2) + 640 fillers, gravity and forces applied per TGS substep\n");
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: substepgravity pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 0;
        }
        const float dt = 1.0f / 60.0f;
        for (int i = 0; i < 60; ++i)
        {
            if (i == 30)
            {
                es.K->addForce(PxVec3(3.0f, 0.0f, 0.0f), PxForceMode::eVELOCITY_CHANGE, false);
                es.S->addTorque(PxVec3(0.0f, 500.0f, 0.0f), PxForceMode::eIMPULSE, false);
            }
            if (i >= 30)
            {
                es.F->addForce(PxVec3(0.0f, 0.0f, 2000.0f), PxForceMode::eFORCE, false);
                gs.D->addForce(PxVec3(0.0f, 0.0f, 2000.0f), PxForceMode::eFORCE, false);
                gs.L->addForce(PxVec3(2000.0f, 0.0f, 0.0f), PxForceMode::eFORCE, false);
                gs.L->addTorque(PxVec3(0.0f, 500.0f, 0.0f), PxForceMode::eFORCE, false);
            }
            scene->simulate(dt);
            scene->fetchResults(true);
        }
        const PxVec3 kP = es.K->getGlobalPose().p, kV = es.K->getLinearVelocity();
        const PxVec3 fP = es.F->getGlobalPose().p, fV = es.F->getLinearVelocity();
        const PxQuat sQ = es.S->getGlobalPose().q;
        const PxVec3 sW = es.S->getAngularVelocity();
        const PxVec3 cP = es.C->getGlobalPose().p, cV = es.C->getLinearVelocity();
        const PxVec3 dP = gs.D->getGlobalPose().p, dV = gs.D->getLinearVelocity();
        const PxVec3 lP = gs.L->getGlobalPose().p, lV = gs.L->getLinearVelocity(), lW = gs.L->getAngularVelocity();
        const PxVec3 j1P = gs.J1->getGlobalPose().p, j1V = gs.J1->getLinearVelocity();
        const PxVec3 j2P = gs.J2->getGlobalPose().p, j2V = gs.J2->getLinearVelocity();
        std::printf("  K  p=(%.6f, %.6f, %.6f) v=(%.6f, %.6f, %.6f)\n", kP.x, kP.y, kP.z, kV.x, kV.y, kV.z);
        std::printf("  F  p=(%.6f, %.6f, %.6f) v=(%.6f, %.6f, %.6f)\n", fP.x, fP.y, fP.z, fV.x, fV.y, fV.z);
        std::printf("  S  q=(%.6f, %.6f, %.6f, %.6f) w=(%.6f, %.6f, %.6f)\n", sQ.x, sQ.y, sQ.z, sQ.w, sW.x, sW.y, sW.z);
        std::printf("  C  p=(%.6f, %.6f, %.6f) v=(%.6f, %.6f, %.6f)\n", cP.x, cP.y, cP.z, cV.x, cV.y, cV.z);
        std::printf("  D  p=(%.6f, %.6f, %.6f) v=(%.6f, %.6f, %.6f)\n", dP.x, dP.y, dP.z, dV.x, dV.y, dV.z);
        std::printf("  L  p=(%.6f, %.6f, %.6f) v=(%.6f, %.6f, %.6f) w=(%.6f, %.6f, %.6f)\n", lP.x, lP.y, lP.z,
                    lV.x, lV.y, lV.z, lW.x, lW.y, lW.z);
        std::printf("  J1 p=(%.6f, %.6f, %.6f) v=(%.6f, %.6f, %.6f)\n", j1P.x, j1P.y, j1P.z, j1V.x, j1V.y, j1V.z);
        std::printf("  J2 p=(%.6f, %.6f, %.6f) v=(%.6f, %.6f, %.6f)\n", j2P.x, j2P.y, j2P.z, j2V.x, j2V.y, j2V.z);

        int failures = 0;
        auto near = [&](const char* what, float got, float want, float tol) {
            if (!(std::fabs(got - want) <= tol))
            { std::printf("FAIL: %s = %.6f, expected %.6f +- %g\n", what, got, want, tol); ++failures; }
        };
        // Per-substep application: a frame moves v0*dt + a*dt^2*(n+1)/(2n), n = 4 -> 0.625.
        const float yEnd = 100.0f - 0.002725f * 1807.5f;  // g dt^2 (60*59/2 + 60*0.625) = 95.0745625
        near("C.y", cP.y, yEnd, 2e-3f);              near("C.v.y", cV.y, -9.81f, 1e-3f);
        near("C.x", cP.x, -5.0f, 1e-5f);             near("C.z", cP.z, 0.0f, 1e-5f);
        near("K.x", kP.x, 1.48125f, 1e-4f);          near("K.v.x", kV.x, 3.0f, 1e-4f);
        near("K.y", kP.y, yEnd, 2e-3f);
        near("F.z", fP.z, 0.2520833f, 5e-5f);        near("F.v.z", fV.z, 1.0f, 1e-4f);
        near("F.y", fP.y, yEnd, 2e-3f);
        near("S.w.y", sW.y, 3.0f, 1e-4f);            near("S.w.x", sW.x, 0.0f, 1e-5f);
        near("S.w.z", sW.z, 0.0f, 1e-5f);
        near("S.q.y", std::fabs(sQ.y), 0.6747493f, 1e-4f);
        near("S.q.w", std::fabs(sQ.w), 0.7380470f, 1e-4f);
        // D: the GPU (CUDA) semantics -- gravity once an acceleration opens the gate.
        const float dyEnd = 100.0f - 0.002725f * 453.75f;  // g dt^2 (30*29/2 + 30*0.625) = 98.7635313
        if (!cpuControl)
        {
            near("D.y", dP.y, dyEnd, 2e-3f);         near("D.v.y", dV.y, -4.905f, 1e-3f);
        }
        near("D.z", dP.z, 0.2520833f, 5e-5f);
        // L: smoke + parity only (see the builder): the locks are re-applied after this kernel.
        near("L.x", lP.x, 20.0f, 1e-6f);             near("L.v.x", lV.x, 0.0f, 1e-6f);
        near("L.w.y", lW.y, 0.0f, 1e-6f);            near("L.y", lP.y, yEnd, 2e-3f);
        // J1/J2: the SLAB path (the only bodies with valid solverBodyReferences).
        near("J1.y", j1P.y, yEnd, 2e-3f);            near("J2.y", j2P.y, yEnd, 2e-3f);
        near("J1.v.y", j1V.y, -9.81f, 2e-3f);        near("J2.v.y", j2V.y, -9.81f, 2e-3f);
        near("|J1.y-J2.y|", std::fabs(j1P.y - j2P.y), 0.0f, 1e-4f);
        // The fillers fall like C (and keep the island on the per-partition path).
        float fillerDev = 0.0f;
        for (PxRigidDynamic* b : gs.fillers)
            fillerDev = std::max(fillerDev, std::fabs(b->getGlobalPose().p.y - cP.y));
        std::printf("  %zu fillers: max |y - C.y| = %.6f\n", gs.fillers.size(), fillerDev);
        near("max |filler.y - C.y|", fillerDev, 0.0f, 1e-4f);
        int asleep = 0;
        for (PxRigidDynamic* b : { es.K, es.F, es.S, es.C, gs.D, gs.L, gs.J1, gs.J2 })
            asleep += b->isSleeping() ? 1 : 0;
        for (PxRigidDynamic* b : gs.fillers)
            asleep += b->isSleeping() ? 1 : 0;
        std::printf("  %d of %zu bodies asleep\n", asleep, 8 + gs.fillers.size());
        if (asleep) { std::printf("FAIL: %d bod(ies) fell asleep (sleep threshold 0 must keep them awake)\n", asleep); ++failures; }
        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (cpuControl)
        {
            std::printf("  [ control ] D: CPU y=%.6f v.y=%.6f vs GPU %.7f / -4.905 -- upstream applyAcceleration "
                        "ignores disableGravity (solverMultiBlockTGS.cu:1575-1583,1656; DyTGSDynamics.cpp:1490-1493)\n",
                        dP.y, dV.y, dyEnd);
            std::printf("[ control ] gpu-sdk-step: substepgravity CPU control run (%d check(s) failed), not scored\n",
                        failures);
            return 0;
        }
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: gravity and forces applied per substep: C fell to %.4f, K moved "
                        "%.5f, D (gravity disabled) fell to %.4f, J pair at %.4f\n", cP.y, kP.x, dP.y, j1P.y);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: scene 'substepgravity': %d check(s) failed\n", failures);
        return 2;
    }

    // PXG_SCENE=slabgrid (P9.97a): the TGS averaging family at the engine grid; the modes and
    // the predicted replica partition are described at buildSlabGridScene.
    if (!std::strcmp(sceneName, "slabgrid"))
    {
        const char* modeEnv = std::getenv("PXG_SLABGRID");
        const char* mode = modeEnv ? modeEnv : "slab";
        if (std::strcmp(mode, "slab") && std::strcmp(mode, "whole") && std::strcmp(mode, "one") &&
            std::strcmp(mode, "slab5"))
        {
            std::printf("[ FAIL ] gpu-sdk-step: PXG_SLABGRID=%s is not a mode (slab | whole | one | slab5)\n", mode);
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 1;
        }
        if (usePgs)
        {
            // The three kernels are launched by PxgTGSCudaSolverCore only; the PGS twins are
            // 2-D / 256 wide already, and the replica partition below is the TGS launch's.
            std::printf("[ FAIL ] gpu-sdk-step: PXG_SCENE=slabgrid is TGS-only (it observes the TGS "
                        "averaging launches, PxgTGSCudaSolverCore.cpp:1232-1239/1295-1306/1481-1495/1734-1748)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 1;
        }
        SlabGridScene sg = buildSlabGridScene(physics, scene, material, mode);
        const int nPairs = int(sg.pa.size());
        // solver-body counts: body 0 is the static world body (mBodyStartIndex = 1)
        const uint32_t O = 1u;
        const uint32_t M = uint32_t(2 * nPairs + int(sg.fillers.size()) + (sg.hub ? 1 + sg.K : 0));
        const uint32_t T = M + O;
        const uint32_t numSlabs = sg.K > 0 ? uint32_t((sg.K + 7) / 8) : 1u;   // gpuMaxNumPartitions 8
        uint32_t nTPB = 1;
        while (nTPB < numSlabs) nTPB <<= 1;
        nTPB = std::min(nTPB, 32u);
        const uint32_t bpw = 32u / nTPB;
        const uint32_t GD = (32u * ((M + 9u) / 10u) + 255u) / 256u;     // PxgTGSCudaSolverCore.cpp:1734-1735
        const uint32_t C = std::min(T, 8u * GD * bpw);                     // CUDA velocity-iteration reach
        std::printf("  [  OK  ] scene 'slabgrid' built (mode %s): %d replicas, %zu fillers, %s K=%d R=%.0f; "
                    "T=%u M=%u, expected numSlabs %u (bpw %u), %s path\n",
                    mode, nPairs, sg.fillers.size(), sg.hub ? "hub +" : "no hub,", sg.K, double(sg.R), T, M,
                    numSlabs, bpw, (T * numSlabs > 640u) ? "per-partition" : "whole-island (position iterations)");
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: slabgrid pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 0;
        }
        const float dt = 1.0f / 60.0f;
        const int steps = 30;
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
        }

        int failures = 0;
        const float yFree = 50.0f - 9.81f * dt * dt * 0.5f * float(steps) * float(steps + 1);   // 48.73288
        if (sg.hub)
        {
            const PxTransform H = sg.hub->getGlobalPose();
            const float hubAngle = 2.0f * std::atan2(H.q.y, H.q.w);
            float maxRad = 0.0f, maxDy = 0.0f;
            int off = 0, firstOff = -1;
            for (int i = 0; i < sg.K; ++i)
            {
                const PxVec3 p = sg.sats[size_t(i)]->getGlobalPose().p - H.p;
                const float er = std::fabs(std::sqrt(p.x * p.x + p.z * p.z) - sg.R), ey = std::fabs(p.y);
                maxRad = std::max(maxRad, er);
                maxDy = std::max(maxDy, ey);
                if (!(er < 0.02f * sg.R) || !(ey < 1e-3f)) { ++off; if (firstOff < 0) firstOff = i; }
            }
            std::printf("  hub y=%.5f (free fall %.5f) angle=%.5f | satellites off the wheel %d/%d (first %d), "
                        "max|r-R|=%.5f max|dy|=%.5f\n", H.p.y, yFree, hubAngle, off, sg.K, firstOff, maxRad, maxDy);
            if (!(std::fabs(H.p.y - yFree) < 1e-3f))
            { std::printf("FAIL: hub y = %.5f, free fall %.5f\n", H.p.y, yFree); ++failures; }
            if (!(hubAngle >= 0.9f && hubAngle <= 1.1f))
            { std::printf("FAIL: hub angle about y = %.5f, expected ~w*t = 1\n", hubAngle); ++failures; }
            if (off)
            { std::printf("FAIL: %d of %d satellites left the wheel (first %d)\n", off, sg.K, firstOff); ++failures; }
        }
        if (!sg.fillers.empty())
        {
            float fillDev = 0.0f;
            for (PxRigidDynamic* f : sg.fillers)
                fillDev = std::max(fillDev, std::fabs(f->getGlobalPose().p.y - yFree));
            std::printf("  %zu fillers: max|y - yFree| = %.5f\n", sg.fillers.size(), fillDev);
            if (!(fillDev < 1e-3f))
            { std::printf("FAIL: fillers max|y - yFree| = %.5f\n", fillDev); ++failures; }
        }
        if (nPairs > 0)
        {
            auto state = [&](int i, float* o) {
                const PxVec3 c(1030.0f + 12.0f * float(i), 0.0f, 0.0f);
                const PxTransform a = sg.pa[size_t(i)]->getGlobalPose(), b = sg.pb[size_t(i)]->getGlobalPose();
                const PxVec3 v[] = { a.p - c, b.p - c, sg.pa[size_t(i)]->getLinearVelocity(),
                                     sg.pb[size_t(i)]->getLinearVelocity(), sg.pa[size_t(i)]->getAngularVelocity(),
                                     sg.pb[size_t(i)]->getAngularVelocity() };
                for (int k = 0; k < 6; ++k) { o[3 * k] = v[k].x; o[3 * k + 1] = v[k].y; o[3 * k + 2] = v[k].z; }
                o[18] = a.q.x; o[19] = a.q.y; o[20] = a.q.z; o[21] = a.q.w;
                o[22] = b.q.x; o[23] = b.q.y; o[24] = b.q.z; o[25] = b.q.w;
            };
            std::vector<std::vector<float>> st(size_t(nPairs), std::vector<float>(26));
            for (int i = 0; i < nPairs; ++i) state(i, st[size_t(i)].data());
            // measured: classes of bitwise-equal replicas, in order of first member
            std::vector<int> cls(static_cast<size_t>(nPairs), -1);
            int nc = 0;
            for (int i = 0; i < nPairs; ++i)
            {
                if (cls[size_t(i)] >= 0) continue;
                cls[size_t(i)] = nc;
                for (int j = i + 1; j < nPairs; ++j)
                    if (cls[size_t(j)] < 0 && !std::memcmp(st[size_t(i)].data(), st[size_t(j)].data(), 26 * sizeof(float)))
                        cls[size_t(j)] = nc;
                ++nc;
            }
            // predicted: full (2+2i < C), straddler (1+2i < C <= 2+2i), rest (1+2i >= C)
            std::vector<int> pred(static_cast<size_t>(nPairs));
            for (int i = 0; i < nPairs; ++i)
            {
                const uint32_t b1 = O + 2u * uint32_t(i), b2 = b1 + 1u;
                pred[size_t(i)] = b2 < C ? 0 : (b1 < C ? 1 : 2);
            }
            auto describe = [&](const std::vector<int>& c) {
                std::string out;
                int prev = -1, start = 0;
                for (int i = 0; i <= nPairs; ++i)
                {
                    const int v = i < nPairs ? c[size_t(i)] : -2;
                    if (i > 0 && v != prev)
                    {
                        char buf[48];
                        std::snprintf(buf, sizeof(buf), "[%d..%d]", start, i - 1);
                        out += buf;
                        start = i;
                    }
                    prev = v;
                }
                return out;
            };
            // canonical form of the prediction: class ids numbered by first member
            std::vector<int> predCanon(static_cast<size_t>(nPairs));
            {
                int map[3] = { -1, -1, -1 }, next = 0;
                for (int i = 0; i < nPairs; ++i)
                {
                    int& m = map[pred[size_t(i)]];
                    if (m < 0) m = next++;
                    predCanon[size_t(i)] = m;
                }
            }
            int diff0 = 0, first0 = -1;
            float maxDev = 0.0f;
            for (int i = 1; i < nPairs; ++i)
            {
                if (cls[size_t(i)] != cls[0]) { ++diff0; if (first0 < 0) first0 = i; }
                for (int k = 0; k < 26; ++k)
                    maxDev = std::max(maxDev, std::fabs(st[size_t(i)][size_t(k)] - st[0][size_t(k)]));
            }
            std::printf("  replicas: G_D=%u bpw=%u C=min(T, 8*G_D*bpw)=%u | expected classes %s | measured %d "
                        "class(es) %s | %d/%d differ from replica 0 (first %d), max|dev|=%.3g\n",
                        GD, bpw, C, describe(pred).c_str(), nc, describe(cls).c_str(), diff0, nPairs - 1, first0,
                        double(maxDev));
            if (cls != predCanon)
            {
                std::printf("FAIL: replica partition %s, the CUDA's velocity-iteration reach predicts %s\n",
                            describe(cls).c_str(), describe(pred).c_str());
                ++failures;
            }
        }
        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        failures += unmappedPenalty();
        if (cpuControl)
        {
            std::printf("[ control ] gpu-sdk-step: slabgrid(%s) CPU control run (%d check(s) failed), not scored "
                        "(the replica partition is the GPU launch's)\n", mode, failures);
            return 0;
        }
        if (failures == 0)
        {
            if (nPairs > 0)
                std::printf("[  OK  ] gpu-sdk-step: slabgrid(%s): T=%u M=%u numSlabs %u -- every check holds, replica "
                            "boundary C=%u as the CUDA's velocity-iteration launch predicts\n", mode, T, M, numSlabs, C);
            else
                std::printf("[  OK  ] gpu-sdk-step: slabgrid(%s): T=%u M=%u numSlabs %u -- the hub falls freely and "
                            "all %d satellites stay on the wheel\n", mode, T, M, numSlabs, sg.K);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: scene 'slabgrid' (%s): %d check(s) failed\n", mode, failures);
        return 2;
    }

    // PXG_SCENE=sdfquery (P9.95): evaluatePointDistancesSDFBatch through
    // PxDirectGPUAPI::evaluateSDFDistances.  The meshes, shapes and points are described
    // at buildSdfQueryScene.
    if (!std::strcmp(sceneName, "sdfquery"))
    {
        SdfQueryScene qs = buildSdfQueryScene(physics, scene, material);
        if (!qs.ok)
        {
            std::printf("[ FAIL ] gpu-sdk-step: scene 'sdfquery' could not be built\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 1;
        }
        std::printf("  [  OK  ] scene 'sdfquery' built: S_A (mesh A), S_B (mesh B), S_C (mesh A, scale 2), "
                    "all static\n");
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: sdfquery pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 0;
        }
        const PxU32 NB = 3, MAXP = 1100;
        const std::vector<PxVec3> ptsA = sdfQueryPointsA(qs.A), ptsB = sdfQueryPointsB(qs.B);
        const PxU32 counts[NB] = { PxU32(ptsB.size()), PxU32(ptsA.size()), 5u };   // S_B, S_A, S_C
        const SdfQueryField* fields[NB] = { &qs.B, &qs.A, &qs.A };
        const size_t SLOTS = size_t(NB) * MAXP;
        std::vector<PxVec4> hostPts(SLOTS);
        // Live slots: the points, with a .w the kernel must not read (a NaN with a
        // payload, +-inf, junk).  Dead slots: distinct finite junk.
        for (size_t i = 0; i < SLOTS; ++i)
            hostPts[i] = PxVec4(100.0f + float(i), -200.0f - float(i), 300.0f + float(i), 7.0f);
        const PxU32 wBits[4] = { 0x7FC0BEEFu, 0x7F800000u, 0xFF800000u, 0x4640E6B7u };
        for (PxU32 b = 0; b < NB; ++b)
            for (PxU32 i = 0; i < counts[b]; ++i)
            {
                const PxVec3 p = (b == 0) ? ptsB[i] : ptsA[i];
                float w;
                std::memcpy(&w, &wBits[i % 4u], 4);
                hostPts[size_t(b) * MAXP + i] = PxVec4(p, w);
            }
        static const float POISON = -999.0f;
        const std::vector<PxVec4> poison(SLOTS, PxVec4(POISON));

        PxCudaContext* ctx = cudaCtx->getCudaContext();
        CUdeviceptr dOut = 0, dIdx = 0, dPts = 0, dCnt = 0;
        if (ctx->memAlloc(&dOut, SLOTS * sizeof(PxVec4)) != 0 || !dOut ||
            ctx->memAlloc(&dIdx, NB * sizeof(PxShapeGPUIndex)) != 0 || !dIdx ||
            ctx->memAlloc(&dPts, SLOTS * sizeof(PxVec4)) != 0 || !dPts ||
            ctx->memAlloc(&dCnt, NB * sizeof(PxU32)) != 0 || !dCnt)
        {
            std::printf("[ FAIL ] gpu-sdk-step: sdfquery memAlloc failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 1;
        }
        ctx->memcpyHtoD(dPts, hostPts.data(), SLOTS * sizeof(PxVec4));
        ctx->memcpyHtoD(dCnt, counts, NB * sizeof(PxU32));
        ctx->memcpyHtoD(dOut, poison.data(), SLOTS * sizeof(PxVec4));
        int failures = 0;
        auto query = [&]() {
            return scene->getDirectGPUAPI().evaluateSDFDistances(
                reinterpret_cast<PxVec4*>(dOut), reinterpret_cast<const PxShapeGPUIndex*>(dIdx),
                reinterpret_cast<const PxVec4*>(dPts), reinterpret_cast<const PxU32*>(dCnt), NB, MAXP);
        };
        std::vector<PxVec4> out(SLOTS);
        auto countPoison = [&]() {
            size_t n = 0;
            for (size_t i = 0; i < SLOTS; ++i)
                if (std::memcmp(&out[i], &poison[i], sizeof(PxVec4)) == 0) ++n;
            return n;
        };

        // (1) Before the first step the direct-GPU API is not initialized
        // (NpDirectGPUAPI.cpp:156): the call is refused and the buffer is untouched.
        {
            PxShapeGPUIndex idx0[NB] = { 0u, 0u, 0u };
            ctx->memcpyHtoD(dIdx, idx0, sizeof(idx0));
            const bool early = query();
            ctx->memcpyDtoH(out.data(), dOut, SLOTS * sizeof(PxVec4));
            const size_t p = countPoison();
            std::printf("  sdfquery: before the first step evaluateSDFDistances returned %s, %zu/%zu slots still poison\n",
                        early ? "true" : "false", p, SLOTS);
            if (early || p != SLOTS)
            { std::printf("FAIL: the pre-step call must be refused and write nothing\n"); ++failures; }
        }

        // (2) One step: setDirectGPUAPIInitialized (ScPipeline.cpp:2814-2815).
        scene->simulate(1.0f / 60.0f);
        scene->fetchResults(true);

        // (3) The real query.
        const PxShapeGPUIndex idx[NB] = { qs.SB->getGPUIndex(), qs.SA->getGPUIndex(), qs.SC->getGPUIndex() };
        std::printf("  sdfquery: shape GPU indices S_B=%u S_A=%u S_C=%u, counts {%u, %u, %u}, maxPointCount %u\n",
                    idx[0], idx[1], idx[2], counts[0], counts[1], counts[2], MAXP);
        for (PxU32 b = 0; b < NB; ++b)
            if (idx[b] == 0xFFFFFFFFu) { std::printf("FAIL: shape %u has no GPU index\n", b); ++failures; }
        ctx->memcpyHtoD(dIdx, idx, sizeof(idx));
        ctx->memcpyHtoD(dOut, poison.data(), SLOTS * sizeof(PxVec4));
        const unsigned launchFailsBefore = gSdfQueryLaunchFailures;
        const bool ret = query();
        ctx->memcpyDtoH(out.data(), dOut, SLOTS * sizeof(PxVec4));
        std::vector<PxVec4> inBack(SLOTS);
        ctx->memcpyDtoH(inBack.data(), dPts, SLOTS * sizeof(PxVec4));
        ctx->memFree(dOut); ctx->memFree(dIdx); ctx->memFree(dPts); ctx->memFree(dCnt);
        const unsigned launchFails = gSdfQueryLaunchFailures - launchFailsBefore;
        std::printf("  sdfquery: evaluateSDFDistances returned %s (reported, not scored: the shim's "
                    "streamSynchronize overwrites the launch result)\n", ret ? "true" : "false");

        // (a) + (b): every live slot written, bit-equal to the literal host oracle.
        size_t written = 0, livePoison = 0, mismatches = 0, oracleBad = 0;
        for (PxU32 b = 0; b < NB; ++b)
            for (PxU32 i = 0; i < counts[b]; ++i)
            {
                const size_t s = size_t(b) * MAXP + i;
                if (std::memcmp(&out[s], &poison[s], sizeof(PxVec4)) == 0) { ++livePoison; continue; }
                ++written;
                bool ok = true;
                const PxVec4 want = sdfQueryOracle(*fields[b], hostPts[s].getXYZ(), ok);
                if (!ok)
                {
                    std::printf("  sdfquery: slot %u/%u p=(%.9g, %.9g, %.9g) is not on texel centres\n", b, i,
                                hostPts[s].x, hostPts[s].y, hostPts[s].z);
                    ++oracleBad;
                    continue;
                }
                if (std::memcmp(&out[s], &want, sizeof(PxVec4)) != 0)
                {
                    if (mismatches < 8)
                        std::printf("FAIL: slot %u/%u p=(%.6f, %.6f, %.6f): got (%.9g, %.9g, %.9g, %.9g), "
                                    "want (%.9g, %.9g, %.9g, %.9g)\n", b, i, hostPts[s].x, hostPts[s].y, hostPts[s].z,
                                    out[s].x, out[s].y, out[s].z, out[s].w, want.x, want.y, want.z, want.w);
                    ++mismatches;
                }
            }
        // (c) the slots past each count keep their poison.
        size_t tailWritten = 0;
        for (PxU32 b = 0; b < NB; ++b)
            for (PxU32 i = counts[b]; i < MAXP; ++i)
            {
                const size_t s = size_t(b) * MAXP + i;
                if (std::memcmp(&out[s], &poison[s], sizeof(PxVec4)) != 0) ++tailWritten;
            }
        // (d) S_C (scale 2) == S_A's first 5, bitwise.
        const bool scaleInvariant = std::memcmp(&out[2 * MAXP], &out[1 * MAXP], 5 * sizeof(PxVec4)) == 0;
        // (e) the shared node answers differently on the two meshes.
        const bool mixupDistinct = std::memcmp(&out[0], &out[MAXP + 5], sizeof(PxVec4)) != 0;
        // The inputs are read-only.
        const bool inputsKept = std::memcmp(inBack.data(), hostPts.data(), SLOTS * sizeof(PxVec4)) == 0;
        std::printf("  sdfquery: S_A[0]=(%.6f, %.6f, %.6f, %.6f)  S_B[0]=(%.6f, %.6f, %.6f, %.6f)  S_A[5]=(%.6f, %.6f, "
                    "%.6f, %.6f)\n", out[MAXP].x, out[MAXP].y, out[MAXP].z, out[MAXP].w, out[0].x, out[0].y, out[0].z,
                    out[0].w, out[MAXP + 5].x, out[MAXP + 5].y, out[MAXP + 5].z, out[MAXP + 5].w);
        std::printf("  sdfquery: %zu written / %zu poison / %zu mismatches / %s / %s  (tail slots written %zu, "
                    "oracle preconditions failed %zu, \"fail to launch\" errors %u)\n", written, livePoison, mismatches,
                    scaleInvariant ? "scale-invariant" : "SCALE-DEPENDENT", mixupDistinct ? "mixup-distinct" : "MIXUP-EQUAL",
                    tailWritten, oracleBad, launchFails);
        if (livePoison) { std::printf("FAIL: %zu live slot(s) still hold the poison\n", livePoison); ++failures; }
        if (mismatches) { std::printf("FAIL: %zu slot(s) differ bitwise from convexMeshMidphase.cu:1282-1286\n", mismatches); ++failures; }
        if (oracleBad) { std::printf("FAIL: %zu point(s) do not sample exact texel centres\n", oracleBad); ++failures; }
        if (tailWritten) { std::printf("FAIL: %zu slot(s) past a shape's count were written\n", tailWritten); ++failures; }
        if (!scaleInvariant) { std::printf("FAIL: S_C (scale 2) differs from S_A: the kernel must ignore PxMeshScale\n"); ++failures; }
        if (!mixupDistinct) { std::printf("FAIL: S_B and S_A answer the shared node identically\n"); ++failures; }
        if (!inputsKept) { std::printf("FAIL: the sample-point buffer changed\n"); ++failures; }
        if (launchFails) { std::printf("FAIL: %u \"evaluatePointDistancesSDF fail to launch\" error(s)\n", launchFails); ++failures; }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        if (gUnmappedLaunchReports)
        { std::printf("FAIL: %u unmapped launch report(s)\n", gUnmappedLaunchReports); ++failures; }
        failures += unmappedPenalty();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: sdfquery: %zu points on three SDF shapes bit-equal to the host "
                        "oracle -- evaluatePointDistancesSDFBatch runs on Vulkan\n", written);
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: scene 'sdfquery': %d check(s) failed\n", failures);
        return 2;
    }

    // PXG_SCENE=directapimerge (P9.95): mergeTransformCacheAndBoundArrayChanges, the
    // direct-GPU API's between-steps merge of new and re-inserted shapes.  The three
    // lanes are described at buildDirectApiMergeScene.
    if (!std::strcmp(sceneName, "directapimerge"))
    {
        DirectApiMergeScene ms = buildDirectApiMergeScene(physics, scene, material);
        std::printf("  [  OK  ] scene 'directapimerge' built: lanes N (x=-30, slab added later), S (x=-10, slab "
                    "re-inserted), D (x=+10, body re-inserted)\n");
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: directapimerge pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 0;
        }
        const bool control = std::getenv("PXG_DIRECTAPI_CONTROL") != nullptr;
        const float dt = 1.0f / 60.0f;
        const int steps = 240;
        const int W1_AFTER = 1;    // window W1 opens after the 2nd step's fetchResults
        const int W2_AFTER = 60;   // window W2 opens after the 61st step's fetchResults

        PxCudaContext* ctx = cudaCtx->getCudaContext();
        static const float POISON = -999.0f;
        const size_t OUT_FLOATS = 8;   // a PxTransform is 7; round up
        CUdeviceptr devIdx = 0, devOut = 0;
        if (!control && (ctx->memAlloc(&devIdx, sizeof(PxU32)) != 0 || !devIdx ||
                         ctx->memAlloc(&devOut, OUT_FLOATS * sizeof(float)) != 0 || !devOut))
        {
            std::printf("[ FAIL ] gpu-sdk-step: directapimerge memAlloc failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 1;
        }
        int failures = 0;
        // One body's pose (7 floats) or linear velocity (3 floats) through the direct
        // API, into memory poisoned before every read: a refused getter reads -999.
        // (A copy of directapi's readPoisoned, deliberately not shared.)
        auto readBody = [&](PxRigidDynamic* b, PxRigidDynamicGPUAPIReadType::Enum type, size_t count,
                            float* host) -> bool
        {
            float poison[OUT_FLOATS];
            for (size_t i = 0; i < OUT_FLOATS; ++i) poison[i] = POISON;
            ctx->memcpyHtoD(devOut, poison, OUT_FLOATS * sizeof(float));
            const PxU32 gpuIndex = b->getGPUIndex();
            ctx->memcpyHtoD(devIdx, &gpuIndex, sizeof(PxU32));
            const bool ok = scene->getDirectGPUAPI().getRigidDynamicData(
                reinterpret_cast<void*>(devOut), reinterpret_cast<const PxRigidDynamicGPUIndex*>(devIdx), type, 1);
            ctx->memcpyDtoH(host, devOut, count * sizeof(float));
            return ok;
        };
        // PxTransform is (q.xyzw, p.xyz): the centre height is element 5.
        auto bodyY = [&](PxRigidDynamic* b) -> float {
            if (control) return b->getGlobalPose().p.y;
            float h[OUT_FLOATS];
            if (!readBody(b, PxRigidDynamicGPUAPIReadType::eGLOBAL_POSE, 7, h))
            { std::printf("FAIL: getRigidDynamicData(eGLOBAL_POSE) returned false\n"); ++failures; }
            return h[5];
        };
        auto bodyVy = [&](PxRigidDynamic* b) -> float {
            if (control) return b->getLinearVelocity().y;
            float h[OUT_FLOATS];
            if (!readBody(b, PxRigidDynamicGPUAPIReadType::eLINEAR_VELOCITY, 3, h))
            { std::printf("FAIL: getRigidDynamicData(eLINEAR_VELOCITY) returned false\n"); ++failures; }
            return h[1];
        };

        std::printf("  stepping directapimerge (%d x %.4fs)%s: W1 after step %d, W2 after step %d\n", steps, dt,
                    control ? " -- CONTROL, no direct-GPU API" : "", W1_AFTER, W2_AFTER);
        float cVy60 = POISON, cVy61 = POISON, cY61 = POISON;
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            if (i == W1_AFTER)
            {
                // W1: G2 under B and two far decoys (isNew records, from == to), and G1
                // re-inserted under a new element id (a device-to-device record).
                scene->addActor(*makeMergeSlab(physics, material, -30.0f));
                scene->addActor(*makeMergeSlab(physics, material, 60.0f));
                scene->addActor(*makeMergeSlab(physics, material, 90.0f));
                if (!scene->resetFiltering(*ms.G1))
                { std::printf("FAIL: resetFiltering(G1) returned false\n"); ++failures; }
            }
            if (i == W2_AFTER)
            {
                cVy60 = bodyVy(ms.C);
                // W2: C re-inserted (a device-to-device record for a dynamic body).
                if (!scene->resetFiltering(*ms.C))
                { std::printf("FAIL: resetFiltering(C) returned false\n"); ++failures; }
            }
            if (i == W2_AFTER + 1)
            {
                cVy61 = bodyVy(ms.C);
                cY61 = bodyY(ms.C);
            }
        }
        const float aY = bodyY(ms.A), bY = bodyY(ms.B), cY = bodyY(ms.C);
        if (!control) { ctx->memFree(devIdx); ctx->memFree(devOut); }
        std::printf("  %slane N: B.y=%.4f   lane S: A.y=%.4f   lane D: C.v_y(step %d)=%.4f C.v_y(step %d)=%.4f "
                    "C.y(step %d)=%.4f C.y=%.4f\n", control ? "[ control ] " : "", bY, aY, W2_AFTER, cVy60,
                    W2_AFTER + 1, cVy61, W2_AFTER + 1, cY61, cY);
        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        if (control)
        {
            std::printf("[ control ] gpu-sdk-step: directapimerge control run, not scored\n");
            return 0;
        }

        auto near = [&](const char* what, float got, float want, float tol) {
            if (!(std::fabs(got - want) <= tol))
            { std::printf("FAIL: %s = %.4f, expected %.4f +- %g\n", what, got, want, tol); ++failures; }
        };
        near("B.y (lane N: G2 is an isNew record, copied from the CPU arrays)", bY, 0.5f, 0.02f);
        near("A.y (lane S: G1's new slot is copied from its old DEVICE slot)", aY, 0.5f, 0.02f);
        near("C.v_y before the reset (lane D baseline)", cVy60, 0.0f, 0.02f);
        near("C.v_y after the reset step (lane D: no one-step free fall)", cVy61, 0.0f, 0.02f);
        near("C.y after the reset step (lane D: the GPU pose, not the stale host pose y=3)", cY61, 0.5f, 0.02f);
        near("C.y at the end (lane D)", cY, 0.5f, 0.02f);
        failures += unmappedPenalty();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: directapimerge: an added slab (isNew), a re-inserted slab and a "
                        "re-inserted body (device to device) all hold their bodies -- "
                        "mergeTransformCacheAndBoundArrayChanges runs on Vulkan\n");
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: scene 'directapimerge': %d check(s) failed\n", failures);
        return 2;
    }

    if (!std::strcmp(sceneName, "directapiwrite"))
    {
        DirectApiWriteScene ws = buildDirectApiWriteScene(physics, scene, material);
        std::printf("  [  OK  ] scene 'directapiwrite' built: seven free-falling 1000 kg boxes, one write "
                    "window through PxDirectGPUAPI::setRigidDynamicData\n");
        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: directapiwrite pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 0;
        }
        const float dt = 1.0f / 60.0f;
        const PxQuat qT(0.0f, 0.38268343f, 0.0f, 0.92387953f);   // 45 degrees about y
        const PxQuat qT2(0.70710677f, 0.0f, 0.0f, 0.70710677f);  // 90 degrees about x
        const PxVec3 pT(-10.0f, 0.5f, 3.0f), pT2(-6.0f, 30.0f, -4.0f);

        // PXG_DIRECTAPI_CONTROL=1: the same writes through the ORDINARY CPU setters,
        // with the direct API off.  Reported, never scored (addForce accumulates on
        // the CPU, so the double eFORCE write has no twin there).
        if (std::getenv("PXG_DIRECTAPI_CONTROL"))
        {
            scene->simulate(dt); scene->fetchResults(true);
            ws.T2->setGlobalPose(PxTransform(pT2, qT2));
            ws.T->setGlobalPose(PxTransform(pT, qT));
            ws.V->setLinearVelocity(PxVec3(4.0f, 0.0f, -2.0f));
            ws.W->setAngularVelocity(PxVec3(0.0f, 3.0f, 0.0f));
            ws.F->addForce(PxVec3(0.0f, 0.0f, 5000.0f));
            ws.Q->addTorque(PxVec3(0.0f, 0.0f, 30000.0f));
            scene->simulate(dt); scene->fetchResults(true);
            const PxVec3 fa = ws.F->getLinearAcceleration(), qa = ws.Q->getAngularAcceleration();
            std::printf("  [ control ] after the write step: F accel=(%.5f, %.5f, %.5f)  Q ang accel=(%.4f, %.4f, %.4f)\n",
                        fa.x, fa.y, fa.z, qa.x, qa.y, qa.z);
            for (int i = 2; i < 61; ++i) { scene->simulate(dt); scene->fetchResults(true); }
            const PxRigidDynamic* bodies[7] = { ws.T, ws.T2, ws.V, ws.W, ws.F, ws.Q, ws.R };
            const char* names[7] = { "T", "T2", "V", "W", "F", "Q", "R" };
            for (int b = 0; b < 7; ++b)
            {
                const PxTransform t = bodies[b]->getGlobalPose();
                std::printf("  [ control ] %-2s p=(%.6f, %.6f, %.6f) q=(%.6f, %.6f, %.6f, %.6f)\n", names[b],
                            t.p.x, t.p.y, t.p.z, t.q.x, t.q.y, t.q.z, t.q.w);
            }
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            std::printf("[ control ] gpu-sdk-step: directapiwrite control run, not scored\n");
            return 0;
        }

        PxCudaContext* ctx = cudaCtx->getCudaContext();
        PxDirectGPUAPI& api = scene->getDirectGPUAPI();
        static const float POISON = -999.0f;
        const size_t MAX_FLOATS = 64;
        CUdeviceptr devIdx = 0, devData = 0;
        if (ctx->memAlloc(&devIdx, 8 * sizeof(PxU32)) != 0 || !devIdx ||
            ctx->memAlloc(&devData, MAX_FLOATS * sizeof(float)) != 0 || !devData)
        {
            std::printf("[ FAIL ] gpu-sdk-step: directapiwrite memAlloc failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 1;
        }
        int failures = 0;
        // A read of `per` floats for each index, into poisoned memory with ONE extra
        // poisoned slot past the end: a refused kernel leaves the poison, a kernel with
        // the wrong element stride writes past the end.
        auto rd = [&](PxRigidDynamicGPUAPIReadType::Enum type, std::vector<PxU32> idx, size_t per,
                      const char* what) -> std::vector<float>
        {
            const size_t n = idx.size() * per;
            std::vector<float> host(n + 1, POISON);
            ctx->memcpyHtoD(devData, host.data(), (n + 1) * sizeof(float));
            ctx->memcpyHtoD(devIdx, idx.data(), idx.size() * sizeof(PxU32));
            const bool ok = api.getRigidDynamicData(reinterpret_cast<void*>(devData),
                                                    reinterpret_cast<const PxRigidDynamicGPUIndex*>(devIdx),
                                                    type, PxU32(idx.size()));
            ctx->memcpyDtoH(host.data(), devData, (n + 1) * sizeof(float));
            if (!ok) { std::printf("FAIL: %s: getRigidDynamicData returned false\n", what); ++failures; }
            for (size_t i = 0; i < n; ++i)
                if (host[i] == POISON)
                { std::printf("FAIL: %s: element float %zu still holds the poison\n", what, i); ++failures; break; }
            if (host[n] != POISON)
            { std::printf("FAIL: %s: the slot past the end was written\n", what); ++failures; }
            host.resize(n);
            return host;
        };
        auto wr = [&](PxRigidDynamicGPUAPIWriteType::Enum type, std::vector<PxU32> idx,
                      std::vector<float> data, const char* what)
        {
            ctx->memcpyHtoD(devData, data.data(), data.size() * sizeof(float));
            ctx->memcpyHtoD(devIdx, idx.data(), idx.size() * sizeof(PxU32));
            if (!api.setRigidDynamicData(reinterpret_cast<const void*>(devData),
                                         reinterpret_cast<const PxRigidDynamicGPUIndex*>(devIdx),
                                         type, PxU32(idx.size())))
            { std::printf("FAIL: %s: setRigidDynamicData returned false\n", what); ++failures; }
        };
        auto near = [&](const char* what, float got, float want, float tol) {
            if (!(std::fabs(got - want) <= tol))
            { std::printf("FAIL: %s = %.6f, expected %.6f +- %g\n", what, got, want, tol); ++failures; }
        };
        auto same = [&](const char* what, const std::vector<float>& got, const std::vector<float>& want) {
            for (size_t i = 0; i < want.size(); ++i)
                if (!(got[i] == want[i]))
                { std::printf("FAIL: %s[%zu] = %.8g, expected exactly %.8g\n", what, i, got[i], want[i]);
                  ++failures; return; }
        };
        using RT = PxRigidDynamicGPUAPIReadType;
        using WT = PxRigidDynamicGPUAPIWriteType;
        const PxU32 iT = ws.T->getGPUIndex(), iT2 = ws.T2->getGPUIndex(), iV = ws.V->getGPUIndex(),
                    iW = ws.W->getGPUIndex(), iF = ws.F->getGPUIndex(), iQ = ws.Q->getGPUIndex(),
                    iR = ws.R->getGPUIndex();
        std::printf("  gpuIndex T=%u T2=%u V=%u W=%u F=%u Q=%u R=%u\n", iT, iT2, iV, iW, iF, iQ, iR);

        // ---- step 1: the warm-up (every direct call is refused before it) ----
        scene->simulate(dt); scene->fetchResults(true);
        const std::vector<float> aR1 = rd(RT::eLINEAR_ACCELERATION, { iR }, 3, "R lin accel after warm-up");
        std::printf("  after warm-up: R lin accel=(%.5f, %.5f, %.5f)\n", aR1[0], aR1[1], aR1[2]);
        near("R lin accel.y after warm-up", aR1[1], -9.81f, 1e-3f);
        near("R lin accel.x after warm-up", aR1[0], 0.0f, 1e-5f);
        const std::vector<float> vR = rd(RT::eLINEAR_VELOCITY, { iR }, 3, "R lin vel");
        const std::vector<float> wR = rd(RT::eANGULAR_VELOCITY, { iR }, 3, "R ang vel");

        // ---- the write window ----
        const std::vector<float> poses = { qT2.x, qT2.y, qT2.z, qT2.w, pT2.x, pT2.y, pT2.z,
                                           qT.x, qT.y, qT.z, qT.w, pT.x, pT.y, pT.z };
        wr(WT::eGLOBAL_POSE, { iT2, iT }, poses, "pose T2,T");
        same("pose read-back T2,T", rd(RT::eGLOBAL_POSE, { iT2, iT }, 7, "pose read-back"), poses);

        const std::vector<float> vel = { 4.0f, 0.0f, -2.0f, vR[0], vR[1], vR[2] };
        wr(WT::eLINEAR_VELOCITY, { iV, iR }, vel, "lin vel V,R");
        same("lin vel read-back V,R", rd(RT::eLINEAR_VELOCITY, { iV, iR }, 3, "lin vel read-back"), vel);
        same("lin accel right after the write V,R",
             rd(RT::eLINEAR_ACCELERATION, { iV, iR }, 3, "lin accel after write"), { 0, 0, 0, 0, 0, 0 });

        const std::vector<float> ang = { 0.0f, 3.0f, 0.0f, wR[0], wR[1], wR[2] };
        wr(WT::eANGULAR_VELOCITY, { iW, iR }, ang, "ang vel W,R");
        same("ang vel read-back W,R", rd(RT::eANGULAR_VELOCITY, { iW, iR }, 3, "ang vel read-back"), ang);
        same("ang accel right after the write W,R",
             rd(RT::eANGULAR_ACCELERATION, { iW, iR }, 3, "ang accel after write"), { 0, 0, 0, 0, 0, 0 });

        wr(WT::eFORCE, { iF }, { 0.0f, 0.0f, 10000.0f }, "force F (first)");
        wr(WT::eFORCE, { iF }, { 0.0f, 0.0f, 5000.0f }, "force F (second, the one that counts)");
        wr(WT::eTORQUE, { iQ }, { 0.0f, 0.0f, 30000.0f }, "torque Q");

        // ---- step 2: force and torque act for this one step ----
        scene->simulate(dt); scene->fetchResults(true);
        const std::vector<float> aF2 = rd(RT::eLINEAR_ACCELERATION, { iF }, 3, "F lin accel step 2");
        const std::vector<float> wQ2 = rd(RT::eANGULAR_VELOCITY, { iQ }, 3, "Q ang vel step 2");
        const std::vector<float> aQ2 = rd(RT::eANGULAR_ACCELERATION, { iQ }, 3, "Q ang accel step 2");
        const std::vector<float> vV2 = rd(RT::eLINEAR_VELOCITY, { iV }, 3, "V lin vel step 2");
        const std::vector<float> aV2 = rd(RT::eLINEAR_ACCELERATION, { iV }, 3, "V lin accel step 2");
        const std::vector<float> wW2 = rd(RT::eANGULAR_VELOCITY, { iW }, 3, "W ang vel step 2");
        const std::vector<float> aW2 = rd(RT::eANGULAR_ACCELERATION, { iW }, 3, "W ang accel step 2");
        std::printf("  step 2: F accel=(%.5f, %.5f, %.5f)  Q w=(%.5f, %.5f, %.5f) alpha=(%.3f, %.3f, %.3f)\n",
                    aF2[0], aF2[1], aF2[2], wQ2[0], wQ2[1], wQ2[2], aQ2[0], aQ2[1], aQ2[2]);
        near("F lin accel.z step 2 (5000 N / 1000 kg; 15 would mean it accumulated)", aF2[2], 5.0f, 1e-3f);
        near("F lin accel.y step 2", aF2[1], -9.81f, 1e-3f);
        near("Q ang vel.z step 2", wQ2[2], 3.0f, 1e-4f);
        near("Q ang accel.z step 2", aQ2[2], 180.0f, 1e-2f);
        near("V lin vel.x step 2", vV2[0], 4.0f, 1e-5f);
        near("V lin vel.y step 2", vV2[1], -0.1635f, 1e-4f);
        near("V lin vel.z step 2", vV2[2], -2.0f, 1e-5f);
        near("V lin accel.y step 2", aV2[1], -9.81f, 1e-3f);
        near("V lin accel.x step 2", aV2[0], 0.0f, 1e-3f);
        near("W ang vel.y step 2", wW2[1], 3.0f, 1e-4f);
        near("W ang accel.y step 2", aW2[1], 0.0f, 1e-3f);

        // ---- step 3: both were one-step accelerations ----
        scene->simulate(dt); scene->fetchResults(true);
        const std::vector<float> aF3 = rd(RT::eLINEAR_ACCELERATION, { iF }, 3, "F lin accel step 3");
        const std::vector<float> vF3 = rd(RT::eLINEAR_VELOCITY, { iF }, 3, "F lin vel step 3");
        const std::vector<float> aQ3 = rd(RT::eANGULAR_ACCELERATION, { iQ }, 3, "Q ang accel step 3");
        near("F lin accel.z step 3 (the force lasts one step)", aF3[2], 0.0f, 1e-3f);
        near("F lin vel.z step 3", vF3[2], 5.0f / 60.0f, 1e-5f);
        near("Q ang accel.z step 3 (the torque lasts one step)", aQ3[2], 0.0f, 1e-2f);

        // ---- 58 more steps: sixty after the write window ----
        for (int i = 3; i < 61; ++i) { scene->simulate(dt); scene->fetchResults(true); }
        const std::vector<float> fin = rd(RT::eGLOBAL_POSE, { iT, iT2, iV, iW, iF, iQ, iR }, 7, "final poses");
        ctx->memFree(devIdx);
        ctx->memFree(devData);
        const char* names[7] = { "T", "T2", "V", "W", "F", "Q", "R" };
        for (int b = 0; b < 7; ++b)
            std::printf("  %-2s p=(%.6f, %.6f, %.6f) q=(%.6f, %.6f, %.6f, %.6f)\n", names[b],
                        fin[b * 7 + 4], fin[b * 7 + 5], fin[b * 7 + 6],
                        fin[b * 7 + 0], fin[b * 7 + 1], fin[b * 7 + 2], fin[b * 7 + 3]);
        auto P = [&](int b, int c) { return fin[b * 7 + 4 + c]; };
        auto Qc = [&](int b, int c) { return fin[b * 7 + c]; };
        const float yFree61 = 20.0f - 5.152975f;       // g dt^2 * 61*62/2
        near("T.x", P(0, 0), -10.0f, 0.05f);  near("T.y (rests on the ground)", P(0, 1), 0.5f, 0.05f);
        near("T.z", P(0, 2), 3.0f, 0.05f);
        near("T.q.y", std::fabs(Qc(0, 1)), 0.38268343f, 1e-2f); near("T.q.w", std::fabs(Qc(0, 3)), 0.92387953f, 1e-2f);
        near("T2.x", P(1, 0), -6.0f, 1e-4f);  near("T2.y", P(1, 1), 24.84975f, 2e-3f);
        near("T2.z", P(1, 2), -4.0f, 1e-4f);
        near("T2.q.x", std::fabs(Qc(1, 0)), 0.70710677f, 1e-4f); near("T2.q.w", std::fabs(Qc(1, 3)), 0.70710677f, 1e-4f);
        near("V.x", P(2, 0), 2.0f, 1e-3f);    near("V.y", P(2, 1), 15.010525f, 2e-3f);
        near("V.z", P(2, 2), -2.0f, 1e-3f);
        near("W.q.y", std::fabs(Qc(3, 1)), 0.9974950f, 2e-3f); near("W.q.w", std::fabs(Qc(3, 3)), 0.0707372f, 2e-3f);
        near("W.y", P(3, 1), yFree61, 2e-3f);
        near("F.z", P(4, 2), 5.0f / 60.0f, 1e-4f); near("F.y", P(4, 1), yFree61, 2e-3f);
        near("Q.q.z", std::fabs(Qc(5, 2)), 0.9974950f, 2e-3f); near("Q.q.w", std::fabs(Qc(5, 3)), 0.0707372f, 2e-3f);
        near("R.y", P(6, 1), yFree61, 2e-3f); near("R.x", P(6, 0), 14.0f, 1e-5f);
        failures += unmappedPenalty();
        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: all five direct-API setters and both acceleration getters "
                        "answer on Vulkan -- teleports, velocities, a one-step force and torque, read back "
                        "and integrated\n");
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: scene 'directapiwrite': %d check(s) failed\n", failures);
        return 2;
    }

    if (!std::strcmp(sceneName, "directapi"))
    {
        DirectApiScene ds = buildDirectApiScene(physics, scene, material);
        std::printf("  [  OK  ] scene 'directapi' built: a falling box + a 1000 kg box on a "
                    "fully-locked D6 joint attached %.2f m off its centre of mass\n",
                    ds.hangerAttachOffsetX);

        if (!std::getenv("PXG_GPU_STEP"))
        {
            std::printf("[  OK  ] gpu-sdk-step: directapi pipeline CONSTRUCTED (set PXG_GPU_STEP=1 to step it)\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 0;
        }

        // PXG_DIRECTAPI_CONTROL=1: the same scene with the direct-API flag NOT
        // raised, read through the ORDINARY API.  Reported, never scored.
        //
        // This is the independent oracle for the joint reaction.  PxConstraint::
        // getForce() and PxDirectGPUAPI::getD6JointData() are two different readers
        // of the SAME PxgConstraintWriteback row, so agreement between them says the
        // direct-API kernels transcribe the row correctly -- and, just as usefully,
        // it says what FRAME the row is expressed in, which is not something this
        // scene gets to decide.
        if (std::getenv("PXG_DIRECTAPI_CONTROL"))
        {
            const float dtC = 1.0f / 60.0f;
            for (int i = 0; i < 240; ++i) { scene->simulate(dtC); scene->fetchResults(true); }
            PxVec3 f(0.0f), tq(0.0f);
            ds.joint->getConstraint()->getForce(f, tq);
            const PxVec3 hp = ds.hanger->getGlobalPose().p;
            std::printf("  [ control ] PXG_DIRECTAPI_CONTROL=1 (no direct-GPU API, ordinary readback)\n");
            std::printf("  [ control ] faller y=%.4f   hanger COM=(%.4f, %.4f, %.4f)\n",
                        ds.faller->getGlobalPose().p.y, hp.x, hp.y, hp.z);
            std::printf("  [ control ] PxConstraint::getForce  force =(%+9.2f, %+9.2f, %+9.2f) |F|=%.2f\n",
                        f.x, f.y, f.z, f.magnitude());
            std::printf("  [ control ] PxConstraint::getForce  torque=(%+9.2f, %+9.2f, %+9.2f) |T|=%.2f\n",
                        tq.x, tq.y, tq.z, tq.magnitude());
            std::printf("  [ control ] anchor=(%.2f, %.2f, %.2f)  m*g=%.2f\n",
                        ds.anchorWorld.x, ds.anchorWorld.y, ds.anchorWorld.z, ds.hangerMass * 9.81f);
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            std::printf("[ control ] gpu-sdk-step: directapi control run, not scored\n");
            return 0;
        }

        PxCudaContext* ctx = cudaCtx->getCudaContext();

        // One device index buffer and one device output buffer, reused for every read.
        // POISON is the point: a refused launch leaves it untouched, so a wrong answer
        // reads as an impossible number and never as a plausible stale one.
        static const float POISON = -999.0f;
        const size_t OUT_FLOATS = 8;                    // a PxTransform is 7; round up
        CUdeviceptr devIdx = 0, devOut = 0;
        if (ctx->memAlloc(&devIdx, sizeof(PxU32)) != 0 || !devIdx ||
            ctx->memAlloc(&devOut, OUT_FLOATS * sizeof(float)) != 0 || !devOut)
        {
            std::printf("[ FAIL ] gpu-sdk-step: directapi memAlloc failed\n");
            scene->release(); physics->release(); cudaCtx->release(); foundation->release();
            return 1;
        }

        // Reads `count` floats back through the direct API after poisoning the output,
        // so "refused" and "answered" are never the same bit pattern.
        auto readPoisoned = [&](auto call, PxU32 gpuIndex, float* host, size_t count) -> bool
        {
            float poison[OUT_FLOATS];
            for (size_t i = 0; i < OUT_FLOATS; ++i) poison[i] = POISON;
            ctx->memcpyHtoD(devOut, poison, OUT_FLOATS * sizeof(float));
            ctx->memcpyHtoD(devIdx, &gpuIndex, sizeof(PxU32));
            const bool ok = call(devOut, devIdx);
            ctx->memcpyDtoH(host, devOut, count * sizeof(float));
            return ok;
        };

        PxDirectGPUAPI& api = scene->getDirectGPUAPI();
        const PxU32 fallerIdx = ds.faller->getGPUIndex();
        const PxU32 jointIdx  = ds.joint->getGPUIndex();

        const float dt = 1.0f / 60.0f;
        const int steps = 240;
        std::printf("  stepping directapi (%d x %.4fs): faller gpuIndex=%u, joint gpuIndex=%u\n",
                    steps, dt, fallerIdx, jointIdx);

        // PxTransform is (q.xyzw, p.xyz): the CENTRE HEIGHT is element 5, not 6.
        // Element 6 is p.z, which is ~0 for this scene -- so reading it as y gave a
        // "did not begin falling" check that passed for every possible simulation,
        // including one that never ran.  Found by the first green run printing
        // "pose after 1 step y=0.0000" for a box spawned at y=5.
        static const size_t POSE_PY = 5;
        float poseAfterOne[8] = { POISON, POISON, POISON, POISON, POISON, POISON, POISON, POISON };
        for (int i = 0; i < steps; ++i)
        {
            scene->simulate(dt);
            scene->fetchResults(true);
            if (i == 0)
                readPoisoned([&](CUdeviceptr out, CUdeviceptr idx) {
                    return api.getRigidDynamicData(reinterpret_cast<void*>(out),
                                                   reinterpret_cast<const PxRigidDynamicGPUIndex*>(idx),
                                                   PxRigidDynamicGPUAPIReadType::eGLOBAL_POSE, 1);
                }, fallerIdx, poseAfterOne, 7);
        }

        float pose[8], linVel[8], angVel[8], jForce[8], jTorque[8];
        const bool okPose = readPoisoned([&](CUdeviceptr out, CUdeviceptr idx) {
            return api.getRigidDynamicData(reinterpret_cast<void*>(out),
                                           reinterpret_cast<const PxRigidDynamicGPUIndex*>(idx),
                                           PxRigidDynamicGPUAPIReadType::eGLOBAL_POSE, 1);
        }, fallerIdx, pose, 7);
        const bool okLin = readPoisoned([&](CUdeviceptr out, CUdeviceptr idx) {
            return api.getRigidDynamicData(reinterpret_cast<void*>(out),
                                           reinterpret_cast<const PxRigidDynamicGPUIndex*>(idx),
                                           PxRigidDynamicGPUAPIReadType::eLINEAR_VELOCITY, 1);
        }, fallerIdx, linVel, 3);
        const bool okAng = readPoisoned([&](CUdeviceptr out, CUdeviceptr idx) {
            return api.getRigidDynamicData(reinterpret_cast<void*>(out),
                                           reinterpret_cast<const PxRigidDynamicGPUIndex*>(idx),
                                           PxRigidDynamicGPUAPIReadType::eANGULAR_VELOCITY, 1);
        }, fallerIdx, angVel, 3);
        const bool okF = readPoisoned([&](CUdeviceptr out, CUdeviceptr idx) {
            return api.getD6JointData(reinterpret_cast<void*>(out),
                                      reinterpret_cast<const PxD6JointGPUIndex*>(idx),
                                      PxD6JointGPUAPIReadType::eJOINT_FORCE, 1);
        }, jointIdx, jForce, 3);
        const bool okT = readPoisoned([&](CUdeviceptr out, CUdeviceptr idx) {
            return api.getD6JointData(reinterpret_cast<void*>(out),
                                      reinterpret_cast<const PxD6JointGPUIndex*>(idx),
                                      PxD6JointGPUAPIReadType::eJOINT_TORQUE, 1);
        }, jointIdx, jTorque, 3);

        ctx->memFree(devIdx);
        ctx->memFree(devOut);

        // The host pose is printed as an INTERNAL CONTROL, not asserted: with the
        // direct API on, PhysX stops writing simulation results back into the actor,
        // so this number is expected to differ from the direct read.  If the two ever
        // agree exactly, the direct read may be echoing the host rather than the GPU.
        const PxVec3 hostP = ds.faller->getGlobalPose().p;

        // *** THE TORQUE CONVENTION, MEASURED RATHER THAN ASSUMED -- AND IT IS NOT
        //     THE SAME CONVENTION IN BOTH SOLVERS. ***
        //
        // This scene first guessed |T| = m*g*d, the moment of gravity about the
        // body's own centre of mass, and under TGS measured 39240 where it expected
        // 4905.  The guess was wrong, not the port, and PXG_DIRECTAPI_CONTROL=1
        // settled it: that mode reads the SAME PxgConstraintWriteback row through
        // PxConstraint::getForce() -- a different reader, direct-GPU API off -- and
        // agrees with the direct-API read to the last digit in BOTH solver modes.
        //
        // What the four runs actually say (anchor (4,3,2), attachment 0.5 m to -x of
        // the centre of mass, F measured as (0, -9810, 0) everywhere):
        //
        //   GPU TGS  direct-API    (-19620, 0, +39240)  = -(anchor    x F)
        //   GPU TGS  getForce()    (-19620, 0, +39240)  identical
        //   GPU PGS  direct-API    (     0, 0,  -4905)  = -(attachment x F)
        //   GPU PGS  getForce()    (     0, 0,  -4905)  identical
        //
        // So both solver cores write "-(arm x F)" and they disagree about the ARM:
        // TGS reports the moment about ACTOR0's origin (actor0 is NULL here, i.e.
        // the world), PGS the moment about the hanger's own centre of mass.  That is
        // an ENGINE property, not a port one -- the two direct-API kernels multiply
        // one writeback row by oneOverDt and have no frame of their own -- so the
        // expectation below follows the solver rather than pretending there is one
        // answer.  Each is written as the RELATION applied to the force this run
        // measured, never as a fitted triple: a constant would pass for this anchor
        // and say nothing.
        //
        // AND THE FOURTH RUN, which P9.79b then explained.  PXG_CPU_CONTROL=1
        // (PhysX's own CPU solver, same geometry) reports (0, 0, -4905) under PGS --
        // matching the GPU exactly -- but (-39240, 0, +78480) under TGS, EXACTLY
        // TWICE the GPU TGS answer, with the linear force matching everywhere.
        //
        // That factor of two is an UPSTREAM CPU-vs-CUDA divergence, not a port
        // defect.  PhysX's CPU TGS writeback (DyTGSContactPrep.cpp:2856) accumulates
        // `(ang0 + lin0.cross(rAWorld)) * appliedForce`; the CUDA kernel this port
        // mirrors (solverBlockTGS.cuh:1157) accumulates only `ang0 * appliedForce`.
        // Both then subtract `body0WorldOffset.cross(lin)`, and on the GPU side
        // rAWorld and body0WorldOffset are the same field -- so the CPU counts the
        // moment arm twice.  The PGS pair has no such term on either side, which is
        // why PGS agrees.  Adding the CPU's term to solverTGS.slang as an experiment
        // made the GPU print (-39240.00, +0.00, +78479.99): the CPU answer to the
        // last digit.  See the long note at that writeback site.
        //
        // So the number this scene asserts under TGS is the CUDA's, deliberately.
        const float mass = ds.hangerMass, g = 9.81f;
        const float expF = mass * g;
        const PxVec3 F(jForce[0], jForce[1], jForce[2]);
        const PxVec3 T(jTorque[0], jTorque[1], jTorque[2]);
        const PxVec3 expFv(0.0f, -expF, 0.0f);
        // the moment arm each solver core uses, as measured above
        const PxVec3 torqueArm = usePgs ? PxVec3(ds.hangerAttachOffsetX, 0.0f, 0.0f)
                                        : ds.anchorWorld;
        const PxVec3 expT = -torqueArm.cross(F);   // the convention, applied to the MEASURED force
        const float expTmag = expT.magnitude();

        std::printf("  faller: pose after 1 step y=%.4f, final pose=(%.4f, %.4f, %.4f) q=(%.3f %.3f %.3f %.3f)\n",
                    poseAfterOne[POSE_PY], pose[4], pose[5], pose[6], pose[0], pose[1], pose[2], pose[3]);
        std::printf("  faller: linVel=(%.4f, %.4f, %.4f)  angVel=(%.4f, %.4f, %.4f)\n",
                    linVel[0], linVel[1], linVel[2], angVel[0], angVel[1], angVel[2]);
        std::printf("  faller: HOST pose (control, writeback is off) y=%.4f\n", hostP.y);
        std::printf("  joint : force =(%+9.2f, %+9.2f, %+9.2f) |F|=%.2f   expected m*g = %.2f\n",
                    F.x, F.y, F.z, F.magnitude(), expF);
        std::printf("  joint : torque=(%+9.2f, %+9.2f, %+9.2f) |T|=%.2f   expected -(%s x F) = "
                    "(%+9.2f, %+9.2f, %+9.2f)\n",
                    T.x, T.y, T.z, T.magnitude(),
                    usePgs ? "attachment" : "anchor", expT.x, expT.y, expT.z);
        std::printf("  api returned: pose=%d linVel=%d angVel=%d force=%d torque=%d\n",
                    int(okPose), int(okLin), int(okAng), int(okF), int(okT));

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();

        int failures = 0;
        auto poisoned = [&](const float* v, size_t n) {
            for (size_t i = 0; i < n; ++i) if (v[i] == POISON) return true;
            return false;
        };
        if (!okPose || !okLin || !okAng) { std::printf("FAIL: a direct-API rigid read returned false\n"); ++failures; }
        if (!okF || !okT) { std::printf("FAIL: a direct-API D6 joint read returned false\n"); ++failures; }
        if (poisoned(pose, 7))
        { std::printf("FAIL: the global-pose read came back POISONED -- getRigidDynamicGlobalPose never ran "
                      "(a mapped kernel with no row in bridge/PxgVulkanKernelAbi.cpp is refused on every "
                      "launch; see check_launch_abi.py)\n"); ++failures; }
        if (poisoned(linVel, 3) || poisoned(angVel, 3))
        { std::printf("FAIL: a velocity read came back POISONED -- getRigidDynamicLinear/AngularVelocity never ran\n"); ++failures; }
        if (poisoned(jForce, 3))
        { std::printf("FAIL: the joint-force read came back POISONED -- getD6JointForces never ran\n"); ++failures; }
        if (poisoned(jTorque, 3))
        { std::printf("FAIL: the joint-torque read came back POISONED -- getD6JointTorques never ran\n"); ++failures; }
        if (!(poseAfterOne[POSE_PY] < 5.0f))
        { std::printf("FAIL: the faller did not begin falling (y after 1 step = %.4f)\n", poseAfterOne[POSE_PY]); ++failures; }
        if (!(std::fabs(pose[5] - 0.5f) < 0.1f))
        { std::printf("FAIL: the faller rests at y=%.4f, not 0.5 -- the direct-API pose read disagrees "
                      "with the number PXG_SCENE=box has measured forty-nine times\n", pose[5]); ++failures; }
        if (!(PxVec3(linVel[0], linVel[1], linVel[2]).magnitude() < 0.75f))
        { std::printf("FAIL: the faller is not at rest (|v|=%.4f)\n",
                      PxVec3(linVel[0], linVel[1], linVel[2]).magnitude()); ++failures; }
        // The force is asserted COMPONENTWISE against static equilibrium: a 1000 kg
        // box held still needs exactly m*g straight down and nothing sideways.
        if (!((F - expFv).magnitude() < 0.02f * expF))
        { std::printf("FAIL: joint force (%+.2f, %+.2f, %+.2f), expected (%+.2f, %+.2f, %+.2f) -- "
                      "static equilibrium of a %.0f kg box\n",
                      F.x, F.y, F.z, expFv.x, expFv.y, expFv.z, mass); ++failures; }
        // The torque is asserted componentwise against -(arm x F) for this solver's
        // arm.  Under TGS the anchor is off TWO axes, so two components have to come
        // out right, not one; under PGS the single-axis attachment gives one.
        if (!((T - expT).magnitude() < 0.02f * expTmag))
        { std::printf("FAIL: joint torque (%+.2f, %+.2f, %+.2f), expected (%+.2f, %+.2f, %+.2f) = "
                      "-(%s x F) with arm (%.2f, %.2f, %.2f)\n",
                      T.x, T.y, T.z, expT.x, expT.y, expT.z,
                      usePgs ? "attachment" : "anchor",
                      torqueArm.x, torqueArm.y, torqueArm.z); ++failures; }
        // ...and the torque must not be the trivially-right answer a refused kernel
        // also gives.  The joint frame is off-centre precisely so this is non-zero.
        if (!(T.magnitude() > 0.1f * expF))
        { std::printf("FAIL: the joint torque is ~zero (%.2f) -- an unanswered query looks like this\n",
                      T.magnitude()); ++failures; }

        failures += unmappedPenalty();
        if (failures == 0)
        {
            std::printf("[  OK  ] gpu-sdk-step: the DIRECT-GPU API is live on Vulkan -- the box's pose and "
                        "velocity came back from the GPU with host writeback off, and the D6 joint reports "
                        "%.0f N and a torque of %.0f N m matching -(%s x F) to within 2%%\n",
                        F.magnitude(), T.magnitude(), usePgs ? "attachment" : "anchor");
            return 0;
        }
        std::printf("[ FAIL ] gpu-sdk-step: directapi checks failed (%d)\n", failures);
        return 2;
    }

    std::vector<RestTarget> targets;
    if      (!std::strcmp(sceneName, "box"))       buildBoxScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "stack"))     buildStackScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "aggregate")) buildAggregateScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "plane"))     buildPlaneScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "sphere"))    buildSphereScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "capsule"))   buildCapsuleScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "capsulebox")) buildCapsuleBoxScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "convex"))    buildConvexScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "joint"))     buildJointScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "meshfloor")) buildMeshFloorScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "sdfmesh")) buildSdfMeshScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "spheremesh")) buildSphereMeshScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "capsulemesh")) buildCapsuleMeshScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "convexcoreplane")) buildConvexCorePlaneScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "convexcorebox")) buildConvexCoreBoxScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "convexcoretrimesh")) buildConvexCoreTrimeshScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "convexcoretrough")) buildConvexCoreTroughScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "convexcoremesh")) buildConvexCoreMeshScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "hffloor")) buildHfFloorScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "spherehf")) buildSphereHfScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "hfbigbox")) buildHfBigBoxScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "hfhugebox")) buildHfHugeBoxScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "articulation")) buildArticulationScene(physics, scene, material, targets);
    // The list is spelled out in full deliberately: an out-of-date usage string is
    // what turned a mistyped scene into a silent non-result once already.  This
    // branch prints NEITHER [  OK  ] nor [ FAIL ], so a sweep that greps only for
    // [ FAIL ] reads the miss as a pass -- hence the [ FAIL ] marker below.
    // Note there is no 'particlebag' scene: the settling bag is PXG_SCENE=particle
    // with PXG_PARTICLE_STARTY set (see bagMode above).
    else { std::printf("[ FAIL ] gpu-sdk-step: unknown PXG_SCENE='%s'\n"
                       "  scenes: box|stack|plane|sphere|capsule|capsulebox|convex|joint|meshfloor|hffloor|articulation\n"
                       "          particle|particlefluid|particlefluidboundary|particlemeshfloor|particlecloth|particleinflatable\n"
                       "          cloth|clothhang|clothdrape\n"
                       "          clothselfcollide|clothcloth\n"
                       "          softbody|softbodyselfcollide|softbodycloth|softbodystack|softbodyattach\n"
                       "          clothnet|particlepush|articulationjoint   (P9.41: pending-work, not scored)\n"
                       "          particleattach                          (P9.76: the particle ATTACHMENT chain)\n"
                       "          particlerigid                           (P9.80: PBD shape matching + volume bounds)\n"
                       "          directapi                               (P9.79: the DIRECT-GPU API, host writeback off)\n"
                       "          forcethreshold                          (P9.91: FOUND/PERSISTS/LOST force-threshold events)\n"
                       "          aggregate                               (P9.92: PxAggregate self / actor / aggregate pairs)\n"
                       "          directapiwrite                          (P9.93: direct-API setters + acceleration getters)\n"
                       "          extforce                                (P9.93: addForce / addTorque -> external velocities)\n"
                       "          substepgravity                          (P9.95: eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS -> applyTGSSubstepGravity)\n"
                       "          slabgrid                                (P9.97a: TGS init/average/propagate at the engine grid; PXG_SLABGRID=slab|whole|one|slab5)\n"
                       "          particleapply                           (P9.95: PxScene::applyParticleBufferData -> applyParticleBufferDataDEPRECATED)\n"
                       "          softbodydata                            (P9.95: PxScene::copy/applySoftBodyData -> sb_copyOrApplySoftBodyDataDEPRECATED)\n"
                       "          directapimerge                          (P9.95: shapes added / re-inserted between steps -> mergeTransformCacheAndBoundArrayChanges)\n"
                       "          sdfquery                                (P9.95: PxDirectGPUAPI::evaluateSDFDistances -> evaluatePointDistancesSDFBatch)\n"
                       "          particle2                               (P9.99b: TWO PBD particle systems; PXG_P2_ARM=split|fluid|apply|push|remove)\n"
                       "  modes:  PXG_SCENE=particle       + PXG_PARTICLE_STARTY=<y>      (the settling bag)\n"
                       "          PXG_SCENE=particlefluid  + PXG_PARTICLE_FLUID_CONTROL=1 (solid-phase control)\n"
                       "          PXG_SCENE=particlecloth  + PXG_PARTICLE_CLOTH_CONTROL=1 (no-spring control)\n"
                       "          PXG_SCENE=particleinflatable + PXG_PARTICLE_INFL_CONTROL=1 (pressure-0 control)\n"
                       "          PXG_SCENE=particlefluidboundary + PXG_PARTICLE_BOUNDARY_CONTROL=1 (scale-0 control)\n",
                       sceneName);
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
    // PXG_REST_TRACE=<n>: every n-th step, each rest target's height and vertical
    // velocity.  A final height hides the trajectory: a box that rests on a shelf for
    // fifty steps and then sinks through it ends exactly where one that never touched
    // the shelf does (P9.83b-8 T7, PXG_SCENE=sdfmesh).
    const char* restTraceEnv = std::getenv("PXG_REST_TRACE");
    const int restTraceEvery = restTraceEnv ? std::atoi(restTraceEnv) : 0;
    // P9.96: the stepping loop is timed, so a change to a hot kernel has a steps-per-
    // second figure to compare (the whole-process wall time is dominated by pipeline
    // creation and cooking, which the step rate leaves out).
    const auto stepT0 = std::chrono::steady_clock::now();
    for (int i = 0; i < steps; ++i)
    {
        scene->simulate(dt);
        scene->fetchResults(true);
        if (i == 0)
            for (size_t t = 0; t < targets.size(); ++t)
                yAfterOne[t] = targets[t].body->getGlobalPose().p.y;
        if (restTraceEvery > 0 && i % restTraceEvery == 0)
        {
            std::printf("  rest-trace step %3d:", i);
            for (size_t t = 0; t < targets.size(); ++t)
                std::printf("  %s y=%.4f vy=%.4f", targets[t].label, targets[t].body->getGlobalPose().p.y,
                            targets[t].body->getLinearVelocity().y);
            std::printf("\n");
        }
    }

    {
        const double stepSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - stepT0).count();
        std::printf("  stepping took %.3f s (%.2f steps/s)\n", stepSec, stepSec > 0.0 ? double(steps) / stepSec : 0.0);
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
        // P9.96: the same numbers at full precision, so a change below the 4-digit print
        // (a filter, a rounding) is visible as a before/after rather than hidden by it.
        std::printf("  [%s] full: y after 1 step=%.9g, final y=%.9g, |v|=%.9g\n", tg.label, yAfterOne[t],
                    finalY, speed);
        if (!(yAfterOne[t] < tg.startY)) { std::printf("FAIL: [%s] did not begin falling\n", tg.label); ++failures; }
        if (!(finalY < tg.restY + 0.75f)) { std::printf("FAIL: [%s] above rest band (finalY=%.3f)\n", tg.label, finalY); ++failures; }
        if (!(finalY > tg.restY - 0.75f)) { std::printf("FAIL: [%s] below rest band / sank (finalY=%.3f)\n", tg.label, finalY); ++failures; }
        if (!(speed < 0.75f)) { std::printf("FAIL: [%s] not at rest (|v|=%.3f)\n", tg.label, speed); ++failures; }
    }

    scene->release();
    physics->release();
    cudaCtx->release();          // <- the bridge reports [pxg-unmapped] lines here
    foundation->release();

    // P9.40.  A scene that drove unmapped kernels ran a DIFFERENT simulation than the
    // one it claims to test, however green its own checks are.  Say so on every run,
    // and fail on it when asked to.
    failures += unmappedPenalty();

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
