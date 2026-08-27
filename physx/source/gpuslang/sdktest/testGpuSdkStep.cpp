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
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <set>
#include <map>
#include <utility>

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
void buildArticulationScene(PxPhysics* physics, PxScene* scene, PxMaterial* material,
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
ParticleScene buildParticleScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                 PxCudaContextManager* cudaCtx, bool fluidMode = false,
                                 bool boundaryMode = false, bool meshFloorMode = false,
                                 bool hfFloorMode = false, bool diffuseMode = false,
                                 bool convexFloorMode = false)
{
    ParticleScene out;
    out.cudaCtx = cudaCtx;

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
        addGroundBox(physics, scene, rigidMat);
    }
    // The engine drives the rigid solver core for the particle island, so one live
    // dynamic rigid body is required (else rigidSumInternalContactAndJointBatches2
    // null-derefs).  Free-fall: box at x=0,y=5 (never meets the y~100 cloud).  Bag
    // mode (PXG_PARTICLE_STARTY set): place it FAR ASIDE at x=20 so the settling
    // pile on the floor never touches it.
    if (std::getenv("PXG_PARTICLE_STARTY") || fluidMode)
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

        phase = ps->createPhase(pbdMat,
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
    const float startY = fluidMode ? 0.6f : (startYEnv ? float(atof(startYEnv)) : 100.0f);
    if (startYEnv || fluidMode)
        ps->setSolverIterationCounts(/*pos*/ 8, /*vel*/ 1);
    std::vector<PxVec4> hostPos(N), hostVel(N, PxVec4(0.0f, 0.0f, 0.0f, 0.0f));
    std::vector<PxU32> hostPhase(N, phase);
    const float cx = 0.5f * float(GX - 1) * particleSpacing;
    const float cz = 0.5f * float(GZ - 1) * particleSpacing;
    PxU32 idx = 0;
    for (int x = 0; x < GX; ++x)
        for (int y = 0; y < GY; ++y)
            for (int z = 0; z < GZ; ++z)
            {
                hostPos[idx++] = PxVec4(float(x) * particleSpacing - cx,
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
    ps->addParticleBuffer(buffer);
    scene->addActor(*ps);

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
                           bool meshFloor = false)
{
    ClothScene out;
    out.cudaCtx = cudaCtx;

    // A minimal rigid body (box on ground) FAR BELOW the cloth.  As with the
    // particle scene, the cloth solve runs inside the rigid PGS island loop and
    // rigidSumInternalContactAndJointBatches2 null-derefs the solver descriptors
    // with zero dynamic rigids, so one live dynamic body is required to keep that
    // proven path valid.  It rests at y=0.5, well below the cloth (spawned y~5).
    if (meshFloor)
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
    // The live dynamic body is moved aside for BOTH non-box floors, for the
    // reason P9.20 measured: it is a PxBoxGeometry, so a box under the cloth is
    // served by the already-mapped box path and would carry the cloth whatever
    // the floor does.
    addDynamic(physics, scene, rigidMat, PxBoxGeometry(0.5f, 0.5f, 0.5f),
               boxInertia(PxVec3(0.5f), 1000.0f), 1.5f,
               (convexFloor || meshFloor) ? 3.0f : 0.0f);

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
    PxCudaContextManager* cudaCtx = nullptr;
};

SoftBodyScene buildSoftBodyScene(PxPhysics* physics, PxScene* scene, PxMaterial* rigidMat,
                                 PxCudaContextManager* cudaCtx, float stretchY)
{
    SoftBodyScene out;
    out.cudaCtx = cudaCtx;

    // ---- the floor ------------------------------------------------------------
    // PXG_SB_FLOOR selects it: "box" (the default), "plane", "sphere" or "none".
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
    const bool floorBox    = !floorNone && !floorPlane && !floorSphere && !floorMesh;

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

    out.hasPlane = !floorNone;      // "there is a floor to settle on", whatever its shape
    out.hasSphere = floorSphere;
    out.hasBox = floorBox;
    out.hasPlaneShape = floorPlane;

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
        !std::strcmp(sceneName, "clothmesh"))
    {
        // `clothconvex` is `clothbox` with the ground BOX replaced by a cooked
        // CONVEX slab at the same height -- see buildClothScene.  It shares every
        // check below, because the two paths run the SAME body behind two engine
        // names, so anything that differs would be the routing or the manifest.
        const bool convexFloor = !std::strcmp(sceneName, "clothconvex");
        const bool meshFloor = !std::strcmp(sceneName, "clothmesh");
        ClothScene csc = buildClothScene(physics, scene, material, cudaCtx, convexFloor,
                                         meshFloor);
        if (!csc.surface || csc.numVerts == 0)
        {
            std::printf("[ FAIL ] gpu-sdk-step: %s scene build failed\n", sceneName);
            scene->release(); physics->release(); cudaCtx->release(); foundation->release(); return 1;
        }
        std::printf("  [  OK  ] scene '%s' built: %u-vertex FEMCloth dropped onto the ground %s\n",
                    sceneName, csc.numVerts,
                    meshFloor ? "cooked triangle mesh" : (convexFloor ? "convex slab" : "box"));
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
        if (!(maxSettleStep < 1e-2f))
        { std::printf("FAIL: the cloth is still moving %.6f per step at the end of the run -- "
                      "it never came to rest\n", maxSettleStep); ++failures; }
        // 2. IT STOPPED ON THE BOX, not somewhere below it.  -73.81 was measured with
        //    cloth_boxVertexContactGenLaunch unmapped, in this same scene.
        if (!(yEnd > -2.0f))
        { std::printf("FAIL: the cloth came to rest at y = %.4f -- it fell through the box "
                      "(need > -2.0; -73.81 measured with the box vertex kernel unmapped)\n",
                      yEnd); ++failures; }
        if (!std::isfinite(yEnd))
        { std::printf("FAIL: mean y is not finite (%.4f)\n", yEnd); ++failures; }

        scene->release(); physics->release(); cudaCtx->release(); foundation->release();
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
                              : sbs.hasBox ? "box" : "mesh";
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

    // PXG_SCENE=softbodyselfcollide (P8.4): the FEM-VOLUME SELF-collision money shot.
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

    std::vector<RestTarget> targets;
    if      (!std::strcmp(sceneName, "box"))       buildBoxScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "stack"))     buildStackScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "plane"))     buildPlaneScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "sphere"))    buildSphereScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "capsule"))   buildCapsuleScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "capsulebox")) buildCapsuleBoxScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "convex"))    buildConvexScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "joint"))     buildJointScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "meshfloor")) buildMeshFloorScene(physics, scene, material, targets);
    else if (!std::strcmp(sceneName, "articulation")) buildArticulationScene(physics, scene, material, targets);
    // The list is spelled out in full deliberately: an out-of-date usage string is
    // what turned a mistyped scene into a silent non-result once already.  This
    // branch prints NEITHER [  OK  ] nor [ FAIL ], so a sweep that greps only for
    // [ FAIL ] reads the miss as a pass -- hence the [ FAIL ] marker below.
    // Note there is no 'particlebag' scene: the settling bag is PXG_SCENE=particle
    // with PXG_PARTICLE_STARTY set (see bagMode above).
    else { std::printf("[ FAIL ] gpu-sdk-step: unknown PXG_SCENE='%s'\n"
                       "  scenes: box|stack|plane|sphere|capsule|capsulebox|convex|joint|meshfloor|articulation\n"
                       "          particle|particlefluid|particlefluidboundary|particlemeshfloor|particlecloth|particleinflatable\n"
                       "          cloth|clothhang|clothdrape\n"
                       "          clothselfcollide|clothcloth\n"
                       "          softbody|softbodyselfcollide|softbodycloth|softbodystack|softbodyattach\n"
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
