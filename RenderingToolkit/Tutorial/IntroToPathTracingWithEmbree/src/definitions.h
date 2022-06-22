#pragma once
#ifndef FILE_DEFINITIONS_SEEN
#define FILE_DEFINITIONS_SEEN

#include <rkcommon/math/vec.h>
/* Added for pathtracer */
#include <rkcommon/math/LinearSpace.h>
#include <rkcommon/math/AffineSpace.h>

using Vec3fa = rkcommon::math::vec_t<float, 3, 1>;
using rkcommon::math::LinearSpace3fa;
using rkcommon::math::AffineSpace3fa;

#ifdef _WIN32
#define alignedMalloc(a, b) _aligned_malloc(a, b)
#define alignedFree(a) _aligned_free(a)
#else
#include <mm_malloc.h>
#define alignedMalloc(a, b) _mm_malloc(a, b)
#define alignedFree(a) _mm_free(a)
#endif

#define TILE_SIZE_X 8
#define TILE_SIZE_Y 8

/* from tutorial_device.h */
/* vertex, quad, and triangle layout */
struct Vertex {
    float x, y, z, r;
};

struct Quad {
    int v0, v1, v2, v3;
};

struct Triangle {
    int v0, v1, v2;
};

/* Added for pathtracer */
enum class MaterialType {
    MATERIAL_MATTE,
    MATERIAL_MIRROR,
    MATERIAL_THIN_DIELECTRIC,
};

using Vec3fa = rkcommon::math::vec_t<float, 3, 1>;
using rkcommon::math::cross;
using rkcommon::math::deg2rad;
using rkcommon::math::normalize;
using std::max;
using std::min;

/* Additions for pathtracer */
using Vec3ff = rkcommon::math::vec4f;
using rkcommon::math::rcp;
using Vec2f = rkcommon::math::vec2f;
using rkcommon::math::dot;
using rkcommon::math::clamp;
using rkcommon::math::rsqrt;

/* Added for pathtracer */
struct DifferentialGeometry
{
    unsigned int instIDs[RTC_MAX_INSTANCE_LEVEL_COUNT];
    unsigned int geomID;
    unsigned int primID;
    float u, v;
    Vec3fa P;
    Vec3fa Ng;
    Vec3fa Ns;
    Vec3fa Tx; //direction along hair
    Vec3fa Ty;
    float eps;
};

/* Added for geometry selection in pathtracer */
enum class SceneSelector {
    SHOW_CUBE_AND_PLANE,
    SHOW_CORNELL_BOX,
};

/* Added for pathtracer */
struct Sample3f
{
    Vec3fa v;
    float pdf;
};

/* Added for pathtracer */
struct InfiniteDirectionalLight {
    Vec3fa dir;
    Vec3fa intensity;
};

/* Added for pathtracer */
enum class LightType {
    INFINITE_DIRECTIONAL_LIGHT,
    POINT_LIGHT
};

struct Light {
    enum LightType type;
    Vec3fa dir;
    Vec3fa intensity;
    Vec3fa pos;
};

AffineSpace3fa positionCamera(Vec3fa from, Vec3fa to, Vec3fa up, float fov,
    size_t width, size_t height) {
    /* There are many ways to set up a camera projection. This one is consolidated
     * from the camera code in the Embree/tutorial/common/tutorial/camera.h object
     */
    AffineSpace3fa camMatrix;
    Vec3fa Z =
        normalize(Vec3fa(to - from));
    Vec3fa U = normalize(
        cross(up,
            Z));
    Vec3fa V = normalize(
        cross(Z,
            U));
    camMatrix.l.vx = U;
    camMatrix.l.vy = V;
    camMatrix.l.vz = Z;
    camMatrix.p = from;

    /* negate for a right handed camera*/
    camMatrix.l.vx = -camMatrix.l.vx;

    const float fovScale = 1.0f / tanf(deg2rad(0.5f * fov));

    camMatrix.l.vz = -0.5f * width * camMatrix.l.vx +
        0.5f * height * camMatrix.l.vy +
        0.5f * height * fovScale * camMatrix.l.vz;
    camMatrix.l.vy = -camMatrix.l.vy;

    return camMatrix;
}

#endif /* !FILE_DEFINITIONS_SEEN */