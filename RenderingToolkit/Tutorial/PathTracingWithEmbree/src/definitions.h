#pragma once
#ifndef FILE_DEFINITIONS_SEEN
#define FILE_DEFINITIONS_SEEN

#include <rkcommon/math/vec.h>
/* Added for pathtracer */
#include <rkcommon/math/LinearSpace.h>
#include <rkcommon/math/AffineSpace.h>
#include <utility>

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

/* minstd_rand is much faster for this application than
ranlux48_base/ranlux24_base. In turn, ranlux was faster than mt19937 or
mt19937_64 on Windows
*/
typedef std::minstd_rand RandomEngine;

/* originally from tutorial_device.h */
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
struct DifferentialGeometry
{
    unsigned int instIDs[RTC_MAX_INSTANCE_LEVEL_COUNT];
    unsigned int geomID;
    unsigned int primID;
    float u, v;
    Vec3fa P;
    Vec3fa Ng;
    /* This sample program does not interpolate normals for normal specific shading. Ns is set to Ng, the regular normal. */
    Vec3fa Ns;
    /* This sample does not use textures. Tx is a place holder. */
    Vec3fa Tx;
    /* This sample does not use textures. Ty is a place holder. */
    Vec3fa Ty;
    float eps;
};

/* Added for pathtracer */
struct Sample3f
{
    Vec3fa v;
    float pdf;
};

/* Added for pathtracer */
inline Vec3fa face_forward(const Vec3fa& dir, const Vec3fa& _Ng) {
    const Vec3fa Ng = _Ng;
    return dot(dir, Ng) < 0.0f ? Ng : -Ng;
}

/* Added for pathtracer: Initializes a rayhit data structure. Used for embree to perform ray to primitive intersect */
inline void init_RayHit(RTCRayHit& rayhit, const Vec3fa& org, const Vec3fa& dir,
    float tnear, float tfar, float time) {
    rayhit.ray.dir_x = dir.x;
    rayhit.ray.dir_y = dir.y;
    rayhit.ray.dir_z = dir.z;
    rayhit.ray.org_x = org.x;
    rayhit.ray.org_y = org.y;
    rayhit.ray.org_z = org.z;
    rayhit.ray.tnear = tnear;
    rayhit.ray.time = time;
    rayhit.ray.tfar = tfar;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
    rayhit.ray.mask = -1;
}

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