#pragma once
#ifndef FILE_DEFINITIONS_SEEN
#define FILE_DEFINITIONS_SEEN

/* The definitions.h file is home to 3D math utility types and 3D math utilty
 * functions. Embree repository tutorial sources often place these into an
 * optics.h or sampling.g */

#include <embree4/rtcore.h>
#include <rkcommon/math/vec.h>

#include <utility>
/* Added for pathtracer */
#include <rkcommon/math/AffineSpace.h>
#include <rkcommon/math/LinearSpace.h>

using Vec3fa = rkcommon::math::vec_t<float, 3, 1>;
using rkcommon::math::AffineSpace3fa;
using rkcommon::math::LinearSpace3fa;

#ifdef _WIN32
#define alignedMalloc(a, b) _aligned_malloc(a, b)
#define alignedFree(a) _aligned_free(a)
#else
#include <mm_malloc.h>
#define alignedMalloc(a, b) _mm_malloc(a, b)
#define alignedFree(a) _mm_free(a)
#endif

/* Here we define the tile size in use for oneTBB tasks */
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
using rkcommon::math::clamp;
using rkcommon::math::dot;
using rkcommon::math::rsqrt;

struct PosInf {
  __forceinline operator float() const {
    return std::numeric_limits<float>::infinity();
  }
};

PosInf inf;

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
struct Normal {
  float x, y, z;
};

/* Added for pathtracer */
struct DifferentialGeometry {
  unsigned int instIDs[RTC_MAX_INSTANCE_LEVEL_COUNT];
  unsigned int geomID;
  unsigned int primID;
  float u, v;
  Vec3fa P;
  Vec3fa Ng;
  /* This sample program does not interpolate normals for normal specific
   * shading. Ns is set to Ng, the regular normal. */
  Vec3fa Ns;
  /* This sample does not use textures. Tx is a place holder. */
  Vec3fa Tx;
  /* This sample does not use textures. Ty is a place holder. */
  Vec3fa Ty;
  float eps;
};

/* Added for pathtracer */
struct Sample3f {
  Vec3fa v;
  float pdf;
};

inline float cosineSampleHemispherePDF(const Vec3fa& dir) {
  return dir.z / float(M_PI);
}

/* Added for pathtracer. The frame function creates a transform from a normal.
 */
LinearSpace3fa frame(const Vec3fa& N) {
  const Vec3fa dx0(0, N.z, -N.y);
  const Vec3fa dx1(-N.z, 0, N.x);

  const Vec3fa dx = normalize((dot(dx0, dx0) > dot(dx1, dx1)) ? dx0 : dx1);
  const Vec3fa dy = normalize(cross(N, dx));

  return LinearSpace3fa(dx, dy, N);
}

inline Vec3fa cartesian(const float phi, const float sinTheta,
                        const float cosTheta) {
  const float sinPhi = sinf(phi);
  const float cosPhi = cosf(phi);
  // sincosf(phi, &sinPhi, &cosPhi);
  return Vec3fa(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
}

inline Vec3fa cartesian(const float phi, const float cosTheta) {
  return cartesian(phi, sqrt(max(0.f, 1.f - (cosTheta * cosTheta))), cosTheta);
}

/// cosine-weighted sampling of hemisphere oriented along the +z-axis
////////////////////////////////////////////////////////////////////////////////

inline Vec3fa cosineSampleHemisphere(const Vec2f s) {
  const float phi = float(2.f * M_PI) * s.x;
  const float cosTheta = sqrt(s.y);
  const float sinTheta = sqrt(1.0f - s.y);
  return cartesian(phi, sinTheta, cosTheta);
}

/*! Cosine weighted hemisphere sampling. Up direction is provided as argument.
 */
inline Vec3fa cosineSampleHemisphere(const float u, const float v,
                                     const Vec3fa& N) {
  /* Determine cartesian coordinate for new Vec3fa */
  const float phi = float(2.0f * M_PI) * u;
  const float cosTheta = sqrt(v);
  const float sinTheta = sqrt(1.0f - v);
  const float sinPhi = sinf(phi);
  const float cosPhi = cosf(phi);

  Vec3fa localDir = Vec3fa(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
  /* Gives the new Vec3fa transformed about the input Vec3fa */

  return frame(N) * localDir;
}

inline Vec3fa cosinePDFHemisphere(const float s) {
  return sqrt(s) / float(M_PI);
}

/// sampling of cone of directions oriented along the +z-axis
////////////////////////////////////////////////////////////////////////////////

inline Vec3fa uniformSampleCone(const float cosAngle, const Vec2f& s) {
  const float phi = float(2.f * M_PI) * s.x;
  const float cosTheta = 1.0f - s.y * (1.0f - cosAngle);
  return cartesian(phi, cosTheta);
}

inline float uniformSampleConePDF(const float cosAngle) {
  return rcp(float(2.f * M_PI) * (1.0f - cosAngle));
}

/// sampling of disk
////////////////////////////////////////////////////////////////////////////////

inline Vec3fa uniformSampleDisk(const float radius, const Vec2f& s) {
  const float r = sqrtf(s.x) * radius;
  const float phi = float(2.0f * float(M_PI)) * s.y;
  const float sinPhi = sinf(phi);
  const float cosPhi = cosf(phi);
  // sincosf(phi, &sinPhi, &cosPhi);
  return Vec3fa(r * cosPhi, r * sinPhi, 0.f);
}

inline float uniformSampleDiskPDF(const float radius) {
  return rcp(float(M_PI) * (radius * radius));
}

/* Added for pathtracer */
inline Vec3fa face_forward(const Vec3fa& dir, const Vec3fa& _Ng) {
  const Vec3fa Ng = _Ng;
  return dot(dir, Ng) < 0.0f ? Ng : -Ng;
}

/* Added for pathtracer: Initializes a rayhit data structure. Used for embree to
 * perform ray to primitive intersect */
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
  Vec3fa Z = normalize(Vec3fa(to - from));
  Vec3fa U = normalize(cross(up, Z));
  Vec3fa V = normalize(cross(Z, U));
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