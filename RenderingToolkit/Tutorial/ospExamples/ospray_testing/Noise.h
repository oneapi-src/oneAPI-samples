// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ospray_testing.h"

using namespace rkcommon::math;

namespace ospray {
namespace testing {

/*
 * Our basic noise primitive.
 * Heavily based on Perlin's Java reference implementation of
 * the improved perlin noise paper from Siggraph 2002 from here
 * https://mrl.nyu.edu/~perlin/noise/
 */
class PerlinNoise
{
  struct PerlinNoiseData
  {
    PerlinNoiseData();

    inline int operator[](size_t idx) const
    {
      return p[idx];
    }
    int p[512];
  };

  static PerlinNoiseData p;

  static inline float smooth(float t)
  {
    return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
  }

  static inline float grad(int hash, float x, float y, float z)
  {
    const int h = hash & 15;
    const float u = h < 8 ? x : y;
    const float v = h < 4 ? y : h == 12 || h == 14 ? x : z;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
  }

 public:
  static float noise(vec3f q, float frequency = 8.f)
  {
    float x = q.x * frequency;
    float y = q.y * frequency;
    float z = q.z * frequency;
    const int X = (int)floor(x) & 255;
    const int Y = (int)floor(y) & 255;
    const int Z = (int)floor(z) & 255;
    x -= floor(x);
    y -= floor(y);
    z -= floor(z);
    const float u = smooth(x);
    const float v = smooth(y);
    const float w = smooth(z);
    const int A = p[X] + Y;
    const int B = p[X + 1] + Y;
    const int AA = p[A] + Z;
    const int BA = p[B] + Z;
    const int BB = p[B + 1] + Z;
    const int AB = p[A + 1] + Z;

    return lerp(w,
        lerp(v,
            lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
            lerp(u, grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z))),
        lerp(v,
            lerp(u,
                grad(p[AA + 1], x, y, z - 1),
                grad(p[BA + 1], x - 1, y, z - 1)),
            lerp(u,
                grad(p[AB + 1], x, y - 1, z - 1),
                grad(p[BB + 1], x - 1, y - 1, z - 1))));
  }
};

// -----------------------------------------------------------------------------
// Hypertexture helpers.
// -----------------------------------------------------------------------------

/*
 * Turbulence - noise at various scales.
 */
inline float turbulence(const vec3f &p, float base_freqency, int octaves)
{
  float value = 0.f;
  float scale = 1.f;
  for (int o = 0; o < octaves; ++o) {
    value += PerlinNoise::noise(scale * p, base_freqency) / scale;
    scale *= 2.f;
  }
  return value;
};

/*
 * A torus indicator function.
 */
inline bool torus(vec3f X, float R, float r)
{
  const float tmp = sqrtf(X.x * X.x + X.z * X.z) - R;
  return tmp * tmp + X.y * X.y - r * r < 0.f;
}

/*
 * Turbulent torus indicator function.
 * We use the turbulence() function to distort space.
 */
inline bool turbulentTorus(vec3f p, float R, float r)
{
  const vec3f X = 2.f * p - vec3f(1.f);
  return torus((1.4f + 0.4 * turbulence(p, 12.f, 12)) * X, R, r);
}

/*
 * Turbulent sphere indicator function.
 */
inline bool turbulentSphere(vec3f p, float r)
{
  vec3f X = 2.f * p - vec3f(1.f);
  return length((1.4f + 0.4 * turbulence(p, 12.f, 12)) * X) < r;
}

} // namespace testing
} // namespace ospray
