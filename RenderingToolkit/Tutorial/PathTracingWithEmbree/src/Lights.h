#pragma once

#ifndef FILE_LIGHTSSEEN
#define FILE_LIGHTSSEEN
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

/* Added for pathtracer */
struct Light_SampleRes {
  Vec3fa weight;  //!< radiance that arrives at the given point divided by pdf
  Vec3fa dir;     //!< direction towards the light source
  float dist;     //!< largest valid t_far value for a shadow ray
  float pdf;      //!< probability density that this sample was taken
};

Light_SampleRes sample_light(const Light& light, DifferentialGeometry& dg,
                             const Vec2f& randomLightSample) {
  Light_SampleRes res;

  switch (light.type) {
    case LightType::INFINITE_DIRECTIONAL_LIGHT:
      res.dir = -light.dir;
      res.dist = std::numeric_limits<float>::infinity();
      res.pdf = std::numeric_limits<float>::infinity();
      res.weight = light.intensity;  // *pdf/pdf cancel
      break;
    case LightType::POINT_LIGHT:
    default:
      // extant light vector from the hit point
      const Vec3fa dir = light.pos - dg.P;
      const float dist2 = dot(dir, dir);
      const float invdist = rsqrt(dist2);

      // normalized light vector
      res.dir = dir * invdist;
      res.dist = dist2 * invdist;

      res.pdf = std::numeric_limits<float>::infinity();  // per default we
                                                         // always take this res

      // convert from power to radiance by attenuating by distance^2
      res.weight = light.intensity * (invdist * invdist);
      break;
  }

  return res;
}

#endif /* FILE_LIGHTSSEEN */