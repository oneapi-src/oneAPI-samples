#pragma once
#ifndef FILE_PATHTRACERSEEN
#define FILE_PATHTRACERSEEN

#include <random>

#include "Lights.h"
#include "RandomSampler.h"
#include "SceneGraph.h"
#include "definitions.h"

struct PathTracer {
 public:
  PathTracer(unsigned int max_path_length);

  PathTracer(unsigned int max_path_length, unsigned int width,
             unsigned int height, unsigned int numLights);

  ~PathTracer();

  /* task that renders a single path pixel */
  Vec3fa render_path(float x, float y, RandomSampler& randomSampler,
                     std::shared_ptr<SceneGraph> sg, unsigned int pxID);

 private:
  unsigned int m_max_path_length;

  /* "Time" set to 0.0f for all rays as there is no motion blur, nor frame
   * interpolation, nor animation */
  const float m_time = 0.0f;

  unsigned int m_numLights;
};

PathTracer::PathTracer(unsigned int max_path_length)
    : m_max_path_length(max_path_length) {}

PathTracer::PathTracer(unsigned int max_path_length, unsigned int width,
                       unsigned int height, unsigned int numLights)
    : m_max_path_length(max_path_length), m_numLights(numLights) {}

/* task that renders a single screen pixel */
Vec3fa PathTracer::render_path(float x, float y, RandomSampler& randomSampler,
                               std::shared_ptr<SceneGraph> sg,
                               unsigned int pxID) {
  Vec3fa dir = sg->get_direction_from_pixel(x, y);
  Vec3fa org = sg->get_camera_origin();

  /* initialize ray */
  RTCRayHit rayhit;
  init_RayHit(rayhit, org, dir, 0.0f, std::numeric_limits<float>::infinity(),
              m_time);

  Vec3fa L = Vec3fa(0.0f);
  Vec3fa Lw = Vec3fa(1.0f);

  Medium medium, nextMedium;
  medium.eta = nextMedium.eta = 1.f;

  DifferentialGeometry dg;

  bool bCoherent = true;
  /* iterative path tracer loop */
  for (int i = 0; i < m_max_path_length; i++) {
    /* terminate if contribution too low */
    if (max(Lw.x, max(Lw.y, Lw.z)) < 0.01f) break;

    /* New for Embree 4: Use coherent ray designation on pramary ray cast with
     * RTCIntersectArguments::flags by passing bCoherent*/
    if (!sg->intersect_path_and_scene(org, dir, rayhit, dg, bCoherent)) break;

    const Vec3fa wo = -dir;
    /* Next is material discovery.
     * Note: the full pathtracer program includes lookup of materials from a
     * scenegraph object. This could include texture lookup transformations that
     * are based on vertex-to-texture assignments. This could include a required
     * tranformation of normals for geometries that have been instanced. Below
     * is a simple option for materials.
     */

    /* default albedo is a pink color for debug */
    Vec3fa albedo = Vec3fa(0.9f, 0.7f, 0.7f);
    MaterialType materialType = MaterialType::MATERIAL_MATTE;
    materialType =
        sg->m_mapGeomToPrim[rayhit.hit.geomID].materialTable[rayhit.hit.primID];
    if (materialType == MaterialType::MATERIAL_EMITTER) {
      std::shared_ptr<Light> light =
          sg->get_light_from_geomID(rayhit.hit.geomID);
      Light_EvalRes le = light->eval(org, dir);
      L = L + Lw * le.value;
      // If we encounter a light emitter we will terminate adding more light
      // from the path.
      // Alternatively, we could move the intersected ray through the light
      // source and continue
      break;
    } else {
      albedo = sg->m_mapGeomToPrim[rayhit.hit.geomID]
                   .primColorTable[rayhit.hit.primID];
      /* An albedo as well as a material type is used */
    }

    /* weight scaling based on material sample. used for attenuation amongst
     * path segments */
    Vec3fa c = Vec3fa(1.0f);

    Vec3fa wi1;
    Vec2f randomMatSample(randomSampler.get_float(), randomSampler.get_float());

    /* Occlusion and Intersect test arguments have changed with Embree 4.
     * Occlusion query flags are set before the shadow ray lookup In this
     * example program, Intersect query flags will be set to
     * RTC_RAY_QUERY_INCOHERENT for all non-primary rays.     */
    bCoherent = false;

    /* For the occlusion test, search for each light in the scene from the hit
     * point. Aggregate the radiance if hit point is not occluded */

    if (Material_direct_illumination(materialType)) {
      /* Cast shadow ray(s) from the hit point */
      sg->cast_shadow_rays(dg, albedo, materialType, Lw, wo, medium, m_time, L,
                           randomSampler, bCoherent);
    }

    /* Sample, Eval, and PDF computation are split and internally perform some
     * redundant calculation */
    wi1 = Material_sample(materialType, Lw, wo, dg, medium, nextMedium,
                          randomMatSample);
    c = c * Material_eval(albedo, materialType, Lw, wo, dg, wi1, medium);
    float nextPDF = Material_pdf(materialType, Lw, wo, dg, medium, wi1);

    if (nextPDF <= 1E-4f) break;
    Lw = Lw * c / nextPDF;

    /* setup secondary ray */
    medium = nextMedium;
    float sign = dot(wi1, dg.Ng) < 0.0f ? -1.0f : 1.0f;
    dg.P = dg.P + sign * dg.eps * dg.Ng;
    org = dg.P;
    dir = normalize(wi1);
    init_RayHit(rayhit, org, dir, dg.eps, inf, m_time);
  }

  return L;
}

PathTracer::~PathTracer() {}

#endif /* FILE_PATHTRACERSEEN */
