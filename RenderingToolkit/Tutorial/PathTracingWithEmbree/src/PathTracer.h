#pragma once
#ifndef FILE_PATHTRACERSEEN
#define FILE_PATHTRACERSEEN

#include <random>

#include "definitions.h"
#include "SceneGraph.h"

struct PathTracer {
public:
    PathTracer(unsigned int max_path_length);

    ~PathTracer();

     /* task that renders a single path pixel */
    Vec3fa PathTracer::render_path(float x, float y, RandomEngine& reng,
        std::uniform_real_distribution<float>& distrib, std::shared_ptr<SceneGraph> sg
    );


private:
 
    unsigned int m_max_path_length;
   
    /* "Time" set to 0.0f for all rays as there is no motion blur, nor frame interpolation, nor animation */
    const float m_time = 0.0f;
};

/* Leave this in for later */
/* struct GuidedPathTracer {
    //
    Device guiding_device;
    Field guding_field;
    SampleDataStorage guiding sample_data_storage;
}
*/

PathTracer::PathTracer(unsigned int max_path_length) : m_max_path_length(max_path_length) {

}



/* task that renders a single screen pixel */
Vec3fa PathTracer::render_path(float x, float y, RandomEngine& reng,
                           std::uniform_real_distribution<float>& distrib, std::shared_ptr<SceneGraph> sg
                           ) {

    Vec3fa dir = sg->get_direction_from_pixel(x, y);
    //normalize(x * m_camera.l.vx + y * m_camera.l.vy + m_camera.l.vz);
    Vec3fa org = sg->get_camera_origin();
    //Vec3fa(m_camera.p.x, m_camera.p.y, m_camera.p.z);

  /* initialize ray */
  RTCRayHit rayhit;
  init_RayHit(rayhit, org, dir, 0.0f, std::numeric_limits<float>::infinity(),
             m_time);

  Vec3fa L = Vec3fa(0.0f);
  Vec3fa Lw = Vec3fa(1.0f);

  Medium medium;
  medium.eta = 1.f;

  DifferentialGeometry dg;

  sg->set_intersect_context_coherent();
  /* iterative path tracer loop */
  for (int i = 0; i < m_max_path_length; i++) {
    /* terminate if contribution too low */
    if (max(Lw.x, max(Lw.y, Lw.z)) < 0.01f) break;

    if (!sg->intersect_path_and_scene(org, dir, rayhit, dg)) break;

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
        g_geomIDs[rayhit.hit.geomID].materialTable[rayhit.hit.primID];
    albedo = g_geomIDs[rayhit.hit.geomID].primColorTable[rayhit.hit.primID];
    /* An albedo as well as a material type is used */

    /* weight scaling based on material sample */
    Vec3fa c = Vec3fa(1.0f);

    Vec3fa wi1;
    Vec2f randomMatSample(distrib(reng), distrib(reng));
    const Vec3fa wo = -dir;

    /* Search for each light in the scene from our hit point. Aggregate the
 * radiance if hit point is not occluded */
    sg->set_intersect_context_incoherent();

    if (Material_direct_illumination(materialType)) {
        /* Cast shadow ray(s) from the hit point */
        sg->cast_shadow_rays(dg, albedo, materialType, Lw, wo, medium, m_time, L, reng,
            distrib);
    }
    //c = c * Material_sample(albedo, materialType, Lw, wo, dg, wi1, medium,
                             //randomMatSample);
    wi1 = Material_sample(materialType, Lw, wo, dg, medium,
                                 randomMatSample);
    c = c * Material_eval(albedo, materialType, Lw, wo, dg, wi1, medium,
                                 randomMatSample);


    //m_sg->cast_shadow_ray(dg, albedo, materialType, wo, m_time, L, Lw, reng, distrib);


    float nextPDF = Material_pdf(materialType, Lw, wo, dg, medium, randomMatSample);
    if (nextPDF <= 1E-4f) break;
    Lw = Lw * c / nextPDF;

    /* setup secondary ray */
    float sign = dot(wi1, dg.Ng) < 0.0f ? -1.0f : 1.0f;
    dg.P = dg.P + sign * dg.eps * dg.Ng;
    org = dg.P;
    dir = normalize(wi1);
    init_RayHit(rayhit, org, dir, dg.eps, std::numeric_limits<float>::infinity(),
               m_time);
  }

  return L;
}



PathTracer::~PathTracer() {

}


#endif /* FILE_PATHTRACERSEEN */