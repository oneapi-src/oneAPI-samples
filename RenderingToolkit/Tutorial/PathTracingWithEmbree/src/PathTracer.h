#pragma once
#ifndef FILE_PATHTRACERSEEN
#define FILE_PATHTRACERSEEN

#include <random>

#include "definitions.h"
#include "SceneGraph.h"
#include "Lights.h"

struct misData {
    Light_SampleRes sam;
    float tfar;
    Vec2f randomLightSample;
};

struct PathTracer {
public:
    PathTracer(unsigned int max_path_length);

    PathTracer(unsigned int max_path_length, unsigned int width, unsigned int height, unsigned int numLights);

    ~PathTracer();

     /* task that renders a single path pixel */
    Vec3fa PathTracer::render_path(float x, float y, RandomEngine& reng,
        std::uniform_real_distribution<float>& distrib, std::shared_ptr<SceneGraph> sg, unsigned int pxID
    );


private:
 
    unsigned int m_max_path_length;
   
    /* "Time" set to 0.0f for all rays as there is no motion blur, nor frame interpolation, nor animation */
    const float m_time = 0.0f;

    /* MIS needs reglar storage at each path segment to capture light PDFs */
    std::vector< std::vector<misData> > m_misLightStorage;

    unsigned int m_numLights;

    void PathTracer::initialize_misData(misData& misOut);
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

void PathTracer::initialize_misData(misData& misOut) {

    misOut.sam.weight = 0.f;
    misOut.sam.dir = Vec3fa(0.f);
    misOut.sam.dist = inf;
    misOut.sam.pdf = 0.f;
    misOut.tfar = 0.f;
    misOut.randomLightSample = Vec2f(0.f);

}


PathTracer::PathTracer(unsigned int max_path_length, unsigned int width, unsigned int height, unsigned int numLights) : m_max_path_length(max_path_length), m_numLights(numLights) {
    //Set up for MIS pdfs for each light at each path segment of each pixel
    std::vector<misData> misLightDefaults;
    misData misDefault;
    initialize_misData(misDefault);

    for (auto i = 0; i < numLights; i++)
        misLightDefaults.push_back(misDefault);

    for (auto i = 0; i < width * height; i++)
        m_misLightStorage.push_back(misLightDefaults);

}



/* task that renders a single screen pixel */
Vec3fa PathTracer::render_path(float x, float y, RandomEngine& reng,
    std::uniform_real_distribution<float>& distrib, std::shared_ptr<SceneGraph> sg, unsigned int pxID
                           ) {

    Vec3fa dir = sg->get_direction_from_pixel(x, y);
    Vec3fa org = sg->get_camera_origin();

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

    /* MIS prep for lights */
      if (true) {
          misData misDefault;
          initialize_misData(misDefault);
          std::fill(m_misLightStorage[pxID].begin(), m_misLightStorage[pxID].end(), misDefault);
      }
    /* terminate if contribution too low */
    if (max(Lw.x, max(Lw.y, Lw.z)) < 0.01f) break;

    if (!sg->intersect_path_and_scene(org, dir, rayhit, dg)) break;

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
        g_geomIDs[rayhit.hit.geomID].materialTable[rayhit.hit.primID];
    if (materialType == MaterialType::MATERIAL_EMITTER) {
        std::shared_ptr<Light> light = sg->get_light_from_geomID(rayhit.hit.geomID);
        Light_EvalRes le = light->eval(org, dir);
        L = L + Lw * le.value;
        // If we encounter a light emitter we will terminate adding more light from the path.
        //Alternatively, we could move the intersected ray through the light source and continue
        break;
    }
    else {
        albedo = g_geomIDs[rayhit.hit.geomID].primColorTable[rayhit.hit.primID];
        /* An albedo as well as a material type is used */
    }

    /* weight scaling based on material sample. used for attenuation amongst path segments */
    Vec3fa c = Vec3fa(1.0f);

    Vec3fa wi1;
    Vec2f randomMatSample(distrib(reng), distrib(reng));


    /* Search for each light in the scene from our hit point. Aggregate the
 * radiance if hit point is not occluded */
    sg->set_intersect_context_incoherent();


    //Not MIS versus MIS
    if (false) {
        if (Material_direct_illumination(materialType)) {
            /* Cast shadow ray(s) from the hit point */
            sg->cast_shadow_rays(dg, albedo, materialType, Lw, wo, medium, m_time, L, reng,
                distrib);
        }

        wi1 = Material_sample(materialType, Lw, wo, dg, medium,
            randomMatSample);
        c = c * Material_eval(albedo, materialType, Lw, wo, dg, wi1, medium,
            randomMatSample);


        float nextPDF = Material_pdf(materialType, Lw, wo, dg, medium, randomMatSample);
        if (nextPDF <= 1E-4f) break;
        Lw = Lw * c / nextPDF;
    }
    else {
            float sumPdf = 0.f;
            size_t misIdx = 0;
            for (std::shared_ptr<Light> light : sg->m_lights) {
                Vec2f randomLightSample(distrib(reng), distrib(reng));
                Light_SampleRes ls = light->sample(dg, randomLightSample);

                m_misLightStorage[pxID][misIdx].sam = ls;
                sumPdf += ls.pdf;
                /* If the sample probability density evaluation is 0 then no need to
                 * consider this shadow ray */
                if (ls.pdf <= 0.0f) continue;

                float tfar = sg->cast_shadow_ray(dg.P, ls.dir, dg.eps, ls.dist, m_time);
                m_misLightStorage[pxID][misIdx].tfar = tfar;
                m_misLightStorage[pxID][misIdx].randomLightSample = randomLightSample;
                misIdx++;
            }
            float nextPDF = Material_pdf(materialType, Lw, wo, dg, medium, randomMatSample);
            if (nextPDF >= 0.f)
                sumPdf += nextPDF;

            Vec3fa misValue(0.f);
            size_t numLights = m_misLightStorage[pxID].size();
            for(auto i = 0; i< numLights; i++) {
                float lightPDF = m_misLightStorage[pxID][i].sam.pdf;
                if (m_misLightStorage[pxID][i].tfar >= 0.f && lightPDF > 0.f) {
                    Vec3fa value = m_misLightStorage[pxID][i].sam.weight *
                        Material_eval(albedo, materialType, Lw, wo, dg, m_misLightStorage[pxID][i].sam.dir, medium, m_misLightStorage[pxID][i].randomLightSample);
                    misValue += lightPDF * value / (sumPdf * lightPDF);
                }
            }
            L = L + Lw * misValue;
    wi1 = Material_sample(materialType, Lw, wo, dg, medium,
        randomMatSample);

    c = c * Material_eval(albedo, materialType, Lw, wo, dg, wi1, medium,
        randomMatSample);


    if (nextPDF <= 1E-4f) break;
    Lw = Lw * (nextPDF * c) / (sumPdf * nextPDF);

    }
    /* setup secondary ray */
    float sign = dot(wi1, dg.Ng) < 0.0f ? -1.0f : 1.0f;
    dg.P = dg.P + sign * dg.eps * dg.Ng;
    org = dg.P;
    dir = normalize(wi1);
    init_RayHit(rayhit, org, dir, dg.eps, inf,
               m_time);


  }

  return L;
}



PathTracer::~PathTracer() {

}


#endif /* FILE_PATHTRACERSEEN */