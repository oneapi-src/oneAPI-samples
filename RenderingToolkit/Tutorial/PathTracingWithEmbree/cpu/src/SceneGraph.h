#pragma once
#include <embree4/rtcore.h>

#include "CornellBox.h"
#include "DefaultCubeAndPlane.h"
#include "Geometry.h"
#include "Lights.h"
#include "Pool.h"
#include "RandomSampler.h"
#include "Sphere.h"

/* Added for geometry selection in pathtracer */
enum class SceneSelector { SHOW_CUBE_AND_PLANE, SHOW_CORNELL_BOX, SHOW_POOL };

/* The most basic scene graph possible for exploratory code... please consider
 * the scene graph from ospray studio or embree tutorials themselves as better
 * production references */
struct SceneGraph {
 public:
  SceneGraph(RTCDevice device, SceneSelector SELECT_SCENE, unsigned int width,
             unsigned int height);

  void init_embree_scene(const RTCDevice device, SceneSelector SELECT_SCENE,
                         const unsigned int width, const unsigned int height);

  bool intersect_path_and_scene(Vec3fa& org, Vec3fa& dir, RTCRayHit& rayhit,
                                DifferentialGeometry& dg, bool bCoherent);

  void cast_shadow_rays(DifferentialGeometry& dg, Vec3fa& albedo,
                        MaterialType materialType, const Vec3fa& Lw,
                        const Vec3fa& wo, const Medium& medium, float time,
                        Vec3fa& L, RandomSampler& randomSampler,
                        bool bCoherent);

  float cast_shadow_ray(const Vec3fa& org, const Vec3fa& dir, float tnear,
                        float tfar, float _time, bool bCoherent);

  Vec3fa get_camera_origin();

  Vec3fa get_direction_from_pixel(float x, float y);

  unsigned int getNumLights();

  std::shared_ptr<Light> get_light_from_geomID(unsigned int geomID);

  ~SceneGraph();

  std::vector<std::shared_ptr<Light>> m_lights;

  /* Added for path tracer: for holding material properties for each geometry id
   */
  std::map<unsigned int, MatAndPrimColorTable> m_mapGeomToPrim;

 private:
  RTCScene m_scene;
  SceneSelector m_sceneSelector;
  rkcommon::math::AffineSpace3fa m_camera;

  std::map<unsigned int, size_t> m_mapGeomToLightIdx;
  void scene_cleanup();

  /* We'll use this 'geometries' container to automatically clean up the data
   * arrays created that are used to create embree geometries */
  std::vector<std::unique_ptr<Geometry>> geometries;
};

SceneGraph::SceneGraph(const RTCDevice device, SceneSelector SELECT_SCENE,
                       const unsigned int width, const unsigned int height) {
  init_embree_scene(device, SELECT_SCENE, width, height);
}

void SceneGraph::init_embree_scene(const RTCDevice device,
                                   SceneSelector SELECT_SCENE,
                                   const unsigned int width,
                                   const unsigned int height) {
  m_sceneSelector = SELECT_SCENE;
  /* create scene */
  m_scene = nullptr;
  m_scene = rtcNewScene(device);

  switch (m_sceneSelector) {
    case SceneSelector::SHOW_CUBE_AND_PLANE:
      /* add cube, add ground plane, and light */
      geometries.push_back(std::make_unique<CubeAndPlane>(
          m_scene, device, m_mapGeomToPrim, m_mapGeomToLightIdx, m_lights,
          m_camera, width, height));

      /* The sphere can be used in the cube and plane scene with a corresponding
       * position for that scene */

      /*
       * geometries.push_back(std::make_unique<Sphere>(m_scene, device,
       m_mapGeomToPrim, MaterialType::MATERIAL_MIRROR, Vec3fa(2.5f, 0.f, 2.5f),
       Vec3fa(0.8f, 0.8f, 0.8f), 1.0f, m_mapGeomToLightIdx, m_lights, m_camera,
       width, height));
      */
      break;
    case SceneSelector::SHOW_POOL:
      geometries.push_back(std::make_unique<Pool>(
          m_scene, device, m_mapGeomToPrim, m_mapGeomToLightIdx, m_lights,
          m_camera, width, height));
      break;
    case SceneSelector::SHOW_CORNELL_BOX:
    default:
      /* add cornell box */
      geometries.push_back(std::make_unique<CornellBoxGeometry>(
          m_scene, device, m_mapGeomToPrim, m_mapGeomToLightIdx, m_lights,
          m_camera, width, height));
      /* If you would like to add an Intel Embree sphere see below for an
       * example... Remember to look for materials properties
       * set in the Sphere source */
      Vec3fa pos = {0.6f, -0.8f, -0.6f};
      Vec3fa color = {1.f, 1.f, 1.f};
      float radius = 0.2f;

      geometries.push_back(std::make_unique<Sphere>(
          m_scene, device, m_mapGeomToPrim, MaterialType::MATERIAL_GLASS, pos,
          color, radius, m_mapGeomToLightIdx, m_lights, m_camera, width,
          height));

      break;
  }

  /* commit changes to scene */
  rtcCommitScene(m_scene);
}

bool SceneGraph::intersect_path_and_scene(Vec3fa& org, Vec3fa& dir,
                                          RTCRayHit& rayhit,
                                          DifferentialGeometry& dg,
                                          bool bCoherent) {
  /* New with Embree 4... RTCIntersectArguments to set ray coherency */
  /* Only primary rays are set as coherent in this example program */
  RTCIntersectArguments iargs;
  rtcInitIntersectArguments(&iargs);
  iargs.flags =
      (bCoherent) ? RTC_RAY_QUERY_FLAG_COHERENT : RTC_RAY_QUERY_FLAG_INCOHERENT;

  /* intersect ray with scene */

  rtcIntersect1(m_scene, &rayhit, &iargs);

  /* if nothing hit the path is terminated, this could be an environment light
   * lookup instead */
  if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) return false;

  Vec3fa Ng = Vec3fa(rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z);
  Vec3fa Ns = normalize(Ng);

  /* compute differential geometry */
  for (int i = 0; i < RTC_MAX_INSTANCE_LEVEL_COUNT; i++)
    dg.instIDs[i] = rayhit.hit.instID[i];

  dg.geomID = rayhit.hit.geomID;
  dg.primID = rayhit.hit.primID;
  dg.u = rayhit.hit.u;
  dg.v = rayhit.hit.v;

  dg.P = org + rayhit.ray.tfar * dir;
  dg.Ng = Ng;
  dg.Ns = Ns;

  /* Reference epsilon value to move away from the plane, avoid artifacts */
  dg.eps =
      32.0f * 1.19209e-07f *
      max(max(abs(dg.P.x), abs(dg.P.y)), max(abs(dg.P.z), rayhit.ray.tfar));

  dg.Ng = face_forward(dir, normalize(dg.Ng));
  dg.Ns = face_forward(dir, normalize(dg.Ns));

  return true;
}

void SceneGraph::cast_shadow_rays(DifferentialGeometry& dg, Vec3fa& albedo,
                                  MaterialType materialType, const Vec3fa& Lw,
                                  const Vec3fa& wo, const Medium& medium,
                                  float time, Vec3fa& L,
                                  RandomSampler& randomSampler,
                                  bool bCoherent) {
  Vec3fa ret;

  RTCOccludedArguments oargs;
  rtcInitOccludedArguments(&oargs);
  /* In this program Occluded rays are never coherent so this is hard coded. */
  oargs.flags = RTC_RAY_QUERY_FLAG_INCOHERENT;
  /* Otherwise, the assignment commented below would apply */
  /* oargs.flags = (bCoherent) ? RTC_RAY_QUERY_FLAG_COHERENT :
   * RTC_RAY_QUERY_FLAG_INCOHERENT; */

  for (std::shared_ptr<Light> light : m_lights) {
    Vec2f randomLightSample(randomSampler.get_float(),
                            randomSampler.get_float());
    Light_SampleRes ls = light->sample(dg, randomLightSample);

    /* If the sample probability density evaluation is 0 then no need to
     * consider this shadow ray */
    if (ls.pdf <= 0.0f) continue;

    RTCRayHit shadow;
    init_RayHit(shadow, dg.P, ls.dir, dg.eps, ls.dist, time);
    rtcOccluded1(m_scene, &shadow.ray, &oargs);
    if (shadow.ray.tfar >= 0.0f) {
      L = L + Lw * ls.weight *
                  Material_eval(albedo, materialType, Lw, wo, dg, ls.dir,
                                medium, randomLightSample);
    }
  }
}

float SceneGraph::cast_shadow_ray(const Vec3fa& org, const Vec3fa& dir,
                                  float tnear, float tfar, float _time,
                                  bool bCoherent) {
  RTCRayHit shadow;
  init_RayHit(shadow, org, dir, tnear, tfar, _time);

  RTCOccludedArguments oargs;
  rtcInitOccludedArguments(&oargs);
  /* In this program Occluded rays are assumed coherent so this is hard coded. */
  oargs.flags = RTC_RAY_QUERY_FLAG_INCOHERENT;
  /* Otherwise, the assignment commented below would apply */
  /* oargs.flags = (bCoherent) ? RTC_RAY_QUERY_FLAG_COHERENT :
   * RTC_RAY_QUERY_FLAG_INCOHERENT; */

  rtcOccluded1(m_scene, &shadow.ray, &oargs);
  return shadow.ray.tfar;
}

Vec3fa SceneGraph::get_camera_origin() {
  return Vec3fa(m_camera.p.x, m_camera.p.y, m_camera.p.z);
}

Vec3fa SceneGraph::get_direction_from_pixel(float x, float y) {
  return normalize(x * m_camera.l.vx + y * m_camera.l.vy + m_camera.l.vz);
}

unsigned int SceneGraph::getNumLights() { return m_lights.size(); }

std::shared_ptr<Light> SceneGraph::get_light_from_geomID(unsigned int geomID) {
  size_t idx = m_mapGeomToLightIdx[geomID];

  return m_lights[idx];
}

/* called by the C++ code for cleanup */
void SceneGraph::scene_cleanup() {
  rtcReleaseScene(m_scene);
  m_scene = nullptr;
}

SceneGraph::~SceneGraph() { scene_cleanup(); }
