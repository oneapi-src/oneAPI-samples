#pragma once
#include "CornellBox.h"
#include "DefaultCubeAndPlane.h"
#include "Sphere.h"
#include "Lights.h"

#include <embree3/rtcore.h>

/* Added for geometry selection in pathtracer */
enum class SceneSelector {
    SHOW_CUBE_AND_PLANE,
    SHOW_CORNELL_BOX,
};

/* The most basic scene graph possible for exploratory code... please consider the scene graph from ospray studio or embree itself as better production references */
struct SceneGraph {

public:
    SceneGraph(RTCDevice device, SceneSelector SELECT_SCENE, unsigned int width, unsigned int height);

    void SceneGraph::init_embree_scene(const RTCDevice device, SceneSelector SELECT_SCENE, const unsigned int width, const unsigned int height);

    void SceneGraph::init_embree_context();
    
    bool SceneGraph::intersect_path_and_scene(Vec3fa& org, Vec3fa& dir, RTCRayHit& rayhit, DifferentialGeometry& dg);

    void SceneGraph::cast_shadow_ray(DifferentialGeometry& dg, Vec3fa& albedo, MaterialType materialType, const Vec3fa& Lw, const Vec3fa& wo, const Medium& medium, float time, Vec3fa& L, RandomEngine& reng,
        std::uniform_real_distribution<float>& distrib);

    void SceneGraph::set_intersect_context_coherent();

    void SceneGraph::set_intersect_context_incoherent();

    Vec3fa SceneGraph::get_camera_origin();
    
    Vec3fa SceneGraph::get_direction_from_pixel(float x, float y);

	~SceneGraph();

    std::vector<struct Light> m_lights;
protected:
    //std::vector<Light> m_lights;
private:
	RTCScene m_scene;
    SceneSelector m_sceneSelector;
    rkcommon::math::AffineSpace3fa m_camera;

    RTCIntersectContext m_context;

    void SceneGraph::scene_cleanup();

};

SceneGraph::SceneGraph(const RTCDevice device, SceneSelector SELECT_SCENE, const unsigned int width, const unsigned int height) {
    
    init_embree_scene(device, SELECT_SCENE, width, height);
    init_embree_context();

}

void SceneGraph::init_embree_scene(const RTCDevice device, SceneSelector SELECT_SCENE, const unsigned int width, const unsigned int height) {
    m_sceneSelector = SELECT_SCENE;
    /* create scene */
    m_scene = nullptr;
    m_scene = rtcNewScene(device);

    switch (m_sceneSelector) {
    case SceneSelector::SHOW_CORNELL_BOX:
        /* add cornell box */
        addCornell(m_scene, device);

        /* If you would like to add an Intel Embree sphere see addSphere(..) as
         * used below for an example... Remember to look for materials properties
         * set in our header files */
        addSphere(m_scene, device, Vec3fa(0.6f, -0.8f, -0.6f), 0.2f);

        cornellCameraLightSetup(m_camera, m_lights, width, height);
        break;
    case SceneSelector::SHOW_CUBE_AND_PLANE:
    default:
        /* add cube */
        addCube(m_scene, device);

        /* add ground plane */
        addGroundPlane(m_scene, device);

        /* The sphere can be used in the cube and plane scene with a corresponding
         * position for that scene */
         // addSphere(m_scene, m_device, Vec3fa(2.5f, 0.f, 2.5f), 1.f);

        cubeAndPlaneCameraLightSetup(m_camera, m_lights, width, height);
        break;
    }

    /* commit changes to scene */
    rtcCommitScene(m_scene);
}

void SceneGraph::init_embree_context() {
    
    rtcInitIntersectContext(&m_context);

}

void SceneGraph::set_intersect_context_coherent() {
    m_context.flags = RTCIntersectContextFlags::RTC_INTERSECT_CONTEXT_FLAG_COHERENT;
}

void SceneGraph::set_intersect_context_incoherent() {
    m_context.flags = RTCIntersectContextFlags::RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;
}



bool SceneGraph::intersect_path_and_scene(Vec3fa& org, Vec3fa& dir, RTCRayHit& rayhit, DifferentialGeometry& dg) {
    /* intersect ray with scene */
    
    rtcIntersect1(m_scene, &m_context, &rayhit);

    /* if nothing hit the path is terminated, this could be an light lookup
     * insteead */
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

void SceneGraph::cast_shadow_ray(DifferentialGeometry& dg, Vec3fa& albedo, MaterialType materialType, const Vec3fa& Lw, const Vec3fa& wo, const Medium& medium, float time, Vec3fa& L, RandomEngine& reng,
    std::uniform_real_distribution<float>& distrib) {

    for (const Light& light : m_lights) {
        Vec2f randomLightSample(distrib(reng), distrib(reng));
        Light_SampleRes ls = sample_light(light, dg, randomLightSample);
        /* If the sample probability density evaluation is 0 then no need to
         * consider this shadow ray */
        if (ls.pdf <= 0.0f) continue;

        RTCRayHit shadow;
        init_RayHit(shadow, dg.P, ls.dir, dg.eps, ls.dist, time);
        rtcOccluded1(m_scene, &m_context, &shadow.ray);
        if (shadow.ray.tfar >= 0.0f) {
            L = L + Lw * ls.weight *
                Material_eval(albedo, materialType, Lw, wo, dg, ls.dir, medium, randomLightSample);
        }
    }

}

Vec3fa SceneGraph::get_camera_origin() {
    return Vec3fa(m_camera.p.x, m_camera.p.y, m_camera.p.z);
}

Vec3fa SceneGraph::get_direction_from_pixel(float x, float y) {
    return normalize(x * m_camera.l.vx + y * m_camera.l.vy + m_camera.l.vz);
}

/* called by the C++ code for cleanup */
void SceneGraph::scene_cleanup() {
    rtcReleaseScene(m_scene);
    m_scene = nullptr;
    switch (m_sceneSelector) {
    case SceneSelector::SHOW_CORNELL_BOX:
        cleanCornell();
        cleanSphere();
        break;
    case SceneSelector::SHOW_CUBE_AND_PLANE:
    default:
        cleanCubeAndPlane();
        // cleanSphere();
        break;
    }
}

SceneGraph::~SceneGraph() {
    scene_cleanup();
}
