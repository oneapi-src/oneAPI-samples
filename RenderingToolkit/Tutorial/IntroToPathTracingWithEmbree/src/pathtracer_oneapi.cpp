#ifdef _MSC_VER
#ifndef NOMINMAX
/* use the c++ library min and max instead of the MSVS macros */
/* rkcommon will define this macro upon CMake import */
#define NOMINMAX
#endif
#endif

#include <embree3/rtcore.h>
#include <rkcommon/math/vec.h>
#include <rkcommon/memory/malloc.h>
#include <tbb/parallel_for.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

/* Addition for pathtracer */
#include <random>
#include <map>
#include <memory>
#include <vector>
#include <rkcommon/math/LinearSpace.h>
#include "CornellBox.h"

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

struct LinearSpace3 {
  Vec3fa vx, vy, vz;
};

typedef struct Affine3fa {
  LinearSpace3 l;
  Vec3fa p;
} ISPCCamera;

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

/* Added for pathtracer */
enum MaterialType {
    MATERIAL_MATTE,
    MATERIAL_MIRROR,
    MATERIAL_THIN_DIELECTRIC,
};

/* Added for geometry selection in pathtracer */
enum SceneSelector {
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

Vec3fa* g_cube_face_colors = nullptr;
Vec3fa* g_cube_vertex_colors = nullptr;
Vec3fa* g_ground_face_colors = nullptr;
Vec3fa* g_ground_vertex_colors = nullptr;

/* Added for pathtracer */
Vec3fa* g_cornell_face_colors = nullptr;
Vec3fa* g_cornell_vertex_colors = nullptr;

std::vector<unsigned int> geomIDs;

/* Addition for pathtracer */
std::map<unsigned int, MaterialType> matMap;

RTCDevice g_device = nullptr;
RTCScene g_scene = nullptr;

unsigned char* g_pixels;

/* Additions for pathtracer */
std::vector<std::shared_ptr<Vec3ff>> g_accu;
unsigned int g_accu_count = 0;
unsigned int g_max_path_length;
unsigned int g_spp;
SceneSelector g_sceneSelector;
std::vector<InfiniteDirectionalLight> g_infDirectionalLights;
unsigned int g_accu_limit;

ISPCCamera g_camera;

void error_handler(void* userPtr, const RTCError code, const char* str) {
  if (code == RTC_ERROR_NONE) return;

  printf("fail: Embree Error: ");
  switch (code) {
    case RTC_ERROR_UNKNOWN:
      printf("RTC_ERROR_UNKNOWN");
      break;
    case RTC_ERROR_INVALID_ARGUMENT:
      printf("RTC_ERROR_INVALID_ARGUMENT");
      break;
    case RTC_ERROR_INVALID_OPERATION:
      printf("RTC_ERROR_INVALID_OPERATION");
      break;
    case RTC_ERROR_OUT_OF_MEMORY:
      printf("RTC_ERROR_OUT_OF_MEMORY");
      break;
    case RTC_ERROR_UNSUPPORTED_CPU:
      printf("RTC_ERROR_UNSUPPORTED_CPU");
      break;
    case RTC_ERROR_CANCELLED:
      printf("RTC_ERROR_CANCELLED");
      break;
    default:
      printf("invalid error code");
      break;
  }
  if (str) {
    printf(" (");
    while (*str) putchar(*str++);
    printf(")\n");
  }
  exit(1);
}



ISPCCamera positionCamera(Vec3fa from, Vec3fa to, Vec3fa up, float fov,
                          size_t width, size_t height) {
  /* There are many ways to set up a camera projection. This one is consolidated
   * from the camera code in the Embree/tutorial/common/tutorial/camera.h object
   */
  ISPCCamera camMatrix;
  Vec3fa Z =
      rkcommon::math::normalize(rkcommon::math::vec_t<float, 3, 1>(to - from));
  Vec3fa U = rkcommon::math::normalize(
      rkcommon::math::cross(rkcommon::math::vec_t<float, 3, 1>(up),
                            rkcommon::math::vec_t<float, 3, 1>(Z)));
  Vec3fa V = rkcommon::math::normalize(
      rkcommon::math::cross(rkcommon::math::vec_t<float, 3, 1>(Z),
                            rkcommon::math::vec_t<float, 3, 1>(U)));
  camMatrix.l.vx = U;
  camMatrix.l.vy = V;
  camMatrix.l.vz = Z;
  camMatrix.p = from;

  /* negate for a right handed camera*/
  camMatrix.l.vx = -camMatrix.l.vx;

  const float fovScale = 1.0f / tanf(rkcommon::math::deg2rad(0.5f * fov));

  camMatrix.l.vz = -0.5f * width * camMatrix.l.vx +
                   0.5f * height * camMatrix.l.vy +
                   0.5f * height * fovScale * camMatrix.l.vz;
  camMatrix.l.vy = -camMatrix.l.vy;

  return camMatrix;
}

/* adds a cube to the scene */
unsigned int addCube(RTCScene _scene) {
  /* create a triangulated cube with 12 triangles and 8 vertices */
  RTCGeometry mesh = rtcNewGeometry(g_device, RTC_GEOMETRY_TYPE_TRIANGLE);

  /* create face and vertex color arrays */
  g_cube_face_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 12, 16);
  g_cube_vertex_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 8, 16);

  /* set vertices and vertex colors */
  Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex), 8);
  g_cube_vertex_colors[0] = Vec3fa(0, 0, 0);
  vertices[0].x = -1;
  vertices[0].y = -1;
  vertices[0].z = -1;
  g_cube_vertex_colors[1] = Vec3fa(0, 0, 1);
  vertices[1].x = -1;
  vertices[1].y = -1;
  vertices[1].z = +1;
  g_cube_vertex_colors[2] = Vec3fa(0, 1, 0);
  vertices[2].x = -1;
  vertices[2].y = +1;
  vertices[2].z = -1;
  g_cube_vertex_colors[3] = Vec3fa(0, 1, 1);
  vertices[3].x = -1;
  vertices[3].y = +1;
  vertices[3].z = +1;
  g_cube_vertex_colors[4] = Vec3fa(1, 0, 0);
  vertices[4].x = +1;
  vertices[4].y = -1;
  vertices[4].z = -1;
  g_cube_vertex_colors[5] = Vec3fa(1, 0, 1);
  vertices[5].x = +1;
  vertices[5].y = -1;
  vertices[5].z = +1;
  g_cube_vertex_colors[6] = Vec3fa(1, 1, 0);
  vertices[6].x = +1;
  vertices[6].y = +1;
  vertices[6].z = -1;
  g_cube_vertex_colors[7] = Vec3fa(1, 1, 1);
  vertices[7].x = +1;
  vertices[7].y = +1;
  vertices[7].z = +1;

  /* set triangles and face colors */
  int tri = 0;
  Triangle* triangles = (Triangle*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle), 12);

  // left side
  g_cube_face_colors[tri] = Vec3fa(1, 0, 0);
  triangles[tri].v0 = 0;
  triangles[tri].v1 = 1;
  triangles[tri].v2 = 2;
  tri++;
  g_cube_face_colors[tri] = Vec3fa(1, 0, 0);
  triangles[tri].v0 = 1;
  triangles[tri].v1 = 3;
  triangles[tri].v2 = 2;
  tri++;

  // right side
  g_cube_face_colors[tri] = Vec3fa(0, 1, 0);
  triangles[tri].v0 = 4;
  triangles[tri].v1 = 6;
  triangles[tri].v2 = 5;
  tri++;
  g_cube_face_colors[tri] = Vec3fa(0, 1, 0);
  triangles[tri].v0 = 5;
  triangles[tri].v1 = 6;
  triangles[tri].v2 = 7;
  tri++;

  // bottom side
  g_cube_face_colors[tri] = Vec3fa(1.0f);
  triangles[tri].v0 = 0;
  triangles[tri].v1 = 4;
  triangles[tri].v2 = 1;
  tri++;
  g_cube_face_colors[tri] = Vec3fa(1.0f);
  triangles[tri].v0 = 1;
  triangles[tri].v1 = 4;
  triangles[tri].v2 = 5;
  tri++;

  // top side
  g_cube_face_colors[tri] = Vec3fa(1.0f);
  triangles[tri].v0 = 2;
  triangles[tri].v1 = 3;
  triangles[tri].v2 = 6;
  tri++;
  g_cube_face_colors[tri] = Vec3fa(1.0f);
  triangles[tri].v0 = 3;
  triangles[tri].v1 = 7;
  triangles[tri].v2 = 6;
  tri++;

  // front side
  g_cube_face_colors[tri] = Vec3fa(0, 0, 1);
  triangles[tri].v0 = 0;
  triangles[tri].v1 = 2;
  triangles[tri].v2 = 4;
  tri++;
  g_cube_face_colors[tri] = Vec3fa(0, 0, 1);
  triangles[tri].v0 = 2;
  triangles[tri].v1 = 6;
  triangles[tri].v2 = 4;
  tri++;

  // back side
  g_cube_face_colors[tri] = Vec3fa(1, 1, 0);
  triangles[tri].v0 = 1;
  triangles[tri].v1 = 5;
  triangles[tri].v2 = 3;
  tri++;
  g_cube_face_colors[tri] = Vec3fa(1, 1, 0);
  triangles[tri].v0 = 3;
  triangles[tri].v1 = 5;
  triangles[tri].v2 = 7;
  tri++;

  rtcSetGeometryVertexAttributeCount(mesh, 1);
  rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
                             RTC_FORMAT_FLOAT3, g_cube_vertex_colors, 0,
                             sizeof(Vec3fa), 8);

  rtcCommitGeometry(mesh);
  unsigned int geomID = rtcAttachGeometry(_scene, mesh);
  rtcReleaseGeometry(mesh);
  return geomID;
}

/* adds a ground plane to the scene */
unsigned int addGroundPlane(RTCScene _scene) {
  /* create a triangulated plane with 2 triangles and 4 vertices */
  RTCGeometry mesh = rtcNewGeometry(g_device, RTC_GEOMETRY_TYPE_TRIANGLE);

  /* create face and vertex color arrays */
  g_ground_face_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 2, 16);
  g_ground_vertex_colors = (Vec3fa*)alignedMalloc(sizeof(Vec3fa) * 4, 16);

  /* set vertices */
  Vertex* vertices = (Vertex*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex), 4);
  g_ground_vertex_colors[0] = Vec3fa(1, 0, 0);
  vertices[0].x = -10;
  vertices[0].y = -1;
  vertices[0].z = -10;
  g_ground_vertex_colors[1] = Vec3fa(1, 0, 1);
  vertices[1].x = -10;
  vertices[1].y = -1;
  vertices[1].z = +10;
  g_ground_vertex_colors[2] = Vec3fa(1, 1, 0);
  vertices[2].x = +10;
  vertices[2].y = -1;
  vertices[2].z = -10;
  g_ground_vertex_colors[3] = Vec3fa(1, 1, 1);
  vertices[3].x = +10;
  vertices[3].y = -1;
  vertices[3].z = +10;

  /* set triangles */
  Triangle* triangles = (Triangle*)rtcSetNewGeometryBuffer(
      mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle), 2);

  g_ground_face_colors[0] = Vec3fa(1, 1, 1);
  triangles[0].v0 = 0;
  triangles[0].v1 = 1;
  triangles[0].v2 = 2;
  g_ground_face_colors[1] = Vec3fa(1, 1, 1);
  triangles[1].v0 = 1;
  triangles[1].v1 = 3;
  triangles[1].v2 = 2;

  rtcSetGeometryVertexAttributeCount(mesh, 1);
  rtcSetSharedGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0,
                             RTC_FORMAT_FLOAT3, g_ground_vertex_colors, 0,
                             sizeof(Vec3fa), 4);

  rtcCommitGeometry(mesh);
  unsigned int geomID = rtcAttachGeometry(_scene, mesh);
  rtcReleaseGeometry(mesh);
  return geomID;
}


/* Added for pathtracer */
inline void initRayHit(RTCRayHit& rayhit, const Vec3fa& org, const Vec3fa& dir, float tnear, float tfar, float time) {
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

/* Added for pathtracer */
inline Vec3fa face_forward(const Vec3fa& dir, const Vec3fa& _Ng) {
    const Vec3fa Ng = _Ng;
    return dot(dir, Ng) < 0.0f ? Ng : -Ng;
}

/// cosine-weighted sampling of hemisphere oriented along the +z-axis
////////////////////////////////////////////////////////////////////////////////

/* Added for pathtracer transforming a normal */
rkcommon::math::LinearSpace3<Vec3fa> frame(const Vec3fa& N) {
    const Vec3fa dx0(0, N.z, -N.y);
    const Vec3fa dx1(-N.z, 0, N.x);

    const Vec3fa dx = normalize((dot(dx0, dx0) > dot(dx1, dx1)) ? dx0 : dx1);
    const Vec3fa dy = normalize(cross(N, dx));

    return rkcommon::math::LinearSpace3<Vec3fa>(dx, dy, N);
}

/*! Cosine weighted hemisphere sampling. Up direction is provided as argument. */
inline Sample3f cosineSampleHemisphere(const float  u, const float  v, const Vec3fa& N)
{
    /* Determine cartesian coordinate for new Vec3fa */
    const float phi = float(2.0 * M_PI) * u;
    const float cosTheta = sqrt(v);
    const float sinTheta = sqrt(1.0f - u);
    const float sinPhi = sinf(phi);
    const float cosPhi = cosf(phi);

    Vec3fa localDir = Vec3fa(cosPhi * sinTheta,
        sinPhi * sinTheta,
        cosTheta);
    /* Gives the new Vec3fa transformed about the input Vec3fa */
    Sample3f s;
    s.v = frame(N) * localDir;

    /* Gives a smooth pdf */
    s.pdf = localDir.z / float(M_PI);
    return s;
}

Vec3fa Lambertian__eval(const Vec3fa& R, const Vec3fa& wo, DifferentialGeometry dg, Vec3fa wi_v) {
    /* The diffuse material. Reflectance (albedo) times the cosign fall off of the vector about the normal. */
    return R * (1.f / float(M_PI)) * clamp(dot(wi_v, dg.Ns));

}

Vec3fa Material__sample(Vec3fa R, enum MaterialType materialType, Vec3fa Lw, Vec3fa wo, DifferentialGeometry dg, Sample3f& wi, Vec2f randomMatSample) {
    Vec3fa c = Vec3fa(0.0f);
    switch (materialType) {
    case MATERIAL_MATTE:
        wi = cosineSampleHemisphere(randomMatSample.x, randomMatSample.y, dg.Ns);
        return Lambertian__eval(R, wo, dg, wi.v);
        break;
        /* Return our debug color if something goes awry */
    default: c = R;
        break;
    }

    return c;
}

Vec3fa Material__eval(Vec3fa R, enum MaterialType materialType, const Vec3fa& wo, const DifferentialGeometry& dg, const Vec3fa& wi) {
    Vec3fa c = Vec3fa(0.0f);
    switch (materialType) {
    case MATERIAL_MATTE:
        return Lambertian__eval(R, wo, dg, wi);
        break;
        /* Return our debug color if something goes awry */
    default: c = R;
        break;
    }
    return c;
}
/* Added for pathtracer */
struct Light_SampleRes
{
    Vec3fa weight;  //!< radiance that arrives at the given point divided by pdf
    Vec3fa dir;     //!< direction towards the light source
    float dist;    //!< largest valid t_far value for a shadow ray
    float pdf;     //!< probability density that this sample was taken
};

Light_SampleRes sample_directional_light(Vec3fa& lightDir, const Vec3fa& lightIntensity, DifferentialGeometry& dg, Vec2f& randomLightSample) {
    Light_SampleRes res;

    res.dir = -lightDir;
    res.dist = std::numeric_limits<float>::infinity();
    res.pdf = std::numeric_limits<float>::infinity();
    res.weight = lightIntensity; // *pdf/pdf cancel

    return res;

}

/* task that renders a single screen pixel */
Vec3fa renderPixelFunction(float x, float y,
    const unsigned int width, const unsigned int height,
    std::default_random_engine& reng, std::uniform_real_distribution<float>& distrib,
    const float time,
    const ISPCCamera& camera) {
    RTCIntersectContext context;
    rtcInitIntersectContext(&context);
    Vec3fa dir = rkcommon::math::normalize(x * camera.l.vx +
        y * camera.l.vy + camera.l.vz);
    Vec3fa org = Vec3fa(camera.p.x, camera.p.y, camera.p.z);

    /* initialize ray */
    RTCRayHit rayhit;
    initRayHit(rayhit, org, dir, 0.0f, std::numeric_limits<float>::infinity(), time);

    Vec3fa L = Vec3fa(0.0f);
    Vec3fa Lw = Vec3fa(1.0f);

    DifferentialGeometry dg;

    /* iterative path tracer loop */
    for (int i = 0; i < g_max_path_length; i++)
    {
        /* terminate if contribution too low */
        if (max(Lw.x, max(Lw.y, Lw.z)) < 0.001f)
            break;

        /* intersect ray with scene */
        context.flags = (i == 0) ? RTCIntersectContextFlags::RTC_INTERSECT_CONTEXT_FLAG_COHERENT : RTCIntersectContextFlags::RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;
        rtcIntersect1(g_scene, &context, &rayhit);
        const Vec3fa wo = -dir;

        /* if nothing hit the path is terminated, this could be an environment map lookup insteead */
        if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
            break;

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

        /* Optionally Next is:
        * material discovery, the full pathtracer program includes objects to lookup materials from a scenegraph object
        * texture lookup transformations based on vertex-texture assignments
        * Interpolating normals from vertex normals is a possibility as well
        * This could include a required tranformation of normals for geometries that have been instanced.
        */

        /* default albedo is a pink color for debug */
        Vec3fa albedo = Vec3fa(0.9f, 0.7f, 0.7f);
        enum MaterialType materialType = MaterialType::MATERIAL_MATTE;
        /* An albedo as well as a material type is used */
        switch (g_sceneSelector) {
        case SceneSelector::SHOW_CORNELL_BOX:
            albedo = g_cornell_face_colors[rayhit.hit.primID];
            materialType = MaterialType::MATERIAL_MATTE;
            break;
        case SceneSelector::SHOW_CUBE_AND_PLANE:
        default:
            if (rayhit.hit.geomID == geomIDs[0]) {
                albedo = g_cube_face_colors[rayhit.hit.primID];
                materialType = MaterialType::MATERIAL_MATTE;
            }
            else if (rayhit.hit.geomID == geomIDs[1]) {
                albedo = g_ground_face_colors[rayhit.hit.primID];
                materialType = MaterialType::MATERIAL_MATTE;
            }
            break;
        }


        /* Reference epsilon value to move away from the plane, avoid artifacts */
        dg.eps = 32.0f * 1.19209e-07f * max(max(abs(dg.P.x), abs(dg.P.y)), max(abs(dg.P.z), rayhit.ray.tfar));


        dg.Ng = face_forward(dir, normalize(dg.Ng));
        dg.Ns = face_forward(dir, normalize(dg.Ns));

        /* weight scaling based on material sample */
        Vec3fa c = Vec3fa(1.0f);

        Sample3f wi1;
        Vec2f randomMatSample(distrib(reng), distrib(reng));
        c = c * Material__sample(albedo, materialType, Lw, wo, dg, wi1, randomMatSample);

        /* Create one light transmitting directionally */
        //Vec3fa lightDir = normalize(Vec3fa(-1, -1, -1));
        Vec3fa lightDir = normalize(Vec3fa(0, -1, 0));
        //Vec3fa lightDir = -(normalize(camera.p));
        Vec3fa lightIntensity(1.0f);
        /* Search for each light in the scene from our hit point. Aggregate the radiance if hit point is not occluded */
        for(auto light : g_infDirectionalLights)
        {
            context.flags = RTCIntersectContextFlags::RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;
            Vec2f randomLightSample(distrib(reng), distrib(reng));
            Light_SampleRes ls = sample_directional_light(light.dir, light.intensity, dg, randomLightSample);
            /* If the sample probability density evaluation is 0 then no need to consider this shadow ray */
            if (ls.pdf <= 0.0f) continue;

            RTCRayHit shadow;
            initRayHit(shadow, dg.P, ls.dir, dg.eps, ls.dist, time);
            rtcOccluded1(g_scene, &context, &shadow.ray);
            if (shadow.ray.tfar >= 0.0f) {
                L = L + Lw * ls.weight * Material__eval(albedo, materialType, wo, dg, ls.dir);
            }
        }

        if (wi1.pdf <= 1E-4f) break;
        Lw = Lw * c / wi1.pdf;


        /* setup secondary ray */
        float sign = dot(wi1.v, dg.Ng) < 0.0f ? -1.0f : 1.0f;
        dg.P = dg.P + sign * dg.eps * dg.Ng;
        org = dg.P;
        dir = normalize(wi1.v);
        initRayHit(rayhit, org, dir, dg.eps, std::numeric_limits<float>::infinity(), time);
    }
    return L;

}

/* task that renders a single screen pixel */
Vec3fa renderPixelStandard(int x, int y, 
    const unsigned int width, const unsigned int height,
    const float time,
    const ISPCCamera& camera) {

    std::default_random_engine reng;
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

    Vec3fa L = Vec3fa(0.0f);

    for (int i = 0; i < g_spp; i++)
    {
        reng.seed(g_accu_count * (y * width +x) * g_spp + (y * width + x) *  g_spp + i);

        /* calculate pixel color */
        float fx = x + distrib(reng); 
        float fy = y + distrib(reng); 
        L = L + renderPixelFunction(fx, fy, width, height, reng, distrib, time, camera);
//#define MY_DEBUG
#ifdef MY_DEBUG
        if (max(max(L.x, L.y), L.z) > 0.0f)
            printf("Hit pixel at %f %f: %f %f %f\n", fx, fy, L.x, L.y, L.z);
#endif
    }
    L = L / (float)g_spp;
    return L;
    
}


/* task that renders a single screen tile */
void renderTileTask(int taskIndex, int threadIndex, unsigned char* pixels, std::vector<std::shared_ptr<Vec3ff>>& accu,
                    const unsigned int width, const unsigned int height,
                    const unsigned int channels, const float time,
                    const ISPCCamera& camera, const int numTilesX,
                    const int numTilesY) {
  const unsigned int tileY = taskIndex / numTilesX;
  const unsigned int tileX = taskIndex - tileY * numTilesX;
  const unsigned int x0 = tileX * TILE_SIZE_X;
  const unsigned int x1 = min(x0 + TILE_SIZE_X, width);
  const unsigned int y0 = tileY * TILE_SIZE_Y;
  const unsigned int y1 = min(y0 + TILE_SIZE_Y, height);

  for (unsigned int y = y0; y < y1; y++)
    for (unsigned int x = x0; x < x1; x++) {
      Vec3fa color = renderPixelStandard(x, y, width, height, time, camera);
//#define MY_DEBUG
#ifdef MY_DEBUG
      if (max(max(color.x, color.y), color.z) > 0.0f)
        printf("Hit pixel at %u %u: %f %f %f\n", x, y, color.x, color.y, color.z);
#endif
      /* write color to accumulation buffer */

      Vec3ff accu_color = *accu[y * width + x] + Vec3ff(color.x, color.y, color.z, 1.0f); *accu[y * width + x] = accu_color;
      float f = rcp(max(0.001f, accu_color.w));
      
      /* write color from accumulation buffer to framebuffer */
      unsigned char r =
          (unsigned char)(255.0f * rkcommon::math::clamp(accu_color.x*f, 0.0f, 1.0f));
      unsigned char g =
          (unsigned char)(255.0f * rkcommon::math::clamp(accu_color.y*f, 0.0f, 1.0f));
      unsigned char b =
          (unsigned char)(255.0f * rkcommon::math::clamp(accu_color.z*f, 0.0f, 1.0f));
      pixels[y * width * channels + x * channels] = r;
      pixels[y * width * channels + x * channels + 1] = g;
      pixels[y * width * channels + x * channels + 2] = b;
    }
}

/* called by the C++ code to render */
void renderFrameStandard(unsigned char* pixels, std::vector<std::shared_ptr<Vec3ff>>& accu, const unsigned int width,
                         const unsigned int height, const unsigned int channels,
                         const float time, const ISPCCamera& camera) {
  const int numTilesX = (width + TILE_SIZE_X - 1) / TILE_SIZE_X;
  const int numTilesY = (height + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
  tbb::task_group_context tgContext;
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, numTilesX * numTilesY, 1),
      [&](const tbb::blocked_range<size_t>& r) {
        const int threadIndex = tbb::this_task_arena::current_thread_index();
        for (size_t i = r.begin(); i < r.end(); i++)
          renderTileTask((int)i, threadIndex, pixels, accu, width, height, channels,
                         time, camera, numTilesX, numTilesY);
      },
      tgContext);
  if (tgContext.is_group_execution_cancelled())
    throw std::runtime_error("fail: oneTBB task cancelled");

  g_accu_count++;
}

/* called by the C++ code for cleanup */
void device_cleanup() {
  rtcReleaseScene(g_scene);
  g_scene = nullptr;
  if (g_cube_face_colors) alignedFree(g_cube_face_colors);
  g_cube_face_colors = nullptr;
  if (g_cube_vertex_colors) alignedFree(g_cube_vertex_colors);
  g_cube_vertex_colors = nullptr;
  if (g_ground_face_colors) alignedFree(g_ground_face_colors);
  g_ground_face_colors = nullptr;
  if (g_ground_vertex_colors) alignedFree(g_ground_vertex_colors);
  g_ground_vertex_colors = nullptr;

  if (g_cornell_face_colors) alignedFree(g_cornell_face_colors);
  g_cornell_face_colors = nullptr;
  if (g_cornell_vertex_colors) alignedFree(g_cornell_vertex_colors);
  g_cornell_vertex_colors = nullptr;
}

void device_init(char* cfg) {
  /* create scene */
  g_scene = nullptr;
  g_scene = rtcNewScene(g_device);

  switch (g_sceneSelector) {
      case SceneSelector::SHOW_CORNELL_BOX:
        /* add cornell box */
        geomIDs.push_back(addCornell(g_scene, g_device, &g_cornell_face_colors, &g_cornell_vertex_colors));
        break;
      case SceneSelector::SHOW_CUBE_AND_PLANE:
      default:
        /* add cube */
        geomIDs.push_back(addCube(g_scene));
        /* add ground plane */
        geomIDs.push_back(addGroundPlane(g_scene));
        break;

  }


  /* commit changes to scene */
  rtcCommitScene(g_scene);
}

int main() {
  /* create device */
  g_device = rtcNewDevice(nullptr);
  error_handler(nullptr, rtcGetDeviceError(g_device),
                "fail: Embree Error Unable to create embree device");

  /* set error handler */
  rtcSetDeviceErrorFunction(g_device, error_handler, nullptr);

  g_sceneSelector = SceneSelector::SHOW_CORNELL_BOX;
  device_init(nullptr);

  /* create an image buffer initialize it with all zeroes */
//  const unsigned int width = 320;
//  const unsigned int height = 200;

  const unsigned int width = 640;
  const unsigned int height = 480;
  const unsigned int channels = 3;

  g_pixels = (unsigned char*)new unsigned char[width * height * channels];
  std::memset(g_pixels, 0, sizeof(unsigned char) * width * height * channels);

  g_accu.resize(width * height);
  for (auto i = 0; i < width * height; i++)
      g_accu[i] = std::make_shared<Vec3ff>(0.0f);

  /* accumulation buffer used for introduction to future applications */
  const unsigned int g_accu_limit = 1;
  g_spp = 4;
  g_max_path_length = 8;
  Vec3fa defaultLightDirection = normalize(Vec3fa(-1.0f, -1.0f, -1.0f));
  Vec3fa defaultLightIntensity = { 1.0f, 1.0f, 1.0f };

  switch (g_sceneSelector) {
  case SceneSelector::SHOW_CORNELL_BOX:
      g_camera = positionCamera(Vec3fa(0.0, 0.0, -2.0f), Vec3fa(0, 0, 0),
          Vec3fa(0, 1, 0), 90.0f, width, height);
      g_infDirectionalLights.resize(1);
      //g_infDirectionalLights[0].dir = normalize(Vec3fa( -0.2f, -0.9f, -0.2f));
      g_infDirectionalLights[0].dir = normalize(Vec3fa(0.0f, 0.0f, 2.0f));
      //g_infDirectionalLights[0].intensity = 4*Vec3fa(0.78f, 0.551f, 0.183f);
      g_infDirectionalLights[0].intensity = 4 * defaultLightIntensity;
      break;
  case SceneSelector::SHOW_CUBE_AND_PLANE:
  default:
      g_camera = positionCamera(Vec3fa(1.5f, 1.5, -1.5f), Vec3fa(0, 0, 0),
          Vec3fa(0, 1, 0), 90.0f, width, height);
      g_infDirectionalLights.resize(1);
      g_infDirectionalLights[0].dir = defaultLightDirection;
      g_infDirectionalLights[0].intensity = 4*defaultLightIntensity;
      break;
  }


 // g_camera = positionCamera(Vec3fa(0, 1.5f, 0), Vec3fa(0, 0, 0),
 //     Vec3fa(0, 1, 0), 90.0f, width, height);

  renderFrameStandard(g_pixels, g_accu, width, height, channels, 0.0, g_camera);
  stbi_write_png("pathtracer_single_oneapi.png", width, height, channels,
                 g_pixels, width * channels);

  for (unsigned int i = 1; i < g_accu_limit; i++)
      renderFrameStandard(g_pixels, g_accu, width, height, channels, 0.0, g_camera);

  stbi_write_png("pathtracer_accu_oneapi.png", width, height, channels,
      g_pixels, width * channels);

  delete[] g_pixels;
  g_pixels = nullptr;

  device_cleanup();
  rtcReleaseDevice(g_device);
  printf("success\n");
  return 0;
}
