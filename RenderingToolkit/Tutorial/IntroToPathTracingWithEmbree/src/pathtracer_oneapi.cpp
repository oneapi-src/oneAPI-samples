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
#include <string>
#include <chrono>

#include "definitions.h"
#include "DefaultCubeAndPlane.h"
#include "CornellBox.h"

using std::string;
/* minstd_rand is much faster for this application than ranlux48_base/ranlux24_base.
In turn, ranlux was faster than mt19937 or mt19937_64 on Windows
*/
typedef std::minstd_rand RandomEngine;

std::vector<unsigned int> g_geomIDs;

RTCDevice g_device = nullptr;
RTCScene g_scene = nullptr;

unsigned char* g_pixels;

/* Additions for pathtracer */
std::vector<std::shared_ptr<Vec3ff>> g_accu;
unsigned int g_accu_count = 0;
unsigned int g_max_path_length;
unsigned int g_spp;
SceneSelector g_sceneSelector;
std::vector<Light> g_lights;
unsigned long long g_accu_limit;

AffineSpace3fa g_camera;

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
LinearSpace3fa frame(const Vec3fa& N) {
    const Vec3fa dx0(0, N.z, -N.y);
    const Vec3fa dx1(-N.z, 0, N.x);

    const Vec3fa dx = normalize((dot(dx0, dx0) > dot(dx1, dx1)) ? dx0 : dx1);
    const Vec3fa dy = normalize(cross(N, dx));

    return LinearSpace3fa(dx, dy, N);
}

/*! Cosine weighted hemisphere sampling. Up direction is provided as argument. */
inline Sample3f cosineSampleHemisphere(const float  u, const float  v, const Vec3fa& N)
{
    /* Determine cartesian coordinate for new Vec3fa */
    const float phi = float(2.0 * M_PI) * u;
    const float cosTheta = sqrt(v);
    const float sinTheta = sqrt(1.0f - v);
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
    return R * (1.f / (float)(float(M_PI))) * clamp(dot(wi_v, dg.Ns));
}

Vec3fa Mirror__sample(const Vec3fa& R, const Vec3fa& Lw, const Vec3fa& wo, const DifferentialGeometry& dg, Sample3f& wi) {
    Sample3f sam;
    sam.pdf = 1.0f;
    /* Compute a reflection vector 2 * N.L * N - L */
    sam.v = normalize(2.0f * dot(wo, dg.Ns) * dg.Ns - wo);
    wi = sam;
    return R;
}

Vec3fa Material__sample(Vec3fa R, enum class MaterialType materialType, Vec3fa Lw, Vec3fa wo, DifferentialGeometry dg, Sample3f& wi, Vec2f randomMatSample) {
    Vec3fa c = Vec3fa(0.0f);
    switch (materialType) {
    case MaterialType::MATERIAL_MATTE:
        wi = cosineSampleHemisphere(randomMatSample.x, randomMatSample.y, dg.Ns);
        return Lambertian__eval(R, wo, dg, wi.v);
        break;

    case MaterialType::MATERIAL_MIRROR:
        return Mirror__sample(R, Lw, wo, dg, wi);
        break;
/* Return our debug color if something goes awry */
    default: c = R;
        break;
    }

    return c;
}

Vec3fa Material__eval(Vec3fa R, enum class MaterialType materialType, const Vec3fa& wo, const DifferentialGeometry& dg, const Vec3fa& wi) {
    Vec3fa c = Vec3fa(0.0f);
    switch (materialType) {
    case MaterialType::MATERIAL_MATTE:
        return Lambertian__eval(R, wo, dg, wi);
        break;
    case MaterialType::MATERIAL_MIRROR:
        return Vec3fa(0.0f);
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

Light_SampleRes sample_light(const Light& light, DifferentialGeometry& dg, const Vec2f& randomLightSample) {
    Light_SampleRes res;

    switch (light.type) {
    case LightType::INFINITE_DIRECTIONAL_LIGHT:
        res.dir = -light.dir;
        res.dist = std::numeric_limits<float>::infinity();
        res.pdf = std::numeric_limits<float>::infinity();
        res.weight = light.intensity; // *pdf/pdf cancel
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

        res.pdf = std::numeric_limits<float>::infinity(); // per default we always take this res

        // convert from power to radiance by attenuating by distance^2
        res.weight = light.intensity * (invdist * invdist);
        break;
    }

    return res;
}

/* task that renders a single screen pixel */
Vec3fa renderPixelFunction(float x, float y,
    const unsigned int width, const unsigned int height,
    RandomEngine& reng, std::uniform_real_distribution<float>& distrib,
    const float time,
    const AffineSpace3fa& camera) {
    RTCIntersectContext context;
    rtcInitIntersectContext(&context);
    Vec3fa dir = normalize(x * camera.l.vx +
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
        if (max(Lw.x, max(Lw.y, Lw.z)) < 0.01f)
            break;

        /* intersect ray with scene */
        context.flags = (i == 0) ? RTCIntersectContextFlags::RTC_INTERSECT_CONTEXT_FLAG_COHERENT : RTCIntersectContextFlags::RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;
        rtcIntersect1(g_scene, &context, &rayhit);
        const Vec3fa wo = -dir;

        /* if nothing hit the path is terminated, this could be an light lookup insteead */
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

        /* Next is material discovery.
        * Note: the full pathtracer program includes lookup of materials from a scenegraph object.
        * This could include texture lookup transformations that are based on vertex-to-texture assignments.
        * This could include a required tranformation of normals for geometries that have been instanced. Below is a simple option for materials.
        */

        /* default albedo is a pink color for debug */
        Vec3fa albedo = Vec3fa(0.9f, 0.7f, 0.7f);
        enum class MaterialType materialType = MaterialType::MATERIAL_MATTE;
        /* An albedo as well as a material type is used */
        switch (g_sceneSelector) {
        case SceneSelector::SHOW_CORNELL_BOX:
            if (rayhit.hit.geomID == g_geomIDs[0]) {
                albedo = g_cornell_face_colors[rayhit.hit.primID];
                materialType = cornellBoxMats[rayhit.hit.primID];
            }
            else if (rayhit.hit.geomID == g_geomIDs[1]) {
                albedo = Vec3fa(1.0f, 1.0f, 1.0f);
                materialType = MaterialType::MATERIAL_MATTE;
            }
            break;
        case SceneSelector::SHOW_CUBE_AND_PLANE:
        default:
            if (rayhit.hit.geomID == g_geomIDs[0]) {
                albedo = g_cube_face_colors[rayhit.hit.primID];
                materialType = MaterialType::MATERIAL_MATTE;
            }
            else if (rayhit.hit.geomID == g_geomIDs[1]) {
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

        /* Search for each light in the scene from our hit point. Aggregate the radiance if hit point is not occluded */
        context.flags = RTCIntersectContextFlags::RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;
        for(const Light& light : g_lights)
        {
            Vec2f randomLightSample(distrib(reng), distrib(reng));
            Light_SampleRes ls = sample_light(light, dg, randomLightSample);
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
    const AffineSpace3fa& camera, RandomEngine& reng, std::uniform_real_distribution<float>& distrib) {

    Vec3fa L = Vec3fa(0.0f);

    for (int i = 0; i < g_spp; i++)
    {
        /* Generate a new seed for a sample. 
        Ray cast position for the sample, materials, and light look ups all use random 0. <-> 1.0 floating point numbers based off this seed. 
        There are better and worse solutions for randomization in rendering depending on your use case. This one is simple to follow.
        
        Note that here 4,294,967,296 distinct seeds are used based on the pixel and sample being computed. At 3840x2160 resolution with 517 samples per pixel, a random number used indiscriminantly, *could* repeat. The result could be visual artifacts.
        The Embree pathtracer full example uses an LCG-based randomizer algorithm written by hand.

        If each sample is given it's own id a0nd it is fed into the randomizer, minstd_rand, you may notice repeated visual patterns in this tutorial. So, below a hash is used.
            function: sample at the start of accumulation + sample at the start of pixel + sample
        id example: unsigned int hseed = g_accu_count * (width * height) * g_spp + (y * width + x) * g_spp + i
        */

        unsigned long long seed = (((unsigned long long)(y * width + x) << 32) | ((g_accu_count * g_spp + i) & 0xFFFFFFFF));
        size_t hseed = std::hash<unsigned long long>{}(seed);

        // Take a 32bit xor for seeding the random number generator with a "unique" enough value in a 32bit size. minstd_rand takes a 32-bit seed.
        if(sizeof(size_t) == 8 ) 
            hseed = (hseed >> 32) ^ (hseed & 0xFFFFFFF);

        reng.seed(hseed);
        /* calculate pixel color, slightly offset the ray cast orientation randomly so each sample occurs at a slightly different location within a pixel */
        /* Note: random offsets for samples within a pixel provide natural anti-aliasing (smoothing) near object edges */
        float fx = x + distrib(reng); 
        float fy = y + distrib(reng); 
        L = L + renderPixelFunction(fx, fy, width, height, reng, distrib, time, camera);
/* If you are not seeing anything, try some printf debug */
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
                    const AffineSpace3fa& camera, const int numTilesX,
                    const int numTilesY, RandomEngine& reng, std::uniform_real_distribution<float>& distrib) {
  const unsigned int tileY = taskIndex / numTilesX;
  const unsigned int tileX = taskIndex - tileY * numTilesX;
  const unsigned int x0 = tileX * TILE_SIZE_X;
  const unsigned int x1 = min(x0 + TILE_SIZE_X, width);
  const unsigned int y0 = tileY * TILE_SIZE_Y;
  const unsigned int y1 = min(y0 + TILE_SIZE_Y, height);


  for (unsigned int y = y0; y < y1; y++)
    for (unsigned int x = x0; x < x1; x++) {
      Vec3fa color = renderPixelStandard(x, y, width, height, time, camera, reng, distrib);
      
      /* In case you run into issues with visibility try manual debug */
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
          (unsigned char)(255.0f * clamp(accu_color.x*f, 0.0f, 1.0f));
      unsigned char g =
          (unsigned char)(255.0f * clamp(accu_color.y*f, 0.0f, 1.0f));
      unsigned char b =
          (unsigned char)(255.0f * clamp(accu_color.z*f, 0.0f, 1.0f));
      pixels[y * width * channels + x * channels] = r;
      pixels[y * width * channels + x * channels + 1] = g;
      pixels[y * width * channels + x * channels + 2] = b;
    }
}

/* called by the C++ code to render */
void renderFrameStandard(unsigned char* pixels, std::vector<std::shared_ptr<Vec3ff>>& accu, const unsigned int width,
                         const unsigned int height, const unsigned int channels,
                         const float time, const AffineSpace3fa& camera) {
  const int numTilesX = (width + TILE_SIZE_X - 1) / TILE_SIZE_X;
  const int numTilesY = (height + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
  tbb::task_group_context tgContext;
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, numTilesX * numTilesY, 1),
      [&](const tbb::blocked_range<size_t>& r) {
        const int threadIndex = tbb::this_task_arena::current_thread_index();

        RandomEngine reng;
        std::uniform_real_distribution<float> distrib(0.0f, 1.0f);
        
        for (size_t i = r.begin(); i < r.end(); i++) {
            
            renderTileTask((int)i, threadIndex, pixels, accu, width, height, channels,
                time, camera, numTilesX, numTilesY, reng, distrib);
        }
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
  switch (g_sceneSelector) {
  case SceneSelector::SHOW_CORNELL_BOX:
      cleanCornell();
      break;
  case SceneSelector::SHOW_CUBE_AND_PLANE:
  default:
    cleanCubeAndPlane();
    break;

  }
}

void device_init(char* cfg, unsigned int width, unsigned int height) {
  /* create scene */
  g_scene = nullptr;
  g_scene = rtcNewScene(g_device);

  switch (g_sceneSelector) {
      case SceneSelector::SHOW_CORNELL_BOX:
        /* add cornell box */
        g_geomIDs.push_back(addCornell(g_scene, g_device));
        /* If you would like to add an Embree sphere see addSphere(..) as used below for an example */
        //g_geomIDs.push_back(addSphere(g_scene, g_device));

        cornellCameraLightSetup(g_camera, g_lights, width, height);
        break;
      case SceneSelector::SHOW_CUBE_AND_PLANE:
      default:
        /* add cube */
        g_geomIDs.push_back(addCube(g_scene, g_device));
        /* add ground plane */
        g_geomIDs.push_back(addGroundPlane(g_scene, g_device));

        cubeAndPlaneCameraLightSetup(g_camera, g_lights, width, height);
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

    /* create an image buffer initialize it with all zeroes */
    const unsigned int width = 512;
    const unsigned int height = 512;
    const unsigned int channels = 3;

    g_pixels = (unsigned char*)new unsigned char[width * height * channels];
    std::memset(g_pixels, 0, sizeof(unsigned char) * width * height * channels);

    /* accumulation buffer used for convenience here, but is critical in interactive/future applications */
    g_accu.resize(width * height);
    for (auto i = 0; i < width * height; i++)
        g_accu[i] = std::make_shared<Vec3ff>(0.0f);

    //g_sceneSelector = SceneSelector::SHOW_CUBE_AND_PLANE;
    g_sceneSelector = SceneSelector::SHOW_CORNELL_BOX;
    device_init(nullptr, width, height);

    /* Control the total number of accumulations, the total number of samples per pixel per accumulation, and the maximum path length of any given traced path.*/
    const unsigned long long g_accu_limit = 500;
    g_spp = 1;
    g_max_path_length = 8;

    /* Use a basic timer to capure compute time per accumulation */
    auto start = std::chrono::high_resolution_clock::now();
    renderFrameStandard(g_pixels, g_accu, width, height, channels, 0.0, g_camera);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> accum_time = end - start;
    printf("Accumulation 1 of %llu: %lf s\n", g_accu_limit, accum_time.count());

    /* Label construction for output files */
    string strscene;
    (g_sceneSelector == SceneSelector::SHOW_CORNELL_BOX) ? strscene = "-cornell" : strscene = "-default";
    string suffix = string("-spp") + std::to_string(g_spp) + string("-plength") + std::to_string(g_max_path_length) + string("-accu") + std::to_string(g_accu_limit) + string("-") + std::to_string(width) + string("x") + std::to_string(height) + string(".png");
    string prefix = "pathtracer_single_oneapi";
    string filename = prefix + strscene + suffix;

    /* Write a single accumulation image (useful for comparison) */
    stbi_write_png(filename.c_str(), width, height, channels,
        g_pixels, width * channels);

    /* Render all remaining accumulations (in addition to the first) */
    for (unsigned long long i = 1; i < g_accu_limit; i++) {
        start = std::chrono::high_resolution_clock::now();
        renderFrameStandard(g_pixels, g_accu, width, height, channels, 0.0, g_camera);
        end = std::chrono::high_resolution_clock::now();
        accum_time = end - start;
        printf("Accumulation %llu of %llu: %lf s\n", i+1, g_accu_limit, accum_time.count());
    }
    /* Label construction for the accumulated output */
    prefix = "pathtracer_accu_oneapi";
    filename = prefix + strscene + suffix;

    /* Write the accumulated image output */
  stbi_write_png(filename.c_str(), width, height, channels,
      g_pixels, width * channels);

  delete[] g_pixels;
  g_pixels = nullptr;

  device_cleanup();
  rtcReleaseDevice(g_device);
  printf("success\n");
  return 0;
}
