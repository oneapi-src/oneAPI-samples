#pragma once
#ifndef FILE_PATHTRACERSEEN
#define FILE_PATHTRACERSEEN

#include "SceneGraph.h"

struct PathTracer {
public:
    PathTracer(unsigned int width, unsigned int height, unsigned int channels, unsigned int samples_per_pixel, unsigned int accumulation_limit, unsigned int max_path_length, SceneSelector SELECTED_SCENE);

    ~PathTracer();

    /* minstd_rand is much faster for this application than
    ranlux48_base/ranlux24_base. In turn, ranlux was faster than mt19937 or
    mt19937_64 on Windows
    */
    typedef std::minstd_rand RandomEngine;

    static void PathTracer::handle_error(void* userPtr, const RTCError code, const char* str);

    void PathTracer::init_device(const char* cfg = nullptr);

    void PathTracer::init_scene(char* cfg, unsigned int width, unsigned int height);

    /* Added for pathtracer: Initializes a rayhit data structure. Used for embree to perform ray triangle intersect */
    inline void PathTracer::init_RayHit(RTCRayHit& rayhit, const Vec3fa& org, const Vec3fa& dir,
        float tnear, float tfar, float time);

    /* called by the C++ code to render */
    void PathTracer::render_accumulation();

    /* task that renders a single screen tile */
    void PathTracer::render_tile_task(int taskIndex, int threadIndex, const int numTilesX,
        const int numTilesY, RandomEngine& reng,
        std::uniform_real_distribution<float>& distrib);

    Vec3fa PathTracer::render_pixel_samples(int x, int y, RandomEngine& reng,
        std::uniform_real_distribution<float>& distrib);

    /* task that renders a single screen pixel */
    Vec3fa PathTracer::render_path(float x, float y, RandomEngine& reng,
        std::uniform_real_distribution<float>& distrib);

    unsigned char* PathTracer::get_pixels();

private:
    /* We might want to use this function outside of our path tracer at somepoint */
    inline Vec3fa face_forward(const Vec3fa& dir, const Vec3fa& _Ng);

    /* create an image buffer initialize it with all zeroes */
    const unsigned int m_width;
    const unsigned int m_height;
    const unsigned int m_channels;

    RTCDevice m_device = nullptr;

    std::shared_ptr<SceneGraph> m_sg;

    unsigned char* m_pixels = nullptr;

    /* Additions for pathtracer */
    std::vector<std::shared_ptr<Vec3ff>> m_accu;
    unsigned int m_accu_count = 0;
    unsigned int m_max_path_length;
    unsigned int m_spp;
    SceneSelector m_sceneSelector;
    std::vector<Light> m_lights;
    unsigned long long m_accu_limit;

    AffineSpace3fa m_camera;
    
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

PathTracer::PathTracer(unsigned int width, unsigned int height, unsigned int channels, unsigned int samples_per_pixel, unsigned int accumulation_limit, unsigned int max_path_length, SceneSelector SELECTED_SCENE) : m_width(width), m_height(height), m_channels(channels), m_spp(samples_per_pixel), m_accu_limit(accumulation_limit), m_max_path_length(max_path_length) {
    m_accu_count = 0;
    m_pixels = (unsigned char*)new unsigned char[m_width * m_height * m_channels];
    std::memset(m_pixels, 0, sizeof(unsigned char) * m_width * m_height * m_channels);

    /* accumulation buffer used for convenience here, but is critical in
     * interactive/future applications */
    m_accu.resize(m_width * m_height);
    for (auto i = 0; i < m_width * m_height; i++)
        m_accu[i] = std::make_shared<Vec3ff>(0.0f);

    init_device(nullptr);

    init_scene(nullptr, m_width, m_height);
}

void PathTracer::handle_error(void* userPtr, const RTCError code, const char* str) {
    if (code == RTC_ERROR_NONE) return;

    std::cout << "fail: Embree Error: ";
    switch (code) {
    case RTC_ERROR_UNKNOWN:
        std::cout << "RTC_ERROR_UNKNOWN";
        break;
    case RTC_ERROR_INVALID_ARGUMENT:
        std::cout << "RTC_ERROR_INVALID_ARGUMENT";
        break;
    case RTC_ERROR_INVALID_OPERATION:
        std::cout << "RTC_ERROR_INVALID_OPERATION";
        break;
    case RTC_ERROR_OUT_OF_MEMORY:
        std::cout << "RTC_ERROR_OUT_OF_MEMORY";
        break;
    case RTC_ERROR_UNSUPPORTED_CPU:
        std::cout << "RTC_ERROR_UNSUPPORTED_CPU";
        break;
    case RTC_ERROR_CANCELLED:
        std::cout << "RTC_ERROR_CANCELLED";
        break;
    default:
        std::cout << "invalid error code";
        break;
    }
    if (str) {
        std::cout << " (" << str << ")\n";
    }
    exit(1);
}

void PathTracer::init_device(const char* cfg = nullptr) {
    /* create device */
    m_device = rtcNewDevice(nullptr);
    handle_error(nullptr, rtcGetDeviceError(m_device),
        "fail: Embree Error Unable to create embree device");

    /* set error handler */
    rtcSetDeviceErrorFunction(m_device, handle_error, nullptr);

}

void PathTracer::init_scene(char* cfg, unsigned int width, unsigned int height) {
    m_sg = std::make_shared<SceneGraph>(m_width, m_height);
}

/* Added for pathtracer */
inline void PathTracer::init_RayHit(RTCRayHit& rayhit, const Vec3fa& org, const Vec3fa& dir,
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



/* called by the C++ code to render */
void PathTracer::render_accumulation() {
    const int numTilesX = (m_width + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int numTilesY = (m_height + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
    tbb::task_group_context tgContext;
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, numTilesX * numTilesY, 1),
        [&](const tbb::blocked_range<size_t>& r) {
            const int threadIndex = tbb::this_task_arena::current_thread_index();

            RandomEngine reng;
            std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

            for (size_t i = r.begin(); i < r.end(); i++) {
                render_tile_task((int)i, threadIndex, numTilesX, numTilesY, reng,
                    distrib);
            }
        },
        tgContext);
    if (tgContext.is_group_execution_cancelled())
        throw std::runtime_error("fail: oneTBB task cancelled");

    m_accu_count++;
}

/* task that renders a single screen tile */
void PathTracer::render_tile_task(int taskIndex, int threadIndex, const int numTilesX,
    const int numTilesY, RandomEngine& reng,
    std::uniform_real_distribution<float>& distrib) {
    const unsigned int tileY = taskIndex / numTilesX;
    const unsigned int tileX = taskIndex - tileY * numTilesX;
    const unsigned int x0 = tileX * TILE_SIZE_X;
    const unsigned int x1 = min(x0 + TILE_SIZE_X, m_width);
    const unsigned int y0 = tileY * TILE_SIZE_Y;
    const unsigned int y1 = min(y0 + TILE_SIZE_Y, m_height);

    for (unsigned int y = y0; y < y1; y++)
        for (unsigned int x = x0; x < x1; x++) {
            Vec3fa color =
                render_pixel_samples(x, y, reng, distrib);

            /* In case you run into issues with visibility try manual debug */
            //#define MY_DEBUG
#ifdef MY_DEBUG
            if (max(max(color.x, color.y), color.z) > 0.0f)
                std::cout << "Hit pixel at :" << x << " , " << y << ": " << color.x
                << " " << color.y << " " << color.z << "\n";
#endif
            /* write color to accumulation buffer */

            Vec3ff accu_color =
                *m_accu[y * m_width + x] + Vec3ff(color.x, color.y, color.z, 1.0f);
            *m_accu[y * m_width + x] = accu_color;
            float f = rcp(max(0.001f, accu_color.w));

            /* write color from accumulation buffer to framebuffer */
            unsigned char r =
                (unsigned char)(255.0f * clamp(accu_color.x * f, 0.0f, 1.0f));
            unsigned char g =
                (unsigned char)(255.0f * clamp(accu_color.y * f, 0.0f, 1.0f));
            unsigned char b =
                (unsigned char)(255.0f * clamp(accu_color.z * f, 0.0f, 1.0f));
            m_pixels[y * m_width * m_channels + x * m_channels] = r;
            m_pixels[y * m_width * m_channels + x * m_channels + 1] = g;
            m_pixels[y * m_width * m_channels + x * m_channels + 2] = b;
        }
}

/* task that renders a single screen pixel */
Vec3fa PathTracer::render_pixel_samples(int x, int y, RandomEngine& reng,
    std::uniform_real_distribution<float>& distrib) {
    Vec3fa L = Vec3fa(0.0f);

    for (int i = 0; i < m_spp; i++) {
        /* Generate a new seed for a sample.
        Ray cast position for the sample, materials, and light look ups all use
        random 0. <-> 1.0 floating point numbers based off this seed. There are
        better and worse solutions for randomization in rendering depending on your
        use case. This one is simple to follow.

        Note that here 4,294,967,296 distinct seeds are used based on the pixel and
        sample being computed. At 3840x2160 resolution with 517 samples per pixel, a
        random number used indiscriminantly, *could* repeat. The result could be
        visual artifacts. The Embree pathtracer full example uses an LCG-based
        randomizer algorithm written by hand.

        If each sample is given it's own id a0nd it is fed into the randomizer,
        minstd_rand, you may notice repeated visual patterns in this tutorial. So,
        below a hash is used. function: sample at the start of accumulation + sample
        at the start of pixel + sample id example: unsigned int hseed = g_accu_count
        * (width * height) * g_spp + (y * width + x) * g_spp + i
        */

        unsigned long long seed = (((unsigned long long)(y * m_width + x) << 32) |
            ((m_accu_count * m_spp + i) & 0xFFFFFFFF));
        size_t hseed = std::hash<unsigned long long>{}(seed);

        // Take a 32bit xor for seeding the random number generator with a "unique"
        // enough value in a 32bit size. minstd_rand takes a 32-bit seed.
        if (sizeof(size_t) == 8) hseed = (hseed >> 32) ^ (hseed & 0xFFFFFFF);

        reng.seed(hseed);
        /* calculate pixel color, slightly offset the ray cast orientation randomly
         * so each sample occurs at a slightly different location within a pixel */
         /* Note: random offsets for samples within a pixel provide natural
          * anti-aliasing (smoothing) near object edges */
        float fx = x + distrib(reng);
        float fy = y + distrib(reng);
        L = L +
            render_path(fx, fy, reng, distrib);
        /* If you are not seeing anything, try some printf debug */
        //#define MY_DEBUG
#ifdef MY_DEBUG
        if (max(max(L.x, L.y), L.z) > 0.0f)
            std::cout << "Hit pixel at " << fx << " , " << fy << ": " << L.x << " "
            << L.y << " " << L.z << "\n";
#endif
    }
    L = L / (float)m_spp;

    return L;
}

/* task that renders a single screen pixel */
Vec3fa PathTracer::render_path(float x, float y, RandomEngine& reng,
                           std::uniform_real_distribution<float>& distrib
                           ) {

  Vec3fa dir = normalize(x * m_camera.l.vx + y * m_camera.l.vy + m_camera.l.vz);
  Vec3fa org = Vec3fa(m_camera.p.x, m_camera.p.y, m_camera.p.z);

  /* initialize ray */
  RTCRayHit rayhit;
  init_RayHit(rayhit, org, dir, 0.0f, std::numeric_limits<float>::infinity(),
             m_time);

  Vec3fa L = Vec3fa(0.0f);
  Vec3fa Lw = Vec3fa(1.0f);
  /* Create vaccum medium.This helps for refractions passing through different
   * mediums... like glass(dielectric) material */
  Medium medium;
  medium.transmission = Vec3fa(1.0f);
  medium.eta = 1.f;

  DifferentialGeometry dg;

  m_sg->set_intersect_context_coherent();
  /* iterative path tracer loop */
  for (int i = 0; i < m_max_path_length; i++) {
    /* terminate if contribution too low */
    if (max(Lw.x, max(Lw.y, Lw.z)) < 0.01f) break;

    if (!m_sg->intersect_path_and_scene(org, dir, rayhit, dg)) break;

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

    /* scale down mediums that do not transmit as much light */
    const Vec3fa transmission = medium.transmission;
    if (transmission != Vec3fa(1.0f))
      c = c * Vec3fa(pow(transmission.x, rayhit.ray.tfar),
                     pow(transmission.y, rayhit.ray.tfar),
                     pow(transmission.z, rayhit.ray.tfar));

    Sample3f wi1;
    Vec2f randomMatSample(distrib(reng), distrib(reng));
    const Vec3fa wo = -dir;
    c = c * Material_sample(albedo, materialType, Lw, wo, dg, wi1, medium,
                             randomMatSample);

    /* Search for each light in the scene from our hit point. Aggregate the
     * radiance if hit point is not occluded */
    m_sg->set_intersect_context_incoherent();

    //m_sg->cast_shadow_ray();

    for (const Light& light : m_lights) {
      Vec2f randomLightSample(distrib(reng), distrib(reng));
      Light_SampleRes ls = sample_light(light, dg, randomLightSample);
      /* If the sample probability density evaluation is 0 then no need to
       * consider this shadow ray */
      if (ls.pdf <= 0.0f) continue;

      RTCRayHit shadow;
      init_RayHit(shadow, dg.P, ls.dir, dg.eps, ls.dist, m_time);
      rtcOccluded1(m_scene, &m_context, &shadow.ray);
      if (shadow.ray.tfar >= 0.0f) {
        L = L + Lw * ls.weight *
                    Material__eval(albedo, materialType, wo, dg, ls.dir);
      }
    }

    if (wi1.pdf <= 1E-4f) break;
    Lw = Lw * c / wi1.pdf;

    /* setup secondary ray */
    float sign = dot(wi1.v, dg.Ng) < 0.0f ? -1.0f : 1.0f;
    dg.P = dg.P + sign * dg.eps * dg.Ng;
    org = dg.P;
    dir = normalize(wi1.v);
    init_RayHit(rayhit, org, dir, dg.eps, std::numeric_limits<float>::infinity(),
               m_time);
  }

  return L;
}

unsigned char* PathTracer::get_pixels() {
    return m_pixels;

}

PathTracer::~PathTracer() {
    //nothing here thus far
}


#endif /* FILE_PATHTRACERSEEN */