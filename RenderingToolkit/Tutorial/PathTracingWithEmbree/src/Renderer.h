#pragma once

#ifndef FILE_RENDERER_SEEN
#define FILE_RENDERER_SEEN

#include <embree3/rtcore.h>
#include <tbb/parallel_for.h>

#include <random>

#include "PathTracer.h"
#include "SceneGraph.h"
#include "definitions.h"

struct Renderer {
 public:
  Renderer(unsigned int width, unsigned int height, unsigned int channels,
           unsigned int samples_per_pixel, unsigned int accumulation_limit,
           unsigned int max_path_length, SceneSelector SELECTED_SCENE);
  ~Renderer();

  static void Renderer::handle_error(void* userPtr, const RTCError code,
                                     const char* str);

  void Renderer::init_device(const char* cfg = nullptr);

  void Renderer::init_scene(char* cfg, unsigned int width, unsigned int height);

  /* called by the C++ code to render */
  void Renderer::render_accumulation();

  /* task that renders a single screen tile */
  void Renderer::render_tile_task(
      int taskIndex, int threadIndex, const int numTilesX, const int numTilesY,
      RandomEngine& reng, std::uniform_real_distribution<float>& distrib);

  Vec3fa Renderer::render_pixel_samples(
      int x, int y, RandomEngine& reng,
      std::uniform_real_distribution<float>& distrib);

  unsigned char* Renderer::get_pixels();

  unsigned char* m_pixels = nullptr;

 private:
  std::shared_ptr<PathTracer> m_pt;
  /* We might want to use this function outside of our path tracer at somepoint
   */
  inline Vec3fa face_forward(const Vec3fa& dir, const Vec3fa& _Ng);

  /* create an image buffer initialize it with all zeroes */
  const unsigned int m_width;
  const unsigned int m_height;
  const unsigned int m_channels;

  RTCDevice m_device = nullptr;

  std::shared_ptr<SceneGraph> m_sg;

  /* Additions for pathtracer */
  std::vector<std::shared_ptr<Vec3ff>> m_accu;
  unsigned int m_accu_count = 0;
  unsigned int m_max_path_length;
  unsigned int m_spp;
  SceneSelector m_sceneSelector;
  std::vector<Light> m_lights;
  unsigned long long m_accu_limit;

  /* "Time" set to 0.0f for all rays as there is no motion blur, nor frame
   * interpolation, nor animation */
  const float m_time = 0.0f;
};

Renderer::Renderer(unsigned int width, unsigned int height,
                   unsigned int channels, unsigned int samples_per_pixel,
                   unsigned int accumulation_limit,
                   unsigned int max_path_length, SceneSelector SELECTED_SCENE)
    : m_width(width),
      m_height(height),
      m_channels(channels),
      m_spp(samples_per_pixel),
      m_accu_limit(accumulation_limit),
      m_max_path_length(max_path_length),
      m_sceneSelector(SELECTED_SCENE) {
  m_accu_count = 0;
  m_pixels = (unsigned char*)new unsigned char[m_width * m_height * m_channels];
  std::memset(m_pixels, 0,
              sizeof(unsigned char) * m_width * m_height * m_channels);

  /* accumulation buffer used for convenience here, but is critical in
   * interactive/future applications */
  m_accu.resize(m_width * m_height);
  for (auto i = 0; i < m_width * m_height; i++)
    m_accu[i] = std::make_shared<Vec3ff>(0.0f);

  init_device(nullptr);

  init_scene(nullptr, m_width, m_height);

  // m_pt = std::make_shared<PathTracer>(max_path_length);
  //  For Multiple Importance sampling we need per pixel storage for light PDFs
  m_pt = std::make_shared<PathTracer>(max_path_length, m_width, m_height,
                                      m_sg->getNumLights());
}

void Renderer::handle_error(void* userPtr, const RTCError code,
                            const char* str) {
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

void Renderer::init_device(const char* cfg) {
  /* create device */
  m_device = rtcNewDevice(nullptr);
  handle_error(nullptr, rtcGetDeviceError(m_device),
               "fail: Embree Error Unable to create embree device");

  /* set error handler */
  rtcSetDeviceErrorFunction(m_device, handle_error, nullptr);
}

void Renderer::init_scene(char* cfg, unsigned int width, unsigned int height) {
  m_sg = std::make_shared<SceneGraph>(m_device, m_sceneSelector, m_width,
                                      m_height);
}

/* called by the C++ code to render */
void Renderer::render_accumulation() {
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
void Renderer::render_tile_task(
    int taskIndex, int threadIndex, const int numTilesX, const int numTilesY,
    RandomEngine& reng, std::uniform_real_distribution<float>& distrib) {
  const unsigned int tileY = taskIndex / numTilesX;
  const unsigned int tileX = taskIndex - tileY * numTilesX;
  const unsigned int x0 = tileX * TILE_SIZE_X;
  const unsigned int x1 = min(x0 + TILE_SIZE_X, m_width);
  const unsigned int y0 = tileY * TILE_SIZE_Y;
  const unsigned int y1 = min(y0 + TILE_SIZE_Y, m_height);

  for (unsigned int y = y0; y < y1; y++)
    for (unsigned int x = x0; x < x1; x++) {
      Vec3fa color = render_pixel_samples(x, y, reng, distrib);

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
Vec3fa Renderer::render_pixel_samples(
    int x, int y, RandomEngine& reng,
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
    visual artifacts. The Embree Renderer full example uses an LCG-based
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
    L = L + m_pt->render_path(fx, fy, reng, distrib, m_sg, y * m_width + x);
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

unsigned char* Renderer::get_pixels() { return m_pixels; }

Renderer::~Renderer() {
  if (m_pixels) {
    delete m_pixels;
    m_pixels = nullptr;
  }
}

#endif /* FILE_RENDERER_SEEN */