#pragma once

#ifndef FILE_RENDERER_SEEN
#define FILE_RENDERER_SEEN

#include <embree4/rtcore.h>
#include <tbb/parallel_for.h>

#include "PathTracer.h"
#include "SceneGraph.h"
#include "definitions.h"

struct Renderer {
 public:
  Renderer(unsigned int width, unsigned int height, unsigned int channels,
           unsigned int samples_per_pixel, unsigned int accumulation_limit,
           unsigned int max_path_length, SceneSelector SELECTED_SCENE);
  ~Renderer();

  static void handle_error(void* userPtr, const RTCError code,
                                     const char* str);

  void init_device(const char* cfg = nullptr);

  void init_scene(char* cfg, unsigned int width, unsigned int height);

  /* called by the C++ code to render */
  void render_accumulation();

  /* task that renders a single screen tile */
  void render_tile_task(
      int taskIndex, int threadIndex, const int numTilesX, const int numTilesY,
      RandomSampler& randomSampler);

  Vec3fa render_pixel_samples(
      int x, int y, RandomSampler& randomSampler);

  unsigned char* get_pixels();

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

        RandomSampler randomSampler;

        for (size_t i = r.begin(); i < r.end(); i++) {
          render_tile_task((int)i, threadIndex, numTilesX, numTilesY, randomSampler);
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
    RandomSampler& randomSampler) {
  const unsigned int tileY = taskIndex / numTilesX;
  const unsigned int tileX = taskIndex - tileY * numTilesX;
  const unsigned int x0 = tileX * TILE_SIZE_X;
  const unsigned int x1 = min(x0 + TILE_SIZE_X, m_width);
  const unsigned int y0 = tileY * TILE_SIZE_Y;
  const unsigned int y1 = min(y0 + TILE_SIZE_Y, m_height);

  for (unsigned int y = y0; y < y1; y++)
    for (unsigned int x = x0; x < x1; x++) {
      Vec3fa Lsample = render_pixel_samples(x, y, randomSampler);

      /* In case you run into issues with visibility try manual debug */
      //#define MY_DEBUG
#ifdef MY_DEBUG
      if (max(max(color.x, color.y), color.z) > 0.0f)
        std::cout << "Hit pixel at :" << x << " , " << y << ": " << color.x
                  << " " << color.y << " " << color.z << "\n";
#endif
      /* write color to accumulation buffer */

      Vec3ff accu_color =
          *m_accu[y * m_width + x] + Vec3ff(Lsample.x, Lsample.y, Lsample.z, 1.0f);
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
    int x, int y, RandomSampler& randomSampler) {
  Vec3fa L = Vec3fa(0.0f);

  for (int i = 0; i < m_spp; i++) {
    /* Generate a new seed for a sample.
    Make sure to use a portable random number generator when necessary. This application uses a hand coded LCG generator from the Embree repository.
    Be careful when considering a cryptographic random number input for one of the C++11 and later random number generators in the <random> library.
    std::hash behavior is implementation defined, therefore using it to seed your random number generator may or may not give a good distribution.
    */


    randomSampler.seed(x, y, m_accu_count * m_spp + i);
    /* calculate pixel color, slightly offset the ray cast orientation randomly
     * so each sample occurs at a slightly different location within a pixel */
    /* Note: random offsets for samples within a pixel provide natural
     * anti-aliasing (smoothing) near object edges */
    float fx = x + randomSampler.get_float();
    float fy = y + randomSampler.get_float();
    L = L + m_pt->render_path(fx, fy, randomSampler, m_sg, y * m_width + x);
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
