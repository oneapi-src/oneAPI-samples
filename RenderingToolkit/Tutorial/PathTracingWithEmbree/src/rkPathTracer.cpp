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

/* Additions for pathtracer */
#include <chrono>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "CornellBox.h"
#include "DefaultCubeAndPlane.h"
#include "Sphere.h"
#include "definitions.h"

#include "Materials.h"
#include "Lights.h"
#include "PathTracer.h"

using std::string;
/* minstd_rand is much faster for this application than
ranlux48_base/ranlux24_base. In turn, ranlux was faster than mt19937 or
mt19937_64 on Windows
*/
typedef std::minstd_rand RandomEngine;

RTCDevice g_device = nullptr;
RTCScene g_scene = nullptr;

unsigned char* g_pixels = nullptr;

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

/* task that renders a single screen pixel */
Vec3fa renderPixelStandard(int x, int y, const unsigned int width,
                           const unsigned int height, const float time,
                           const AffineSpace3fa& camera, RandomEngine& reng,
                           std::uniform_real_distribution<float>& distrib) {
  Vec3fa L = Vec3fa(0.0f);

  for (int i = 0; i < g_spp; i++) {
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

    unsigned long long seed = (((unsigned long long)(y * width + x) << 32) |
                               ((g_accu_count * g_spp + i) & 0xFFFFFFFF));
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
        PathTracer_renderPixelFunction(fx, fy, width, height, reng, distrib, time, camera);
/* If you are not seeing anything, try some printf debug */
//#define MY_DEBUG
#ifdef MY_DEBUG
    if (max(max(L.x, L.y), L.z) > 0.0f)
      std::cout << "Hit pixel at " << fx << " , " << fy << ": " << L.x << " "
                << L.y << " " << L.z << "\n";
#endif
  }
  L = L / (float)g_spp;

  return L;
}

/* task that renders a single screen tile */
void renderTileTask(int taskIndex, int threadIndex, unsigned char* pixels,
                    std::vector<std::shared_ptr<Vec3ff>>& accu,
                    const unsigned int width, const unsigned int height,
                    const unsigned int channels, const float time,
                    const AffineSpace3fa& camera, const int numTilesX,
                    const int numTilesY, RandomEngine& reng,
                    std::uniform_real_distribution<float>& distrib) {
  const unsigned int tileY = taskIndex / numTilesX;
  const unsigned int tileX = taskIndex - tileY * numTilesX;
  const unsigned int x0 = tileX * TILE_SIZE_X;
  const unsigned int x1 = min(x0 + TILE_SIZE_X, width);
  const unsigned int y0 = tileY * TILE_SIZE_Y;
  const unsigned int y1 = min(y0 + TILE_SIZE_Y, height);

  for (unsigned int y = y0; y < y1; y++)
    for (unsigned int x = x0; x < x1; x++) {
      Vec3fa color =
          renderPixelStandard(x, y, width, height, time, camera, reng, distrib);

      /* In case you run into issues with visibility try manual debug */
      //#define MY_DEBUG
#ifdef MY_DEBUG
      if (max(max(color.x, color.y), color.z) > 0.0f)
        std::cout << "Hit pixel at :" << x << " , " << y << ": " << color.x
                  << " " << color.y << " " << color.z << "\n";
#endif
      /* write color to accumulation buffer */

      Vec3ff accu_color =
          *accu[y * width + x] + Vec3ff(color.x, color.y, color.z, 1.0f);
      *accu[y * width + x] = accu_color;
      float f = rcp(max(0.001f, accu_color.w));

      /* write color from accumulation buffer to framebuffer */
      unsigned char r =
          (unsigned char)(255.0f * clamp(accu_color.x * f, 0.0f, 1.0f));
      unsigned char g =
          (unsigned char)(255.0f * clamp(accu_color.y * f, 0.0f, 1.0f));
      unsigned char b =
          (unsigned char)(255.0f * clamp(accu_color.z * f, 0.0f, 1.0f));
      pixels[y * width * channels + x * channels] = r;
      pixels[y * width * channels + x * channels + 1] = g;
      pixels[y * width * channels + x * channels + 2] = b;
    }
}

/* called by the C++ code to render */
void renderFrameStandard(unsigned char* pixels,
                         std::vector<std::shared_ptr<Vec3ff>>& accu,
                         const unsigned int width, const unsigned int height,
                         const unsigned int channels, const float time,
                         const AffineSpace3fa& camera) {
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
          renderTileTask((int)i, threadIndex, pixels, accu, width, height,
                         channels, time, camera, numTilesX, numTilesY, reng,
                         distrib);
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
      cleanSphere();
      break;
    case SceneSelector::SHOW_CUBE_AND_PLANE:
    default:
      cleanCubeAndPlane();
      // cleanSphere();
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
      addCornell(g_scene, g_device);

      /* If you would like to add an Intel Embree sphere see addSphere(..) as
       * used below for an example... Remember to look for materials properties
       * set in our header files */
      addSphere(g_scene, g_device, Vec3fa(0.6f, -0.8f, -0.6f), 0.2f);

      cornellCameraLightSetup(g_camera, g_lights, width, height);
      break;
    case SceneSelector::SHOW_CUBE_AND_PLANE:
    default:
      /* add cube */
      addCube(g_scene, g_device);

      /* add ground plane */
      addGroundPlane(g_scene, g_device);

      /* The sphere can be used in the cube and plane scene with a corresponding
       * position for that scene */
      // addSphere(g_scene, g_device, Vec3fa(2.5f, 0.f, 2.5f), 1.f);

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
  /* Control the total number of accumulations, the total number of samples per
   * pixel per accumulation, and the maximum path length of any given traced
   * path.*/
  const unsigned long long g_accu_limit = 500;
  g_spp = 1;
  g_max_path_length = 8;

  g_pixels = (unsigned char*)new unsigned char[width * height * channels];
  std::memset(g_pixels, 0, sizeof(unsigned char) * width * height * channels);

  /* accumulation buffer used for convenience here, but is critical in
   * interactive/future applications */
  g_accu.resize(width * height);
  for (auto i = 0; i < width * height; i++)
    g_accu[i] = std::make_shared<Vec3ff>(0.0f);

  // g_sceneSelector = SceneSelector::SHOW_CUBE_AND_PLANE;
  g_sceneSelector = SceneSelector::SHOW_CORNELL_BOX;
  device_init(nullptr, width, height);

  /* Use a basic timer to capure compute time per accumulation */
  auto start = std::chrono::high_resolution_clock::now();
  renderFrameStandard(g_pixels, g_accu, width, height, channels, 0.0, g_camera);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> accum_time = end - start;
  std::cout << "Accumulation 1 of " << g_accu_limit << ": "
            << accum_time.count() << "s\n";

  /* Label construction for output files */
  string strscene;
  (g_sceneSelector == SceneSelector::SHOW_CORNELL_BOX) ? strscene = "-cornell"
                                                       : strscene = "-default";
  string suffix = string("-spp") + std::to_string(g_spp) + string("-accu") +
                  std::to_string(g_accu_limit) + string("-plength") +
                  std::to_string(g_max_path_length) + string("-") +
                  std::to_string(width) + string("x") + std::to_string(height) +
                  string(".png");
  string prefix = "pathtracer-single";
  string singleFilename = prefix + strscene + suffix;

  /* Write a single accumulation image (useful for comparison) */
  stbi_write_png(singleFilename.c_str(), width, height, channels, g_pixels,
                 width * channels);

  /* Render all remaining accumulations (in addition to the first) */
  for (unsigned long long i = 1; i < g_accu_limit; i++) {
    start = std::chrono::high_resolution_clock::now();
    renderFrameStandard(g_pixels, g_accu, width, height, channels, 0.0,
                        g_camera);
    end = std::chrono::high_resolution_clock::now();
    accum_time = end - start;
    std::cout << "Accumulation " << i + 1 << " of " << g_accu_limit << ": "
              << accum_time.count() << "s" << std::endl;
  }
  /* Label construction for the accumulated output */
  prefix = "pathtracer-accu";
  string accumFilename = prefix + strscene + suffix;

  /* Write the accumulated image output */
  stbi_write_png(accumFilename.c_str(), width, height, channels, g_pixels,
                 width * channels);
  std::cout << "Output images: \n  '" << singleFilename << "'\n  '"
            << accumFilename << "'\n... written to disk\n";

  delete[] g_pixels;
  g_pixels = nullptr;

  device_cleanup();
  rtcReleaseDevice(g_device);
  std::cout << "success\n";
  return 0;
}
