#ifdef _MSC_VER
#ifndef NOMINMAX
/* use the c++ library min and max instead of the MSVS macros */
/* rkcommon will define this macro upon CMake import */
#define NOMINMAX
#endif
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

/* Additions for pathtracer */
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Renderer.h"
#include "definitions.h"

using std::string;

int write_image_first_accumulation(SceneSelector sceneSelector,
                                   unsigned int width, unsigned int height,
                                   unsigned int channels, unsigned int spp,
                                   unsigned int accu_limit,
                                   unsigned int max_path_length,
                                   unsigned char* pixels) {
  int ret = -1;

  /* Label construction for output files */
  string strscene;

  switch (sceneSelector) {
    case SceneSelector::SHOW_CORNELL_BOX:
      strscene = "-cornell";
      break;
    case SceneSelector::SHOW_CUBE_AND_PLANE:
      strscene = "-cubeandplane";
      break;
    case SceneSelector::SHOW_POOL:
      strscene = "-pool";
      break;
    default:
      strscene = "-default";
      break;
  };
  string suffix = string("-spp") + std::to_string(spp) + string("-accu") +
                  std::to_string(accu_limit) + string("-plength") +
                  std::to_string(max_path_length) + string("-") +
                  std::to_string(width) + string("x") + std::to_string(height) +
                  string(".png");
  string prefix = "pathtracer-single";
  string singleFilename = prefix + strscene + suffix;

  /* Write a single accumulation image (useful for comparison) */
  if ((ret = stbi_write_png(singleFilename.c_str(), width, height, channels,
                           pixels, width * channels)))
    std::cout << "Output image: '" << singleFilename
              << "'... written to disk\n";

  return ret;
}

int write_image_all_accumulations(SceneSelector sceneSelector,
                                  unsigned int width, unsigned int height,
                                  unsigned int channels, unsigned int spp,
                                  unsigned int accu_limit,
                                  unsigned int max_path_length,
                                  unsigned char* pixels) {
  int ret = -1;

  /* Label construction for output files */
  string strscene;
  switch (sceneSelector) {
    case SceneSelector::SHOW_CORNELL_BOX:
      strscene = "-cornell";
      break;
    case SceneSelector::SHOW_CUBE_AND_PLANE:
      strscene = "-cubeandplane";
      break;
    case SceneSelector::SHOW_POOL:
      strscene = "-pool";
      break;
    default:
      strscene = "-default";
      break;
  };

  string suffix = string("-spp") + std::to_string(spp) + string("-accu") +
                  std::to_string(accu_limit) + string("-plength") +
                  std::to_string(max_path_length) + string("-") +
                  std::to_string(width) + string("x") + std::to_string(height) +
                  string(".png");

  /* Label construction for the accumulated output */
  string prefix = "pathtracer-accu";
  string accumFilename = prefix + strscene + suffix;

  /* Write the accumulated image output */
  if ((ret = stbi_write_png(accumFilename.c_str(), width, height, channels,
                           pixels, width * channels)))
    std::cout << "Output image: '" << accumFilename << "'... written to disk\n";
  return ret;
}

int main() {
  /* create an image buffer initialize it with all zeroes */
  const unsigned int width = 512;
  const unsigned int height = 512;
  const unsigned int channels = 3;
  /* Control the total number of accumulations, the total number of samples per
   * pixel per accumulation, and the maximum path length of any given traced
   * path.*/
  const unsigned long long accu_limit = 500;
  const unsigned int spp = 1;
  const unsigned int max_path_length = 8;

  std::unique_ptr<Renderer> r;

  // SceneSelector sceneSelector = SceneSelector::SHOW_POOL;
  SceneSelector sceneSelector = SceneSelector::SHOW_CORNELL_BOX;
  // SceneSelector sceneSelector = SceneSelector::SHOW_CUBE_AND_PLANE;
  r = std::make_unique<Renderer>(width, height, channels, spp, accu_limit,
                                 max_path_length, sceneSelector);

  /* Use a basic timer to capure compute time per accumulation */
  auto start = std::chrono::high_resolution_clock::now();

  r->render_accumulation();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> accum_time = end - start;
  std::cout << "Accumulation 1 of " << accu_limit << ": " << accum_time.count()
            << "s\n";

  write_image_first_accumulation(sceneSelector, width, height, channels, spp,
                                 accu_limit, max_path_length, r->get_pixels());

  /* Render all remaining accumulations (in addition to the first) */
  for (unsigned long long i = 1; i < accu_limit; i++) {
    start = std::chrono::high_resolution_clock::now();

    r->render_accumulation();

    end = std::chrono::high_resolution_clock::now();
    accum_time = end - start;
    std::cout << "Accumulation " << i + 1 << " of " << accu_limit << ": "
              << accum_time.count() << "s" << std::endl;
  }

  write_image_all_accumulations(sceneSelector, width, height, channels, spp,
                                accu_limit, max_path_length, r->get_pixels());

  std::cout << "success\n";
  return 0;
}
