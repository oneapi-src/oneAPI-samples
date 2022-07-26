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

  std::unique_ptr<PathTracer> pt;
  SceneSelector sceneSelector = SceneSelector::SHOW_CORNELL_BOX;
  //SceneSelector sceneSelector = SceneSelector::SHOW_CUBE_AND_PLANE;
  pt = std::make_unique<PathTracer>(width, height, spp, channels, accu_limit, max_path_length, SceneSelector::SHOW_CORNELL_BOX);

  /* Use a basic timer to capure compute time per accumulation */
  auto start = std::chrono::high_resolution_clock::now();
  pt->render_accumulation();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> accum_time = end - start;
  std::cout << "Accumulation 1 of " << accu_limit << ": "
            << accum_time.count() << "s\n";

  /* Label construction for output files */
  string strscene;
  (sceneSelector == SceneSelector::SHOW_CORNELL_BOX) ? strscene = "-cornell"
                                                       : strscene = "-default";
  string suffix = string("-spp") + std::to_string(spp) + string("-accu") +
                  std::to_string(accu_limit) + string("-plength") +
                  std::to_string(max_path_length) + string("-") +
                  std::to_string(width) + string("x") + std::to_string(height) +
                  string(".png");
  string prefix = "pathtracer-single";
  string singleFilename = prefix + strscene + suffix;

  /* Write a single accumulation image (useful for comparison) */
  stbi_write_png(singleFilename.c_str(), width, height, channels, pt->get_pixels(),
                 width * channels);

  /* Render all remaining accumulations (in addition to the first) */
  for (unsigned long long i = 1; i < accu_limit; i++) {
    start = std::chrono::high_resolution_clock::now();
    pt->render_accumulation();
    end = std::chrono::high_resolution_clock::now();
    accum_time = end - start;
    std::cout << "Accumulation " << i + 1 << " of " << accu_limit << ": "
              << accum_time.count() << "s" << std::endl;
  }
  /* Label construction for the accumulated output */
  prefix = "pathtracer-accu";
  string accumFilename = prefix + strscene + suffix;

  /* Write the accumulated image output */
  stbi_write_png(accumFilename.c_str(), width, height, channels, pt->get_pixels(),
                 width * channels);
  std::cout << "Output images: \n  '" << singleFilename << "'\n  '"
            << accumFilename << "'\n... written to disk\n";

  std::cout << "success\n";
  return 0;
}
