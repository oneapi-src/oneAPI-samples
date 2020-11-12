#include <CL/sycl.hpp>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"
#include "monte_carlo_pi.hpp"
#define STB_IMAGE_IMPLEMENTATION 
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

using namespace sycl;

// Number of samples
constexpr int size_n = 10000;  // Must be greater than size_wg
// Size of parallel work groups
constexpr int size_wg = 32;
// Number of parallel work groups
const int num_wg = (int)ceil((double)size_n / (double)size_wg);

// Output image dimensions
constexpr int img_dimensions = 1024;
// Consts for drawing the image plot
constexpr double circle_outline = 0.025;
// Radius of the circle in the image plot
constexpr int radius = img_dimensions / 2;

// Returns the pixel index corresponding to a set of simulation coordinates
SYCL_EXTERNAL int GetPixelIndex(double x, double y) {
  int img_x = x * radius + radius;
  int img_y = y * radius + radius;
  return img_y * img_dimensions + img_x;
}

// Returns a random double between -1.0 and 1.0
double GetRandCoordinate() { return (double)rand() / (RAND_MAX / 2.0) - 1.0; }

// Creates an array representing the image data and inscribes a circle
void DrawPlot(rgb image_plot[]) {
  for (int i = 0; i < img_dimensions * img_dimensions; ++i) {
    // calculate unit coordinates relative to the center of the image
    double x = (double)(i % img_dimensions - radius) / radius;
    double y = (double)(i / img_dimensions - radius) / radius;
    // draw the circumference of the circle
    if ((x * x + y * y) > 1 - circle_outline && (x * x + y * y) < 1) {
      image_plot[i].red = 255;
      image_plot[i].green = 255;
      image_plot[i].blue = 255;
    }
  }
}

// Performs the Monte Carlo simulation procedure for calculating pi, with size_n
// number of samples.
double MonteCarloPi(rgb image_plot[]) {
  int total = 0;  // Stores the total number of simulated points falling within
                  // the circle
  coordinate coords[size_n];  // Array for storing the RNG coordinates

  // Generate Random Coordinates
  for (int i = 0; i < size_n; ++i) {
    coords[i].x = GetRandCoordinate();
    coords[i].y = GetRandCoordinate();
  }

  // Set up sycl queue
  queue q(default_selector{}, dpc_common::exception_handler);
  std::cout << "\nRunning on "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  try {
    // Set up buffers
    buffer imgplot_buf(image_plot, range(img_dimensions * img_dimensions));
    buffer coords_buf(coords, range(size_n));
    buffer total_buf(&total, range(1));

    // Perform Monte Carlo simulation and reduce results
    q.submit([&](handler& h) {
      // Set up accessors
      auto imgplot_acc = imgplot_buf.get_access(h);
      auto coords_acc = coords_buf.get_access(h);
      auto total_acc = total_buf.get_access(h);

      // Monte Carlo Procedure + Reduction
      h.parallel_for(nd_range<1>(num_wg * size_wg, size_wg),
                     sycl::ONEAPI::reduction(total_acc, 0, std::plus<int>()),
                     [=](nd_item<1> it, auto& total_acc) {
                       // Index for accessing buffers
                       int i = it.get_global_id();

                       if (i < size_n) {  // Only runs if a work item's ID has a
                                          // corresponding sample coordinate
                         // Get random coords
                         double x = coords_acc[i].x;
                         double y = coords_acc[i].y;

                         // Check if coordinates are bounded by a circle of
                         // radius 1
                         double hypotenuse_sqr = x * x + y * y;
                         if (hypotenuse_sqr <= 1.0) {  // If bounded
                           // increment total
                           total_acc += 1;
                           // Draw sample point in image plot
                           imgplot_acc[GetPixelIndex(x, y)].red = 0;
                           imgplot_acc[GetPixelIndex(x, y)].green = 255;
                           imgplot_acc[GetPixelIndex(x, y)].blue = 0;
                         } else {
                           // Draw sample point in image plot
                           imgplot_acc[GetPixelIndex(x, y)].red = 255;
                           imgplot_acc[GetPixelIndex(x, y)].green = 0;
                           imgplot_acc[GetPixelIndex(x, y)].blue = 0;
                         }
                       }
                     });
    });
    q.wait_and_throw();

  } catch (sycl::exception e) {
    std::cout << "SYCL exception caught: " << e.what() << "\n";
    exit(1);
  }

  // return calculated value of pi
  return 4.0 * (double)total / size_n;
}

int main() {
  // Validate constants
  if (size_n < size_wg) {
    std::cout << "ERROR: size_n must be greater than or equal to size_wg\n";
    exit(1);
  }

  // Initialize random seed
  srand(time(NULL));

  // Allocate memory for the output image
  std::vector<rgb> image_plot(img_dimensions * img_dimensions);

  // Draw the inscribed circle for the image plot
  DrawPlot(image_plot.data());

  // Perform Monte Carlo simulation to estimate pi (with timing)
  std::cout << "Calculating estimated value of pi...\n";
  dpc_common::TimeInterval t;
  double pi = MonteCarloPi(image_plot.data());
  double proc_time = t.Elapsed();
  std::cout << "The estimated value of pi (N = " << size_n << ") is: " << pi
            << "\n";
  std::cout << "\nComputation complete. The processing time was " << proc_time
            << " seconds.\n";

  // Write image to file
  stbi_write_bmp("MonteCarloPi.bmp", img_dimensions, img_dimensions, 3,
                 image_plot.data());
  std::cout
      << "The simulation plot graph has been written to 'MonteCarloPi.bmp'\n";

  return 0;
}
