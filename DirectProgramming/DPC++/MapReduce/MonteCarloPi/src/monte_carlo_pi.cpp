#include <CL/sycl.hpp>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "dpc_common.hpp"

#include "monte_carlo_pi.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

using namespace sycl;

// Size of parallel work groups
constexpr int size_wg = 32;
// Number of parallel work groups
constexpr int num_wg = 128;
// Number of sample points
constexpr int size_n = size_wg * num_wg; // Must be a multiple of size_wg
// Output image dimensions
constexpr int img_dimensions = 1024;

// Consts for drawing the image plot
constexpr double circle_outline = 0.025;
constexpr int radius = img_dimensions / 2;

// Returns the pixel index corresponding to a set of simulation coordinates
SYCL_EXTERNAL int GetPixelIndex(double x, double y){
    int img_x = x * radius + radius;
    int img_y = y * radius + radius;
    return img_y * img_dimensions + img_x;
}

// Returns a random double between -1.0 and 1.0
double GetRandCoordinate(){
    return (double)rand() / (RAND_MAX / 2.0) - 1.0;
}

// Creates an array representing the image data and inscribes a circle
rgb* DrawPlot(rgb * image_plot){
    for (int i = 0; i < img_dimensions * img_dimensions; ++i){
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
    return image_plot;
}

// Performs the Monte Carlo simulation procedure for calculating pi, with size_n number of samples.
void MonteCarloPi(rgb * image_plot){
    int total = 0; // Stores the total number of simulated points falling within the circle
    coordinate coords[size_n]; // Array for storing the RNG coordinates

    // Generate Random Coordinates
    for (int i = 0; i < size_n; ++i){
        coords[i].x = GetRandCoordinate();
        coords[i].y = GetRandCoordinate();
    }

    // Set up sycl queue
    queue q(default_selector{}, dpc_common::exception_handler);
    std::cout << "Running on "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    
    try{
        // Set up buffers
        buffer imgplot_buf((rgb*)image_plot, range(img_dimensions * img_dimensions));
        buffer coords_buf((coordinate*)coords, range(size_n));
        buffer total_buf((int*)(&total), range(1));

        // Perform Monte Carlo simulation and reduce results
        q.submit([&](handler& h){
            // Set up accessors
            auto imgplot_acc = imgplot_buf.get_access<access::mode::write>(h);
            auto coords_acc = coords_buf.get_access<access::mode::read_write>(h);
            auto total_acc = total_buf.get_access<access::mode::read_write>(h);

            // Monte Carlo Procedure + Reduction
            h.parallel_for(nd_range<1>(size_n, size_wg), sycl::intel::reduction(total_acc, 0, std::plus<int>()), [=](nd_item<1> it, auto& total_acc)
            {
                int i = it.get_global_id(); // Index for accessing external buffers

                // Get random coords
                double x = coords_acc[i].x;
                double y = coords_acc[i].y;

                // Check if coordinates are bounded by a circle of radius 1
                double hypotenuse_sqr = (x * x + y * y);
                if (hypotenuse_sqr <= 1.0){ // If bounded
                    // increment total
                    total_acc += 1;
                    // Draw sample point in image plot
                    imgplot_acc[GetPixelIndex(x, y)].red = 0;
                    imgplot_acc[GetPixelIndex(x, y)].green = 255;
                    imgplot_acc[GetPixelIndex(x, y)].blue = 0;
                }
                else{
                    // Draw sample point in image plot
                    imgplot_acc[GetPixelIndex(x, y)].red = 255;
                    imgplot_acc[GetPixelIndex(x, y)].green = 0;
                    imgplot_acc[GetPixelIndex(x, y)].blue = 0;
                }
            });

        });
        q.wait_and_throw();


    } catch (sycl::exception e) {
        std::cout << "SYCL exception caught: " << e.what() << std::endl;
        exit(1);
    }

    // Print calculated value of pi
    double pi = 4.0 * (double) total / size_n;
    std::cout << "The estimated value of pi (N = " << size_n << ") is: " << pi << std::endl;
}

int main(){
    // Initialize random seed
    srand(time(NULL));

    // Allocate memory for the output image
    rgb* image_plot = (rgb*) calloc(img_dimensions * img_dimensions, sizeof(rgb));

    // Draw the inscribed circle for the image plot
    DrawPlot(image_plot);

    // Perform Monte Carlo simulation to estimate pi (with timing)
    std::cout << "Calculating estimated value of pi...\n" << std::endl;
    dpc_common::TimeInterval t;
    MonteCarloPi(image_plot);
    double proc_time = t.Elapsed();
    std::cout << "\nComputation complete. The processing time was " << proc_time << " seconds." << std::endl;

    // Write image to file
    stbi_write_bmp("MonteCarloPi.bmp", img_dimensions, img_dimensions, 3, image_plot);
    std::cout << "The simulation plot graph has been written to 'MonteCarloPi.bmp'\n" << std::endl;
    free(image_plot);

    return 0;
}