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

constexpr int size_n = 100000;
constexpr int img_dimensions = 1024;
constexpr int radius = img_dimensions / 2;
constexpr double circle_outline = 0.025;
constexpr int seed = 777;

// Returns the pixel index corresponding to a set of simulation coordinates
SYCL_EXTERNAL int GetIndex(double x, double y){
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

// performs the Monte Carlo simulation procedure for calculating pi, with size_n number of samples.
void MonteCarloPi(rgb * image_plot){
    coordinate coords[size_n]; // array for storing the RNG coordinates
    int reduction_arr[size_n]; // this array will be used in the reduction stage to sum all the simulated points which fall within the circle

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
        buffer<coordinate, 1> coords_buf((coordinate*)coords, range<1>(size_n));
        buffer<int, 1> reduction_buf((int*)reduction_arr, range<1>(size_n));

        // Initialize random coordinates buffer on the host
        /*q.submit([=](handler& h){
            auto coords_acc = coords_buf.get_host_access<access::mode::read_write>(h);

            h.codeplay_host_task([=]() {
                for (int i = 0; i < size_n; ++i){
                    coords_acc[i].x = GetRandCoordinate();
                    coords_acc[i].y = GetRandCoordinate();
                }
            });
        });
        q.wait_and_throw();*/

        // Perform Monte Carlo Procedure on the device
        q.submit([&](handler& h){
            auto imgplot_acc = imgplot_buf.get_access<access::mode::read_write>(h);
            auto coords_acc = coords_buf.get_access<access::mode::read_write>(h);
            auto reduction_acc = reduction_buf.get_access<access::mode::read_write>(h);

            h.parallel_for(size_n, [=](id<1> idx){
                double x = coords_acc[idx].x;
                double y = coords_acc[idx].y;
                double hypotenuse_sqr = (x * x + y * y);
                if (hypotenuse_sqr <= 1.0){
                    reduction_acc[idx] = 1;
                    imgplot_acc[GetIndex(x, y)].red = 0;
                    imgplot_acc[GetIndex(x, y)].green = 255;
                    imgplot_acc[GetIndex(x, y)].blue = 0;
                }
                else{
                    reduction_acc[idx] = 0;
                    imgplot_acc[GetIndex(x, y)].red = 255;
                    imgplot_acc[GetIndex(x, y)].green = 0;
                    imgplot_acc[GetIndex(x, y)].blue = 0;
                }
            });
        });
    } catch (sycl::exception e) {
        std::cout << "SYCL exception caught: " << e.what() << std::endl;
        exit(1);
    }

    // Print calculated value of pi
    int count = 0;
    for (int i = 0; i < size_n; ++i){
        count += reduction_arr[i];
    }
    double pi = 4.0 * (double) count / size_n;
    std::cout << "The estimated value of pi is: " << pi << std::endl;
}

int main(){
    // Initialize random seed
    srand(time(NULL));

    // Allocate memory for the output image
    rgb* image_plot = (rgb*) calloc(img_dimensions * img_dimensions, sizeof(rgb));

    // Draw the inscribed circle for the image plot
    DrawPlot(image_plot);

    // Perform Monte Carlo simulation to estimate pi (with timing)
    std::cout << "Calculating estimated value of pi..." << std::endl;
    dpc_common::TimeInterval t;
    MonteCarloPi(image_plot);
    double proc_time = t.Elapsed();
    std::cout << "Computation complete. The processing time was " << proc_time << " seconds." << std::endl;

    // Write image to file
    stbi_write_bmp("MonteCarloPi.bmp", img_dimensions, img_dimensions, 3, image_plot);
    std::cout << "The simulation plot graph has been written to 'MonteCarloPi.bmp'" << std::endl;
    free(image_plot);

    return 0;
}