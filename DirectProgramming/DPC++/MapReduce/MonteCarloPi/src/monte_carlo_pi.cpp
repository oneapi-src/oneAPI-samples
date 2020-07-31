#include <CL/sycl.hpp>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "dpc_common.hpp"

#include "rgb.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image_write.h"

using namespace sycl;

constexpr int size_n = 100000;
constexpr int img_dimensions = 1024;
constexpr int radius = img_dimensions / 2;
constexpr double circle_outline = 0.025;

// Returns the pixel index corresponding to a set of simulation coordinates
int GetIndex(double x, double y){
    int img_x = x * radius + radius;
    int img_y = y * radius + radius;
    return img_y * img_dimensions + img_x;
}

// Returns a random double between -1 and 1
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
    // Set up sycl queue
    queue q(default_selector{}, dpc_common::exception_handler);
    std::cout << "Running on "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    
    try{
        // Array for reduction stage
        float reduction_arr[size_n];

        // Set up buffers
        buffer<rgb, 1> imgplot_buf((rgb*)image_plot, range<1>(img_dimensions * img_dimensions));
        buffer<float, 1> reduce_buf((float*)reduction_arr, range<1>(size_n));

        //Monte Carlo sim procedure
        int count = 0;
        for (int i = 0; i < size_n; ++i){
            double rand_x = GetRandCoordinate();
            double rand_y = GetRandCoordinate();
            double hypotenuse_sqr = (rand_x * rand_x + rand_y * rand_y);
            if (hypotenuse_sqr <= 1.0){
                ++count;
                image_plot[GetIndex(rand_x, rand_y)].red = 0;
                image_plot[GetIndex(rand_x, rand_y)].green = 255;
                image_plot[GetIndex(rand_x, rand_y)].blue = 0;
            }
            else{
                image_plot[GetIndex(rand_x, rand_y)].red = 255;
                image_plot[GetIndex(rand_x, rand_y)].green = 0;
                image_plot[GetIndex(rand_x, rand_y)].blue = 0;
            }
        }

        // Print calculated value of pi
        double pi = 4.0 * (double) count / size_n;
        std::cout << "The estimated value of pi is: " << pi << std::endl;
    } catch (sycl::exception e) {
        std::cout << "SYCL exception caught: " << e.what() << std::endl;
        exit(1);
    }
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