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

#define SIZE_N 10000 // number of simulated samples
#define IMG_DIMENSIONS 1024 //must be even
#define CIRCLE_OUTLINE 0.025

constexpr int radius = IMG_DIMENSIONS / 2;

// Returns the pixel index corresponding to a set of simulation coordinates
int GetIndex(double x, double y){
    int img_x = x * radius + radius;
    int img_y = y * radius + radius;
    return img_y * IMG_DIMENSIONS + img_x;
}

// Returns a random double between -1 and 1
double GetRandCoordinate(){
    return (double)rand() / (RAND_MAX / 2.0) - 1.0;
}

// Creates an array representing the image data and inscribes a circle
rgb* CreatePlot(rgb * plot){
    for (int i = 0; i < IMG_DIMENSIONS * IMG_DIMENSIONS; ++i){
        // calculate unit coordinates relative to the center of the image
        double x = (double)(i % IMG_DIMENSIONS - radius) / radius;
        double y = (double)(i / IMG_DIMENSIONS - radius) / radius;
        // draw the circumference of the circle
        if ((x * x + y * y) > 1 - CIRCLE_OUTLINE && (x * x + y * y) < 1) {
            plot[i].red = 255;
            plot[i].green = 255;
            plot[i].blue = 255;
        }
    }
    return plot;
}

int main(){
    // Initialize random seed
    srand(time(NULL));

    // Create image plot, and draw the circle
    rgb* image_plot = (rgb*) calloc(IMG_DIMENSIONS * IMG_DIMENSIONS, sizeof(rgb));
    CreatePlot(image_plot);

    // Perform Monte Carlo simulation to estimate pi
    int count = 0;
    for (int i = 0; i < SIZE_N; ++i){
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

    std::cout << "The estimated value of pi is: " << 4.0 * (double) count / SIZE_N << std::endl;


    // Write image to file
    stbi_write_bmp("MonteCarloPi.bmp", IMG_DIMENSIONS, IMG_DIMENSIONS, 3, image_plot);

    return 0;
}