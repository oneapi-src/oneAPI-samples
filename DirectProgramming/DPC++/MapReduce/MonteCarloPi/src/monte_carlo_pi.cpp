#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "rgb.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image_write.h"

#define SIZE_N 5 // number of simulated samples
#define IMG_DIMENSIONS 512 //must be even
#define CIRCLE_OUTLINE 0.025

#define PI 3.14159265

// Returns a random float between -1 and 1
float GetRandCoordinate(){
    return (float)rand() / (RAND_MAX / 2) - 1.0;
}

// Creates an array representing the image data and inscribes a circle
rgb* CreatePlot(){
    constexpr int radius = IMG_DIMENSIONS / 2;
    rgb* plot = (rgb*) calloc(IMG_DIMENSIONS * IMG_DIMENSIONS, sizeof(rgb));
    for (int i = 0; i < IMG_DIMENSIONS * IMG_DIMENSIONS; ++i){
        // calculate unit coordinates relative to the center of the image
        float x = (float)(i % IMG_DIMENSIONS - radius) / radius;
        float y = (float)(i / IMG_DIMENSIONS - radius) / radius;
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
    rgb* image_plot = CreatePlot();


    int count = 0;
    for (int i = 0; i < SIZE_N; ++i){
        float rand_x = GetRandCoordinate();
        float rand_y = GetRandCoordinate();
        std::cout << "X: " << rand_x << " Y: " << rand_y << std::endl;
    }




    // Write image to file
    stbi_write_bmp("MonteCarloPi.bmp", IMG_DIMENSIONS, IMG_DIMENSIONS, 3, image_plot);

    return 0;
}