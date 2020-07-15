#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "rgb.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image_write.h"

#define SIZE_N 1000 // number of simulated samples
#define IMG_DIMENSIONS 512 //must be even
#define RADIUS IMG_DIMENSIONS / 2
#define CIRCLE_OUTLINE 0.025

#define PI 3.14159265

// Returns the pixel index corresponding to a set of simulation coordinates
int GetIndex(float x, float y){
    int img_x = x * RADIUS + RADIUS;
    int img_y = y * RADIUS + RADIUS;
    return img_y * IMG_DIMENSIONS + img_x;
}

// Returns a random float between -1 and 1
float GetRandCoordinate(){
    return (float)rand() / (RAND_MAX / 2.0) - 1.0;
}

// Creates an array representing the image data and inscribes a circle
rgb* CreatePlot(){
    rgb* plot = (rgb*) calloc(IMG_DIMENSIONS * IMG_DIMENSIONS, sizeof(rgb));
    for (int i = 0; i < IMG_DIMENSIONS * IMG_DIMENSIONS; ++i){
        // calculate unit coordinates relative to the center of the image
        float x = (float)(i % IMG_DIMENSIONS - RADIUS) / RADIUS;
        float y = (float)(i / IMG_DIMENSIONS - RADIUS) / RADIUS;
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

    // Perform Monte Carlo simulation to estimate pi
    /*int count = 0;
    for (int i = 0; i < SIZE_N; ++i){
        float rand_x = GetRandCoordinate();
        float rand_y = GetRandCoordinate();
        float hypotenuse_sqr = (rand_x * rand_x + rand_y * rand_y);
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
    }*/




    // Write image to file
    stbi_write_bmp("MonteCarloPi.bmp", IMG_DIMENSIONS, IMG_DIMENSIONS, 3, image_plot);

    return 0;
}