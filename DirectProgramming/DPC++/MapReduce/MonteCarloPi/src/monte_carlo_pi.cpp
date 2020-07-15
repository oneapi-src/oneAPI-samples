#include <iostream>
#include <math.h>

#include "rgb.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image_write.h"

#define IMG_DIMENSIONS 512 //must be even

#define PI 3.14159265

int main(){
    std::cout << "Hello World!" << std::endl;

    // Create image plot, and draw the circle
    constexpr int radius = IMG_DIMENSIONS / 2;
    rgb* image_plot = (rgb*) calloc(IMG_DIMENSIONS * IMG_DIMENSIONS, sizeof(rgb));
    for (int i = 0; i < IMG_DIMENSIONS * IMG_DIMENSIONS; i++){
        // calculate unit coordinates relative to the center of the image
        float x = (i % IMG_DIMENSIONS - radius) / radius;
        float y = (i / IMG_DIMENSIONS - radius) / radius;
        // draw the circumference of the circle
        std::cout << "x is " << x << " y is " << y << std::endl;


        if ((x * x + y * y) <= IMG_DIMENSIONS / 2) {
            image_plot[i].red = 255;
            image_plot[i].green = 255;
            image_plot[i].blue = 255;
        }
    }






    // Write image to file
    stbi_write_bmp("MonteCarloPi.bmp", IMG_DIMENSIONS, IMG_DIMENSIONS, 3, image_plot);

    return 0;
}