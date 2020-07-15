#include <iostream>
#include <math.h>

#include "rgb.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image_write.h"

#define IMG_DIMENSIONS 512

#define PI 3.14159265

int main(){
    std::cout << "Hello World!" << std::endl;

    // Create image
    rgb* image_plot = (rgb*) calloc(IMG_DIMENSIONS * IMG_DIMENSIONS, sizeof(rgb));
    for (int i = 0; i < IMG_DIMENSIONS * IMG_DIMENSIONS; i++){
        int x = i % IMG_DIMENSIONS - (IMG_DIMENSIONS / 2);
        int y = i / IMG_DIMENSIONS - (IMG_DIMENSIONS / 2);
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