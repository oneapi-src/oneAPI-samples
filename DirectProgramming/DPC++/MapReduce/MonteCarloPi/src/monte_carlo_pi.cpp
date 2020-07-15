#include <iostream>

#include "rgb.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb/stb_image_write.h"

#define IMG_DIMENSIONS 512

int main(){
    std::cout << "Hello World!" << std::endl;

    // Create image
    rgb* image_plot = (rgb*) malloc(IMG_DIMENSIONS * IMG_DIMENSIONS * sizeof(rgb));
    for (int i = 0; i < IMG_DIMENSIONS; i++){
        image_plot[i * 512 + 20].red = 255;
    }






    // Write image to file
    stbi_write_bmp("MonteCarloPi.bmp", IMG_DIMENSIONS, IMG_DIMENSIONS, 3, image_plot);

    return 0;
}