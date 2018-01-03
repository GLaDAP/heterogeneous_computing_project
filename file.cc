/*
 * File: file.cc
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: File containing functions to open and save images using the STB
 *              library by Sean Barrett: https://github.com/nothings/stb
 */
 
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "stb-master/stb_image.h"
#include "stb-master/stb_image_write.h"

#include <stdlib.h>
using namespace std;

/* Image input/output settings. */
#define PNG_STRIDE_DEFAULT 0
#define NUM_CHANNELS_RGB 3
#define NUM_CHANNELS_GREYSCALE 1

unsigned char * open_rgb_image(char *file_in, int* width, int* height,
                           int* num_channels) {
    return stbi_load(file_in, width, height, num_channels, NUM_CHANNELS_RGB);
}

void free_image (unsigned char * image) {
    stbi_image_free(image);
}

void write_grey_png(char* file_out, int width, int height, \
                 unsigned char * image_data) {
    stbi_write_png(file_out, width, height, NUM_CHANNELS_GREYSCALE,
                   image_data, PNG_STRIDE_DEFAULT);
}
