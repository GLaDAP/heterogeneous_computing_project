/*
 * File: contrast.cu
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: This file contains sequential implementations of different
 * image processing functions.
 */

#include <stdbool.h>
#include <stdlib.h>
#include <iostream>

#include "greyscale.h"
#include "contrast.h"
#include "smoothing.h"


using namespace std;

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "stb-master/stb_image.h"
#include "stb-master/stb_image_write.h"

/* Image input/output settings. */
#define PNG_STRIDE_DEFAULT 0
#define NUM_CHANNELS_RGB 3
#define NUM_CHANNELS_GREYSCALE 1
//
// /* Greyscale component weighting factors. */
// #define GREYSCALE_R 0.2126
// #define GREYSCALE_G 0.7152
// #define GREYSCALE_B 0.0722
//
// /* The maximum value we can use as RGB component. */
// #define RGB_MAX_VALUE 255
//
// /* Kernel dimension values. */
// #define KERNEL_WIDTH 5
// #define KERNEL_OFFSET KERNEL_WIDTH / 2
// #define KERNEL_MULTIPLIER (1.0 / 81.0)

/* Triangular smoothing kernel. */
// const int kernel[KERNEL_WIDTH][KERNEL_WIDTH] = {{1, 2, 3, 2, 1},
//                                                 {2, 4, 6, 4, 2},
//                                                 {3, 6, 9, 6, 3},
//                                                 {2, 4, 6, 4, 2},
//                                                 {1, 2, 3, 2, 1}};
//

/* Check whether two arguments are supplied. */
int check_argc(int argc) {
    if(argc != 3) {
        cout << "Error: wrong argument count.\n";
        cout << "Usage: ./main input_file output_file\n";

        return false;
    }

    return true;
}

/* Check whether the image is valid. */
int check_image(unsigned char *image_data, int num_channels) {
    if(!image_data || num_channels != NUM_CHANNELS_RGB) {
        cout << "Error reading file.\n";

        return false;
    }

    return true;
}

/* Process the entire image. Return true on success, false on failure. */
int process_image(char *file_in, char *file_out) {
    int width, height, num_channels;
    unsigned char *image_data = stbi_load(file_in, &width, &height,
                                          &num_channels, NUM_CHANNELS_RGB);

    /* We stop processing if the image is invalid or we don't get the
    specific amount of channels we want. */
    if(!check_image(image_data, num_channels))
        return false;

    int num_pixels = width * height;

    /* Apply filter 1 and swap the newly acquired image data. */
    unsigned char *temp_image_data = filter_greyscale_cuda(image_data, num_pixels);
    stbi_image_free(image_data);
    image_data = temp_image_data;

    /* Apply filter 2. */
    filter_contrast_cuda(image_data, num_pixels);
    /* Apply filter 3. */
    // filter_smoothing(image_data, num_pixels, width, height);

    stbi_write_png(file_out, width, height, NUM_CHANNELS_GREYSCALE, image_data, PNG_STRIDE_DEFAULT);
    stbi_image_free(image_data);

    return true;
}

int main(int argc, char *argv[]) {
    if(!check_argc(argc))
        return EXIT_FAILURE;

    char *file_in = argv[1], *file_out = argv[2];

    if(!process_image(file_in, file_out))
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
