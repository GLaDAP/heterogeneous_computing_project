/*
 * File: main.cpp
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

/* Greyscale component weighting factors. */
#define GREYSCALE_R 0.2126
#define GREYSCALE_G 0.7152
#define GREYSCALE_B 0.0722

/* The maximum value we can use as RGB component. */
#define RGB_MAX_VALUE 255

/* Kernel dimension values. */
#define KERNEL_WIDTH 5
#define KERNEL_OFFSET KERNEL_WIDTH / 2
#define KERNEL_MULTIPLIER (1.0 / 81.0)

/* Triangular smoothing kernel. */
const int kernel[KERNEL_WIDTH][KERNEL_WIDTH] = {{1, 2, 3, 2, 1},
                                                {2, 4, 6, 4, 2},
                                                {3, 6, 9, 6, 3},
                                                {2, 4, 6, 4, 2},
                                                {1, 2, 3, 2, 1}};


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

/* Perform a greyscale filter on an image. Return a new image data array
containing only 1 color channel. */
unsigned char *filter_greyscale(unsigned char *image_data, int num_pixels) {
    unsigned char *new_image_data = (unsigned char *) malloc(num_pixels);
    int total_size = num_pixels * NUM_CHANNELS_RGB;

    for(int i = 0, j = 0; i < total_size; i += NUM_CHANNELS_RGB, j++) {
        int greyscale_value = (image_data[i] * GREYSCALE_R
                               + image_data[i + 1] * GREYSCALE_G
                               + image_data[i + 2] * GREYSCALE_B);

        new_image_data[j] = greyscale_value;
    }

    return new_image_data;
}

/* Perform a contrast filter on an image. */
void filter_contrast(unsigned char *image_data, int num_pixels) {
    long brightness_sum = 0;

    for(int i = 0; i < num_pixels; i++) {
        brightness_sum += image_data[i];
    }

    float brightness_mean = (double) brightness_sum / (double) num_pixels;
    float denominator = sqrt(RGB_MAX_VALUE - brightness_mean);

    for(int i = 0; i < num_pixels; i++) {
        int contrast_value = 0;

        if(image_data[i] >= brightness_mean) {
            contrast_value = (sqrt(image_data[i] - brightness_mean)
                              / denominator * RGB_MAX_VALUE);
        }

        image_data[i] = contrast_value;
    }
}

/* Perform a triangular smoothing filter on an image. */
void filter_smoothing(unsigned char *image_data, int num_pixels, int width, int height) {
    unsigned char *temp_image_data = (unsigned char *) malloc(num_pixels);
    memcpy(temp_image_data, image_data, num_pixels);

    for(int i = 0; i < num_pixels; i++) {
        int col = i % width;
        int row = i / width;

        /* Boundary check for the current kernel center. */
        if(row < KERNEL_OFFSET || col < KERNEL_OFFSET
           || row > (height - KERNEL_OFFSET) || col > (width - KERNEL_OFFSET)) {
               continue;
           }

        int accumulator = 0;
        for (int ky = -KERNEL_OFFSET; ky <= KERNEL_OFFSET; ky++) {
            for (int kx = -KERNEL_OFFSET; kx <= KERNEL_OFFSET; kx++) {
                accumulator += (kernel[ky + KERNEL_OFFSET][kx + KERNEL_OFFSET]
                                * temp_image_data[i + kx + width * ky]);
            }
        }
        accumulator *= KERNEL_MULTIPLIER;

        image_data[i] = accumulator;
    }

    free(temp_image_data);
}

/* Process the entire image. Return true on success, false on failure. */
int process_image(char *file_in, char *file_out) {
    int width, height, num_channels;
    unsigned char *image_data = stbi_load(file_in, &width, &height, &num_channels, NUM_CHANNELS_RGB);

    /* We stop processing if the image is invalid or we don't get the
    specific amount of channels we want. */
    if(!check_image(image_data, num_channels))
        return false;

    int num_pixels = width * height;

    /* Apply filter 1 and swap the newly acquired image data. */
    unsigned char *temp_image_data = filter_greyscale(image_data, num_pixels);
    stbi_image_free(image_data);
    image_data = temp_image_data;

    /* Apply filter 2. */
    filter_contrast(image_data, num_pixels);
    /* Apply filter 3. */
    filter_smoothing(image_data, num_pixels, width, height);

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
