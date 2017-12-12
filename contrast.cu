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
// 
// #include "stb-master/stb_image.h"
// #include "stb-master/stb_image_write.h"

/* The maximum value we can use as RGB component. */
#define RGB_MAX_VALUE 255


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
