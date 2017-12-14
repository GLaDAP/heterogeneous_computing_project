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
// #include "timer.h"
#include <omp.h>
// #include "contrast.h"
#include <math.h>

using namespace std;

/* The maximum value we can use as RGB component. */
#define RGB_MAX_VALUE 255
/* Perform a contrast filter on an image. */
void filter_contrast_omp(unsigned char *image_data, int num_pixels) {
    long brightness_sum = 0;

    omp_set_num_threads(4);
    // omp_get_max_threads();
    /* CUDA Reduction or OpenMP. */
    for(int i = 0; i < num_pixels; i++) {
        brightness_sum += image_data[i];
    }
    cout << brightness_sum << endl;

    float brightness_mean = (double) brightness_sum / (double) num_pixels;
    float denominator = sqrt(RGB_MAX_VALUE - brightness_mean);

    #pragma omp parallel for schedule( dynamic, (num_pixels/4) )
    for(int i = 0; i < num_pixels; i++) {
        int contrast_value = 0;

        if(image_data[i] >= brightness_mean) {
            contrast_value = (sqrt(image_data[i] - brightness_mean)
                              / denominator * RGB_MAX_VALUE);
        }

        image_data[i] = contrast_value;
    }
    #pragma omp barrier
}
