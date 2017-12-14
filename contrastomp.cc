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
#include "timer.h"
#include <omp.h>
#include <math.h>

using namespace std;

/* The maximum value we can use as RGB component. */
#define RGB_MAX_VALUE 255
/* Perform a contrast filter on an image. */
void filter_contrast_omp(unsigned char *image_data, int num_pixels, int min_index) {
    long brightness_sum = 0;

    omp_set_num_threads(4);
    /* CUDA Reduction. Brightness must be calculated first. This for loop must
     * be removed.
     * NOTE: TO REMOVE.
     */
    #pragma omp parallel for schedule( dynamic, ((num_pixels)/4)) reduction(+:brightness_sum)
    for(int i = 0; i < num_pixels; i++) {
        brightness_sum += image_data[i];
    }
    // cout << brightness_sum << endl;
    #pragma omp barrier

    float brightness_mean = (double) brightness_sum / (double) num_pixels;
    float denominator = sqrt(RGB_MAX_VALUE - brightness_mean);

    #pragma omp parallel for schedule( dynamic, ((num_pixels-min_index)/4))
    for(int i = min_index; i < num_pixels; i++) {
        int contrast_value = 0;

        if(image_data[i] >= brightness_mean) {
            contrast_value = (sqrt(image_data[i] - brightness_mean)
                              / denominator * RGB_MAX_VALUE);
        }

        image_data[i] = contrast_value;
    }
    #pragma omp barrier


}
