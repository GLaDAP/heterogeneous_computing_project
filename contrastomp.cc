/*
 * File: contrastomp.cc
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: Applies the contrast filter on the image using OpenMP.
 */
#include <cstdlib>
#include <iostream>
#include "timer.h"
#include <omp.h>
#include <math.h>

using namespace std;

/* The maximum value we can use as RGB component. */
#define RGB_MAX_VALUE 255
/* Perform a contrast filter on an image. */
void filter_contrast_omp(unsigned char *image_data, int num_pixels,
                         int min_index, int num_threads) {
    long brightness_sum = 0;

    omp_set_num_threads(num_threads);
    /* CUDA Reduction. Brightness must be calculated first. This for loop must
     * be removed.
     * NOTE: TO REMOVE.
     */
    #pragma omp parallel for schedule (dynamic, (num_pixels / num_threads)) \
        reduction (+:brightness_sum)
    for (int i = 0; i < num_pixels; i++) {
        brightness_sum += image_data[i];
    }
    #pragma omp barrier
    cout << "Brightness OMP: " << brightness_sum << endl;

    float brightness_mean = (double) brightness_sum / (double) num_pixels;
    float denominator = sqrt(RGB_MAX_VALUE - brightness_mean);
    int total_indices = num_pixels - min_index;

    #pragma omp parallel for schedule (dynamic, (total_indices / num_threads))
    for (int i = min_index; i < num_pixels; i++) {
        int contrast_value = 0;

        if (image_data[i] >= brightness_mean) {
            contrast_value = (sqrt(image_data[i] - brightness_mean)
                              / denominator * RGB_MAX_VALUE);
        }
        image_data[i] = contrast_value;
    }
    #pragma omp barrier
}
