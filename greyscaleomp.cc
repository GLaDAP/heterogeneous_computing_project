/*
 * File: greyscaleomp.cc
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: Applies the greyscale filter on the image using OpenMP.
 */
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include "timer.h"

using namespace std;


/* Image input/output settings. */
#define NUM_CHANNELS_RGB 3

/* Greyscale component weighting factors. */
#define GREYSCALE_R 0.2126
#define GREYSCALE_G 0.7152
#define GREYSCALE_B 0.0722

/* Perform a greyscale filter on an image. Return a new image data array
 * containing only 1 color channel.
*/
void filter_greyscale_omp(unsigned char *image_data, int num_pixels,
                                     int min_index, int num_threads,
                                     unsigned char** result) {

    unsigned char *new_image_data = (unsigned char *) malloc(num_pixels \
                                  * sizeof(unsigned char));
    if (new_image_data == NULL) {
        cout << "Could not allocate memory" << endl;
        return;
    }

    omp_set_num_threads(num_threads);
    /* Since the OpenMP takes the second part of the array, the total size is
     * equal to the picture.
     */
    int total_indices = num_pixels - min_index;
    int total_size = num_pixels * NUM_CHANNELS_RGB;

    #pragma omp parallel for schedule (dynamic, (total_indices / num_threads))
    for(int i = min_index* NUM_CHANNELS_RGB; i < total_size; \
        i += NUM_CHANNELS_RGB) {
        int j = i / NUM_CHANNELS_RGB;
        int greyscale_value = (image_data[i] * GREYSCALE_R \
                               + image_data[i + 1] * GREYSCALE_G \
                               + image_data[i + 2] * GREYSCALE_B);

        new_image_data[j] = greyscale_value;
    }
    #pragma omp barrier
    *result = *&new_image_data;
}
