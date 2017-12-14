#include <stdbool.h>
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
containing only 1 color channel.*/
unsigned char *filter_greyscale_omp(unsigned char *image_data, int num_pixels) {
    
    unsigned char *new_image_data = (unsigned char *) malloc(num_pixels \
                                  * sizeof(unsigned char));
    if (new_image_data == NULL) {
        cout << "Could not allocate memory" << endl;
        return NULL;
    }

    omp_set_num_threads(4);


    int total_size = num_pixels * NUM_CHANNELS_RGB;
    // int j = 0;
    #pragma omp parallel for schedule( dynamic, (num_pixels/4) ) // private(j)
    for(int i = 0; i < total_size; i += NUM_CHANNELS_RGB) {
        int j = i / NUM_CHANNELS_RGB;
        int greyscale_value = (image_data[i] * GREYSCALE_R
                               + image_data[i + 1] * GREYSCALE_G
                               + image_data[i + 2] * GREYSCALE_B);

        new_image_data[j] = greyscale_value;
        // j++;
    }
    #pragma omp barrier
    return new_image_data;
}
