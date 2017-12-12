#include <stdbool.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

// #include "stb-master/stb_image.h"
// #include "stb-master/stb_image_write.h"

/* Image input/output settings. */
#define NUM_CHANNELS_RGB 3

/* Greyscale component weighting factors. */
#define GREYSCALE_R 0.2126
#define GREYSCALE_G 0.7152
#define GREYSCALE_B 0.0722

/* Perform a greyscale filter on an image. Return a new image data array
containing only 1 color channel. */
unsigned char *filter_greyscale(unsigned char *image_data, int num_pixels) {

    unsigned char *new_image_data = (unsigned char *) malloc(num_pixels);
    if (new_image_data == NULL) {
        cout << "Could not allocate memory" << endl;
        return NULL;
    }
    int total_size = num_pixels * NUM_CHANNELS_RGB;

    for(int i = 0, j = 0; i < total_size; i += NUM_CHANNELS_RGB, j++) {
        int greyscale_value = (image_data[i] * GREYSCALE_R
                               + image_data[i + 1] * GREYSCALE_G
                               + image_data[i + 2] * GREYSCALE_B);

        new_image_data[j] = greyscale_value;
    }

    return new_image_data;
}
