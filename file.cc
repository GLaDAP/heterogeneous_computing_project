
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


unsigned char * open_image(char *file_in, int* width, int* height,
                           int* num_channels, int NUM_CHANNELS_RGB) {
    return stbi_load(file_in, width, height, num_channels, NUM_CHANNELS_RGB);
}

void free_image (unsigned char * image) {
    stbi_image_free(image);
}

void write_image(char* file_out, int width, int height, \
                 int NUM_CHANNELS_GREYSCALE, unsigned char * image_data, \
                 int PNG_STRIDE_DEFAULT) {
    stbi_write_png(file_out, width, height, NUM_CHANNELS_GREYSCALE,
                   image_data, PNG_STRIDE_DEFAULT);

}
