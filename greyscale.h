// unsigned char *filter_greyscale(unsigned char *image_data, int num_pixels);
#pragma once

unsigned char * filter_greyscale_cuda(unsigned char *image_data, int num_pixels, int max_index);
unsigned char * filter_greyscale_omp(unsigned char *image_data, int num_pixels, int min_index);
