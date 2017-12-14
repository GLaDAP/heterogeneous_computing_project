// void filter_smoothing(unsigned char *image_data, int num_pixels, int width, int height);
#pragma once
void filter_smoothing_omp(unsigned char *image_data, int num_pixels, int width, int height);
void filter_smoothing_cuda(unsigned char *image_data, int num_pixels, int width, int height);