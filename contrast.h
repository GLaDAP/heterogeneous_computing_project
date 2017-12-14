#pragma once

void filter_contrast_omp(unsigned char *image_data, int num_pixels, int min_index);
void filter_contrast_cuda(unsigned char *image_data, int num_pixels, int max_index);
