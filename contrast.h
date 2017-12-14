#pragma once

void filter_contrast_omp(unsigned char *image_data, int num_pixels);
void filter_contrast_cuda(unsigned char *image_data, int num_pixels);
