/*
 * File: contrast.h
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, david.puroja@student.uva.nl
 * Studentnumber: 11320788, 10469036
 *
 * Description: Applies the contrast filter on the image using CUDA and OpenMP.
 */

void filter_contrast_omp(unsigned char *image_data, int num_pixels,
                         long brightness, int min_index, int num_threads);
void filter_contrast_cuda(unsigned char *image_data, int num_pixels,
                          long brightness, int max_index,
                          int thread_block_size);
