/*
 * File: greyscale.h
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: Applies the greyscale filter on the image using CUDA and OpenMP.
 */

void filter_greyscale_cuda(unsigned char *image_data, int num_pixels,
                           int max_index, int thread_block_size,
                           unsigned char** result);
void filter_greyscale_omp(unsigned char *image_data, int num_pixels,
                          int min_index, int num_threads,
                          unsigned char** result);
