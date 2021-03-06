/*
 * File: smoothing.h
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, david.puroja@student.uva.nl
 * Studentnumber: 11320788, 10469036
 *
 * Description: Applies the smoothing filter on the image using CUDA and OpenMP.
 */

#pragma once

void filter_smoothing_omp(unsigned char *image_data, int num_pixels, int width,
                          int height, int min_index, int num_threads);
void filter_smoothing_cuda(unsigned char *image_data, int num_pixels, int width,
                           int height, int max_index, int thread_block_size);
