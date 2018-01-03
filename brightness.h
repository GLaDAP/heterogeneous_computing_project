/*
 * File: brightness.h
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: Calculates the sum of brightness of the image using
 *              reduction on CUDA and OpenMP. Returns the sum of the brightness.
 *
 *              NOTE: The CUDA program uses the __shfl_down() function which is
 *              only available on Nvidia GPUs with a compute capability of 3.0
 *              or higher.
 */

long calculate_brightness_omp(unsigned char *image_data, int num_pixels,
                               int num_threads, int min_index);

unsigned long long int calculate_brightness_cuda(unsigned char *image_data,
                                                 int num_pixels, int max_index,
                                                 int thread_block_size);
