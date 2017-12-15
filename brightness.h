/*
 * File: brightness.h
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: Calculates the sum of brightness of the image using
 *              reduction. Returns the sum of the brightness.
 */
unsigned long long int calculate_brightness_cuda(unsigned char *device_image,
                                                 int num_pixels,
                                                 int thread_block_size);
