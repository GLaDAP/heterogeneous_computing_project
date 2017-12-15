/*
 * File: file.h
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: File containing functions to open and save images using the STB
 *              library by Sean Barrett: https://github.com/nothings/stb
 */
unsigned char * open_rgb_image(char *file_in, int* width, int* height, \
                               int* num_channels);
void free_image (unsigned char * image);
void write_grey_png(char* file_out, int width, int height,
                    unsigned char * image_data);
