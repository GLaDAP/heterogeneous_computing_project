/*
 * File: smoothing.cu
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: This file contains sequential implementations of different
 * image processing functions.
 */
#include <stdbool.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

/* Kernel dimension values. */
#define KERNEL_WIDTH 5
#define KERNEL_OFFSET KERNEL_WIDTH / 2
#define KERNEL_MULTIPLIER (1.0 / 81.0)

/* Triangular smoothing kernel. */
const int kernel[KERNEL_WIDTH][KERNEL_WIDTH] = {{1, 2, 3, 2, 1},
                                                {2, 4, 6, 4, 2},
                                                {3, 6, 9, 6, 3},
                                                {2, 4, 6, 4, 2},
                                                {1, 2, 3, 2, 1}};


/* Perform a triangular smoothing filter on an image. */
void filter_smoothing(unsigned char *image_data, int num_pixels, int width, int height) {
    unsigned char *temp_image_data = (unsigned char *) malloc(num_pixels);
    if (temp_image_data == NULL) {
        cout << "Could not allocate memory in smoothin function." << endl;
        exit(1);
    }
    memcpy(temp_image_data, image_data, num_pixels);

    /* OpenMP since the IF-statement is too complex */
    for(int i = 0; i < num_pixels; i++) {
        int col = i % width;
        int row = i / width;

        /* Boundary check for the current kernel center. */
        if(row < KERNEL_OFFSET || col < KERNEL_OFFSET
           || row > (height - KERNEL_OFFSET) || col > (width - KERNEL_OFFSET)) {
               continue;
           }

        int accumulator = 0;
        for (int ky = -KERNEL_OFFSET; ky <= KERNEL_OFFSET; ky++) {
            for (int kx = -KERNEL_OFFSET; kx <= KERNEL_OFFSET; kx++) {
                accumulator += (kernel[ky + KERNEL_OFFSET][kx + KERNEL_OFFSET]
                                * temp_image_data[i + kx + width * ky]);
            }
        }
        accumulator *= KERNEL_MULTIPLIER;

        image_data[i] = accumulator;
    }

    free(temp_image_data);
}
