/*
 * File: smoothingomp.cc
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: Applies the smoothing filter on the image using OpenMP.
 */

#include <stdbool.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <cstring>
#include "timer.h"

using namespace std;

/* Kernel dimension values. */
#define KERNEL_WIDTH 5
#define KERNEL_SIZE KERNEL_WIDTH * KERNEL_WIDTH
#define KERNEL_OFFSET KERNEL_WIDTH / 2
#define KERNEL_MULTIPLIER (1.0 / 81.0)

/* Triangular smoothing kernel. */
const int kernel[KERNEL_SIZE] = {1, 2, 3, 2, 1,
                                 2, 4, 6, 4, 2,
                                 3, 6, 9, 6, 3,
                                 2, 4, 6, 4, 2,
                                 1, 2, 3, 2, 1};
/* Perform a triangular smoothing filter on an image. */
void filter_smoothing_omp(unsigned char *image_data, int num_pixels, int width,
                          int height, int min_index, int num_threads) {

    unsigned char *temp_image_data = (unsigned char *) malloc(num_pixels \
                                   * sizeof (unsigned char*));
    if (temp_image_data == NULL) {
        cout << "Could not allocate memory in smoothin function." << endl;
        return;
    }

    memcpy(temp_image_data, image_data, num_pixels);

    omp_set_num_threads(num_threads);
    int total_indices = num_pixels - min_index;

    timer ompTimeSmooth = timer("omptime");
    ompTimeSmooth.start();

    #pragma omp parallel for schedule (dynamic, (total_indices / num_threads))
    for (int i = min_index; i < num_pixels; i++) {

        int col = i % width;
        int row = i / width;

        /* Boundary check for the current kernel center. */
        if (!(row < KERNEL_OFFSET || col < KERNEL_OFFSET \
           || row > (height - KERNEL_OFFSET) \
           || col > (width - KERNEL_OFFSET))) {
            int accumulator = 0;
            for (int kernel_index = 0; kernel_index < 25; kernel_index++) {
                int kx = (kernel_index % KERNEL_WIDTH);
                int ky = (kernel_index / KERNEL_WIDTH);

                int index = i + (kx - KERNEL_OFFSET) + width \
                              * (ky - KERNEL_OFFSET);
                accumulator += (kernel[ky * KERNEL_WIDTH + kx] \
                                * temp_image_data[index]);
            }
            accumulator *= KERNEL_MULTIPLIER;
            image_data[i] = accumulator;
        }
        else {
            image_data[i] = 0;
        }
    }
    #pragma omp barrier
    ompTimeSmooth.stop();

    /* Print elapsed parallel time. */
    cout << fixed << setprecision(6);
    cout << "smoothing (OMP): \t\t" << ompTimeSmooth.getElapsed() \
          << " seconds." << endl;

    free(temp_image_data);
}
