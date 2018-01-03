/*
 * File: brightness.cc
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: Calculates the sum of brightness of the image using OpenMP
 *              reduction. Returns the sum of the brightness values.
 */

#include <stdbool.h>
#include <stdlib.h>
#include <iostream>
#include "timer.h"
#include <omp.h>

using namespace std;
/* Brightness reduction function using OpenMP reduction. Calculates the sum of
 * the brightness of all the pixels until a given index.
*/
long calculate_brightness_omp(unsigned char *image_data, int num_pixels,
                              int num_threads, int min_index) {
    long brightness_sum = 0;

    omp_set_num_threads(num_threads);
    timer ompTime = timer("omptime");
    ompTime.start();
    #pragma omp parallel for schedule (dynamic, (num_pixels / num_threads)) \
        reduction (+:brightness_sum)
    for (int i = min_index; i < num_pixels; i++) {
        brightness_sum += image_data[i];
    }
    #pragma omp barrier
    ompTime.stop();

    /* Print elapsed parallel time. */
    cout << fixed << setprecision(6);
    cout << "brightness (OMP): \t\t" << ompTime.getElapsed() \
          << " seconds." << endl;

    return brightness_sum;
}
