/*
 * File: main.cu
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: This file contains functions to apply greyscale, contrast and
 *              smoothing filters on a given image. The image must be an image
 *              with 3 RGB channels.
 *
 *              The image is divided, by an given percentage, in two parts:
 *              one part is calculated on the GPU using CUDA and the other part
 *              on the CPU using OpenMP. For each filter, the CUDA and OpenMP
 *              functions are executed in parallel using 2 threads:
 *
 *                            /---> CUDA ---> \
 *        filter-function -->|                |--> join results -> next filter
 *                            \--> OpenMP --> /
 *
 *              When the filters are applied, an PNG image is saved to disk.
 *
 * TODO: Combining brightness reduction functions. NOTE: CUDA brightness
 *       reduction uses the same array for the contrast.
 */

#include <cstdbool>
#include <cstdlib>
#include <iostream>
#include <thread> //Used for threads
#include <future> //used for async, since greyscale has a return value.
#include "greyscale.h"
#include "contrast.h"
#include "smoothing.h"
#include "timer.h"
// #include "cuda_helper.h"
#include "file.h"

using namespace std;

/* Image input/output settings. */
#define PNG_STRIDE_DEFAULT 0
#define NUM_CHANNELS_RGB 3
#define NUM_CHANNELS_GREYSCALE 1

/* Check whether two arguments are supplied. */
int check_argc(int argc) {
    if (argc != 6) {
        cout << "Error: wrong argument count.\n";
        cout << "Usage: ./main input_file output_file workload_gpu "<< \
        "(0-100 with increment of 10) num_blocks num_threads\n";

        return false;
    }

    return true;
}

/* Check whether the image is valid. */
int check_image(unsigned char *image_data, int num_channels) {
    if (!image_data || num_channels != NUM_CHANNELS_RGB) {
        cout << "Error reading file.\n";

        return false;
    }

    return true;
}

/* Apply the greyfilter using two threads: one for CUDA and one for OpenMP.
 * Since greyscale does not do any operation on image_data-memory, only one copy
 * of the array is used.
*/
unsigned char* apply_grey_filter(unsigned char* image_data, int num_pixels,
                      int gpu_end_index, int cpu_start_index, int block_size,
                      int num_threads) {

    cout << "Greyscale filter" << endl;
    /* Calculate the filter in parallel by using separate threads. */
    auto cuda_con = std::async(filter_greyscale_cuda, image_data, num_pixels, \
                               gpu_end_index, block_size);
    auto omp_con = std::async(filter_greyscale_omp, image_data, num_pixels, \
                              cpu_start_index, num_threads);

    /* Retrieve the values of the proceses. */
    unsigned char *temp_image_data_cuda = cuda_con.get();
    unsigned char *temp_image_data_omp = omp_con.get();;

    /* Copy both results into one array for further calculation. */
    memcpy(temp_image_data_omp, temp_image_data_cuda, \
           gpu_end_index * sizeof (unsigned char));
    return temp_image_data_omp;
}

/* Apply the contrast filter using two threads: one for CUDA and one for
 * OpenMP. TODO: Allocate GPU-memory in the main function so it can be used for
 * brightness, contrast and smoothing.
 */
void apply_contrast_filter(unsigned char* image_data,
                           unsigned char* image_data2, int num_pixels,
                           int gpu_end_index, int cpu_start_index,
                           int block_size, int num_threads) {

    cout << "Contrast filter" << endl;
    memcpy(image_data2, image_data, num_pixels * sizeof (unsigned char));
    std::thread cuda_con (filter_contrast_cuda, image_data2, num_pixels, \
                          gpu_end_index, block_size);
    std::thread omp_con (filter_contrast_omp, image_data, num_pixels, \
                         cpu_start_index, num_threads);
    cuda_con.join();
    omp_con.join();
    memcpy(image_data, image_data2, gpu_end_index * sizeof (unsigned char));
    memcpy(image_data2, image_data, num_pixels * sizeof (unsigned char));
}

/* Apply the smoothing filter using two threads: one for CUDA and one for
 * OpenMP.
 */
void apply_smoothing_filter(unsigned char* image_data,
                            unsigned char* image_data2, int num_pixels,
                            int width, int height, int gpu_end_index,
                            int cpu_start_index, int block_size,
                            int num_threads) {

    cout << "Smoothing filter" << endl;
    std::thread cuda_con (filter_smoothing_cuda, image_data2, num_pixels, width,
                       height, gpu_end_index, block_size);
    std::thread omp_con (filter_smoothing_omp, image_data, num_pixels, width,
                        height, cpu_start_index, num_threads);
    cuda_con.join();
    omp_con.join();
    memcpy(image_data, image_data2, gpu_end_index * sizeof (unsigned char));
    free(image_data2);
}

/* Process the entire image. Return true on success, false on failure. */
int process_image(char *file_in, char *file_out, int workload_gpu,
                  int block_size, int num_threads) {
    int width, height, num_channels;

    unsigned char *image_data = open_rgb_image(file_in, &width, &height,
                                           &num_channels);
    /* We stop processing if the image is invalid or we don't get the
    specific amount of channels we want. */
    if (!check_image(image_data, num_channels))
        return false;
    int num_pixels = width * height;

    /* Divide the number of pixels over the gpu and cpu. */
    int gpu_end_index = (int) num_pixels * ((float)workload_gpu/100.0f);
    int cpu_start_index = num_pixels - (num_pixels - gpu_end_index);

    unsigned char* temp = apply_grey_filter(image_data, num_pixels,
                                            gpu_end_index, cpu_start_index,
                                            block_size, num_threads);
    free_image(image_data);
    image_data = temp;

    unsigned char* image_data2 = (unsigned char*) malloc(num_pixels \
                                  * sizeof(unsigned char));
    if (image_data2 == NULL) {
        free_image(image_data);
        cout << "Could not allocate memory (malloc) in process image." << endl;
        return false;
    }
    apply_contrast_filter(image_data, image_data2, num_pixels, gpu_end_index,
                          cpu_start_index, block_size, num_threads);
    apply_smoothing_filter(image_data, image_data2, num_pixels, width, height,
                           gpu_end_index, cpu_start_index, block_size,
                           num_threads);

    write_grey_png(file_out, width, height, image_data);
    free_image(image_data);
    return true;
}

/* Main function. Checks the given arguments and passes it to the image
 * processing function.
 */
int main(int argc, char *argv[]) {
    if (!check_argc(argc))
        return EXIT_FAILURE;

    char *file_in = argv[1], *file_out = argv[2];
    int block_size = atoi(argv[3]);
    int num_threads = atoi(argv[4]);
    int workload_gpu = atoi(argv[5]);
    /* Checks if the workload parameter is within the correct interval. */
    if (workload_gpu > 100 || workload_gpu < 0){
        cout << "Workload must be an increment of 10" << endl;
        return EXIT_FAILURE;
    }

    if (!process_image(file_in, file_out, workload_gpu, block_size,
        num_threads)) {
            return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
