/*
 * File: main.cc
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * USAGE:       ./rgb2grey source.jpg output.png num_blocks num_threads
 *              workload_gpu[0-100]
 *
 * NOTE:        This program can only load 3-channel image files.
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
 */

#include <cstdlib>
#include <iostream>
#include <thread>
#include "greyscale.h"
#include "contrast.h"
#include "smoothing.h"
#include "timer.h"
#include "brightness.h"
#include "file.h"

using namespace std;

/* Image input/output settings. */
#define PNG_STRIDE_DEFAULT 0
#define NUM_CHANNELS_RGB 3
#define NUM_CHANNELS_GREYSCALE 1

/* Check whether enough arguments are supplied. */
int check_argc(int argc) {
    if (argc != 6) {
        cout << "Error: wrong argument count.\n";
        cout << "Usage: ./main input_file output_file "<< \
        "block_size num_threads workload_gpu (between 1 and 100)\n";

        return false;
    }

    return true;
}

/* Check whether the image is valid. */
int check_image(unsigned char *image_data, int num_channels) {
    if (!image_data) {
        cout << "Error reading file. No valid data pointer returned.\n";

        return false;
    } else if (num_channels != NUM_CHANNELS_RGB) {
        cout << "Error reading file. Image does not contain 3 channels.\n";

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
    unsigned char *temp_image_data_cuda;
    unsigned char *temp_image_data_omp;

    /* Calculate the filter in parallel by using separate threads. */
    std::thread cuda_con (filter_greyscale_cuda, image_data, num_pixels, \
                          gpu_end_index, block_size, &temp_image_data_cuda);
    std::thread omp_con (filter_greyscale_omp, image_data, num_pixels, \
                         cpu_start_index, num_threads, &temp_image_data_omp);

    /* Wait for the processes to finish. */
    cuda_con.join();
    omp_con.join();

    /* Copy both results into one array for further calculation. */
    memcpy(temp_image_data_omp, temp_image_data_cuda, \
           gpu_end_index * sizeof (unsigned char));
    return temp_image_data_omp;
}

/* Apply the contrast filter using two threads: one for CUDA and one for
 * OpenMP.
 */
void apply_contrast_filter(unsigned char* image_data,
                           unsigned char* image_data2, int num_pixels,
                           int gpu_end_index, int cpu_start_index,
                           long brightness, int block_size, int num_threads) {

    cout << "Contrast filter" << endl;
    memcpy(image_data2, image_data, num_pixels * sizeof (unsigned char));
    std::thread cuda_con (filter_contrast_cuda, image_data2, num_pixels, \
                          brightness, gpu_end_index, block_size);
    std::thread omp_con (filter_contrast_omp, image_data, num_pixels, \
                         brightness, cpu_start_index, num_threads);

    /* Wait for the processes to finish. */
    cuda_con.join();
    omp_con.join();
    /* Copy the images from both processes to one image array. */
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

    /* Wait for the processes to finish. */
    cuda_con.join();
    omp_con.join();

    memcpy(image_data, image_data2, gpu_end_index * sizeof (unsigned char));
    free(image_data2);
}

/* Calculate the brightness used by the contrast filter with CUDA and OpenMP.
 */
long calculate_brightness(unsigned char* image_data, int num_pixels,
                          int cpu_index, int gpu_index, int num_threads,
                          int block_size) {

    cout << "Brightness calculation" << endl;

    long brightness_sum;
    long brightness_omp = calculate_brightness_omp(image_data, num_pixels,
                                                   num_threads, cpu_index);
    long brightness_cuda = calculate_brightness_cuda(image_data, num_pixels,
                                                     gpu_index, block_size);
    brightness_sum = brightness_omp + brightness_cuda;

    return brightness_sum;
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
    /* Only time when the filters are applied. */
    timer progTimer = timer("programtimer");
    progTimer.start();

    unsigned char* temp = apply_grey_filter(image_data, num_pixels,
                                            gpu_end_index, cpu_start_index,
                                            block_size, num_threads);
    free_image(image_data);
    image_data = temp;

    unsigned char* image_data2 = (unsigned char*) malloc(num_pixels \
                                  * sizeof (unsigned char));
    if (image_data2 == NULL) {
        free_image(image_data);
        cout << "Could not allocate memory (malloc) in process image." << endl;
        return false;
    }

    /* First calculate the brightness, then apply the contrast filter. */
    long brightness = calculate_brightness(image_data, num_pixels,
                                           cpu_start_index, gpu_end_index,
                                           num_threads, block_size);

    apply_contrast_filter(image_data, image_data2, num_pixels, gpu_end_index,
                          cpu_start_index, brightness, block_size, num_threads);

    apply_smoothing_filter(image_data, image_data2, num_pixels, width, height,
                           gpu_end_index, cpu_start_index, block_size,
                           num_threads);
    progTimer.stop();
    /* Print elapsed parallel time. */
    cout << fixed << setprecision(6);
    cout << "Program Timer: \t\t" << progTimer.getElapsed() \
         << " seconds." << endl;

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
    if (workload_gpu > 100 || workload_gpu < 1){
        cout << "GPU-workload must be between 1 and 100." << endl;
        return EXIT_FAILURE;
    }

    cout << "GPU workload (percentage): " << workload_gpu << \
    " Number of threads OMP: " << num_threads << " BlockSize CUDA: " \
    << block_size << endl;


    if (!process_image(file_in, file_out, workload_gpu, block_size,
        num_threads)) {
            return EXIT_FAILURE;
    }


    return EXIT_SUCCESS;
}
