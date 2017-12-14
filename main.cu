/*
 * File: contrast.cu
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: This file contains sequential implementations of different
 * image processing functions.
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
#include "cuda_helper.h"
#include "file.h"

using namespace std;

/* Image input/output settings. */
#define PNG_STRIDE_DEFAULT 0
#define NUM_CHANNELS_RGB 3
#define NUM_CHANNELS_GREYSCALE 1

/* Check whether two arguments are supplied. */
int check_argc(int argc) {
    if(argc != 6) {
        cout << "Error: wrong argument count.\n";
        cout << "Usage: ./main input_file output_file workload_gpu "<< \
        "(0-100 with increment of 10) num_blocks num_threads\n";

        return false;
    }

    return true;
}

/* Check whether the image is valid. */
int check_image(unsigned char *image_data, int num_channels) {
    if(!image_data || num_channels != NUM_CHANNELS_RGB) {
        cout << "Error reading file.\n";

        return false;
    }

    return true;
}

/* Process the entire image. Return true on success, false on failure. */
int process_image(char *file_in, char *file_out, int workload_gpu,
                  int cuda_block_size, int num_blocks) {

    int width, height, num_channels;
    unsigned char *image_data = open_image(file_in, &width, &height,
                                           &num_channels, NUM_CHANNELS_RGB);

    /* We stop processing if the image is invalid or we don't get the
    specific amount of channels we want. */
    if(!check_image(image_data, num_channels))
        return false;

    int num_pixels = width * height;

    /* First divide the number of pixels over the gpu and cpu. The GPU always
     * gets the first part of the array since it makes index comparison in the
     * kernel easier.
     */
    int GPU_START_INDEX = 0;
    int GPU_END_INDEX = (int) num_pixels * ((float)workload_gpu/100.0f);
    int CPU_START_INDEX = num_pixels - (num_pixels - GPU_END_INDEX);
    int CPU_END_INDEX = num_pixels;
    cout << "Workload is divided in: GPU: " << GPU_START_INDEX <<"-" \
    << GPU_END_INDEX << " | CPU: " << CPU_START_INDEX << "-" << CPU_END_INDEX \
    << endl;

    /* Now the array is divided according to the given workload-division, the
     * first filter is called. The brightness_sum is the only function NOT
     * implemented heterogeneous.
     */
    /* Allocate the image on the GPU for ALL the calculations. */


    /* Brightness sum on CUDA reduction kernel.. */

    /* Apply filter 1 and swap the newly acquired image data. */
    cout << "Greyscale filter" << endl;

    auto cuda_one = std::async(filter_greyscale_cuda, image_data, num_pixels, GPU_END_INDEX);     // spawn new thread that calls foo()
    auto omp_one = std::async(filter_greyscale_omp, image_data, num_pixels, CPU_START_INDEX);  // spawn new thread that calls bar(0)

    // filter_greyscale_cuda(image_data, temp_image_data_cuda, num_pixels, GPU_END_INDEX);
    // filter_greyscale_omp(image_data, temp_image_data_omp, num_pixels, CPU_START_INDEX);
                  // pauses until first finishes
                   // pauses until second finishes
    unsigned char *temp_image_data_cuda = cuda_one.get();
    unsigned char *temp_image_data_omp = omp_one.get();;
    memcpy(temp_image_data_omp, temp_image_data_cuda, (GPU_END_INDEX) * sizeof (unsigned char));

    free_image(image_data);
    image_data = temp_image_data_omp;
    cout << "Contrast filter" << endl;
    unsigned char* image_data2 = (unsigned char*) malloc(num_pixels \
                                  * sizeof(unsigned char));
    memcpy(image_data2, image_data, num_pixels * sizeof(unsigned char));
    /* Apply filter 2. */


    // filter_contrast_cuda(image_data2, num_pixels, (GPU_END_INDEX));
    // filter_contrast_omp(image_data, num_pixels, CPU_START_INDEX);
    std::thread first (filter_contrast_cuda, image_data2, num_pixels, (GPU_END_INDEX));
    std::thread second (filter_contrast_omp, image_data, num_pixels, CPU_START_INDEX);

    first.join();                // pauses until first finishes
    second.join();               // pauses until second finishes
    memcpy(image_data, image_data2, (GPU_END_INDEX) * sizeof (unsigned char));
    memcpy(image_data2, image_data, (num_pixels * sizeof (unsigned char)));

    /* Apply filter 3. */
    cout << "Smoothing filter" << endl;
    std::thread third (filter_smoothing_cuda, image_data2, num_pixels, width, height, GPU_END_INDEX);
    std::thread fourth (filter_smoothing_omp, image_data, num_pixels, width, height, CPU_START_INDEX);

    // filter_smoothing_cuda(image_data2, num_pixels, width, height, GPU_END_INDEX);
    // filter_smoothing_omp(image_data, num_pixels, width, height, CPU_START_INDEX);
    third.join();                // pauses until first finishes
    fourth.join();               // pauses until second finishes
    memcpy(image_data, image_data2, (GPU_END_INDEX) * sizeof (unsigned char));

    write_image(file_out, width, height, NUM_CHANNELS_GREYSCALE,
                image_data, PNG_STRIDE_DEFAULT);
    free_image(image_data);
    return true;
}

/* Main function. Checks the given arguments and  */
int main(int argc, char *argv[]) {
    if(!check_argc(argc))
        return EXIT_FAILURE;

    char *file_in = argv[1], *file_out = argv[2];
    int workload_gpu = atoi(argv[3]);
    int cuda_block_size = atoi(argv[4]);
    int num_threads = atoi(argv[5]);

    if(workload_gpu > 100 || workload_gpu < 0 || workload_gpu % 10 != 0){
        cout << "Workload must be an increment of 10" << endl;
        return EXIT_FAILURE;
    }

    if(!process_image(file_in, file_out, workload_gpu, cuda_block_size, num_threads))
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
