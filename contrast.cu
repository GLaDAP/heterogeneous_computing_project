/*
 * File: contrast.cu
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, david.puroja@student.uva.nl
 * Studentnumber: 11320788, 10469036
 *
 * Description: Applies the contrast filter on the image using CUDA.
 */

#include <stdbool.h>
#include <stdlib.h>
#include <iostream>
#include "timer.h"
#include "cuda_helper.h"

using namespace std;

/* The maximum value we can use as RGB component. */
#define RGB_MAX_VALUE 255

/* CUDA contrast filter kernel. Calculates for each pixel in its range the
 * new value using the mean and denominator of the brightness sum. This kernel
 * executes in two steps: the first step are the threads with a pixel which
 * satisfies the requirement of the second if-statement. The second stage is the
 * else. This is due the architecture of the GPU where all the threads in a warp
 * executes the same instruction.
 */
__global__ void filter_contrast_kernel(unsigned char *image_data, int size,
                                       double mean, double denominator) {
    unsigned int index = (blockIdx.x * blockDim.x + threadIdx.x);
    if (index < size) {
        if (image_data[index] >= mean) {
            image_data[index] = (sqrt(image_data[index] - mean)
                              / denominator * RGB_MAX_VALUE);
        }
        else {
            image_data[index] = 0;
        }
    }
}

/* Allocates resources on the GPU, times the memory and kernel operations,
 * and executes the kernel.
 */
void filter_contrast_cuda(unsigned char *image_data, int num_pixels,
                          long brightness_sum, int max_index,
                          int thread_block_size) {
    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    /* Copy the image to the device. */
    unsigned char* device_image = (unsigned char*) allocateDeviceMemory( \
        num_pixels * sizeof (unsigned char));
    memcpyHostToDevice(device_image, image_data, \
                       num_pixels * sizeof (unsigned char));

    double brightness_mean = (double) (brightness_sum / (double) num_pixels);
    double denominator = sqrt(RGB_MAX_VALUE - brightness_mean);

    int num_blocks = (max_index + thread_block_size - 1) / thread_block_size;

    kernelTime1.start();
    filter_contrast_kernel<<<num_blocks, thread_block_size>>> \
        (device_image, max_index, brightness_mean, denominator);
        cudaDeviceSynchronize();
    kernelTime1.stop();
    checkCudaCall(cudaGetLastError());

    /* Copy the result image back to the GPU. */
    memoryTime.start();
    memcpyDeviceToHost(image_data, device_image, \
                       num_pixels * sizeof (unsigned char));
    memoryTime.stop();

    /* Free used memory on the GPU. */
    freeDeviceMemory(device_image);

    /* Print the elapsed time. */
    cout << fixed << setprecision(6);
    cout << "contrast (kernel): \t\t" << kernelTime1.getElapsed() \
          << " seconds." << endl;
    cout << "contrast (memory): \t\t" << memoryTime.getElapsed() \
         << " seconds." << endl;
}
