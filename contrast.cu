/*
 * File: main.cpp
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
#include "timer.h"
#include "cuda_helper.h"
#include "brightness.h"
using namespace std;

/* The maximum value we can use as RGB component. */
#define RGB_MAX_VALUE 255

/* Compare with a OpenMP implementation? */
__global__ void filter_contrast_kernel(unsigned char *image_data,
                                       int size, float mean,
                                       float denominator) {
    unsigned int index = (blockIdx.x * blockDim.x + threadIdx.x);
    if(index < size) {
        if(image_data[index] >= mean) {
            image_data[index] = (sqrt(image_data[index] - mean)
                              / denominator * RGB_MAX_VALUE);
        }
        else {
            image_data[index] = 0;
        }
    }
}

void filter_contrast_cuda(unsigned char *image_data, int num_pixels, int max_index) {
    int thread_block_size = 512;
    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    unsigned char* device_image = (unsigned char*) allocateDeviceMemory( \
        num_pixels * sizeof (unsigned char));
    memcpyHostToDevice(device_image, image_data, num_pixels * sizeof(unsigned char));

    int brightness_sum = calculate_brightness_cuda(device_image, num_pixels);
    int num_blocks = (num_pixels + thread_block_size - 1) / thread_block_size;

    /* And now the contrast */
    float brightness_mean = (double) brightness_sum / (double) num_pixels;
    float denominator = sqrt(RGB_MAX_VALUE - brightness_mean);
    kernelTime1.start();
    filter_contrast_kernel<<<num_blocks, thread_block_size>>> \
        (device_image, max_index, brightness_mean, denominator);
        cudaDeviceSynchronize();

    kernelTime1.stop();
    checkCudaCall(cudaGetLastError());

    /* Copy the result image back to the GPU. */
    memoryTime.start();
    memcpyDeviceToHost(image_data, device_image, num_pixels * sizeof (unsigned char));
    memoryTime.stop();

    /* Free used memory on the GPU. */
    freeDeviceMemory(device_image);
    cout << fixed << setprecision(6);
    cout << "contrast (kernel): \t\t" << kernelTime1.getElapsed() \
          << " seconds." << endl;
    cout << "contrast (memory): \t\t" << memoryTime.getElapsed() \
         << " seconds." << endl;
}
