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
// #include "timer.h"
// #include "contrast.h"

using namespace std;

/* The maximum value we can use as RGB component. */
#define RGB_MAX_VALUE 255
/* Utility function, use to do error checking.
 * Use this function like this:
 * checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));
 * And to check the result of a kernel invocation:
 * checkCudaCall(cudaGetLastError());
 */
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}


__global__ void brightness_reduction_kernel(unsigned char *data, int size, int* result) {
    int sum = 0;
    unsigned int index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    for(unsigned int i = index; i < index + 4 && i < size; i++) {
        sum += int(data[i]);
    }
    /* Shuffle down. Shifts the register by adding the sum of the half of the
     * threads to the other half until one thread contains the sum. Since a
     * warp contains 32 threads, the shuffle-down operation starts at 16.
     */
    for(int i = 16; i > 0; (i >>= 1)){
        sum += __shfl_down(sum, i);
    }

    /* Add all the sums of the warps within the block to one variable. */
    __shared__ int block_sum;
    block_sum = 0;
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&block_sum, sum);
    }
    __syncthreads();
    /* Add the sum of the blocks to the result variable" Decoded: " \. */
    if (threadIdx.x == 0) {
        atomicAdd(result, block_sum);
    }
}

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

void filter_contrast_cuda(unsigned char *image_data, int num_pixels) {
    int thread_block_size = 512;

    unsigned char* device_image = NULL;
    checkCudaCall(cudaMalloc((void **) &device_image, \
                  num_pixels * sizeof(unsigned char)));
    if (device_image == NULL) {
        cout << "could not allocate memory on the GPU." << endl;
        exit(1);
    }
    int brightness_sum;
    int* device_brightness_sum = NULL;
    checkCudaCall(cudaMalloc((void **) &device_brightness_sum, sizeof (int)));
    if (device_brightness_sum == NULL) {
        checkCudaCall(cudaFree(device_image));
        cout << "could not allocate memory on the GPU." << endl;
        exit(1);
    }
    int zero[] = {0};
    /* Initialize the timers used to measure the kernel invocation time and
     * memory transfer time.
     */
    // timer kernelTime1 = timer("kernelTime");
    // timer memoryTime = timer("memoryTime");
    checkCudaCall(cudaMemcpy(device_brightness_sum, &zero, sizeof (int),
                         cudaMemcpyHostToDevice));


    // memoryTime.start();
    checkCudaCall(cudaMemcpy(device_image, image_data, \
                            num_pixels * sizeof(unsigned char), \
                             cudaMemcpyHostToDevice));
    // memoryTime.stop();


    int num_blocks = (num_pixels + thread_block_size - 1) / thread_block_size;
    // kernelTime1.start();
    brightness_reduction_kernel<<<num_blocks, thread_block_size>>> \
        (device_image, num_pixels, device_brightness_sum);
    cudaDeviceSynchronize();
    // kernelTime1.stop();
    checkCudaCall(cudaGetLastError());
    // memoryTime.start();
    checkCudaCall(cudaMemcpy(&brightness_sum, device_brightness_sum, sizeof (int),
                             cudaMemcpyDeviceToHost));
    // memoryTime.stop();

    /* And now the contrast */
    float brightness_mean = (double) brightness_sum / (double) num_pixels;
    cout << brightness_sum << endl;
    float denominator = sqrt(RGB_MAX_VALUE - brightness_mean);
    // for(int i = 0; i < num_pixels; i++) {
    //     int contrast_value = 0;
    //
    //     if(image_data[i] >= brightness_mean) {
    //         contrast_value = (sqrt(image_data[i] - brightness_mean)
    //                           / denominator * RGB_MAX_VALUE);
    //     }
    //
    //     image_data[i] = contrast_value;
    // }
    // kernelTime1.start();
    filter_contrast_kernel<<<num_blocks, thread_block_size>>> \
        (device_image, num_pixels, brightness_mean, denominator);
        cudaDeviceSynchronize();

        // kernelTime1.stop();
        checkCudaCall(cudaGetLastError());

    /* Copy the result image back to the GPU. */
    // memoryTime.start();
    checkCudaCall(cudaMemcpy(image_data, device_image, \
                             num_pixels * sizeof(unsigned char), \
                             cudaMemcpyDeviceToHost));
    // memoryTime.stop();

    /* Free used memory on the GPU. */
    checkCudaCall(cudaFree(device_image));
    checkCudaCall(cudaFree(device_brightness_sum));
    // cout << fixed << setprecision(6);
    // cout << "filter (kernel): \t\t" << kernelTime1.getElapsed() \
          << " seconds." << endl;
    // cout << "filter (memory): \t\t" << memoryTime.getElapsed() \
         << " seconds." << endl;
}
