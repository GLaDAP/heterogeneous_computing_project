/*
 * File: brightness.cu
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: Calculates the sum of brightness of the image using CUDA
 *              reduction. Returns the sum of the brightness. This program uses
 *              the __shfl_down() functionwhich is only available on
 *              Nvidia GPUs with a compute capability of 3.0 or higher.
 *
 */
#include <stdbool.h>
#include <stdlib.h>
#include <iostream>
#include "timer.h"
#include "cuda_helper.h"

using namespace std;

/* Brightness reduction kernel. Calculates the sum of the brightness of all the
 * pixels until a given index. This function uses the __shfl_down() function
 * which is only available on Nvidia GPUs with a compute capability of 3.0 or
 * higher.
 */
__global__ void brightness_reduction_kernel(unsigned char *data, int size,
                                            unsigned long long int* result) {
    int sum = 0;
    unsigned int index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    for (unsigned int i = index; i < index + 4 && i < size; i++) {
        sum += int(data[i]);
    }
    /* Shuffle down. Shifts the register by adding the sum of the half of the
     * threads to the other half until one thread contains the sum. Since a
     * warp contains 32 threads, the shuffle-down operation starts at 16.
     */
    for (int i = 16; i > 0; (i >>= 1)){
        sum += __shfl_down(sum, i, 32);
    }

    /* Add all the sums of the warps within the block to one variable. */
    __shared__ unsigned long long int block_sum;
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

/* Calculates the brightness of the picture used by the contrast filter.
 * This function copies the image into the GPU memory and times the duration
 * of the operations. Returns a unsigned long long int to prevent wrap around
 * when large images are used in the calculation.
 */
unsigned long long int calculate_brightness_cuda(unsigned char *image_data,
                                                 int num_pixels, int max_index,
                                                 int thread_block_size) {
    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    unsigned char* device_image = (unsigned char*) allocateDeviceMemory( \
        num_pixels * sizeof (unsigned char));
    memcpyHostToDevice(device_image, image_data, \
                       num_pixels * sizeof (unsigned char));

    unsigned long long int brightness_sum;
    unsigned long long int* device_brightness_sum = (unsigned long long int*) \
        allocateDeviceMemory(sizeof (unsigned long long int));
    unsigned long long int zero[] = {0};

    /* Initialize the timers used to measure the kernel invocation time and
     * memory transfer time.
     */
    memoryTime.start();
    memcpyHostToDevice(device_brightness_sum, &zero,
                       sizeof (unsigned long long int));
    memoryTime.stop();

    int num_blocks = (max_index + thread_block_size - 1) / thread_block_size;
    kernelTime1.start();
    brightness_reduction_kernel<<<num_blocks, thread_block_size>>> \
        (device_image, max_index, device_brightness_sum);
    cudaDeviceSynchronize();
    kernelTime1.stop();
    checkCudaCall(cudaGetLastError());

    memoryTime.start();
    memcpyDeviceToHost(&brightness_sum, device_brightness_sum,
                       sizeof (unsigned long long int));
    memoryTime.stop();
    cout << "Brightness cuda: " << brightness_sum << endl;
    freeDeviceMemory(device_brightness_sum);
    freeDeviceMemory(device_image);

    /* Print the elapsed time. */
    cout << fixed << setprecision(6);
    cout << "brightness (kernel): \t\t" << kernelTime1.getElapsed() \
          << " seconds." << endl;
    cout << "brightness (memory): \t\t" << memoryTime.getElapsed() \
         << " seconds." << endl;

    return brightness_sum;
}
