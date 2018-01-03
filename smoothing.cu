/*
 * File: smoothingomp.cc
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: Applies the smoothing filter on the image using CUDA.
 */
#include <stdbool.h>
#include <stdlib.h>
#include <iostream>
#include "timer.h"
#include "cuda_helper.h"

using namespace std;

/* Kernel dimension values. */
#define KERNEL_WIDTH 5
#define KERNEL_SIZE KERNEL_WIDTH * KERNEL_WIDTH
#define KERNEL_OFFSET KERNEL_WIDTH / 2
#define KERNEL_MULTIPLIER (1.0 / 81.0)

/* Triangular smoothing kernel. */
const int kernel_1D[KERNEL_WIDTH * KERNEL_WIDTH] = {1, 2, 3, 2, 1,
                                                    2, 4, 6, 4, 2,
                                                    3, 6, 9, 6, 3,
                                                    2, 4, 6, 4, 2,
                                                    1, 2, 3, 2, 1};

/* Smoothing filter kernel. */
__global__ void smoothing_kernel(unsigned char* image_data,
                                 unsigned char* temp_image_data, int* kernel,
                                 int num_pixels, int width, int height) {
    /* Calculate thread index. */
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
    int col = i % width;
    int row = i / width;

    /* Boundary check for the current kernel center. */
    if (i < num_pixels && !(row < KERNEL_OFFSET || col < KERNEL_OFFSET ||
         row > (height - KERNEL_OFFSET) || col > (width - KERNEL_OFFSET))) {
        int accumulator = 0;
        for (int kernel_index = 0; kernel_index < KERNEL_SIZE; kernel_index++) {
            int kx = (kernel_index % KERNEL_WIDTH);
            int ky = (kernel_index / KERNEL_WIDTH);

            int index = i + (kx - KERNEL_OFFSET) + width * (ky - KERNEL_OFFSET);
            accumulator += (kernel[ky * KERNEL_WIDTH + kx] \
                            * temp_image_data[index]);
        }
        accumulator *= KERNEL_MULTIPLIER;
        image_data[i] = accumulator;
    }
}

/* Allocates the necessary memory on the GPU and executes the CUDA-kernel.  */
void filter_smoothing_cuda(unsigned char *image_data, int num_pixels,
                             int width, int height, int max_index,
                             int thread_block_size) {

    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");
    /* Allocate the image used to calculate the smoothing. */

    unsigned char* device_image = (unsigned char*) allocateDeviceMemory( \
        num_pixels * sizeof (unsigned char));
    unsigned char* device_temp_image = (unsigned char*) allocateDeviceMemory( \
        num_pixels * sizeof (unsigned char));

    int* device_kernel_array = (int*) allocateDeviceMemory( \
        KERNEL_SIZE * sizeof (int));

    /* Copy the data to the device. */
    memoryTime.start();
    memcpyHostToDevice(device_kernel_array, (int*) kernel_1D, \
                       KERNEL_SIZE * sizeof (int));
    memcpyHostToDevice(device_image, image_data, \
                       num_pixels * sizeof (unsigned char));
    memcpyDeviceToDevice(device_temp_image, device_image, \
                         num_pixels * sizeof (unsigned char));
    memoryTime.stop();

    int num_blocks = (num_pixels + thread_block_size - 1) / thread_block_size;
    /* Start smoothing kernel for all the pixels. This can be changed if OpenMP
     * is used to split the image size into different kernels. The kernels can
     * be executed sequentially on the GPU.
     */
    kernelTime1.start();
    smoothing_kernel<<<num_blocks, thread_block_size>>> \
        (device_image, device_temp_image, device_kernel_array, max_index, \
         width, height);
    kernelTime1.stop();
    checkCudaCall(cudaGetLastError());

    /* Copy the result back to the host. Only the pixels actually calculated
     * by the GPU are copied back.
     */
    memoryTime.start();
    memcpyDeviceToHost(image_data, device_image,
                       max_index * sizeof (unsigned char));
    memoryTime.stop();

    freeDeviceMemory(device_image);
    freeDeviceMemory(device_kernel_array);

    /* Print the elapsed time. */
    cout << fixed << setprecision(6);
    cout << "smoothing (kernel): \t\t" << kernelTime1.getElapsed() \
         << " seconds." << endl;
    cout << "smoothing (memory): \t\t" << memoryTime.getElapsed() \
         << " seconds." << endl;
}
