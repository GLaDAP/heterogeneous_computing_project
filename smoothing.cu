/*
 * File: smoothing.cu
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: This file contains sequential implementations of different
 * image processing functions.

NOTE: THE NUM-PIXEL MUST BE SEPARATED BY A PERCENTAGE SO THE CPU AND GPU CAN
CONDUCT THE SAME CALCULATION IN PARALLEL. RESEARCH: FIND THE SWEETSPOT/PERCENTAGE
WHERE THE GPU AND CPU HETEROGENOUOS COMPUTATION IS OPTIMAL.

DO THIS FOR ALL THE FUNCTIONS BY CONVERTING THE SEQUENTIAL VERSION TO AN
OPENMP VERSION AND A CUDA VERSION WITH PERCENTAGE.

THE ARRAYS WITH THE PIXELS CAN BE MERGED LATER SINCE THE CUT-OFF POINT OF THE
ARRAY IS KNOWN. (USE MEMCPY TO PERFORM THE OPERATION)

EACH FUNCTION IN PARALLEL HAS THE WHOLE IMAGE IN THEIR MEMORY TO DO THE COMPUTATION.





 */
#include <stdbool.h>
#include <stdlib.h>
#include <iostream>
#include "timer.h"

using namespace std;

/* Kernel dimension values. */
#define KERNEL_WIDTH 5
#define KERNEL_OFFSET KERNEL_WIDTH / 2
#define KERNEL_MULTIPLIER (1.0 / 81.0)

/* Triangular smoothing kernel. */
const int kernel_1D[KERNEL_WIDTH * KERNEL_WIDTH] = {1, 2, 3, 2, 1,
                                                    2, 4, 6, 4, 2,
                                                    3, 6, 9, 6, 3,
                                                    2, 4, 6, 4, 2,
                                                    1, 2, 3, 2, 1};
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

/* Smoothing filter kernel. */
__global__ void smoothing_kernel(unsigned char* image_data,
                                 unsigned char* temp_image_data, int* kernel,
                                 int num_pixels, int width, int height) {

    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
    int col = i % width;
    int row = i / width;

    /* Boundary check for the current kernel center. */
    if(!(row < KERNEL_OFFSET || col < KERNEL_OFFSET ||
         row > (height - KERNEL_OFFSET) || col > (width - KERNEL_OFFSET))) {
        int accumulator = 0;
        for (int kernel_index = 0; kernel_index < 25; kernel_index++) {
            int kx = (kernel_index % KERNEL_WIDTH);
            int ky = (kernel_index / KERNEL_WIDTH);

            int index = i + (kx - KERNEL_OFFSET) + width * (ky - KERNEL_OFFSET);
            accumulator += (kernel[ky*KERNEL_WIDTH+kx] * temp_image_data[index]);
        }
        accumulator *= KERNEL_MULTIPLIER;
        image_data[i] = accumulator;
    }
}

/* Allocates the necessary memory on the GPU and executes the CUDA-kernel.  */
void filter_smoothing_cuda(unsigned char *image_data, int num_pixels,
                             int width, int height) {

    int thread_block_size = 512;
    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");
    /* Allocate the image used to calculate the smoothing. */
    unsigned char* device_image = NULL;
    checkCudaCall(cudaMalloc((void **) &device_image, \
                  num_pixels * sizeof(unsigned char)));
    if (device_image == NULL) {
        cout << "could not allocate memory on the GPU." << endl;
        exit(1);
    }

    unsigned char* device_temp_image = NULL;
    checkCudaCall(cudaMalloc((void **) &device_temp_image, \
                  num_pixels * sizeof(unsigned char)));
    if (device_temp_image == NULL) {
        cout << "could not allocate memory on the GPU." << endl;
        exit(1);
    }

    int* device_kernel_array = NULL;
    checkCudaCall(cudaMalloc((void **) &device_kernel_array, \
                  KERNEL_WIDTH * KERNEL_WIDTH * sizeof(int)));
    if (device_kernel_array == NULL) {
        cout << "could not allocate memory on the GPU." << endl;
        exit(1);
    }


    memoryTime.start();
    checkCudaCall(cudaMemcpy(device_kernel_array, kernel_1D, \
                            25 * sizeof(int), \
                            cudaMemcpyHostToDevice));

    checkCudaCall(cudaMemcpy(device_image, image_data, \
                            num_pixels * sizeof(unsigned char), \
                            cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(device_temp_image, device_image, \
                            num_pixels * sizeof(unsigned char), \
                            cudaMemcpyDeviceToDevice));
    memoryTime.stop();

    int num_blocks = (num_pixels + thread_block_size - 1) / thread_block_size;
    /* Start smoothing kernel for all the pixels. This can be changed if OpenMP
     * is used to split the image size into different kernels. The kernels can
     * be executed sequentially on the GPU.
     */
    kernelTime1.start();
    smoothing_kernel<<<num_blocks, thread_block_size>>> \
        (device_image, device_temp_image, device_kernel_array, num_pixels, width, height);
    kernelTime1.stop();

    checkCudaCall(cudaGetLastError());

    memoryTime.start();
    checkCudaCall(cudaMemcpy(image_data, device_image, \
                             num_pixels * sizeof(unsigned char), \
                             cudaMemcpyDeviceToHost));
    memoryTime.stop();
    checkCudaCall(cudaFree(device_image));
    checkCudaCall(cudaFree(device_kernel_array));

    cout << fixed << setprecision(6);
    cout << "smoothing (kernel): \t\t" << kernelTime1.getElapsed() \
         << " seconds." << endl;
    cout << "smoothing (memory): \t\t" << memoryTime.getElapsed() \
         << " seconds." << endl;
}
