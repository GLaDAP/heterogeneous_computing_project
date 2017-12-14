/*
 * file: brightness.cu
 *
 */
#include <stdbool.h>
#include <stdlib.h>
#include <iostream>
#include "timer.h"
#include "cuda_helper.h"

using namespace std;



__global__ void brightness_reduction_kernel(unsigned char *data, int size,
                                            int* result) {
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

int calculate_brightness_cuda(unsigned char *device_image, int num_pixels) {
    int thread_block_size = 512;

    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    int brightness_sum;
    int* device_brightness_sum = (int*) allocateDeviceMemory(sizeof (int));
    int zero[] = {0};
    /* Initialize the timers used to measure the kernel invocation time and
     * memory transfer time.
     */
    memoryTime.start();
    memcpyHostToDevice(device_brightness_sum, &zero, sizeof (int));
    memoryTime.stop();

    int num_blocks = (num_pixels + thread_block_size - 1) / thread_block_size;
    kernelTime1.start();
    brightness_reduction_kernel<<<num_blocks, thread_block_size>>> \
        (device_image, num_pixels, device_brightness_sum);
    cudaDeviceSynchronize();
    kernelTime1.stop();
    checkCudaCall(cudaGetLastError());

    memoryTime.start();
    memcpyDeviceToHost(&brightness_sum, device_brightness_sum, sizeof (int));
    memoryTime.stop();

    freeDeviceMemory(device_brightness_sum);
    cout << fixed << setprecision(6);
    cout << "brightness (kernel): \t\t" << kernelTime1.getElapsed() \
          << " seconds." << endl;
    cout << "brightness (memory): \t\t" << memoryTime.getElapsed() \
         << " seconds." << endl;

    return brightness_sum;
}
