#include <stdbool.h>
#include <stdlib.h>
#include <iostream>
#include "timer.h"
#include "cuda_helper.h"

using namespace std;


/* Image input/output settings. */
#define NUM_CHANNELS_RGB 3

/* Greyscale component weighting factors. */
#define GREYSCALE_R 0.2126
#define GREYSCALE_G 0.7152
#define GREYSCALE_B 0.0722


/* Kernel converting the image to greyscale. */
__global__ void greyscale_kernel(unsigned char* image_data, \
                               unsigned char* device_result, int size) {
    /* Index to store the calculated value. */
    unsigned int target_index = (blockIdx.x * blockDim.x + threadIdx.x);
    /* Index for retrieving image data. */
    unsigned int index = target_index * NUM_CHANNELS_RGB;
    if (index < size) {
        device_result[target_index] = (image_data[index] * GREYSCALE_R \
                                    + image_data[index + 1] * GREYSCALE_G \
                                    + image_data[index + 2] * GREYSCALE_B);
    }
}


unsigned char * filter_greyscale_cuda(unsigned char *image_data, int num_pixels, int max_index) {

    int thread_block_size = 512;
    /* Allocate CPU memory to store the image after CUDA is ready. */
    unsigned char *new_image_data = (unsigned char *) malloc(num_pixels \
                                  * sizeof(unsigned char));
    if (new_image_data == NULL) {
        cout << "Could not allocate memory" << endl;
        exit(1);
    }
    /* Total pixels in the image data array. */
    int total_size = num_pixels * NUM_CHANNELS_RGB;

    unsigned char* device_new_image = (unsigned char*) allocateDeviceMemory( \
        num_pixels * sizeof (unsigned char));

    unsigned char* device_image_data = (unsigned char*) allocateDeviceMemory( \
        total_size * sizeof (unsigned char));

    /* Initialize the timers used to measure the kernel invocation time and
     * memory transfer time.
     */
    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    memoryTime.start();
    memcpyHostToDevice(device_image_data, image_data, total_size* sizeof(unsigned char));
    memoryTime.stop();

    /* Start calculation on the GPU. */
    int num_blocks = (num_pixels + thread_block_size - 1) / thread_block_size;
    kernelTime1.start();
    greyscale_kernel<<<num_blocks, thread_block_size>>> \
        (device_image_data, device_new_image, max_index* NUM_CHANNELS_RGB);//total_size);
    cudaDeviceSynchronize();
    kernelTime1.stop();
    checkCudaCall(cudaGetLastError());

    /* Copy the result image back to the GPU. */
    memoryTime.start();
    memcpyDeviceToHost(new_image_data, device_new_image, num_pixels * sizeof(unsigned char));
    memoryTime.stop();

    /* Free used memory on the GPU. */
    freeDeviceMemory(device_new_image);
    freeDeviceMemory(device_image_data);

    cout << fixed << setprecision(6);
    cout << "Greyscale (kernel): \t\t" << kernelTime1.getElapsed() \
         << " seconds." << endl;
    cout << "Greyscale (memory): \t\t" << memoryTime.getElapsed() \
         << " seconds." << endl;
    return new_image_data;
}
