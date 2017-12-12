#include <stdbool.h>
#include <stdlib.h>
#include <iostream>
#include "timer.h"

using namespace std;


/* Image input/output settings. */
#define NUM_CHANNELS_RGB 3

/* Greyscale component weighting factors. */
#define GREYSCALE_R 0.2126
#define GREYSCALE_G 0.7152
#define GREYSCALE_B 0.0722

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

/* Perform a greyscale filter on an image. Return a new image data array
containing only 1 color channel. */
unsigned char *filter_greyscale(unsigned char *image_data, int num_pixels) {

    unsigned char *new_image_data = (unsigned char *) malloc(num_pixels \
                                  * sizeof(unsigned char));
    if (new_image_data == NULL) {
        cout << "Could not allocate memory" << endl;
        return NULL;
    }
    int total_size = num_pixels * NUM_CHANNELS_RGB;

    /* CUDA */
    for(int i = 0, j = 0; i < total_size; i += NUM_CHANNELS_RGB, j++) {
        int greyscale_value = (image_data[i] * GREYSCALE_R
                               + image_data[i + 1] * GREYSCALE_G
                               + image_data[i + 2] * GREYSCALE_B);

        new_image_data[j] = greyscale_value;
    }

    return new_image_data;
}

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


unsigned char* filter_greyscale_cuda(unsigned char *image_data, int num_pixels) {

    int thread_block_size = 512;
    /* Allocate CPU memory to store the image after CUDA is ready. */
    unsigned char *new_image_data = (unsigned char *) malloc(num_pixels \
                                  * sizeof(unsigned char));
    if (new_image_data == NULL) {
        cout << "Could not allocate memory" << endl;
        return NULL;
    }

    /* Total pixels in the image data array. */
    int total_size = num_pixels * NUM_CHANNELS_RGB;

    unsigned char* device_new_image = NULL;
    checkCudaCall(cudaMalloc((void **) &device_new_image, \
                  num_pixels * sizeof(unsigned char)));
    if (device_new_image == NULL) {
        free(new_image_data);
        cout << "could not allocate memory on the GPU." << endl;
        return NULL;
    }

    unsigned char* device_image_data = NULL;
    checkCudaCall(cudaMalloc((void **) &device_image_data, \
                  total_size * sizeof(unsigned char)));
    if (device_image_data == NULL) {
        free(new_image_data);
        checkCudaCall(cudaFree(device_new_image));
        cout << "could not allocate memory on the GPU." << endl;
        return NULL;
    }

    /* Initialize the timers used to measure the kernel invocation time and
     * memory transfer time.
     */
    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    memoryTime.start();
    checkCudaCall(cudaMemcpy(device_image_data, image_data, \
                            total_size * sizeof(unsigned char), \
                             cudaMemcpyHostToDevice));
    memoryTime.stop();

    /* Start calculation on the GPU. */
    int num_blocks = (num_pixels + thread_block_size - 1) / thread_block_size;
    kernelTime1.start();
    greyscale_kernel<<<num_blocks, thread_block_size>>> \
        (device_image_data, device_new_image, total_size);
    cudaDeviceSynchronize();
    kernelTime1.stop();
    checkCudaCall(cudaGetLastError());

    /* Copy the result image back to the GPU. */
    memoryTime.start();
    checkCudaCall(cudaMemcpy(new_image_data, device_new_image, \
                             num_pixels * sizeof(unsigned char), \
                             cudaMemcpyDeviceToHost));
    memoryTime.stop();

    /* Free used memory on the GPU. */
    checkCudaCall(cudaFree(device_new_image));
    checkCudaCall(cudaFree(device_image_data));

    cout << fixed << setprecision(6);
    cout << "Greyscale (kernel): \t\t" << kernelTime1.getElapsed() \
         << " seconds." << endl;
    cout << "Greyscale (memory): \t\t" << memoryTime.getElapsed() \
         << " seconds." << endl;

    return new_image_data;
}
