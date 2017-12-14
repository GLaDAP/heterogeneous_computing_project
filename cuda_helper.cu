/*
 * file: cuda_helper.cu
 *
 * DESCRIPTION: File containing cuda-functions to allocate arrays and cleanup
 *              those arrays.
 *
 *
 */
#include <iostream>

using namespace std;


/* Utility function, use to do error checking.
 * Use this function like this:
 * checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));
 * And to check the result of a kernel invocation:
 * checkCudaCall(cudaGetLastError());
 */
void checkCudaCall(cudaError_t result){
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}

/* Allocates a int (array) in the device memory. */
void* allocateDeviceMemory(unsigned int size) {
    void* pointer = NULL;
    checkCudaCall(cudaMalloc((void **) &pointer, size));
    if (pointer == NULL) {
        cout << "could not allocate memory on the GPU." << endl;
        exit(1);
    }
    else {
        return pointer;
    }
}

void memcpyDeviceToDevice(void* target, void* source, unsigned int size) {
    checkCudaCall(cudaMemcpy(target, source, size, \
        cudaMemcpyDeviceToDevice));
}

void memcpyHostToDevice(void* target, void* source, unsigned int size) {
    checkCudaCall(cudaMemcpy(target, source, size, \
        cudaMemcpyHostToDevice));
}

void memcpyDeviceToHost(void* target, void* source, unsigned int size) {
    checkCudaCall(cudaMemcpy(target, source, size, \
        cudaMemcpyDeviceToHost));
}

/* Frees memory from the GPU. */
void freeDeviceMemory(void* pointer) {
    checkCudaCall(cudaFree(pointer));
}
