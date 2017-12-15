/*
 * File: cuda_helper.cu
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: File containing functions used to allocate, copy and free
 *              device memory and to check is a call is succesful.
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

/* Copies memory on the device to another memory adress on the device. */
void memcpyDeviceToDevice(void* target, void* source, unsigned int size) {
    checkCudaCall(cudaMemcpy(target, source, size, \
        cudaMemcpyDeviceToDevice));
}

/* Copies memory on the host to a memory adress on the device. */
void memcpyHostToDevice(void* target, void* source, unsigned int size) {
    checkCudaCall(cudaMemcpy(target, source, size, \
        cudaMemcpyHostToDevice));
}

/* Copies memory on the device to the host. */
void memcpyDeviceToHost(void* target, void* source, unsigned int size) {
    checkCudaCall(cudaMemcpy(target, source, size, \
        cudaMemcpyDeviceToHost));
}

/* Frees memory from the GPU. */
void freeDeviceMemory(void* pointer) {
    checkCudaCall(cudaFree(pointer));
}
