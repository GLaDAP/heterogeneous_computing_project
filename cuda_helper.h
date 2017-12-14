/*
 * file: cuda_helper.h
 *
 * DESCRIPTION: File containing cuda-functions to allocate arrays and cleanup
 *              those arrays.
 *
 *
 */
void checkCudaCall(cudaError_t result);
void* allocateDeviceMemory(unsigned int size);
void freeDeviceMemory(void* pointer);
void memcpyDeviceToDevice(void* target, void* source, unsigned int size);
void memcpyHostToDevice(void* target, void* source, unsigned int size);
void memcpyDeviceToHost(void* target, void* source, unsigned int size);
