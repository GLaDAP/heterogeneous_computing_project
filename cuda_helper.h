/*
 * File: cuda_helper.h
 * Assignment: 5
 * Students: Teun Mathijssen, David Puroja
 * Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
 * Studentnumber: 11320788, 10469036
 *
 * Description: File containing functions used to allocate, copy and free
 *              device memory and to check is a call is succesful.
 */
 
void checkCudaCall(cudaError_t result);
void* allocateDeviceMemory(unsigned int size);
void freeDeviceMemory(void* pointer);
void memcpyDeviceToDevice(void* target, void* source, unsigned int size);
void memcpyHostToDevice(void* target, void* source, unsigned int size);
void memcpyDeviceToHost(void* target, void* source, unsigned int size);
