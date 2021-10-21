# Heterogeneous computing project

Code part of elective CUDA assignment for the course Concurrency and Parallel Programming

The goal of this project is to explore which workload distribution between the CPU and GPU delivers the best performance for image processing. The image processing consists of three operations:

1. Convert the image to greyscale and compute the mean brightness of the image
2. Apply contrast filter using the computed mean brightness
3. Smooth the image using a convolution kernel

For this, CUDA and C code is written to run the application in parallel on the CPU (OpenMP) and GPU (CUDA). The project was run on the [DAS4](https://www.cs.vu.nl/das4/) Supercomputer, using a node with the NVIDIA Tesla K20 accelerator. 

### How to run

`./assign_5 <source_img> <output_img>.png blocksize num_threads workload_gpu[1-100]`

- `<source_img>`: Source image. Not guaranteed to be usable by our program. For these limitations, see https://github.com/nothings/stb/blob/master/stb_image.h

- `<output_img>.png`: Will be stored in the same folder. It always is a PNG file.

- `blocksize`: The CUDA block size.

- `num_threads`: The amount of threads for OpenMP.

- `workload_gpu`: Defines the workload for the GPU. Consequently, the workload for the CPU is (100-workload_gpu). Workload_gpu must be at least 1 and at most 100.

### Make

First, the program needs to be compiled. When compiling on the DAS4, the following modules needs to be loaded:

#### DAS4

- openmpi/gcc/64/1.4.4
- sge
- cuda55
- prun
- gcc/4.8.2

Then run `make && make run`

#### On a notebook with NVIDIA GPU

It is also possible to run the code on a notebook with Nvidia GPU (tested with Debian 8 Jessie) in combination with Bumblebee:

`make && make runlocal`

