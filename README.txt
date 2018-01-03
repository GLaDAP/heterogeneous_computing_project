File: README.txt
Assignment: 5
Students: Teun Mathijssen, David Puroja
Student email: teun.mathijssen@student.uva.nl, dpuroja@gmail.com
Studentnumber: 11320788, 10469036

-----

Welcome to the starting point of our elective assignment! In this assignment,
we explore the utilization of hybrid multithreading on a series of
computationally intensive image operations.

-----

The following list of commands needs to be run in the following order to allow
the program to build correctly:

1. module load openmpi/gcc/64/1.4.4
module load sge
module load cuda55
module load prun
2. module add gcc/4.8.2
3. make

-----

You can now run it by using the following command:

./assign_5 <source_img> <output_img>.png blocksize num_threads workload_gpu[1-100]

<source_img>: Source image. Not guaranteed to be usable by our program. For these
limitations, see https://github.com/nothings/stb/blob/master/stb_image.h

<output_img>.png: Will be stored in the same folder. It always is a PNG file.

blocksize: The CUDA block size.

num_threads: The amount of threads for OpenMP.

workload_gpu: Defines the workload for the GPU. Consequently, the
workload for the CPU is (100-workload_gpu). Workload_gpu must be at least 1 and
at most 100.
