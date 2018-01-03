PROJ_BASE	:= .

INCLUDES	:= -I"$(PROJ_BASE)"

CUDA_LIBS	:= $(LIBS) -L"$(CUDA_INSTALL_PATH)/lib64"

# This program only works with a GPU with Compute Capability 3.0 or higher.
NVCCFLAGS	:= -O3 -g --ptxas-options=-v \
			   -gencode arch=compute_30,code=compute_30 \
			   -gencode arch=compute_30,code=sm_30

CFLAGS		= -O3 -g3 -Wall -g -Werror -fopenmp

LDFLAGS         := -lm -lX11 -lpthread -lgomp
CUDA_LDFLAGS	:= $(LDFLAGS) -lrt -lcudart

CU_SOURCES	= greyscale.cu smoothing.cu contrast.cu cuda_helper.cu brightness.cu
CC_SOURCES	= contrastomp.cc greyscaleomp.cc smoothingomp.cc timer.cc file.cc \
			  brightnessomp.cc main.cc

CU_OBJECTS	= $(CU_SOURCES:%.cu=%.o)
CU_PTX		= $(CU_SOURCES:%.cu=%.ptx)
CC_OBJECTS	= $(CC_SOURCES:%.cc=%.o)

# -std=c++11 used to use thread header of C++ library. On the DAS4, it requires
# the gcc/4.8.2 module to be added.
CC			:= g++
NVCC		:= nvcc
LINKER		:= g++ -std=c++11

all: clean assign_5

assign_5: $(CU_SOURCES) $(CC_SOURCES)
	$(NVCC) -c $(CU_SOURCES) $(NVCCFLAGS) $(INCLUDES)
	$(LINKER) -o $(PROJ_BASE)/assign_5 $(CU_OBJECTS) $(CC_SOURCES) $(INCLUDES) \
	$(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)

#Runlocal works only on computers with configured bumblebee.
runlocal: $(PROGNAME)
	optirun $(PROJ_BASE)/assign_5 kit0.jpg kit_res.png 512 4 90

# Run on the DAS4 with test-image of the cat.
run: $(PROGNAME)
	prun -v -np 1 -native '-l fat,gpu=K20' $(PROJ_BASE)/assign_5 kit0.jpg \
	kit_res.png 512 4 90

clean:
		rm -fv $(PROGNAME) $(TARNAME) assign_5 main.o greyscale.o smoothing.o \
		contrast.o timer.o contrastomp.o greyscaleomp.o smoothingomp.o \
		cuda_helper.o brightness.o brightnessomp.o file.o
