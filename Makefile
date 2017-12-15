debug		:= 0
PROJ_BASE	:= .

INCLUDES	:= -I"$(PROJ_BASE)"
LIBS		:=

CUDA_LIBS	:= $(LIBS) -L"$(CUDA_INSTALL_PATH)/lib64"

NVCCFLAGS	:= -O3 -g --ptxas-options=-v \
			  -gencode arch=compute_50,code=compute_50 \
			  -gencode arch=compute_50,code=sm_50

			  # -gencode arch=compute_30,code=compute_30 \
			  # -gencode arch=compute_30,code=sm_30

CFLAGS		= -O3 -g3 -Wall -g -Werror -fopenmp

LDFLAGS         := -lm -lX11 -lpthread -lgomp
CUDA_LDFLAGS	:= $(LDFLAGS) -lrt -lcudart

CU_SOURCES	= main.cu greyscale.cu smoothing.cu contrast.cu cuda_helper.cu brightness.cu
CC_SOURCES	= contrastomp.cc greyscaleomp.cc smoothingomp.cc timer.cc file.cc


CU_OBJECTS	= $(CU_SOURCES:%.cu=%.o)
CU_PTX		= $(CU_SOURCES:%.cu=%.ptx)
CC_OBJECTS	= $(CC_SOURCES:%.cc=%.o)

CC			:= g++-5
NVCC		:= nvcc -std=c++11
LINKER		:= g++-5 -std=c++11


all: clean rgb2grey

rgb2grey: $(CU_SOURCES) $(CC_SOURCES)
	$(NVCC) -c $(CU_SOURCES) $(NVCCFLAGS) $(INCLUDES)
	$(LINKER) -o $(PROJ_BASE)/rgb2grey $(CU_OBJECTS) $(CC_SOURCES) $(INCLUDES) \
	$(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)

clean:
		rm -fv $(PROGNAME) $(TARNAME) rgb2grey main.o greyscale.o smoothing.o \
		contrast.o timer.o contrastomp.o greyscaleomp.o smoothingomp.o cuda_helper.o brightness.o file.o
