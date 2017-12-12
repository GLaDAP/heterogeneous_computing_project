PROGNAME = RGB2GREY
TARNAME = Group_7_-_Assignment_5.tgz

NVCC 	= nvcc

CC		= g++

CU_FLAGS	= -O3 -g --ptxas-options=-v \
			  -gencode arch=compute_30,code=compute_30 \
			  -gencode arch=compute_30,code=sm_30 \
			  -gencode arch=compute_50,code=compute_50 \
			  -gencode arch=compute_50,code=sm_50

CC_FLAGS	= -O3 -m64 -Wall -Wextra -Werror

CU_SOURCES	= greyscale.cu smoothing.cu contrast.cu
CC_SOURCES	= main.cc timer.cc

CU_OBJECTS	= $(CU_SOURCES:%.cu=%.o)
CU_PTX		= $(CU_SOURCES:%.cu=%.ptx)
CC_OBJECTS	= $(CC_SOURCES:%.cc=%.o)

%.o:		%.cu
		$(NVCC) $(CU_FLAGS) -c $< -o $@

%.o:		%.cc
		$(CC) $(CC_FLAGS) -c $< -o $@

%.ptx:		%.cu
		$(NVCC) $(CU_FLAGS) --ptx $< -o $@

rgb2grey:	$(CU_OBJECTS) $(CC_OBJECTS)
		$(NVCC) $^ -o $@

ptx:		$(CU_PTX)

dist:
	tar cvzf $(TARNAME) Makefile *.cc *.cu *.h

clean:
		rm -fv $(PROGNAME) $(TARNAME) rgb2grey main.o greyscale.o smoothing.o \
		contrast.o timer.o
