COMPILER_NVCC = nvcc
COMPILER_CC = gcc
CUDAFLAGS = -I /usr/local/cuda-11.4/samples/common/inc
CFLAGS = -Wall -pedantic
COBJS = bmpfile.o
CEXES = julia

all: ${CEXES}

julia: julia.cu ${COBJS}
	${COMPILER_NVCC} ${CUDAFLAGS} -o julia julia.cu ${COBJS} -lm

bmpfile.o: bmpfile.c bmpfile.h
	${COMPILER_CC} ${CFLAGS} -I /usr/local/cuda-11.4/samples/common/inc -c bmpfile.c -o bmpfile.o

clean:
	rm -f *.o *~ ${CEXES}