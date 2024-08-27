CUDACOMPILER = nvcc
CUDAFLAGS = -I /usr/local/cuda-11.4/samples/common/inc
COMPILER = gcc
CFLAGS = -Wall -pedantic
COBJS = bmpfile.o
EXES = julia

all: ${EXES}

julia: julia.cu ${COBJS}
	${CUDACOMPILER} ${CUDAFLAGS} -o julia julia.cu ${COBJS} -lm

bmpfile.o: bmpfile.c bmpfile.h
	${COMPILER} ${CFLAGS} -c bmpfile.c -o bmpfile.o

clean:
	rm -f *.o *~ ${EXES}