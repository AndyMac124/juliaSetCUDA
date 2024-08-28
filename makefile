CUDACOMPILER = nvcc
CUDAFLAGS = -I /usr/local/cuda-11.4/samples/common/inc
COMPILER = gcc
CFLAGS = -Wall -pedantic
COBJS = bmpfile.o julia_function_utils.o julia_utils.o
EXES = julia

all: ${EXES}

julia: julia.cu ${COBJS}
	${CUDACOMPILER} ${CUDAFLAGS} -o julia julia.cu ${COBJS} -lm

julia_function_utils.o: julia_function_utils.cu julia_function_utils.cuh
	${CUDACOMPILER} ${CUDAFLAGS} -c julia_function_utils.cu -o julia_function_utils.o

julia_utils.o: julia_utils.cu julia_utils.h bmpfile.c bmpfile.h
	${CUDACOMPILER} ${CUDAFLAGS} -c julia_utils.cu -o julia_utils.o

bmpfile.o: bmpfile.c bmpfile.h
	${COMPILER} ${CFLAGS} -c bmpfile.c -o bmpfile.o

run: julia
	${COMPILER} ${ARGS}

clean:
	rm -f *.o *~ ${EXES}