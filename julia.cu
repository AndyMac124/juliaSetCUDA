/*H*
 * FILENAME: julia.cu
 *
 * AUTHOR: Andrew McKenzie
 * UNE EMAIL: amcken33@myune.edu.au
 * STUDENT NUMBER: 220263507
 *
 * PURPOSE: Generate a bitmap image of a fractal representation of the
 * Julia set.
 *
 * This program takes in a width and height for the image. It then uses
 * CUDA C to calculate each pixel's RGB values from the julia algorithm and
 * once they are returned to the host the program sets the pixels in the
 * bitmap and saves the file.
 *
 * COMPILING: The included makefile can be run with the 'make' command.
 *
 * RUNNING: The program is run by the following:
 *      ./julia <width> <height>
 *
 * Run Target Example: make run <width> <height>
 *
 * Author Recommendation: run with width 2160 and height 3240
 * (This image is now my desktop background)
 *
 * As per the Linux Kernel C programming guide:
 * - Function names use snake case for emphasis.
 * - Variables use camel case for brevity.
 * - Global variables use snake case.
 * - Constants and macros use snake case and are upper case.
 * - Everything except function declarations use K&R style braces.
 * - Functions use Allman style braces.
 *
 *H*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "bmpfile.h"
#include "julia_utils.h"
#include "julia_function_utils.cuh"
#include "macros.h"

#define FILENAME "my_julia_fractal.bmp"
#define PIXEL_BITS 32

/**
 * main() - Main function for the julia program.
 * @arg1: Number of args from the terminal.
 * @arg2: Array of the args from the terminal.
 *
 * The function follows these generic steps:
 * - It sets up device and host arrays for the pixels
 * - Calculates the pixels from the given function
 * - Sets the pixels in the bitmap and saves the file
 *
 * Reference: This program is heavily reliant on the provided
 * julia.c program and the examples used in COSC330 lectures.
 *
 * Return: Int, zero on success, non-zero on failure.
 */
int main(int argc, char **argv) {
        // Error code to check return values for CUDA calls
        cudaError_t err = cudaSuccess;

        int width, height;
        parse_args(argc, argv, &width, &height);

        // Bitmap with 32 bits for each pixel (RGBA)
        bmpfile_t *bmp = bmp_create(width, height, PIXEL_BITS);

        // Offset for the Julia image in the bitmap image
        int xOffSet = -(width - 1) / 2;
        int yOffSet = (height - 1) / 2;

        // Size of image
        size_t size = width * height * RGB_LENGTH * sizeof(float);

        // Pointer to memory for hosts pixels
        float* hPixels = (float*)malloc(size);
        if (hPixels == NULL)
        {
                fprintf(stderr, "Failed to allocate host vectors!\n");
                exit(EXIT_FAILURE);
        }

        // Pointer to devices pixels
        float* dPixels;
        err = cudaMalloc((void**)&dPixels, size);
        check_error(err, "Failed to allocate memory for device");

        // Multiple of 32 for warp sizes
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
                     (height + dimBlock.y - 1) / dimBlock.y);

        // Kernel call
        julia_set_kernel<<<dimGrid, dimBlock>>>(dPixels,
                                              width,
                                              height,
                                              xOffSet,
                                              yOffSet);

        err = cudaGetLastError();
        check_error(err, "Failed in call to juliaSetKernel");

        // Copying results back to host
        err = cudaMemcpy(hPixels, dPixels, size, cudaMemcpyDeviceToHost);
        check_error(err, "Failed to copy d_result to Host");

        set_pixels(height, width, hPixels, bmp);

        // Attempting to save new file
        if (bmp_save(bmp, FILENAME) == 0) {
                fprintf(stderr, "Failed to save bmp file (error code %s)!\n",
                        cudaGetErrorString(err));
        }

        // Freeing up resources
        bmp_destroy(bmp);
        free(hPixels);
        err = cudaFree(dPixels);
        check_error(err, "Failed to free device memory");

        // Deinitialising for good practice
        err = cudaDeviceReset();
        check_error(err, "Failed to deinitialise the device");

        return 0;
}
