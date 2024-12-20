/*H*
 * FILENAME: julia_function_utils.cu
 *
 * AUTHOR: Andrew McKenzie
 * UNE EMAIL: amcken33@myune.edu.au
 * STUDENT NUMBER: 220263507
 *
 * PURPOSE: Utility functions to aid the CUDA parts of the julia program.
 *
 *H*/

#include <cuda_runtime.h>
#include "julia_function_utils.cuh"
#include "macros.h"

/**
 * ground_color_mix() - Computes the color gradiant
 *
 * @arg1: Pointer to the color array for the RGBA
 * @arg2: The gradiant (between 0 and 360)
 * @arg3: min variation of the RGB channels
 * @arg4: max variation of the RGB channels
 *
 * Reference: Code provided with the juila.c program for this assignment
 *
 * Sets the RGB values based on the gradiant and min/max values.
 *
 * Return: void (pass by reference).
 */
__device__ void ground_color_mix(double* color, double x, double min,
                                 double max)
{
        double posSlope = (max-min)/60;
        double negSlope = (min-max)/60;

        if ( x < 60 ) {
                color[R] = posSlope * x + min;
                color[G] = max;
                color[B] = max;
                return;
        } else if ( x < 120 ) {
                color[R] = max;
                color[G] = negSlope * x + 2.0 * max + min;
                color[B] = max;
                return;
        } else if ( x < 180  ) {
                color[R] = max;
                color[G] = min;
                color[B] = posSlope * x - 2.0 * max + min;
                return;
        } else if ( x < 240  ) {
                color[R] = negSlope * x + 4.0 * max + min;
                color[G] = min;
                color[B] = min;
                return;
        } else if ( x < 300  ) {
                color[R] = min;
                color[G] = posSlope * x - 4.0 * max + min;
                color[B] = min;
                return;
        } else {
                color[R] = min;
                color[G] = max;
                color[B] = negSlope * x + 6 * max;
                return;
        }
}


/**
 * julia_set_kernel() - kernel function for performing the julia set function
 *
 * @arg1: Pointer to the pixel to calculate
 * @arg2: width of the image
 * @arg3: height of the image
 * @arg4: x offset of the julia image
 * @arg5: y offset of the julia image
 *
 * Reference: Base structure modified from the julia.c program.
 *
 * Executes the julia function for a single pixel to calculate and
 * set its RGB value.
 *
 * Return: void (pass by reference).
 */
__global__ void julia_set_kernel(float* pixel, int width, int height,
                                 int xOffSet, int yOffSet)
{
        // For pre-calculating ground colour mix to avoid
        // repetitive computation
        __shared__ double shared_colour_mix[360][RGB_LENGTH];

        // First thread will set the colours
        if (threadIdx.x == 0 && threadIdx.y == 0) {
                for (int i = 0; i < 360; i++) {
                        double x = (double) i;
                        ground_color_mix(shared_colour_mix[i], x, 1,
                                         COLOUR_DEPTH);
                }
        }

        // Syncing all threads so the colours are set
        __syncthreads();

        // Calculate the x and y position for the pixel
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;

        // If pixel is within image dimensions
        if (idx < width && idy < height) {
                // Pixel index in 1D array
                int pixelIndex = idy * width + idx;

                // Determine where in the Julia set the pixel is referencing
                double x = XCENTER + (xOffSet + idx) / RESOLUTION;
                double y = YCENTER + (yOffSet - idy) / RESOLUTION;

                // Values for this iteration
                double a = 0;
                double b = 0;

                // Previous value in the iteration
                double aOld = x;
                double bOld = y;

                // Magnitude squared of the complex number (z)
                double zMagSqr = 0;

                // Tracks number of iterations to influence pixel colour
                int iter = 0;

                /*
                 * If max iterations hasn't been reached and magnitude squared
                 * is within max bound squared continues to perform the
                 * Julia algorithm
                 */
                while (iter < MAX_ITER && zMagSqr <= MAX_BOUND * MAX_BOUND) {
                        ++iter;
                        a = (aOld * aOld) - (bOld * bOld) + X0;
                        b = 2.0 * aOld * bOld + Y0;

                        zMagSqr = a * a + b * b;

                        aOld = a;
                        bOld = b;
                }

                // Value used to determine the pixel's colour based on
                // the number of iterations
                double xCol;

                // Generate the colour of the pixel from the iter value
                xCol = (COLOUR_MAX - ((((float) iter / ((float) MAX_ITER) *
                                        GRADIENT_COLOUR_MAX))));

                // Map the colour to an index in the shared colour mixes
                int xColIndex = (int)(xCol) % 360;
                // Set the pixel colours from the shared colour mixes
                double* color = shared_colour_mix[xColIndex];

                // Setting the pixel values in the 1D array
                pixel[pixelIndex * RGB_LENGTH + R] = color[R];
                pixel[pixelIndex * RGB_LENGTH + G] = color[G];
                pixel[pixelIndex * RGB_LENGTH + B] = color[B];
        }
}
