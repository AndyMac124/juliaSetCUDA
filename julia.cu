#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "bmpfile.h"

/*Julia values*/
#define RESOLUTION 1000.0
#define XCENTER 0
#define YCENTER 0
#define MAX_ITER 1000
#define MAX_BOUND 10.0
#define X0 0.285
#define Y0 0.01

/*Colour Values*/
#define COLOUR_DEPTH 255
#define COLOUR_MAX 240.0
#define GRADIENT_COLOUR_MAX 230.0

#define FILENAME "my_julia_fractal.bmp"

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>

/**
   * Computes the color gradiant
   * color: the output vector
   * x: the gradiant (beetween 0 and 360)
   * min and max: variation of the RGB channels (Move3D 0 -> 1)
   * Check wiki for more details on the colour science: en.wikipedia.org/wiki/HSL_and_HSV
   */
__device__ void GroundColorMix(double* color, double x, double min, double max)
{
        /*
         * Red = 0
         * Green = 1
         * Blue = 2
         */
        double posSlope = (max-min)/60;
        double negSlope = (min-max)/60;

        if( x < 60 )
        {
                color[0] = max;
                color[1] = posSlope*x+min;
                color[2] = min;
                return;
        }
        else if ( x < 120 )
        {
                color[0] = negSlope*x+2.0*max+min;
                color[1] = max;
                color[2] = min;
                return;
        }
        else if ( x < 180  )
        {
                color[0] = min;
                color[1] = max;
                color[2] = posSlope*x-2.0*max+min;
                return;
        }
        else if ( x < 240  )
        {
                color[0] = min;
                color[1] = negSlope*x+4.0*max+min;
                color[2] = max;
                return;
        }
        else if ( x < 300  )
        {
                color[0] = posSlope*x-4.0*max+min;
                color[1] = min;
                color[2] = max;
                return;
        }
        else
        {
                color[0] = max;
                color[1] = min;
                color[2] = negSlope*x+6*max;
                return;
        }
}


__global__ void juliaSetKernel(float* output, int width, int height, int xoffset, int yoffset){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;

        if (idx < width && idy < height) {
                int pixelIndex = idy * width + idx;
                // Determine where in the Julia set the pixel is referencing
                // XCENTER/YCENTER is center of julia set
                // RESOLUTION is the scaling of image
                double x = XCENTER + (xoffset + idx) / RESOLUTION;
                double y = YCENTER + (yoffset - idy) / RESOLUTION;

                // Julia stuff

                // Real and imaginary parts of complex number
                // During iteration.
                double a = 0;
                double b = 0;

                // Previous value in the iteration
                double aold = x;
                double bold = y;

                // Magniture squared of the complex number
                double zmagsqr = 0;

                // Tracks number of iterations to influence pixel colour
                int iter = 0;

                // Check if the x,y coord are part of the Julia set - refer to the algorithm
                // Repeatedly applics quadratic function until max iterations is reached
                // or magnitude squared exceeds boundary.
                while (iter < MAX_ITER && zmagsqr <= MAX_BOUND * MAX_BOUND) {
                        ++iter;
                        a = (aold * aold) - (bold * bold) + X0;
                        b = 2.0 * aold * bold + Y0;

                        zmagsqr = a * a + b * b;

                        aold = a;
                        bold = b;
                }

                // Value used to determine the pixel's colour based on the number
                // of iterations
                double x_col;

                // RGB colours
                double color[3];

                /* Generate the colour of the pixel from the iter value */
                /* You can mess around with the colour settings to use different gradients */
                /* Colour currently maps from royal blue to red */
                x_col = (COLOUR_MAX - ((((float) iter / ((float) MAX_ITER) *
                                         GRADIENT_COLOUR_MAX))));
                GroundColorMix(color, x_col, 1, COLOUR_DEPTH);

                output[pixelIndex * 3 + 0] = color[0];
                output[pixelIndex * 3 + 1] = color[1];
                output[pixelIndex * 3 + 2] = color[2];
        }
}

int parse_args(int argc, char *argv[], int *width, int *height)
{
        if ((argc != 3) || ((*width = atoi(argv[1])) <= 0) || ((*height = atoi(argv[2])) <= 0)) {
                fprintf(stderr, "Usage: %s <image width> <image height>\n", argv[0]);
                return(-1);
        }

        return(0);
}

int main(int argc, char **argv) {
        int width, height;
        if (parse_args(argc, argv, &width, &height) != 0) {
                exit(EXIT_FAILURE);
        }

        bmpfile_t *bmp;

        // Offset for the Julia image in the bitmap image
        int xoffset = -(width - 1) / 2;
        int yoffset = (height - 1) / 2;

        // Create bitmap image of WxH with 32 bits for each pixel,
        // with 8 bits for RGBA channels
        bmp = bmp_create(width, height, 32);

        const int NUM_ELEMENTS = (width * height);
        size_t size = NUM_ELEMENTS * 3 * sizeof(float);

        float* h_result = (float*)malloc(size);
        float* d_result;
        cudaMalloc((void**)&d_result, size);

        dim3 threadsPerBlock(8, 8);
        dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
        printf("Launching kernel with %d blocks and %d threads per block\n",
               blocksPerGrid.x * blocksPerGrid.y, threadsPerBlock.x * threadsPerBlock.y);

        juliaSetKernel<<<blocksPerGrid, threadsPerBlock>>>(d_result, width, height, xoffset, yoffset);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }

        cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }

        rgb_pixel_t pixel = {0, 0, 0, 0};
        for (int row = 0; row < height; ++row) {
                for (int col = 0; col < width; ++col) {
                        int index = (row * width + col) * 3;
                        pixel.red = h_result[index];
                        pixel.green = h_result[index + 1];
                        pixel.blue = h_result[index + 2];

                        // Set the pixel in the bitmap
                        bmp_set_pixel(bmp, col, row, pixel);
                }
        }

        bmp_save(bmp, FILENAME);
        bmp_destroy(bmp);

        free(h_result);
        cudaFree(d_result);

        return 0;
}