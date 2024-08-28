/*H*
 * FILENAME: julia_utils.c
 *
 * AUTHOR: Andrew McKenzie
 * UNE EMAIL: amcken33@myune.edu.au
 * STUDENT NUMBER: 220263507
 *
 * PURPOSE: C based Utility functions to aid the julia program.
 *
 *H*/

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "bmpfile.h"
#include "julia_utils.h"
#include "macros.h"


/**
 * parse_args() - Processes command line arguments.
 *
 * @arg1: Number of args from the terminal.
 * @arg2: Array of the args from the terminal.
 * @arg3: Pointer to width of image
 * @arg4: Pointer to height of image
 *
 * Checks for the correct number of args and checks that
 * the args for height and width are int values.
 *
 * Return: void (pass by reference).
 */
void parse_args(int argc, char *argv[], int *width, int *height)
{
        if ((argc != 3) || ((*width = atoi(argv[WIDTH_ARG])) <= 0)
                || ((*height = atoi(argv[HEIGHT_ARG])) <= 0)) {
                fprintf(stderr,
                        "Usage: %s <image width> <image height>\n", argv[0]);
                exit(EXIT_FAILURE);
        }
}


/**
 * check_error() - Checks for a cuda error
 *
 * @arg1: The error value
 * @arg2: Message to respond with
 *
 * Checks if the error is not cudaSuccess and if not it will respond
 * with the given error message.
 *
 * Return: void (pass by reference).
 */
void check_error(cudaError_t error, const char *errorMessage)
{
        if (error != cudaSuccess) {
                fprintf(stderr, "%s (error code %s)!\n", errorMessage,
                        cudaGetErrorString(error));
                exit(EXIT_FAILURE);
        }
}

/**
 * set_pixels() - Sets the pixels in the bitmap image
 *
 * @arg1: Height of image
 * @arg2: Width of image
 * @arg3: Pointer to the start of the pixel values
 * @arg4: Pointer to the image
 *
 * Iterates through the pixels in the image setting them
 * with the values from the pointer of pixel values.
 *
 * Return: void (pass by reference).
 */
void set_pixels(int height, int width, float *hPixels, bmpfile_t *bmp)
{
        // Initialising values to zero
        rgb_pixel_t pixel = {0, 0, 0, 0};

        for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                        // Calculate index in 1D array
                        int index = (y * width + x) * RGB_LENGTH;

                        // Offset for each RGB value
                        pixel.red = hPixels[index + R];
                        pixel.green = hPixels[index + G];
                        pixel.blue = hPixels[index + B];

                        // Set the pixel in the bitmap
                        if (bmp_set_pixel(bmp, x, y, pixel) == 0) {
                                fprintf(stderr, "Failed to set pixel: %s!\n",
                                                cudaGetErrorString(
                                                        cudaGetLastError()));
                        }
                }
        }
}