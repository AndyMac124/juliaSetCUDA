/*H*
 * FILENAME: julia_utils.h
 *
 * AUTHOR: Andrew McKenzie
 * UNE EMAIL: amcken33@myune.edu.au
 * STUDENT NUMBER: 220263507
 *
 * PURPOSE: Header file for the juila_utils.c file
 *
 *H*/

#ifndef A4_JULIA_UTILS_H
#define A4_JULIA_UTILS_H

void parse_args(int argc, char *argv[], int *width, int *height);
void check_error(cudaError_t error, const char *errorMessage);
void set_pixels(int height, int width, float *hPixels, bmpfile_t *bmp);

#endif //A4_JULIA_UTILS_H
