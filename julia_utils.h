
#ifndef A4_JULIA_UTILS_H
#define A4_JULIA_UTILS_H

void parse_args(int argc, char *argv[], int *width, int *height);
void check_error(cudaError_t error, const char *errorMessage);
void set_pixels(int height, int width, float *hPixels, bmpfile_t *bmp);

#endif //A4_JULIA_UTILS_H
