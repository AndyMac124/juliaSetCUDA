
#ifndef A4_JULIA_UTILS_H
#define A4_JULIA_UTILS_H

void parse_args(int argc, char *argv[], int *width, int *height);
void check_error(cudaError_t err, const char *error_message);
void set_pixels(int height, int width, float *h_result, bmpfile_t *bmp);

#endif //A4_JULIA_UTILS_H
