

#ifndef A4_JULIA_FUNCTION_UTILS_CUH
#define A4_JULIA_FUNCTION_UTILS_CUH

// Julia values
#define RESOLUTION 1200.0
#define XCENTER 0
#define YCENTER 0
#define MAX_ITER 1100
#define MAX_BOUND 10.0
#define X0 0.2835
#define Y0 0.01

// Colour Values
#define COLOUR_DEPTH 245
#define COLOUR_MAX 240.0
#define GRADIENT_COLOUR_MAX 225.0

__device__ void ground_color_mix(double* color, double x, double min,
                                 double max);
__global__ void julia_set_kernel(float* pixel, int width, int height,
                                 int xoffset, int yoffset);

#endif //A4_JULIA_FUNCTION_UTILS_CUH
