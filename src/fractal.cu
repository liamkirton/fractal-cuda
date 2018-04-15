#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory.h>

#include "fractal.h"

__global__ void mandelbrot_kernel(const unsigned int image_width, const unsigned int image_height, unsigned int *point_buffer) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int pixel_x = tid % image_width;
    const int pixel_y = tid / image_width;

    const double image_center_re = -0.66;
    const double image_center_im = 0.15;
    const double image_scale = 1.66;

    const double re_c = image_center_re + (-2.0 + pixel_x * 3.0 / image_width) / image_scale;
    const double im_c = image_center_im + (1.0 - pixel_y * 2.0 / image_height) / image_scale;

    const unsigned int escape_limit = 2048;

    double re_z = re_c;
    double im_z = im_c;
    double abs_z = 0.0;

    unsigned int escape = 0;

    for (escape = 0; escape < escape_limit; ++escape) {
        double re_z_i = re_z;
        re_z = (re_z * re_z) - (im_z * im_z) + re_c;
        im_z = (2.0 * re_z_i * im_z) + im_c;
        abs_z = re_z * re_z + im_z * im_z;
        if (abs_z > 4.0) {
            break;
        }
    }

    abs_z = sqrt(abs_z);

    double hue = escape + 1.0 - (log(log(abs_z)) / log(2.0));
    double sat = 0.85;
    double val = 0.33 + log(1.0 * escape) / log(1.0 * escape_limit);
    if (val > 1.0) {
        val = 1.0;
    }

    hue += 0;
    hue = fmod(hue, 360.0);
    hue /= 360;
    hue += 4;

    double hue_fract = hue - floor(hue);
    double p = val * (1.0 - sat);
    double q = val * (1.0 - sat * hue_fract);
    double t = val * (1.0 - sat * (1.0 - hue_fract));

    double r = 0;
    double g = 0;
    double b = 0;

    if (escape < escape_limit) {
        switch (static_cast<unsigned char>(floor(hue))) {
        case 0:
            r = val; g = t; b = p;
            break;
        case 1:
            r = q; g = val; b = p;
            break;
        case 2:
            r = p; g = val; b = t;
            break;
        case 3:
            r = p; g = q; b = val;
            break;
        case 4:
            r = t; g = p; b = val;
            break;
        case 5:
            r = val; g = p; b = q;
            break;
        default:
            break;
        }
        r = floor(r * 255); g = floor(g * 255); b = floor(b * 255);
    }

    point_buffer[tid] = ((unsigned char)(r) << 0) |
        ((unsigned char)(g) << 8) |
        ((unsigned char)(b) << 16) |
        (255 << 24);
}

void mandelbrot(unsigned int image_width, unsigned int image_height, unsigned int *result_buffer) {
    size_t point_buffer_size = sizeof(unsigned int) * image_width * image_height;
    unsigned int *point_buffer{ nullptr };
    cudaMalloc(&point_buffer, point_buffer_size);

    mandelbrot_kernel<<<image_width * image_height / 1024, 1024>>>(image_width, image_height, point_buffer);

    cudaError_t cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess) {
        std::wcout << L"cudaError: " << cudaError << std::endl;
        return;
    }

    cudaMemcpy(result_buffer, point_buffer, point_buffer_size, cudaMemcpyDeviceToHost);
}