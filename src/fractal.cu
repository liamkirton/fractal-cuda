#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory.h>

#include "fractal.h"

__global__ void mandelbrot_kernel(unsigned int *point_buffer) {
    const double image_shift_x = 0;// 0.485;
    const double image_shift_y = 0;// 0.94385;
    const double image_scale = 1;// 18.0;

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int pixel_x = tid % image_width;
    int pixel_y = tid / image_width;

    double re_c = (-2.0 + image_shift_x) + pixel_x * 2 / (image_width * image_scale);
    double im_c = (1.0 - image_shift_y) - pixel_y * 1.5 / (image_height * image_scale);

    double re_z = re_c;
    double im_z = im_c;
    double abs_z = (re_z * re_z + im_z * im_z);

    unsigned short escape = 0xffff;

    for (unsigned int i = 0; i < 256; ++i) {
        double re_z_i = (re_z * re_z) - (im_z * im_z) + re_c;
        double im_z_i = (2.0 * re_z * im_z) + im_c;
        re_z = re_z_i;
        im_z = im_z_i;

        double abs_z_i = (re_z * re_z + im_z * im_z);
        abs_z = abs_z_i;

        if ((abs_z > 4) && (escape == 0xffff)) {
            escape = i;
        }
        else if ((abs_z < 4) && (escape != 0xffff)) {
            escape = 0xffff;
        }
    }

    unsigned char r = (escape != 0xffff) ? 0 : 0;
    unsigned char g = (escape != 0xffff) ? min(255, escape) : 0;
    unsigned char b = (escape != 0xffff) ? min(255, escape) : 0;

    point_buffer[tid] = (r << 0) | (g << 8) | (b << 16) | (255 << 24);
}

void mandelbrot(unsigned int *result_buffer) {
    size_t point_buffer_size = sizeof(unsigned int) * image_width * image_height;
    unsigned int *point_buffer{ nullptr };
    cudaMalloc(&point_buffer, point_buffer_size);

    mandelbrot_kernel<<<image_width * image_height / 1024, 1024>>>(point_buffer);

    cudaError_t cudaError = cudaDeviceSynchronize();
    if (cudaError != cudaSuccess) {
        std::wcout << L"cudaError: " << cudaError << std::endl;
        return;
    }

    cudaMemcpy(result_buffer, point_buffer, point_buffer_size, cudaMemcpyDeviceToHost);
}