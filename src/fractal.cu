#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "fractal.h"

__global__ void mandelbrot_kernel(const uint64_t image_width, const uint64_t image_height, const uint64_t image_chunk, uint32_t *image_chunk_buffer) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int pixel_x = (image_chunk + tid) % image_width;
    const int pixel_y = (image_chunk + tid) / image_width;

    /*const double image_center_re = -0.66;
    const double image_center_im = 0.15;
    const double image_scale = 1.66;*/

    /*const double image_center_re = -0.7440;
    const double image_center_im = 0.1102;
    const double image_scale = 1/0.005;*/

    //const double image_center_re = -0.74516;
    //const double image_center_im = 0.112575;
    //const double image_scale = 1 / 6.5E-4;

    const double image_center_re = -0.235125;
    const double image_center_im = 0.827215;
    const double image_scale = 1 / 4.0E-5;

    const double re_c = image_center_re + (-2.0 + pixel_x * 3.0 / image_width) / image_scale;
    const double im_c = image_center_im + (1.0 - pixel_y * 2.0 / image_height) / image_scale;

    const unsigned int escape_limit = 16384;

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

    image_chunk_buffer[tid] = ((unsigned char)(r) << 0) |
        ((unsigned char)(g) << 8) |
        ((unsigned char)(b) << 16) |
        (255 << 24);
}

int mandelbrot(const uint64_t image_width, const uint64_t image_height, uint32_t *image) {
    cudaError_t cuda_status;
    if ((cuda_status = cudaSetDevice(0)) != cudaSuccess) {
        std::wcout << L"ERROR: cudaSetDevice() Failed. [" << cuda_status << L"]" << std::endl << std::endl;
        return -1;
    }

    uint64_t groups = 2048;
    uint64_t threads = 1024;

    uint32_t *image_chunk_buffer{ nullptr };
    cudaMalloc(&image_chunk_buffer, groups * threads * sizeof(uint32_t));

    std::wcout << L"[+] Chunks: " << image_width * image_height / (groups * threads) << L" " << std::flush;

    for (uint64_t image_chunk = 0; image_chunk < (image_width * image_height); image_chunk += (groups * threads)) {
        uint64_t chunk_size = std::min((image_width * image_height) - image_chunk, groups * threads);
        uint64_t chunk_groups = chunk_size / threads;

        mandelbrot_kernel<<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(threads)>>>(image_width, image_height, image_chunk, image_chunk_buffer);

        cudaError_t cudaError = cudaDeviceSynchronize();
        if (cudaError != cudaSuccess) {
            std::wcout << std::endl << "cudaError: " << cudaError << std::endl;
            return -1;
        }

        cudaMemcpy(&image[image_chunk], image_chunk_buffer, chunk_groups * threads * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        std::wcout << L"." << std::flush;
    }

    std::wcout << std::endl;

    cudaFree(image_chunk_buffer);
    image_chunk_buffer = nullptr;

    if ((cuda_status = cudaDeviceReset()) != cudaSuccess) {
        std::wcout << L"ERROR: cudaDeviceReset() Failed. [" << cuda_status << L"]" << std::endl << std::endl;
        return -1;
    }

    return 0;
}