#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "fractal.h"

const uint32_t escape_block = 4096;
const uint32_t escape_limit = 65536;

__global__ void mandelbrot_kernel(uint64_t *chunk_buffer, const uint64_t image_width, const uint64_t image_height, const double image_re, const double image_im, const double image_scale, const uint64_t image_chunk) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int pixel_x = (image_chunk + tid) % image_width;
    const int pixel_y = (image_chunk + tid) / image_width;

    const double re_c = image_re + (-2.0 + pixel_x * 3.0 / image_width) / image_scale;
    const double im_c = image_im + (1.0 - pixel_y * 2.0 / image_height) / image_scale;

    uint32_t escape = chunk_buffer[tid * 5];
    double re_z = reinterpret_cast<double *>(chunk_buffer)[tid * 5 + 2];
    double im_z = reinterpret_cast<double *>(chunk_buffer)[tid * 5 + 3];
    double abs_z = reinterpret_cast<double *>(chunk_buffer)[tid * 5 + 4]; 

    if (chunk_buffer[tid * 5 + 1] == 0) {
        escape = escape_limit;
        re_z = re_c;
        im_z = im_c;
        abs_z = 0.0;
    }

    if (escape == escape_limit) {
        for (uint32_t i = 0; i < escape_block; ++i) {
            double re_z_i = re_z;
            re_z = (re_z * re_z) - (im_z * im_z) + re_c;
            im_z = (2.0 * re_z_i * im_z) + im_c;
            abs_z = re_z * re_z + im_z * im_z;
            if (abs_z > 4.0) {
                escape = i + chunk_buffer[tid * 5 + 1];
                break;
            }
        }
    }

    chunk_buffer[tid * 5] = escape;
    chunk_buffer[tid * 5 + 1] += escape_block;
    reinterpret_cast<double *>(chunk_buffer)[tid * 5 + 2] = re_z;
    reinterpret_cast<double *>(chunk_buffer)[tid * 5 + 3] = im_z;
    reinterpret_cast<double *>(chunk_buffer)[tid * 5 + 4] = abs_z;
}

__global__ void mandelbrot_kernel_colour(uint64_t *chunk_buffer, uint32_t *image_chunk_buffer, const uint64_t image_width, const uint64_t image_height, const uint64_t image_chunk) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    uint64_t escape = chunk_buffer[tid * 5];
    double abs_z = reinterpret_cast<double *>(chunk_buffer)[tid * 5 + 4];

    abs_z = sqrtf(abs_z);

    double hue = 360.0*log(1.0*escape)/log(1.0*escape_limit) + 1.0 - (log(log(abs_z)) / log(2.0));
    double sat = 0.85;
    double val = 1.0;

    hue = fmod(hue, 360.0);
    hue /= 60;

    double hue_fract = hue - floor(hue);
    double p = val * (1.0 - sat);
    double q = val * (1.0 - sat * hue_fract);
    double t = val * (1.0 - sat * (1.0 - hue_fract));

    double r = 0;
    double g = 0;
    double b = 0;

    if (escape < escape_limit) {
        switch ((static_cast<unsigned char>(floor(hue)) + 3) % 6) {
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

uint64_t groups{ 0 };
uint64_t threads{ 0 };

uint64_t *chunk_buffer{ nullptr };
uint32_t *image_chunk_buffer{ nullptr };

int init(uint64_t image_width, uint64_t image_height) {
    cudaError_t cuda_status;
    if ((cuda_status = cudaSetDevice(0)) != cudaSuccess) {
        std::wcout << L"ERROR: cudaSetDevice() Failed. [" << cuda_status << L"]" << std::endl << std::endl;
        return -1;
    }

    threads = 1024;
    groups = 8192;

    cudaMalloc(&chunk_buffer, groups * threads * 5 * sizeof(uint64_t));
    cudaMalloc(&image_chunk_buffer, groups * threads * sizeof(uint32_t));

    return 0;
}

int uninit(uint64_t image_width, uint64_t image_height) {
    cudaFree(chunk_buffer);
    cudaFree(image_chunk_buffer);
    chunk_buffer = nullptr;
    image_chunk_buffer = nullptr;

    cudaError_t cuda_status;
    if ((cuda_status = cudaDeviceReset()) != cudaSuccess) {
        std::wcout << L"ERROR: cudaDeviceReset() Failed. [" << cuda_status << L"]" << std::endl << std::endl;
        return -1;
    }

    return 0;
}

int mandelbrot(uint32_t *image, const uint64_t image_width, const uint64_t image_height, const double image_center_re, const double image_center_im, const double image_scale) {
    cudaError_t cudaError;

    std::wcout << "[+] z = " << image_center_re << " + " << image_center_im << "i; scale = " << (1.0 / image_scale) << "; "
        << (image_center_re + (-2.0 / image_width) / image_scale) << " : "
        << (image_center_im + (1.0 / image_height) / image_scale) << std::endl;

    std::wcout << L"[+] Chunks: "
        << 1 + image_width * image_height / (groups * threads)
        << L" " << std::flush;

    for (uint64_t image_chunk = 0; image_chunk < (image_width * image_height); image_chunk += (groups * threads)) {
        uint64_t chunk_size = std::min((image_width * image_height) - image_chunk, groups * threads);
        uint64_t chunk_groups = chunk_size / threads;
        cudaMemset(chunk_buffer, 0, groups * threads * 5 * sizeof(uint64_t));
        cudaMemset(image_chunk_buffer, 0, groups * threads * sizeof(uint32_t));

        std::wcout << L"+" << std::flush;

        for (uint32_t i = 0; i < (escape_limit / escape_block); ++i) {
            std::wcout << L"." << std::flush;
            mandelbrot_kernel<<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(threads)>>>(chunk_buffer, image_width, image_height, image_center_re, image_center_im, image_scale, image_chunk);
            if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                std::wcout << std::endl << "[!] cudaDeviceSynchronize(): cudaError: " << cudaError << std::endl;
                return -1;
            }
        }

        std::wcout << L"*" << std::flush;
        mandelbrot_kernel_colour<<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(threads)>>>(chunk_buffer, image_chunk_buffer, image_width, image_height, image_chunk);
        if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
            std::wcout << std::endl << "[!] cudaDeviceSynchronize(): cudaError: " << cudaError << std::endl;
            return -1;
        }

        cudaMemcpy(&image[image_chunk], image_chunk_buffer, chunk_groups * threads * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    std::wcout << std::endl;

    return 0;
}