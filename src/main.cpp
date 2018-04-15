#include <windows.h>

#include <cmath>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "fractal.h"

#include <png.h>

int main() {
    //
    // Initialise CUDA
    //

    cudaError_t cuda_status;
    if ((cuda_status = cudaSetDevice(0)) != cudaSuccess) {
        std::wcout << L"ERROR: cudaSetDevice() Failed. [" << cuda_status << L"]" << std::endl << std::endl;
        return -1;
    }

    LARGE_INTEGER p_freq;
    QueryPerformanceFrequency(&p_freq);

    LARGE_INTEGER p_t0;
    QueryPerformanceCounter(&p_t0);

    unsigned int image_width = 2560;
    unsigned int image_height = 1440;

    image_width *= 1.5;
    image_height *= 1.5;

    unsigned int *result_buffer = new unsigned int[image_width * image_height];
    mandelbrot(image_width, image_height, result_buffer);

    LARGE_INTEGER p_t1;
    QueryPerformanceCounter(&p_t1);

    std::wcout << L"[+] Complete. " << (p_t1.QuadPart - p_t0.QuadPart) / p_freq.QuadPart << std::endl;

    //
    // Finish CUDA
    //

    if ((cuda_status = cudaDeviceReset()) != cudaSuccess) {
        std::wcout << L"ERROR: cudaDeviceReset() Failed. [" << cuda_status << L"]" << std::endl << std::endl;
        return -1;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);

    if (setjmp(png_jmpbuf(png_ptr))) {
        std::wcout << L"ERROR: libpng" << std::endl << std::endl;
        return -1;
    }

    FILE *fp = fopen("out.png", "wb");

    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr,
        info_ptr,
        image_width,
        image_height,
        8,
        PNG_COLOR_TYPE_RGBA,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    auto row = (png_bytep)malloc(4 * image_width * sizeof(png_byte));

    for (int y = 0; y < image_height; y++) {
        for (int x = 0; x < image_width; x++) {
            *((unsigned int *)&row[x * 4]) = result_buffer[x + y * image_width];
        }
        png_write_row(png_ptr, row);
    }

    png_write_end(png_ptr, NULL);
    fclose(fp);

    return 0;
}