#include <windows.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <png.h>

#include "fractal.h"

int write_png(uint64_t image_width, uint64_t image_height, uint32_t *image) {
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

    auto row = new uint32_t[image_width];
    std::memset(row, 0, sizeof(uint32_t) * image_width);

    for (int y = 0; y < image_height; y++) {
        for (int x = 0; x < image_width; x++) {
            row[x] = image[x + y * image_width];
        }
        png_write_row(png_ptr, reinterpret_cast<png_const_bytep>(row));
    }

    png_write_end(png_ptr, NULL);
    fclose(fp);

    return 0;
}