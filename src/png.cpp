#include <windows.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <png.h>

#include "fractal.h"

int write_png(uint32_t *image, uint64_t image_width, uint64_t image_height, std::string &name) {
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);

    if (setjmp(png_jmpbuf(png_ptr))) {
        std::wcout << L"ERROR: libpng" << std::endl << std::endl;
        return -1;
    }

    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    std::stringstream fn; 
    fn << std::put_time(std::localtime(&now), "out\\%Y%m%d-%H%M%S ") << name << ".png";

    FILE *fp = fopen(fn.str().c_str(), "wb");

    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr,
        info_ptr,
        static_cast<png_uint_32>(image_width),
        static_cast<png_uint_32>(image_height),
        8,
        PNG_COLOR_TYPE_RGBA,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    auto row = new uint32_t[image_width];
    std::memset(row, 0, sizeof(uint32_t) * image_width);

    for (uint64_t y = 0; y < image_height; y++) {
        for (uint64_t x = 0; x < image_width; x++) {
            row[x] = image[x + y * image_width];
        }
        png_write_row(png_ptr, reinterpret_cast<png_const_bytep>(row));
    }

    png_write_end(png_ptr, NULL);
    fclose(fp);

    return 0;
}