//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <windows.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <tuple>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <png.h>

#include "fractal.h"
#include "png.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern HANDLE g_ExitEvent;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

png::png(std::string directory) : directory_(directory) {
    DWORD directory_attributes = GetFileAttributes(directory.c_str());
    if ((directory_attributes == INVALID_FILE_ATTRIBUTES) || !(directory_attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        CreateDirectory(directory_.c_str(), nullptr);
    }

    for (uint32_t i = 0; i < 4; ++i) {
        threads_.push_back(std::thread([this]() {
            while (true) {
                if (WaitForSingleObject(g_ExitEvent, 1000) == WAIT_OBJECT_0) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (queue_.empty()) {
                        break;
                    }
                }
                std::tuple<uint64_t, uint64_t, const uint32_t *, std::string> image{ 0, 0, nullptr, "" };
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (queue_.empty()) {
                        continue;
                    }
                    image = queue_.front();
                    queue_.pop();
                }
                write(image);
                delete[] std::get<2>(image);
            }
        }));
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

png::~png() {
    for (auto &t : threads_) {
        t.join();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void png::write(std::tuple<uint64_t, uint64_t, const uint32_t *, std::string> &image) {
    uint64_t image_width = std::get<0>(image);
    uint64_t image_height = std::get<1>(image);
    const uint32_t *image_buffer = std::get<2>(image);
    std::string suffix = std::get<3>(image);

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);

    if (setjmp(png_jmpbuf(png_ptr))) {
        std::cout << L"[!] ERROR: png::write(): libpng Exception" << std::endl;
        return;
    }

    tm tm_now{ 0 }; 
    __time64_t time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    _localtime64_s(&tm_now, &time_now);

    std::stringstream fn;
    fn << directory_ << "\\" << std::put_time(&tm_now, "%Y%m%d-%H%M%S") << " " << suffix << ".png";

    FILE *file{ nullptr };
    if (fopen_s(&file, fn.str().c_str(), "wb") != 0) {
        std::cout << L"[!] ERROR: png::write(): fopen_s Failed" << std::endl;
        return;
    }

    png_init_io(png_ptr, file);
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

    uint32_t *row_buffer = new uint32_t[image_width];
    if (row_buffer == nullptr) {
        return;
    }
    std::memset(row_buffer, 0, image_width * sizeof(uint32_t));

    for (uint64_t y = 0; y < image_height; y++) {
        for (uint64_t x = 0; x < image_width; x++) {
            row_buffer[x] = image_buffer[x + y * image_width];
        }
        png_write_row(png_ptr, reinterpret_cast<png_const_bytep>(row_buffer));
    }

    delete[] row_buffer;
    png_write_end(png_ptr, NULL);
    fclose(file);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
