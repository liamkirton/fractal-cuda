#include <windows.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "fractal.h"

int main() {
    LARGE_INTEGER p_freq{ 0 };
    QueryPerformanceFrequency(&p_freq);

    const unsigned int image_width = 1920;
    const unsigned int image_height = 1024;

    const double image_center_re = -1.16403856759996471;
    const double image_center_im = 2.29637178821327975e-01;
    double image_scale = 1024;

    init(image_width, image_height);

    std::vector<std::thread> png_threads;

    for (uint32_t i = 0; i < 256; ++i) {
        std::wcout << L"[+] Generating Mandelbrot: " << image_width << L"x" << image_height << L" At " << image_scale << " (" << image_width * image_height * sizeof(uint32_t) << L" bytes)" << std::endl;

        LARGE_INTEGER p_t0{ 0 };
        LARGE_INTEGER p_t1{ 0 };

        uint32_t *image = new uint32_t[image_width * image_height];

        QueryPerformanceCounter(&p_t0);
        if (mandelbrot(image, image_width, image_height, image_center_re, image_center_im, image_scale) < 0) {
            break;
        }
        QueryPerformanceCounter(&p_t1);

        image_scale *= 2.0;

        png_threads.push_back(std::thread([p_freq, image_width, image_height](uint32_t *image) {
            std::stringstream name;
            name << 0;
            write_png(image, image_width, image_height, name.str());
            delete[] image;
        }, image));

        break;
    }

    for (auto &t : png_threads) {
        t.join();
    }

    uninit(image_width, image_height);

    return 0;
}