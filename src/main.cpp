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

    const unsigned int image_width = 2048;
    const unsigned int image_height = 2048;

    //const double image_center_re = 0;
    //const double image_center_im = 0;
    //const double image_scale = 1.0;

    /*const double image_center_re = -1.16403856759996471;
    const double image_center_im = 2.29637178821327975e-01;
    double image_scale = 1024;*/

    const double image_center_re = -0.74516;
    const double image_center_im = 0.112575;
    const double image_scale = 1 / 6.5E-4;

    init(image_width, image_height);

    std::vector<std::thread> png_threads;

    std::wcout << L"[+] Generating Mandelbrot: " << image_width << L"x" << image_height << L" (" << image_width * image_height * sizeof(uint32_t) << L" bytes)" << std::endl;

    LARGE_INTEGER p_t0{ 0 };
    LARGE_INTEGER p_t1{ 0 };

    uint32_t *image = new uint32_t[image_width * image_height];

    QueryPerformanceCounter(&p_t0);
    mandelbrot(image, image_width, image_height, image_center_re, image_center_im, image_scale);
    QueryPerformanceCounter(&p_t1);

    png_threads.push_back(std::thread([p_freq, image_width, image_height](uint32_t *image) {
        std::stringstream name;
        name << 0;
        write_png(image, image_width, image_height, name.str());
        delete[] image;
    }, image));

    for (auto &t : png_threads) {
        t.join();
    }

    uninit(image_width, image_height);

    return 0;
}