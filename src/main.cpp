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

    unsigned int image_width = 1024*2;
    unsigned int image_height = 768;

    init(image_width, image_height);

    std::vector<std::thread> png_threads;

    for (double i = 1.0; i < 2.0; i += 1.0) {
        uint32_t *image = new uint32_t[image_width * image_height];

        std::wcout << L"[+] Generating Mandelbrot: " << image_width << L"x" << image_height << L" (" << image_width * image_height * sizeof(uint32_t) << L" bytes)" << std::endl;

        LARGE_INTEGER p_t0{ 0 };
        LARGE_INTEGER p_t1{ 0 };

        QueryPerformanceCounter(&p_t0);
        mandelbrot(image_width, image_height, i, image);
        QueryPerformanceCounter(&p_t1);

        std::wcout << L"[+] Generation Complete: " << (p_t1.QuadPart - p_t0.QuadPart) / p_freq.QuadPart << L" secs." << std::endl;

        png_threads.push_back(std::thread([p_freq, image_width, image_height, i](uint32_t *image) {
            std::stringstream name;
            name << i;
            write_png(image_width, image_height, image, name.str());
            delete[] image;
        }, image));
    }

    for (auto &t : png_threads) {
        t.join();
    }

    uninit(image_width, image_height);

    return 0;
}