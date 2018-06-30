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
#include "png.h"

int main() {
    LARGE_INTEGER p_freq{ 0 };
    QueryPerformanceFrequency(&p_freq);

    std::vector<std::thread> png_threads;

    LARGE_INTEGER p_t0{ 0 };
    LARGE_INTEGER p_t1{ 0 };

    fractal<1, 2> f;
    f.specify(0.25, 0, 20);
    std::wcout << L"[+] Generating Mandelbrot: " << f.image_width() << L"x" << f.image_height() << " (" << f.image_size() << L" bytes)" << std::endl;

    QueryPerformanceCounter(&p_t0); 
    f.generate(true);
    QueryPerformanceCounter(&p_t1);
    std::wcout << "[+] Time: " << (1.0 * (p_t1.QuadPart - p_t0.QuadPart)) / p_freq.QuadPart << std::endl;

    png_threads.push_back(std::thread([](const uint32_t *image, uint64_t image_width, uint64_t image_height) {
        std::stringstream name;
        name << 0;
        write_png(image, image_width, image_height, name.str());
    }, f.image(), f.image_width(), f.image_height()));

    for (auto &t : png_threads) {
        t.join();
    }

    return 0;
}