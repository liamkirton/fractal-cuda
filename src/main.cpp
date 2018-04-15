#include <windows.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "fractal.h"

int main() {
    LARGE_INTEGER p_freq{ 0 };
    QueryPerformanceFrequency(&p_freq);

    LARGE_INTEGER p_t0{ 0 };
    LARGE_INTEGER p_t1{ 0 };

    unsigned int image_width = 2560 * 4;
    unsigned int image_height = 1440 * 4;
    uint32_t *image = new uint32_t[image_width * image_height];

    std::wcout << L"[+] Generating Mandelbrot: " << image_width << L"x" << image_height << L" (" << image_width * image_height * sizeof(uint32_t) << L" bytes)" << std::endl;

    QueryPerformanceCounter(&p_t0);
    mandelbrot(image_width, image_height, image);
    QueryPerformanceCounter(&p_t1);

    std::wcout << L"[+] Generation Complete: " << (p_t1.QuadPart - p_t0.QuadPart) / p_freq.QuadPart << L" secs." << std::endl;

    QueryPerformanceCounter(&p_t0);
    write_png(image_width, image_height, image);
    QueryPerformanceCounter(&p_t1);

    std::wcout << L"[+] PNG Complete: " << (p_t1.QuadPart - p_t0.QuadPart) / p_freq.QuadPart << L" secs." << std::endl;

    delete[] image;
    image = nullptr;

    return 0;
}