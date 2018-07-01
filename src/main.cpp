#include <windows.h>

#include <cmath>
#include <cstdint>
#include <iomanip>
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

    LARGE_INTEGER p_t0{ 0 };
    LARGE_INTEGER p_t1{ 0 };

    std::vector<std::thread> png_threads;
    auto png = [&png_threads](const uint32_t *image, uint64_t image_width, uint64_t image_height) {
        png_threads.push_back(std::thread([](const uint32_t *image, uint64_t image_width, uint64_t image_height) {
            std::stringstream name;
            name << 0;
            write_png(image, image_width, image_height, name.str());
        }, image, image_width, image_height));
    };

    /*fixed_point<2, 2> a(0.123456789012345);
    double b = 1.0;
    uint64_t c = (*reinterpret_cast<uint64_t *>(&a.data[2 - 2]) >> 12);
    *reinterpret_cast<uint64_t *>(&b) |= c;
    b -= 1;

    std::cout << std::hex << std::setfill('0') << std::setw(16) << *reinterpret_cast<uint64_t *>(&b) << std::endl
        << std::dec << b << std::endl;*/

    fixed_point<2, 8> re("-0.08312772914623438");
    fixed_point<2, 8> im("0.8292469389114127");
    fixed_point<2, 8> scale(1.0 / 1.5095605350e+08);
    fixed_point<2, 8> scale_factor(0.5);
    for (uint32_t i = 0; i < 48; ++i) {
        scale.multiply(scale_factor);
    }

    fractal<fixed_point<2, 8>> f(5120, 2880);
    //fractal<double> f(32768, 32768);
    f.limits(4096, 32768);
    //f.specify(-0.74516, 0.112575, 1 / 6.5E-4);
    //f.specify(-0.66, 0.15, 1.66);
    //.specify(-0.7440, 0.1102, 1.0/200);
    //f.specify(-1.16403856759996471, 0.229637178821327975, scale);
    f.specify(re, im, scale);
        

    std::wcout << L"[+] Generating Fractal: " << f.image_width() << L"x" << f.image_height() << " (" << f.image_size() << L" bytes)" << std::endl;

    QueryPerformanceCounter(&p_t0);
    if (!f.generate()) {
        std::wcout << L"[!] Generation Failed." << std::endl;
    }
    QueryPerformanceCounter(&p_t1);

    std::wcout << "[+] Time: " << (1.0 * (p_t1.QuadPart - p_t0.QuadPart)) / p_freq.QuadPart << std::endl;

    png(f.image(true), f.image_width(), f.image_height());

    for (auto &t : png_threads) {
        t.join();
    }

    return 0;
}