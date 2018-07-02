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

    uint64_t image_width = 5120;
    uint64_t image_height = 2880;

    fractal<double> f(image_width, image_height);
    //fractal<fixed_point<2, 4>> f(image_width, image_height);
    //f.specify(-0.74516, 0.112575, 6.5E-4);
    //f.specify(-0.4706839696164857, -0.5829393990803547, 1/8.6797568398e+10);
    //f.specify(-0.7440, 0.1102, 1.0/200);
    //f.specify(-0.13856524454488, -0.64935990748190, 0.00000000045);
    //f.specify(-1.16403856759996471, 0.229637178821327975, 1.0/2.0);

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