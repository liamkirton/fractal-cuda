//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <windows.h>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <tuple>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "fractal.h"
#include "png.h"
#include "timer.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct run_params {
    uint64_t I = 0;
    uint64_t F = 0;

    uint64_t image_width = default_image_width;
    uint64_t image_height = default_image_height;

    std::string re = "0.0";
    std::string im = "0.0";
    std::string scale = "1.0";
    std::string scale_factor = "0.5";

    uint64_t count = 1;
    uint64_t skip = 0;

    uint8_t colour_method = 0; 
    uint64_t escape_limit = default_escape_limit;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
void run(png &png_writer, run_params &params);

template<>
void run<0, 0>(png &png_writer, run_params &params);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

HANDLE g_ExitEvent{ nullptr };

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
    g_ExitEvent = CreateEvent(nullptr, TRUE, FALSE, nullptr); 

    auto usage = []() {
        std::cout
            << std::endl
            << "Usage: cuda-fractal.exe -r|-re <re> -i|-im <im>" << std::endl
            << "                        -s|-scale <scale> -sf|-scale-factor <scale-factor>" << std::endl
            << "                        -c|-count <count> -el|-escape-limit <escape-limit> -cm|-colour-method <colour-method>" << std::endl
            << "                        -fp|-fixed-point I/F" << std::endl
            << "                        -w|-width <width> -h|-height <height>" << std::endl
            << "                        -q|-quick -d|-detailed" << std::endl
            << std::endl;
        return -1;
    };

    run_params params;
    std::string directory = "output";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string param = "";
        if ((i + 1) < argc) {
            param = argv[i + 1];
        }
        if ((arg == "-r") || (arg == "-re")) {
            params.re = param;
            ++i;
        }
        else if ((arg == "-i") || (arg == "-im")) {
            params.im = param;
            ++i;
        }
        else if ((arg == "-s") || (arg == "-scale")) {
            params.scale = param;
            ++i;
        }
        else if ((arg == "-sf") || (arg == "-scale-factor")) {
            params.scale_factor = param;
            ++i;
        }
        else if ((arg == "-c") || (arg == "-count")) {
            params.count = std::atoll(param.c_str());
            ++i;
        }
        else if ((arg == "-el") || (arg == "-escape-limit")) {
            params.escape_limit = std::atoll(param.c_str());
            ++i;
        }
        else if ((arg == "-cm") || (arg == "-colour-method")) {
            params.colour_method = static_cast<uint8_t>(std::atoi(param.c_str()));
            ++i;
        }
        else if (arg == "-skip") {
            params.skip = std::atoll(param.c_str());
            ++i;
        }
        else if ((arg == "-w") || (arg == "-width")) {
            params.image_width = std::atoll(param.c_str());
            ++i;
        }
        else if ((arg == "-h") || (arg == "-height")) {
            params.image_height = std::atoll(param.c_str());
            ++i;
        }
        else if (((arg == "-fp") || (arg == "-fixed-point")) && (param.find("/") != std::string::npos)) {
            params.I = std::atoll(param.substr(0, param.find("/")).c_str());
            params.F = std::atoll(param.substr(param.find("/") + 1).c_str());
            if ((params.I == 0) || (params.I > 4) || (params.F == 0) || (params.F > 32)) {
                return usage();
            }
            ++i;
        }
        else if ((arg == "-d") || (arg == "-detailed")) {
            params.image_width = 5120;
            params.image_height = 2880;
            params.escape_limit = 4 * 1048576;
        }
        else if ((arg == "-q") || (arg == "-quick")) {
            params.image_width = 320;
            params.image_height = 240;
            params.escape_limit = 256;
        }
        else if ((arg == "-o") || (arg == "-output")) {
            directory = param;
            ++i;
        }
        else {
            std::cout << arg << std::endl;
            return usage();
        }
    }

    png png_writer(directory);

    if ((params.I == 0) && (params.F == 0)) {
        run<0, 0>(png_writer, params);
    }
    else {
        switch (params.I) {
        case 1:
            switch (params.F) {
            case 1: run<1, 1>(png_writer, params); break;
            case 2: run<1, 2>(png_writer, params); break;
            default: return usage();
            }
            break;
        case 2:
            switch (params.F) {
            case 2: run<2, 2>(png_writer, params); break;
            case 4: run<2, 4>(png_writer, params); break;
            case 8: run<2, 8>(png_writer, params); break;
            case 16: run<2, 16>(png_writer, params); break;
            case 24: run<2, 24>(png_writer, params); break;
            case 32: run<2, 32>(png_writer, params); break;
            default: return usage();
            }
            break;
        default:
            return usage();
        }
    }

    SetEvent(g_ExitEvent);

    std::cout << std::endl;

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
void run<0, 0>(png &png_writer, run_params &params) {
    fractal<double> f(params.image_width, params.image_height);
    f.colour(params.colour_method);
    f.limits(params.escape_limit);

    double re = std::stod(params.re);
    double im = std::stod(params.im);
    double scale = std::stod(params.scale);
    double scale_factor = std::stod(params.scale_factor);

    for (uint64_t i = 0; i < params.count; ++i) {
        if (i >= params.skip) {
            std::cout << std::endl << "[+] Generating Fractal #" << i << std::endl;
            f.specify(re, im, scale);

            timer gen_timer;
            if (!f.generate()) {
                break;
            }
            gen_timer.stop();
            gen_timer.print();

            png_writer.write(f, i);
        }
        scale *= scale_factor;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
void run(png &png_writer, run_params &params) {
    fractal<fixed_point<I, F>> f(params.image_width, params.image_height);
    f.colour(params.colour_method);
    f.limits(params.escape_limit); 

    fixed_point<I, F> re(params.re);
    fixed_point<I, F> im(params.im);
    fixed_point<I, F> scale(params.scale);
    fixed_point<I, F> scale_factor(params.scale_factor);

    for (uint64_t i = 0; i < params.count; ++i) {
        if (i >= params.skip) {
            std::cout << std::endl << "[+] Generating Fractal #" << i << std::endl;
            f.specify(re, im, scale);

            timer gen_timer;
            if (!f.generate()) {
                break;
            }
            gen_timer.stop();
            gen_timer.print();

            png_writer.write(f, i);
        }
        scale.multiply(scale_factor);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
