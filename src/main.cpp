//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
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
    uint64_t cuda_groups = 0;
    uint64_t cuda_threads = 0;

    uint64_t image_width = default_image_width;
    uint64_t image_height = default_image_height;

    std::string re = "0.0";
    std::string im = "0.0";
    std::string scale = "1.0";
    std::string scale_factor = "0.5";

    bool follow_variance = false;
    bool random = false;
    bool reset = false;

    uint64_t count = 1;
    uint64_t skip = 0;

    uint8_t colour_method = 0;
    std::string palette_file = "config\\palette_00.txt";
    std::vector <std::tuple<double, double, double>> palette;

    uint64_t escape_limit = default_escape_limit;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void load_palette(run_params &params);

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
            << "                        -fv|-follow-variance -r|-random -reset" << std::endl
            << "                        -c|-count <count> -el|-escape-limit <escape-limit>" << std::endl
            << "                        -cm|-colour-method <colour-method> -pf|-palette-file <palette-file.txt>" << std::endl
            << "                        -fp|-fixed-point <I/F> -cuda <G/T>" << std::endl
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
        else if ((arg == "-fv") || (arg == "-follow-variance")) {
            params.follow_variance = true;
        }
        else if ((arg == "-r") || (arg == "-random")) {
            params.random = true;
        }
        else if (arg == "-reset") {
            params.reset = true;
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
        else if ((arg == "-pf") || (arg == "-palette-file")) {
            params.palette_file = param;
            ++i;
        }
        else if (arg == "-skip") {
            params.skip = std::atoll(param.c_str());
            ++i;
        }
        else if ((arg == "-w") || (arg == "-width")) {
            params.image_width = std::atoll(param.c_str());
            if (params.image_width == 0) {
                return usage();
            }
            ++i;
        }
        else if ((arg == "-h") || (arg == "-height")) {
            params.image_height = std::atoll(param.c_str());
            if (params.image_height == 0) {
                return usage();
            }
            ++i;
        }
        else if (((arg == "-fp") || (arg == "-fixed-point")) && (param.find("/") != std::string::npos)) {
            params.I = std::atoll(param.substr(0, param.find("/")).c_str());
            params.F = std::atoll(param.substr(param.find("/") + 1).c_str());
            if ((params.I == 0) || (params.I > 4) || (params.F == 0) || (params.F > 128)) {
                return usage();
            }
            ++i;
        }
        else if ((arg == "-cuda") && (param.find("/") != std::string::npos)) {
            params.cuda_groups = std::atoll(param.substr(0, param.find("/")).c_str());
            params.cuda_threads = std::atoll(param.substr(param.find("/") + 1).c_str());
            if ((params.cuda_groups == 0) || ((params.cuda_groups > 1) && ((params.cuda_groups % 2) != 0)) ||
                (params.cuda_threads == 0) || (params.cuda_threads > 1024) || ((params.cuda_threads > 1) && ((params.cuda_threads % 2) != 0))) {
                return usage();
            }
            ++i;
        }
        else if ((arg == "-d") || (arg == "-detailed")) {
            params.image_width = 5120;
            params.image_height = 2880;
            params.escape_limit = default_escape_limit;
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

    load_palette(params);

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

void load_palette(run_params &params) {
    std::vector <std::tuple<double, double, double>> parse_palette;

    std::ifstream f(params.palette_file, std::ios::in);
    std::string l;
    while (f.is_open() && !f.eof() && std::getline(f, l)) {
        if ((l.length() == 0) || (l.at(0) == '#')) {
            continue;
        }

        double hue = 0.0;
        double sat = 0.0;
        double val = 0.0;

        std::stringstream ls(l);
        ls >> hue >> sat >> val;

        if (hue > 1.0) hue /= 360.0;
        if (sat > 1.0) sat /= 100.0;
        if (val > 1.0) val /= 100.0;

        parse_palette.push_back(std::make_tuple(hue, sat, val));
    }

    auto binomial = [](uint32_t k, uint32_t n) {
        double r = 1;
        for (uint32_t i = 1; i <= k; ++i) {
            r *= 1.0 * (n + 1 - i) / i;
        }
        return r;
    };

    for (uint32_t i = 0; i < parse_palette.size(); ++i) {
        auto &p = parse_palette.at(i);
        double hue = std::get<0>(p);
        double sat = std::get<1>(p);
        double val = std::get<2>(p);
        double coeff = binomial(i, parse_palette.size() - 1);
        params.palette.push_back(std::make_tuple(
            hue * coeff,
            sat * coeff,
            val * coeff
        ));
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
void run<0, 0>(png &png_writer, run_params &params) {
    fractal<double> f(params.image_width, params.image_height);
    if ((params.cuda_groups != 0) && (params.cuda_threads != 0)) {
        f.initialise(params.cuda_groups, params.cuda_threads);
    }

    f.colour(params.colour_method, params.palette);
    f.limits(params.escape_limit);

    double re{ 0 };
    double im{ 0 };
    double scale{ 0 };
    double scale_factor{ 0 };

    auto reset = [&]() {
        re = std::stod(params.re);
        im = std::stod(params.im);
        scale = std::stod(params.scale);
        if (scale > 10.0) {
            scale = 1.0 / scale;
        }
        scale_factor = std::stod(params.scale_factor);
        if (params.random) {
            std::random_device random;
            std::mt19937 gen(random());
            re = std::uniform_real_distribution<>(-2.0, 1.0)(gen);
            im = std::uniform_real_distribution<>(-1.0, 1.0)(gen);
            scale = std::uniform_real_distribution<>(0.00005, 1.0)(gen);
        }
    };

    reset();

    for (uint64_t i = 0; i < params.count; ++i) {
        if (i >= params.skip) {
            std::cout << std::endl << "[+] Generating Fractal #" << i << std::endl;
            f.specify(re, im, scale);

            timer gen_timer;
            if (f.generate()) {
                gen_timer.stop();
                gen_timer.print();

                png_writer.write(f, i);

                if (params.follow_variance) {
                    re = f.re_max_variance();
                    im = f.im_max_variance();
                }
            }
            else {
                if (!params.reset) {
                    break;
                }
                --i;
                reset();
            }
        }
        scale *= scale_factor;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
void run(png &png_writer, run_params &params) {
    fractal<fixed_point<I, F>> f(params.image_width, params.image_height);
    if ((params.cuda_groups != 0) && (params.cuda_threads != 0)) {
        f.initialise(params.cuda_groups, params.cuda_threads);
    }
    f.colour(params.colour_method, params.palette);
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

            if (params.follow_variance) {
                re.set(f.re_max_variance());
                im.set(f.im_max_variance());
            }
        }
        scale.multiply(scale_factor);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
