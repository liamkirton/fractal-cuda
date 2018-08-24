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

#include <yaml-cpp/yaml.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "fractal.h"
#include "png.h"
#include "timer.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::tuple<double, double, double>> create_palette(YAML::Node &palette);

void run(YAML::Node &run_config);

template<uint32_t I, uint32_t F> bool run_step(YAML::Node &run_config, uint32_t ix, std::vector<std::tuple<double, double, double>> &palette, png &png_writer);
template<> bool run_step<0, 0>(YAML::Node &run_config, uint32_t ix, std::vector<std::tuple<double, double, double>> &palette, png &png_writer);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
    std::cout << std::endl
        << "CUDA Fractal Generator" << std::endl
        << "(C)2018 Liam Kirton <liam@int3.ws>" << std::endl
        << std::endl;

    YAML::Node default_config;
    std::vector<std::string> inst_config_files;

    try {
        default_config = YAML::LoadFile("config/default.yaml");
    }
    catch (YAML::BadFile &) {
        std::cout << "[!] ERROR: Cannot Load config/default.yaml" << std::endl;
        return -1;
    }
    catch (YAML::ParserException &) {
        std::cout << "[!] ERROR: Cannot Parse config/default.yaml" << std::endl;
        return -1;
    }

    for (int i = 1; i < argc; ++i) {
        std::string p = argv[i];
        std::string v = "true"; 

        if (p.at(0) != '-') {
            inst_config_files.push_back(p);
            continue;
        }

        p = p.substr(1);
        if (i < argc - 1) {
            std::string parse_v = argv[i + 1];
            if ((parse_v.at(0) != '-') || std::all_of(parse_v.begin(), parse_v.end(), [](char c) { return isdigit(c) || (c == '.') || (c == '-') || (c == 'e') || (c == 'E'); })) {
                v = argv[++i];
            }
        }

        if (default_config[p] && (default_config[p].Type() == YAML::NodeType::Scalar)) {
            default_config[p] = v;
        }
        else {
            std::cout << "[!] ERROR: Unrecognised Option \"" << p << "\" = \"" << v << "\"" << std::endl;
            return -1;
        }
    }

    try {
        if (inst_config_files.size() == 0) {
            run(default_config);
        }
        else {
            for (auto &f : inst_config_files) {
                YAML::Node load_config;
                try {
                    load_config = YAML::LoadFile(f);
                }
                catch (YAML::BadFile &) {
                    std::cout << "[!] ERROR: Cannot Load " << f << std::endl;
                    continue;
                }
                catch (YAML::ParserException &) {
                    std::cout << "[!] ERROR: Cannot Parse " << f << std::endl;
                    continue;
                }

                YAML::Node run_config = default_config;
                for (auto &c : load_config) {
                    run_config[c.first.as<std::string>()] = c.second;
                }

                run(run_config);
            }
        }
    }
    catch (std::exception &e) {
        std::cout << "[!] ERROR: Caught Unexpected Exception - " << e.what() << std::endl;
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::tuple<double, double, double>> create_palette(YAML::Node &run_config) {
    std::vector <std::tuple<double, double, double>> palette;

    for (auto &i : run_config["colour_palette"]) {
        if (i.size() != 3) {
            continue;
        }

        double hue = i[0].as<double>();
        double sat = i[1].as<double>();
        double val = i[2].as<double>();

        if (hue > 1.0) hue /= 360.0;
        if (sat > 1.0) sat /= 100.0;
        if (val > 1.0) val /= 100.0;

        palette.push_back(std::make_tuple(hue, sat, val));
    }

    auto binomial = [](uint32_t k, uint32_t n) {
        double r = 1;
        for (uint32_t i = 1; i <= k; ++i) {
            r *= 1.0 * (n + 1 - i) / i;
        }
        return r;
    };

    for (uint32_t i = 1; i < palette.size(); ++i) {
        double coeff = binomial(i - 1, static_cast<uint32_t>(palette.size() - 2));
        auto &p = palette.at(i);
        std::get<0>(p) = std::get<0>(p) * coeff;
        std::get<1>(p) = std::get<1>(p) * coeff;
        std::get<2>(p) = std::get<2>(p) * coeff;
    }

    return palette;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void run(YAML::Node &run_config) {
    std::vector<std::tuple<double, double, double>> palette = create_palette(run_config);
    png png_writer(run_config);

    fixed_point<2, 32> scale(run_config["scale"].as<std::string>());
    fixed_point<2, 32> scale_factor(run_config["scale_factor"].as<std::string>());

    uint32_t image_width = run_config["image_width"].as<uint32_t>();
    uint32_t image_height = run_config["image_height"].as<uint32_t>();

    double step_re = (re_max - re_min) / image_width;
    double step_im = (im_max - im_min) / image_height;
    fixed_point<2, 32> step((step_re > step_im) ? step_re : step_im);

    for (uint32_t i = 0; i < run_config["count"].as<uint32_t>(); ++i) {
        if (i >= run_config["skip"].as<uint32_t>()) {
            fixed_point<2, 32> precision_test(step);
            precision_test.multiply(scale);
            uint32_t precision_bit = precision_test.get_fractional_significant_bit();

            std::cout << "[+] Generating Fractal #" << i << std::endl
                << "  [+] Precison Bit: " << precision_bit;

            if (precision_bit < 53) {
                run_step<0, 0>(run_config, i, palette, png_writer);
            }
            else if (precision_bit < 32 * 2) {
                run_step<1, 2>(run_config, i, palette, png_writer);
            }
            else if (precision_bit < 32 * 3) {
                run_step<1, 3>(run_config, i, palette, png_writer);
            }
            else if (precision_bit < 32 * 4) {
                run_step<1, 4>(run_config, i, palette, png_writer);
            }
            else if (precision_bit < 32 * 6) {
                run_step<1, 6>(run_config, i, palette, png_writer);
            }
            else if (precision_bit < 32 * 8) {
                run_step<1, 8>(run_config, i, palette, png_writer);
            }
            else if (precision_bit < 32 * 12) {
                run_step<1, 12>(run_config, i, palette, png_writer);
            }
            else if (precision_bit < 32 * 16) {
                run_step<1, 16>(run_config, i, palette, png_writer);
            }
            else if (precision_bit < 32 * 20) {
                run_step<1, 20>(run_config, i, palette, png_writer);
            }
            else if (precision_bit < 32 * 24) {
                run_step<1, 24>(run_config, i, palette, png_writer);
            }
            else if (precision_bit < 32 * 28) {
                run_step<1, 28>(run_config, i, palette, png_writer);
            }
            else if (precision_bit < 32 * 32) {
                run_step<1, 32>(run_config, i, palette, png_writer);
            }
            else {
                std::cout << " - UNSUPPORTED" << std::endl;
            }
        }

        std::cout << std::endl;
        scale.multiply(scale_factor);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> bool run_step<0, 0>(YAML::Node &run_config, uint32_t ix, std::vector<std::tuple<double, double, double>> &palette, png &png_writer) {
    fractal<double> f(run_config["image_width"].as<uint32_t>(), run_config["image_height"].as<uint32_t>());
    if ((run_config["cuda_groups"].as<uint32_t>() != 0) && (run_config["cuda_threads"].as<uint32_t>() != 0)) {
        f.initialise(run_config["cuda_groups"].as<uint32_t>(), run_config["cuda_threads"].as<uint32_t>());
    }

    f.colour(run_config["colour_method"].as<uint32_t>(), palette);
    f.limits(run_config["escape_limit"].as<uint32_t>(), run_config["escape_block"].as<uint32_t>());

    double re = run_config["re"].as<double>();
    double im = run_config["im"].as<double>();
    double scale = run_config["scale"].as<double>();
    double scale_factor = run_config["scale_factor"].as<double>();

    for (uint32_t i = 0; i < ix; ++i) {
        scale *= scale_factor;
    }

    std::cout << ", Type: double" << std::endl;
    f.specify(re, im, scale);

    timer gen_timer;
    if (f.generate(run_config["trial"].as<bool>(), true)) {
        gen_timer.stop();
        gen_timer.print();
        png_writer.write(f, ix);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F> bool run_step(YAML::Node &run_config, uint32_t ix, std::vector<std::tuple<double, double, double>> &palette, png &png_writer) {
    fractal<fixed_point<I, F>> f(run_config["image_width"].as<uint32_t>(), run_config["image_height"].as<uint32_t>());
    if ((run_config["cuda_groups"].as<uint32_t>() != 0) && (run_config["cuda_threads"].as<uint32_t>() != 0)) {
        f.initialise(run_config["cuda_groups"].as<uint32_t>(), run_config["cuda_threads"].as<uint32_t>());
    }

    f.colour(run_config["colour_method"].as<uint32_t>(), palette);
    f.limits(run_config["escape_limit"].as<uint32_t>(), run_config["escape_block"].as<uint32_t>());

    fixed_point<I, F> re(run_config["re"].as<std::string>());
    fixed_point<I, F> im(run_config["im"].as<std::string>());
    fixed_point<I, F> scale(run_config["scale"].as<std::string>());
    fixed_point<I, F> scale_factor(run_config["scale_factor"].as<std::string>());

    for (uint32_t i = 0; i < ix; ++i) {
        scale.multiply(scale_factor);
    }

    std::cout << ", Type: fixed_point<" << I << ", " << F << ">" << std::endl;
    f.specify(re, im, scale);

    timer gen_timer;
    if (f.generate(run_config["trial"].as<bool>(), true)) {
        gen_timer.stop();
        gen_timer.print();
        png_writer.write(f, ix);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
