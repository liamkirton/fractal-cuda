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
#include "image.h"
#include "png_writer.h"
#include "timer.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::tuple<double, double, double>> create_palette(YAML::Node &palette);

bool run(YAML::Node &run_config);
bool run_interactive(YAML::Node &run_config);

bool run_generate(YAML::Node &run_config, std::function<void(bool, image &, std::string &, uint32_t)> image_callback);
template<uint32_t I, uint32_t F> bool run_step(YAML::Node &run_config, uint32_t ix, std::vector<std::tuple<double, double, double>> &palette, std::function<void(bool, image &, std::string &, uint32_t)> image_callback);
template<> bool run_step<0, 0>(YAML::Node &run_config, uint32_t ix, std::vector<std::tuple<double, double, double>> &palette, std::function<void(bool, image &, std::string &, uint32_t)> image_callback);

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

    if (default_config["interactive"].as<bool>()) {
        YAML::Node interactive_config = default_config;
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
            for (auto &c : load_config) {
                interactive_config[c.first.as<std::string>()] = c.second;
            }
        }
        run_interactive(interactive_config);
    }
    else {
        try {
            if (inst_config_files.size() == 0) {
                inst_config_files.push_back("");
            }

            for (auto &f : inst_config_files) {
                YAML::Node load_config;

                if (f.size() > 0) {
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
                }

                YAML::Node run_config = default_config;
                for (auto &c : load_config) {
                    run_config[c.first.as<std::string>()] = c.second;
                }

                run(run_config);
            }
        }
        catch (std::exception &e) {
            std::cout << "[!] ERROR: Caught Unexpected Exception - " << e.what() << std::endl;
        }
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

bool run(YAML::Node &run_config) {
    std::unique_ptr<png_writer> png(new png_writer(run_config));
    return run_generate(run_config, [&png](bool complete, image &i, std::string &suffix, uint32_t ix) {
        if (complete) {
            png->write(i, suffix, ix);
        }
    });
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool run_interactive(YAML::Node &run_config) {
    HWND hWnd = NULL;

    uint32_t image_width = run_config["image_width"].as<uint32_t>();
    uint32_t image_height = run_config["image_height"].as<uint32_t>();

    uint32_t *img_buf = new uint32_t[image_width * image_height];
    memset(img_buf, 0, sizeof(uint32_t) * image_width * image_height);

    std::mutex mutex;
    std::queue<std::tuple<uint64_t, uint64_t>> queue;

    queue.push(std::make_tuple(image_width / 2, image_height / 2));

    auto gen_thread = std::thread([&]() {
        while (true) {
            std::tuple<uint64_t, uint64_t> coords{ 0, 0 };
            Sleep(1000);
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (queue.empty()) {
                    continue;
                }
                coords = queue.front();
                queue.pop();
            }

            run_config["coords_x"] = std::get<0>(coords);
            run_config["coords_y"] = std::get<1>(coords);

            std::cout << "[+] Generate... " << run_config["coords_x"].as<std::string>() << ", " << run_config["coords_y"].as<std::string>() << std::endl;

            run_generate(run_config, [&](bool complete, image &i, std::string &suffix, uint32_t ix) {
                memcpy(img_buf, i.image_buffer(), sizeof(uint32_t) * image_width * image_height);
                InvalidateRect(hWnd, NULL, TRUE);
            });
        }
    });

    std::function<LRESULT(HWND, UINT, WPARAM, LPARAM)> wndproc = [&](HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) -> LRESULT {
        switch (uMsg) {
        case WM_CLOSE:
            DestroyWindow(hWnd);
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            break;
        case WM_PAINT:
            {
                PAINTSTRUCT ps;
                HDC hdc = BeginPaint(hWnd, &ps);

                HDC hmdc = CreateCompatibleDC(hdc);

                RECT rect;
                GetWindowRect(hWnd, &rect);

                uint32_t width = rect.right - rect.left;
                uint32_t height = rect.bottom - rect.top;
            
                BITMAPINFO bi;
                ZeroMemory(&bi, sizeof(BITMAPINFO));
                bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
                bi.bmiHeader.biWidth = static_cast<LONG>(width);
                bi.bmiHeader.biHeight = -static_cast<LONG>(height);
                bi.bmiHeader.biPlanes = 1;
                bi.bmiHeader.biBitCount = 32; 
            
                uint32_t *lpBitmapBits{ nullptr };
                HBITMAP hbm = ::CreateDIBSection(hmdc, &bi, DIB_RGB_COLORS, (VOID**)&lpBitmapBits, NULL, 0);

                memcpy(lpBitmapBits, img_buf, sizeof(uint32_t) * width * height);
                HGDIOBJ old = SelectObject(hmdc, hbm);
                BitBlt(hdc, 0, 0, width, height, hmdc, 0, 0, SRCCOPY);

                SelectObject(hmdc, old);
                DeleteObject(hbm);
                DeleteDC(hmdc);

                EndPaint(hWnd, &ps);
            }
            break;
        case WM_LBUTTONUP:
        {
            std::lock_guard<std::mutex> lock(mutex);
            run_config["count"] = run_config["count"].as<uint32_t>() + 1;
            run_config["skip"] = run_config["skip"].as<uint32_t>() + 1;
            queue.push(std::make_tuple(LOWORD(lParam), HIWORD(lParam)));
            break;
        }
        case WM_RBUTTONUP:
        {
            std::lock_guard<std::mutex> lock(mutex);
            int32_t count = run_config["count"].as<int32_t>() - 1;
            int32_t skip = run_config["skip"].as<int32_t>() - 1;
            run_config["count"] = count >= 1 ? count : 1;
            run_config["skip"] = skip >= 0 ? skip : 0;
            queue.push(std::make_tuple(LOWORD(lParam), HIWORD(lParam)));
            break;
        }
        default:
            return DefWindowProc(hWnd, uMsg, wParam, lParam);
        }
        return 0;
    };

    WNDCLASSEX wc{ 0 };
    wc.cbSize = sizeof(wc);
    wc.cbClsExtra = 0; 
    wc.cbWndExtra = sizeof(size_t);
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = "fractal_interactive";
    wc.style = CS_HREDRAW | CS_VREDRAW;

    wc.lpfnWndProc = [](HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)->LRESULT {
        static std::function<LRESULT(HWND, UINT, WPARAM, LPARAM)> *wndproc{ nullptr };
        if (uMsg == WM_CREATE) {
            wndproc = reinterpret_cast<std::function<LRESULT(HWND, UINT, WPARAM, LPARAM)> *>(reinterpret_cast<LPCREATESTRUCT>(lParam)->lpCreateParams);
        }
        return (wndproc) ? (*wndproc)(hWnd, uMsg, wParam, lParam) : DefWindowProc(hWnd, uMsg, wParam, lParam);
    };

    if (RegisterClassEx(&wc) == 0) {
        std::cout << "[!] ERROR: RegisterClassEx() Failed - " << GetLastError() << std::endl;
        return false;
    }

    hWnd = CreateWindow(wc.lpszClassName,
        "CUDA Fractal",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        image_width,
        image_height,
        NULL,
        NULL,
        wc.hInstance,
        &wndproc);

    if (hWnd == NULL) {
        std::cout << "[!] ERROR: CreateWindow() Failed - " << GetLastError() << std::endl;
        return false;
    }

    ShowWindow(hWnd, SW_SHOW);
    UpdateWindow(hWnd);

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    gen_thread.join();
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool run_generate(YAML::Node &run_config, std::function<void(bool, image &, std::string &, uint32_t)> image_callback) {
    std::vector<std::tuple<double, double, double>> palette = create_palette(run_config);

    fixed_point<2, 32> scale(run_config["scale"].as<std::string>());
    fixed_point<2, 32> scale_factor(run_config["scale_factor"].as<std::string>());

    uint32_t image_width = run_config["image_width"].as<uint32_t>();
    uint32_t image_height = run_config["image_height"].as<uint32_t>();

    double step_re = (re_max - re_min) / image_width;
    double step_im = (im_max - im_min) / image_height;
    fixed_point<2, 32> step((step_re > step_im) ? step_re : step_im);

    for (uint32_t i = 0; i < run_config["count"].as<uint32_t>(); ++i) {
        if (i == run_config["skip"].as<uint32_t>() - 1) {
            if (run_config["coords_x"] && run_config["coords_y"]) {
                fixed_point<2, 32> re_c(run_config["re"].as<std::string>());
                fixed_point<2, 32> im_c(run_config["im"].as<std::string>());

                fixed_point<2, 32> re;
                fixed_point<2, 32> im;

                re.set(re_min + run_config["coords_x"].as<uint32_t>() * (re_max - re_min) / image_width);
                im.set(im_max - run_config["coords_y"].as<uint32_t>() * (im_max - im_min) / image_height);
                re.multiply(scale);
                im.multiply(scale);
                re.add(re_c);
                im.add(im_c);

                run_config["re"] = static_cast<std::string>(re);
                run_config["im"] = static_cast<std::string>(im);
                run_config["coords_x"] = image_width / 2;
                run_config["coords_y"] = image_height / 2;
            }
        }
        else if (i >= run_config["skip"].as<uint32_t>()) {
            fixed_point<2, 32> precision_test(step);
            precision_test.multiply(scale);
            uint32_t precision_bit = precision_test.get_fractional_significant_bit();

            std::cout << "[+] Generating Fractal #" << i << std::endl
                << "  [+] Precison Bit: " << precision_bit;

            if (precision_bit < 53) {
                run_step<0, 0>(run_config, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 2) {
                run_step<1, 2>(run_config, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 3) {
                run_step<1, 3>(run_config, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 4) {
                run_step<1, 4>(run_config, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 6) {
                run_step<1, 6>(run_config, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 8) {
                run_step<1, 8>(run_config, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 12) {
                run_step<1, 12>(run_config, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 16) {
                run_step<1, 16>(run_config, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 20) {
                run_step<1, 20>(run_config, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 24) {
                run_step<1, 24>(run_config, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 28) {
                run_step<1, 28>(run_config, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 32) {
                run_step<1, 32>(run_config, i, palette, image_callback);
            }
            else {
                std::cout << " - UNSUPPORTED" << std::endl;
            }
        }

        scale.multiply(scale_factor);
    }

    std::cout << std::endl;

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> bool run_step<0, 0>(YAML::Node &run_config, uint32_t ix, std::vector<std::tuple<double, double, double>> &palette, std::function<void(bool, image &, std::string &, uint32_t)> image_callback) {
    fractal<double> f(run_config["image_width"].as<uint32_t>(), run_config["image_height"].as<uint32_t>());
    if ((run_config["cuda_groups"].as<uint32_t>() != 0) && (run_config["cuda_threads"].as<uint32_t>() != 0)) {
        f.initialise(run_config["cuda_groups"].as<uint32_t>(), run_config["cuda_threads"].as<uint32_t>());
    }

    f.colour(run_config["colour_method"].as<uint32_t>(), palette);
    f.limits(run_config["escape_limit"].as<uint32_t>(), run_config["escape_block"].as<uint32_t>());

    std::cout << ", Type: double" << std::endl;

    double re = run_config["re"].as<double>();
    double im = run_config["im"].as<double>();
    double scale = run_config["scale"].as<double>();
    double scale_factor = run_config["scale_factor"].as<double>();

    for (uint32_t i = 0; i < ix; ++i) {
        scale *= scale_factor;
    }
    f.specify(re, im, scale);

    auto re_c = run_config["re_c"].as<std::string>();
    auto im_c = run_config["im_c"].as<std::string>();

    if (re_c.size() > 0) {
        std::cout << "  [+] Julia: " << re_c << ", " << im_c << std::endl;
        f.specify_julia(std::stod(re_c), std::stod(im_c));
    }
    else {
        std::cout << "  [+] Mandelbrot" << std::endl;
    }

    std::stringstream suffix;
    suffix << std::setfill('0')
        << "ix=" << ix << "_"
        << "re=" << std::setprecision(12) << f.re() << "_"
        << "im=" << std::setprecision(12) << f.im() << "_"
        << "scale=" << std::setprecision(12) << f.scale();

    auto do_callback = [&](bool complete) {
        image i(f.image_width(), f.image_height(), f.image());
        image_callback(complete, i, suffix.str(), ix);
    };

    timer gen_timer;
    if (f.generate(run_config["trial"].as<bool>(), [&]() { do_callback(false); })) {
        gen_timer.stop();
        gen_timer.print();
        do_callback(true);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F> bool run_step(YAML::Node &run_config, uint32_t ix, std::vector<std::tuple<double, double, double>> &palette, std::function<void(bool, image &i, std::string &suffix, uint32_t ix)> image_callback) {
    fractal<fixed_point<I, F>> f(run_config["image_width"].as<uint32_t>(), run_config["image_height"].as<uint32_t>());
    if ((run_config["cuda_groups"].as<uint32_t>() != 0) && (run_config["cuda_threads"].as<uint32_t>() != 0)) {
        f.initialise(run_config["cuda_groups"].as<uint32_t>(), run_config["cuda_threads"].as<uint32_t>());
    }

    f.colour(run_config["colour_method"].as<uint32_t>(), palette);
    f.limits(run_config["escape_limit"].as<uint32_t>(), run_config["escape_block"].as<uint32_t>());

    std::cout << ", Type: fixed_point<" << I << ", " << F << ">" << std::endl;

    fixed_point<I, F> re(run_config["re"].as<std::string>());
    fixed_point<I, F> im(run_config["im"].as<std::string>());
    fixed_point<I, F> scale(run_config["scale"].as<std::string>());
    fixed_point<I, F> scale_factor(run_config["scale_factor"].as<std::string>());

    for (uint32_t i = 0; i < ix; ++i) {
        scale.multiply(scale_factor);
    }
    f.specify(re, im, scale);

    auto re_c = run_config["re_c"].as<std::string>();
    auto im_c = run_config["im_c"].as<std::string>();

    if (re_c.size() > 0) {
        std::cout << "  [+] Julia: " << re_c << ", " << im_c << std::endl;
        f.specify_julia(re_c, im_c);
    }
    else {
        std::cout << "  [+] Mandelbrot" << std::endl;
    }

    std::stringstream suffix;
    suffix << std::setfill('0')
        << "ix=" << ix << "_"
        << "re=" << std::setprecision(12) << f.re() << "_"
        << "im=" << std::setprecision(12) << f.im() << "_"
        << "scale=" << std::setprecision(12) << f.scale();

    auto do_callback = [&](bool complete) {
        image i(f.image_width(), f.image_height(), f.image());
        image_callback(complete, i, suffix.str(), ix);
    };

    timer gen_timer;
    if (f.generate(run_config["trial"].as<bool>(), [&]() { do_callback(false); })) {
        gen_timer.stop();
        gen_timer.print();
        do_callback(true);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
