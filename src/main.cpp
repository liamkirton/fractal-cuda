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

struct run_state {
    run_state() : count(1), skip(0), julia(false),
        image_width(1024), image_height(768),
        cuda_groups(default_cuda_groups), cuda_threads(default_cuda_threads),
        trial(true), colour_method(2), escape_limit(default_escape_limit), escape_block(default_escape_block) {}

    YAML::Node run_config;

    uint32_t count;
    uint32_t skip;

    fixed_point<2, 32> re;
    fixed_point<2, 32> im;
    fixed_point<2, 32> scale;
    fixed_point<2, 32> scale_factor;
    fixed_point<2, 32> step;

    bool julia;
    fixed_point<2, 32> re_c;
    fixed_point<2, 32> im_c;

    uint32_t image_width;
    uint32_t image_height;

    uint32_t cuda_groups;
    uint32_t cuda_threads;

    bool trial;
    uint32_t colour_method;

    uint32_t escape_limit;
    uint32_t escape_block;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::tuple<double, double, double>> create_palette(run_state &r);
void create_run_state(run_state &r, YAML::Node &run_config);

bool run(run_state &r);
bool run_interactive(run_state &r);

bool run_generate(run_state &r, std::function<void(bool, image &, std::string &, uint32_t)> image_callback);
template<uint32_t I, uint32_t F> bool run_step(run_state &r, uint32_t ix, std::vector<std::tuple<double, double, double>> &palette, std::function<void(bool, image &, std::string &, uint32_t)> image_callback);
template<> bool run_step<0, 0>(run_state &r, uint32_t ix, std::vector<std::tuple<double, double, double>> &palette, std::function<void(bool, image &, std::string &, uint32_t)> image_callback);

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

        run_state r;
        create_run_state(r, interactive_config);
        run_interactive(r);
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

                run_state r;
                create_run_state(r, run_config);
                run(r);
            }
        }
        catch (std::exception &e) {
            std::cout << "[!] ERROR: Caught Unexpected Exception - " << e.what() << std::endl;
        }
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::tuple<double, double, double>> create_palette(run_state &r) {
    std::vector <std::tuple<double, double, double>> palette;

    for (auto &i : r.run_config["colour_palette"]) {
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

void create_run_state(run_state &r, YAML::Node &run_config) {
    r.run_config = run_config;

    r.re.set(run_config["re"].as<std::string>());
    r.im.set(run_config["im"].as<std::string>());
    r.scale.set(run_config["scale"].as<std::string>());
    r.scale_factor.set(run_config["scale_factor"].as<std::string>());

    if ((run_config["re_c"].as<std::string>().size() > 0) && (run_config["im_c"].as<std::string>().size() > 0)) {
        r.julia = true;
        r.re_c.set(run_config["re_c"].as<std::string>());
        r.im_c.set(run_config["im_c"].as<std::string>());
    }

    r.image_width = run_config["image_width"].as<uint32_t>();
    r.image_height = run_config["image_height"].as<uint32_t>();

    double step_re = (re_max - re_min) / r.image_width;
    double step_im = (im_max - im_min) / r.image_height;
    r.step.set((step_re > step_im) ? step_re : step_im);

    r.count = r.run_config["count"].as<uint32_t>();
    r.skip = r.run_config["skip"].as<uint32_t>();

    r.colour_method = r.run_config["colour_method"].as<uint32_t>();

    r.escape_block = r.run_config["escape_block"].as<uint32_t>(); 
    r.escape_limit = r.run_config["escape_limit"].as<uint32_t>();

    r.cuda_groups = r.run_config["cuda_groups"].as<uint32_t>();
    r.cuda_threads = r.run_config["cuda_threads"].as<uint32_t>();

    r.trial = r.run_config["trial"].as<bool>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool run(run_state &r) {
    std::unique_ptr<png_writer> png(new png_writer(r.run_config));
    return run_generate(r, [&png](bool complete, image &i, std::string &suffix, uint32_t ix) {
        if (complete) {
            png->write(i, suffix, ix);
        }
    });
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool run_interactive(run_state &r) {
    HWND hWnd{ nullptr };

    std::mutex mutex;
    std::queue<std::tuple<bool, uint64_t, uint64_t>> queue;
    queue.push(std::make_tuple(false, r.image_width / 2, r.image_height / 2));

    uint32_t *img_buf = new uint32_t[r.image_width * r.image_height];
    memset(img_buf, 0, sizeof(uint32_t) * r.image_width * r.image_height);

    auto gen_thread = std::thread([&]() {
        while (true) {
            std::tuple<bool, uint64_t, uint64_t> coords{ false, 0, 0 };
            Sleep(1000);
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (queue.empty()) {
                    continue;
                }
                coords = queue.front();
                queue.pop();
            }

            fixed_point<2, 32> re_c(r.re);
            fixed_point<2, 32> im_c(r.im);
            fixed_point<2, 32> scale_c(r.scale);

            for (uint32_t i = 0; i < r.skip; ++i) {
                scale_c.multiply(r.scale_factor);
            }

            r.re.set(re_min + std::get<1>(coords) * (re_max - re_min) / r.image_width);
            r.im.set(im_max - std::get<2>(coords) * (im_max - im_min) / r.image_height);
            r.re.multiply(scale_c);
            r.im.multiply(scale_c);
            r.re.add(re_c);
            r.im.add(im_c);

            int32_t count = r.count;
            int32_t skip = r.skip;

            if (std::get<0>(coords)) {
                count += 1;
                skip += 1;
            }
            else {
                count -= 1;
                skip -= 1;
            }

            r.count = count >= 1 ? count : 1;
            r.skip = skip >= 0 ? skip : 0;

            run_generate(r, [&](bool complete, image &i, std::string &suffix, uint32_t ix) {
                memcpy(img_buf, i.image_buffer(), sizeof(uint32_t) * r.image_width * r.image_height);
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
            queue.push(std::make_tuple(true, LOWORD(lParam), HIWORD(lParam)));
            break;
        }
        case WM_RBUTTONUP:
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push(std::make_tuple(false, LOWORD(lParam), HIWORD(lParam)));
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
        r.image_width,
        r.image_height,
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

bool run_generate(run_state &r, std::function<void(bool, image &, std::string &, uint32_t)> image_callback) {
    std::vector<std::tuple<double, double, double>> palette = create_palette(r);

    fixed_point<2, 32> scale(r.scale);

    for (uint32_t i = 0; i < r.count; ++i) {
        if (i >= r.skip) {
            fixed_point<2, 32> precision_test(r.step);
            precision_test.multiply(scale);
            uint32_t precision_bit = precision_test.get_fractional_significant_bit();

            std::cout << "[+] Generating Fractal #" << i << std::endl
                << "  [+] Precison Bit: " << precision_bit;

            if (precision_bit < 53) {
                run_step<0, 0>(r, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 2) {
                run_step<1, 2>(r, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 3) {
                run_step<1, 3>(r, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 4) {
                run_step<1, 4>(r, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 6) {
                run_step<1, 6>(r, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 8) {
                run_step<1, 8>(r, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 12) {
                run_step<1, 12>(r, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 16) {
                run_step<1, 16>(r, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 20) {
                run_step<1, 20>(r, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 24) {
                run_step<1, 24>(r, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 28) {
                run_step<1, 28>(r, i, palette, image_callback);
            }
            else if (precision_bit < 32 * 32) {
                run_step<1, 32>(r, i, palette, image_callback);
            }
            else {
                std::cout << " - UNSUPPORTED" << std::endl;
            }
        }

        scale.multiply(r.scale_factor);
    }

    std::cout << std::endl;

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<> bool run_step<0, 0>(run_state &r, uint32_t ix, std::vector<std::tuple<double, double, double>> &palette, std::function<void(bool, image &, std::string &, uint32_t)> image_callback) {
    fractal<double> f(r.image_width, r.image_height);
    if ((r.cuda_groups != 0) && (r.cuda_threads != 0)) {
        f.initialise(r.cuda_groups, r.cuda_threads);
    }

    f.colour(r.colour_method, palette);
    f.limits(r.escape_limit, r.escape_block);

    std::cout << ", Type: double" << std::endl;

    double re = static_cast<double>(r.re);
    double im = static_cast<double>(r.im);
    double scale = static_cast<double>(r.scale);
    double scale_factor = static_cast<double>(r.scale_factor);

    for (uint32_t i = 0; i < ix; ++i) {
        scale *= scale_factor;
    }
    f.specify(re, im, scale);

    if (r.julia) {
        double re_c = std::stod(r.re_c);
        double im_c = std::stod(r.im_c);
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
    if (f.generate(r.trial, [&]() { do_callback(false); })) {
        gen_timer.stop();
        gen_timer.print();
        do_callback(true);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F> bool run_step(run_state &r, uint32_t ix, std::vector<std::tuple<double, double, double>> &palette, std::function<void(bool, image &i, std::string &suffix, uint32_t ix)> image_callback) {
    fractal<fixed_point<I, F>> f(r.image_width, r.image_height);
    if ((r.cuda_groups != 0) && (r.cuda_threads != 0)) {
        f.initialise(r.cuda_groups, r.cuda_threads);
    }

    f.colour(r.colour_method, palette);
    f.limits(r.escape_limit, r.escape_block);

    std::cout << ", Type: fixed_point<" << I << ", " << F << ">" << std::endl;

    fixed_point<I, F> re(r.re);
    fixed_point<I, F> im(r.im);
    fixed_point<I, F> scale(r.scale);
    fixed_point<I, F> scale_factor(r.scale_factor);

    for (uint32_t i = 0; i < ix; ++i) {
        scale.multiply(scale_factor);
    }
    f.specify(re, im, scale);

    if (r.julia) {
        fixed_point<I, F> re_c(r.re_c);
        fixed_point<I, F> im_c(r.im_c);
        std::cout << "  [+] Julia: " << std::setprecision(12) << static_cast<double>(re_c) << ", " << std::setprecision(12) << static_cast<double>(im_c) << std::endl;
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
    if (f.generate(r.trial  , [&]() { do_callback(false); })) {
        gen_timer.stop();
        gen_timer.print();
        do_callback(true);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
