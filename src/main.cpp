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

    //uint64_t image_width = 640;
    //uint64_t image_height = 480;

    //uint64_t image_width = 1024;// 5120;
    //uint64_t image_height = 576;// 2880;

    uint64_t image_width = 2560;// 5120;
    uint64_t image_height = 1440;// 2880;

    //fractal<double> f(image_width, image_height);
    fractal<double> f(image_width, image_height);

    //fixed_point<2, 4> re("-1.74995768370609350360221450607069970727110579726252077930242837820286008082972804887218672784431700831100544507655659531379747541999999995");
    //fixed_point<2, 4> im("0.00000000000000000278793706563379402178294753790944364927085054500163081379043930650189386849765202169477470552201325772332454726999999995");

    //fixed_point<2, 4> re("0.013438870532012129028364919004019686867528573314565492885548699");
    //fixed_point<2, 4> im("0.655614218769465062251320027664617466691295975864786403994151735");

    //fixed_point<2, 4> re("0.27533764774673799358866712482462788156671406989542628591627436306743751013023030130967197535665363986058288420463735384997362663584446169657773339617717365950286959762265485804783047336923365261060963100721927003791989610861331863571141065592841226995797739723012374298589823921181693139824190379745910243872940870200527114596661654505");
    //fixed_point<2, 4> im("0.006759649405327850670181700456194929502189750234614304846357269137106731032582471677573582008294494705826194131450773107049670717146785957633119244225710271178867840504202402362491296317894835321064971518673775630252745135294700216673815790733343134984120108524001799351076577642283751627469315124883962453013093853471898311683555782404");

    //fixed_point<2, 4> re("0.39739358836206895");
    //fixed_point<2, 4> im("0.1334986193426724");

    //fixed_point<2, 4> scale("0.25");
    //fixed_point<2, 4> scale_factor(0.75);

    //f.limits(65536);
    f.limits(4 * 1048576);

    f.initialise(256, 768);
    f.specify(-1.2069164549178608, -0.14040002560478732, 1.25 * 0.000000096);
    f.specify(f.re() + 4.3 * (f.re(image_width) - f.re()) / 7.0, f.im() + 1.0 * (f.im(image_height) - f.im()) / 10.0, f.scale());

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