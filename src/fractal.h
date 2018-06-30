//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "fixed_point.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct kernel_block {
    uint64_t escape_;
    T re_;
    T im_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct kernel_params {
    kernel_params(const uint64_t &image_width,
            const uint64_t &image_height,
            const uint64_t &escape_block,
            const uint64_t &escape_limit,
            const T &re,
            const T &im,
            const T &scale) :
                image_width_(image_width), image_height_(image_height),
                escape_block_(escape_block), escape_limit_(escape_limit),
                image_chunk_(0), i_(0), re_(re), im_(im), scale_(scale) {};
    uint64_t image_width_;
    uint64_t image_height_;
    uint64_t escape_block_;
    uint64_t escape_limit_;
    uint64_t image_chunk_;
    uint64_t i_;
    T re_;
    T im_;
    T scale_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef _DEBUG
    constexpr uint64_t default_cuda_groups = 64;
    constexpr uint64_t default_cuda_threads = 256;
    constexpr uint64_t default_escape_block = 256;
    constexpr uint64_t default_escape_limit = 256;
    constexpr uint64_t default_image_width = 640;
    constexpr uint64_t default_image_height = 480;
#else
    constexpr uint64_t default_cuda_groups = 512;
    constexpr uint64_t default_cuda_threads = 1024;
    constexpr uint64_t default_escape_block = 16384;
    constexpr uint64_t default_escape_limit = 65536;
    constexpr uint64_t default_image_width = 1024;
    constexpr uint64_t default_image_height = 768;
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
class fractal {
public:
    fractal() : fractal(default_image_width, default_image_height, default_escape_block, default_escape_limit, default_cuda_groups, default_cuda_threads) {
    }

    fractal(const uint64_t image_width, const uint64_t image_height) : fractal(image_width, image_height, default_escape_block, default_escape_limit, default_cuda_groups, default_cuda_threads) {
    }

    fractal(const uint64_t image_width, const uint64_t image_height, uint64_t escape_block, uint64_t escape_limit) : fractal(image_width, image_height, escape_block, escape_limit, default_cuda_groups, default_cuda_threads) {
    }

    fractal(const uint64_t image_width, const uint64_t image_height, uint64_t escape_block, uint64_t escape_limit, uint64_t cuda_groups, uint64_t cuda_threads) :
            image_(nullptr), block_double_(nullptr), block_fixed_point_(nullptr), block_image_(nullptr),
            image_width_(image_width), image_height_(image_height),
            cuda_groups_(cuda_groups), cuda_threads_(cuda_threads),
            escape_block_(escape_block), escape_limit_(escape_limit) {
        initialise();
        specify(0.0, 0.0, 1.0);
    }

    ~fractal() {
        uninitialise();
        if (image_ != nullptr) {
            delete[] image_;
        }
    }

    bool initialise() {
        return initialise(cuda_groups_, cuda_threads_);
    }

    bool initialise(uint64_t cuda_groups, uint64_t cuda_threads);
    void uninitialise();

    void limits(uint64_t escape_block, uint64_t escape_limit) {
        escape_block_ = escape_block;
        escape_limit_ = escape_limit;
    }

    void resize(const uint64_t image_width, const uint64_t image_height);

    void specify(const double re, const double im, const double scale);
    void specify(const fixed_point<I, F> &re, const fixed_point<I, F> &im, const fixed_point<I, F> &scale);

    bool generate(bool use_fixed_point = false);

    const uint64_t image_width() {
        return image_width_;
    }

    const uint64_t image_height() {
        return image_height_;
    }

    const uint64_t image_size() {
        return image_width_ * image_height_ * sizeof(uint32_t);
    }

    const uint32_t *image(bool release = false) {
        uint32_t *image_buffer = image_;
        if (release) {
            image_ = nullptr;
        }
        return image_buffer;
    }

private:
    uint64_t cuda_groups_;
    uint64_t cuda_threads_;
    uint64_t escape_block_;
    uint64_t escape_limit_;
    uint64_t image_width_;
    uint64_t image_height_;

    double re_d_;
    fixed_point<I, F> re_fp_;
    double im_d_;
    fixed_point<I, F> im_fp_;
    double scale_d_;
    fixed_point<I, F> scale_fp_;

    uint32_t *image_;
    kernel_block<double> *block_double_;
    kernel_block<fixed_point<I, F>> *block_fixed_point_;
    uint32_t *block_image_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
