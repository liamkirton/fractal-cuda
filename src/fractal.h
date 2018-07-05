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
    T abs_;
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
                image_chunk_(0), escape_i_(0), escape_range_min_(0), escape_range_max_(0),
                re_(re), im_(im), scale_(scale) {};
    uint64_t image_width_;
    uint64_t image_height_;

    uint64_t escape_block_;
    uint64_t escape_limit_;
    uint64_t escape_i_;
    uint64_t escape_range_min_;
    uint64_t escape_range_max_;

    uint64_t image_chunk_;

    T re_;
    T im_;
    T scale_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//constexpr double im_min = 1.525;
//constexpr double im_max = -1.525;
//constexpr double re_min = (im_max - im_min);
//constexpr double re_max = -(im_max - im_min) / 2.0;

constexpr double static_scale = 1.5;
constexpr double im_min = 1.0 * static_scale;
constexpr double im_max = -1.0 * static_scale;
constexpr double re_min = -2.0 * static_scale;
constexpr double re_max = 1.0 * static_scale;

#ifdef _DEBUG
    constexpr uint64_t default_cuda_groups = 64;
    constexpr uint64_t default_cuda_threads = 256;
    
    constexpr uint64_t default_escape_block = 256;
    constexpr uint64_t default_escape_limit = 256;

    constexpr uint64_t default_image_width = 640;
    constexpr uint64_t default_image_height = 480;
    constexpr uint64_t preview_image_width = 32;
    constexpr uint64_t preview_image_height = 32;
#else
    constexpr uint64_t default_cuda_groups = 256;
    constexpr uint64_t default_cuda_threads = 896;

    constexpr uint64_t default_escape_block = 16384;
    constexpr uint64_t default_escape_limit = 1048576;

    constexpr uint64_t default_image_width = 1024;
    constexpr uint64_t default_image_height = 768;
    constexpr uint64_t default_preview_image_width = default_cuda_threads;
    constexpr uint64_t default_preview_image_height = default_cuda_groups;
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class fractal {
public:
    fractal() : fractal(default_image_width, default_image_height, default_escape_block, default_escape_limit, default_cuda_groups, default_cuda_threads) {
    }

    fractal(const uint64_t image_width, const uint64_t image_height) : fractal(image_width, image_height, default_escape_block, default_escape_limit, default_cuda_groups, default_cuda_threads) {
    }

    fractal(const uint64_t image_width, const uint64_t image_height, uint64_t escape_block, uint64_t escape_limit) : fractal(image_width, image_height, escape_block, escape_limit, default_cuda_groups, default_cuda_threads) {
    }

    fractal(const uint64_t image_width, const uint64_t image_height, uint64_t escape_block, uint64_t escape_limit, uint64_t cuda_groups, uint64_t cuda_threads) :
            image_(nullptr),
            image_width_(image_width), image_height_(image_height),
            preview_image_width_(default_preview_image_width), preview_image_height_(default_preview_image_height),
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

    void limits(uint64_t escape_limit, uint64_t escape_block = default_escape_block) {
        if (escape_block > escape_limit) {
            escape_block = escape_limit;
        }
        escape_block_ = escape_block;
        escape_limit_ = escape_limit;
    }

    void resize(const uint64_t image_width, const uint64_t image_height);
    void specify(const T &re, const T &im, const T &scale);

    bool generate();

    uint64_t image_width() {
        return image_width_;
    }

    uint64_t image_height() {
        return image_height_;
    }

    uint64_t image_size() {
        return image_width_ * image_height_ * sizeof(uint32_t);
    }

    const uint32_t *image(bool release = false) {
        uint32_t *image_buffer = image_;
        if (release) {
            image_ = nullptr;
        }
        return image_buffer;
    }

    double re() {
        return re(image_width_ / 2);
    }

    double re(uint64_t x) {
        if (x > image_width_) {
            x = 0;
        }
        return static_cast<double>(re_) + (re_min + x * (re_max - re_min) / image_width_) * static_cast<double>(scale_);
    }

    double im() {
        return im(image_height_ / 2);
    }

    double im(uint64_t y) {
        if (y > image_height_) {
            y = 0;
        }
        return static_cast<double>(im_) + (im_max - y * (im_max - im_min) / image_height_) * static_cast<double>(scale_);
    }

    double scale() {
        return static_cast<double>(scale_);
    }

private:
    bool generate(kernel_params<T> &params, kernel_block<T> *block, bool colour);

private:
    uint64_t cuda_groups_;
    uint64_t cuda_threads_;
    uint64_t escape_block_;
    uint64_t escape_limit_;

    uint64_t image_width_;
    uint64_t image_height_;
    uint64_t preview_image_width_;
    uint64_t preview_image_height_;

    T re_;
    T im_;
    T scale_;

    uint32_t *image_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
