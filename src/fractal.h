//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "fixed_point.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct kernel_chunk {
    uint64_t escape_;
    T re_c_;
    T im_c_;
    T re_;
    T im_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct kernel_chunk_perturbation_reference {
    uint64_t escape_;
    uint64_t index_;
    T re_c_;
    T im_c_;
    T re_;
    T im_;
    double re_d_[2048];
    double im_d_[2048];
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct kernel_chunk_perturbation {
    uint32_t rid_;
    uint64_t escape_;
    double re_delta_0;
    double im_delta_0;
    double re_delta_n;
    double im_delta_n;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct kernel_params {
    kernel_params(const uint32_t &image_width,
            const uint32_t &image_height,
            const uint64_t &escape_block,
            const uint64_t &escape_limit,
            const uint8_t &colour_method,
            const T &re,
            const T &im,
            const T &scale,
            const T &re_c,
            const T &im_c) :
                image_width_(image_width), image_height_(image_height),
                escape_block_(escape_block), escape_limit_(escape_limit), colour_method_(colour_method),
                escape_i_(0), escape_range_min_(0), escape_range_max_(escape_limit),
                chunk_offset_(0),
                re_(re), im_(im), scale_(scale), re_c_(re_c), im_c_(im_c),
                set_hue_(0.0), set_sat_(0.0), set_val_(0.0), palette_(nullptr), palette_count_(0) {};
    uint32_t image_width_;
    uint32_t image_height_;

    uint64_t escape_block_;
    uint64_t escape_limit_;
    uint64_t escape_i_;
    uint64_t escape_range_min_;
    uint64_t escape_range_max_;

    uint8_t colour_method_;
    uint32_t chunk_offset_;

    T re_;
    T im_;
    T scale_;
    T re_c_;
    T im_c_;

    double set_hue_;
    double set_sat_;
    double set_val_;

    double *palette_;
    uint32_t palette_count_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct kernel_reduce_params {
    uint64_t escape_reduce_;
    uint64_t escape_reduce_min_;
    uint64_t escape_reduce_max_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr double im_min = 3.0;
constexpr double im_max = -3.0;
constexpr double re_min = -3.0;
constexpr double re_max = 3.0;

#ifdef _DEBUG
    constexpr uint32_t default_cuda_groups = 128;
    constexpr uint32_t default_cuda_threads = 128;

    constexpr uint32_t default_escape_block = 1024;
    constexpr uint32_t default_escape_limit = 2048;
    constexpr double default_escape_radius = 16.0;

    constexpr uint32_t default_image_width = 320;
    constexpr uint32_t default_image_height = 320;
#else
    constexpr uint32_t default_cuda_groups = 256;
    constexpr uint32_t default_cuda_threads = 1024;

    constexpr uint32_t default_escape_block = 65536;
    constexpr uint32_t default_escape_limit = 1048576;
    constexpr double default_escape_radius = 16.0;

    constexpr uint32_t default_image_width = 1024;
    constexpr uint32_t default_image_height = 768;
#endif

constexpr double default_escape_radius_square = default_escape_radius * default_escape_radius;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class fractal {
public:
    fractal() : fractal(default_image_width, default_image_height, default_escape_block, default_escape_limit, default_cuda_groups, default_cuda_threads) {
    }

    fractal(const uint32_t image_width, const uint32_t image_height) : fractal(image_width, image_height, default_escape_block, default_escape_limit, default_cuda_groups, default_cuda_threads) {
    }

    fractal(const uint32_t image_width, const uint32_t image_height, uint64_t escape_block, uint64_t escape_limit) : fractal(image_width, image_height, escape_block, escape_limit, default_cuda_groups, default_cuda_threads) {
    }

    fractal(const uint32_t image_width, const uint32_t image_height, uint64_t escape_block, uint64_t escape_limit, uint32_t cuda_groups, uint32_t cuda_threads) :
            chunk_buffer_(nullptr), chunk_buffer_device_(nullptr), image_(nullptr), image_device_(nullptr),
            image_width_(image_width), image_height_(image_height),
            trial_image_width_(cuda_threads), trial_image_height_(cuda_groups),
            cuda_groups_(cuda_groups), cuda_threads_(cuda_threads),
            escape_block_(escape_block), escape_limit_(escape_limit),
            colour_method_(0),
            julia_(false), re_c_(0), im_c_(0) {
        initialise();
        specify(0.0, 0.0, 1.0);
    }

    ~fractal() {
        uninitialise();
    }

    bool initialise() {
        return initialise(cuda_groups_, cuda_threads_);
    }

    bool initialise(uint32_t cuda_groups, uint32_t cuda_threads);
    void uninitialise();

    void limits(uint64_t escape_limit, uint64_t escape_block = default_escape_block) {
        if (escape_block > escape_limit) {
            escape_block = escape_limit;
        }
        escape_block_ = escape_block;
        escape_limit_ = escape_limit;
    }

    void colour(const uint8_t method, std::vector<std::tuple<double, double, double>> &palette) {
        colour_method_ = method;
        palette_ = palette;
    }

    void resize(const uint32_t image_width, const uint32_t image_height);
    void specify(const T &re, const T &im, const T &scale);
    void specify_julia(const T &re_c, const T &im_c);

    bool generate(bool trial, bool interactive, std::function<bool(bool)> callback = []() {});

    uint32_t image_width() {
        return image_width_;
    }

    uint32_t image_height() {
        return image_height_;
    }

    uint32_t image_size() {
        return image_width_ * image_height_ * sizeof(uint32_t);
    }

    uint32_t *image() {
        return image_;
    }

    T re() {
        return re_;
    }

    T im() {
        return im_;
    }

    T scale() {
        return scale_;
    }

private:
    bool dev_perturbation(kernel_params<T> &params, bool interactive, std::function<bool(bool)> callback);
    bool generate(kernel_params<T> &params, bool interactive, std::function<bool(bool)> callback);
    void pixel_to_coord(uint32_t x, uint32_t image_width, T &re, uint32_t y, uint32_t image_height, T &im);
    bool process_trial(kernel_params<T> &params_trial, kernel_params<T> &params, kernel_chunk<T> *preview);

private:
    uint32_t cuda_groups_;
    uint32_t cuda_threads_;
    uint64_t escape_block_;
    uint64_t escape_limit_;

    uint32_t image_width_;
    uint32_t image_height_;
    uint32_t trial_image_width_;
    uint32_t trial_image_height_;

    uint8_t colour_method_;
    std::vector<std::tuple<double, double, double>> palette_;

    T re_;
    T im_;
    T scale_;

    bool julia_;
    T re_c_;
    T im_c_;

    kernel_chunk<T> *chunk_buffer_;
    kernel_chunk<T> *chunk_buffer_device_;

    uint32_t *image_;
    uint32_t *image_device_; 
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
