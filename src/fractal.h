//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Fractal
// (C)2018-20 Liam Kirton <liam@int3.ws>
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "fixed_point.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Constants
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr double kImMin = 3.0;
constexpr double kImMax = -3.0;
constexpr double kReMin = -3.0;
constexpr double kReMax = 3.0;

#ifdef _DEBUG
constexpr uint32_t kDefaultCudaGroups = 128;
constexpr uint32_t kDefaultCudaThreads = 128;
constexpr uint32_t kDefaultEscapeBlock = 1024;
constexpr uint32_t kDefaultEscapeLimit = 2048;
constexpr uint32_t kDefaultImageWidth = 320;
constexpr uint32_t kDefaultImageHeight = 320;
constexpr size_t kKernelChunkPerturbationReferenceBlock = 256;
#else
constexpr uint32_t kDefaultCudaGroups = 256;
constexpr uint32_t kDefaultCudaThreads = 1024;
constexpr uint32_t kDefaultEscapeBlock = 65536;
constexpr uint32_t kDefaultEscapeLimit = 1048576;
constexpr uint32_t kDefaultImageWidth = 1024;
constexpr uint32_t kDefaultImageHeight = 768;
constexpr size_t kKernelChunkPerturbationReferenceBlock = 4096;
#endif

constexpr double kDefaultEscapeRadius = 16.0;
constexpr double kDefaultEscapeRadiusSquared = kDefaultEscapeRadius * kDefaultEscapeRadius;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Structs
//
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

struct kernel_chunk_perturbation {
    uint64_t escape_;
    uint32_t ref_;
    union {
        double re_delta_0_;
        double re_;
    };
    union {
        double im_delta_0_;
        double im_;
    };
    double re_delta_n_;
    double im_delta_n_;
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
#ifdef _DEBUG
    double re_d_[kKernelChunkPerturbationReferenceBlock];
    double im_d_[kKernelChunkPerturbationReferenceBlock];
#else
    double re_d_[kKernelChunkPerturbationReferenceBlock];
    double im_d_[kKernelChunkPerturbationReferenceBlock];
#endif
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
            const T &im_c,
            const uint32_t grid_x,
            const uint32_t grid_y) :
                image_width_(image_width), image_height_(image_height),
                image_viewport_((image_width > image_height) ? image_width : image_height),
                escape_block_(escape_block), escape_limit_(escape_limit), colour_method_(colour_method),
                escape_i_(0), escape_range_min_(0), escape_range_max_(escape_limit),
                chunk_offset_(0),
                re_(re), im_(im), scale_(scale), re_c_(re_c), im_c_(im_c),
                grid_x_(grid_x), grid_y_(grid_y),
                have_ref_(false), re_ref_(0), im_ref_(0),
                set_hue_(0.0), set_sat_(0.0), set_val_(0.0), palette_device_(nullptr), palette_count_(0) {};

    uint32_t image_width_;
    uint32_t image_height_;
    uint32_t image_viewport_;

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

    uint32_t grid_x_;
    uint32_t grid_y_;

    bool have_ref_;
    T re_ref_;
    T im_ref_;

    double set_hue_;
    double set_sat_;
    double set_val_;

    double *palette_device_;
    uint32_t palette_count_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct kernel_reduce_params {
    uint64_t escape_reduce_;
    uint64_t escape_reduce_min_;
    uint64_t escape_reduce_max_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Fractal Class
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class fractal {
public:
    fractal() : fractal(kDefaultImageWidth, kDefaultImageHeight,
            kDefaultEscapeBlock, kDefaultEscapeLimit, kDefaultCudaGroups, kDefaultCudaThreads) {
    }

    fractal(const uint32_t image_width, const uint32_t image_height) :
            fractal(image_width, image_height, kDefaultEscapeBlock, kDefaultEscapeLimit,
                kDefaultCudaGroups, kDefaultCudaThreads) {
    }

    fractal(const uint32_t image_width, const uint32_t image_height, uint64_t escape_block, uint64_t escape_limit) :
            fractal(image_width, image_height, escape_block, escape_limit, kDefaultCudaGroups, kDefaultCudaThreads) {
    }

    fractal(const uint32_t image_width, const uint32_t image_height, uint64_t escape_block,
                uint64_t escape_limit, uint32_t cuda_groups, uint32_t cuda_threads) :
            image_(nullptr),
            image_width_(image_width), image_height_(image_height),
            image_viewport_((image_width > image_height) ? image_width : image_height),
            cuda_groups_(cuda_groups), cuda_threads_(cuda_threads),
            escape_block_(escape_block), escape_limit_(escape_limit),
            colour_method_(0),
            perturbation_(false), grid_x_(32), grid_y_(32), grid_levels_(1), grid_steps_(1),
            julia_(false), re_c_(0), im_c_(0) {
        initialise();
        specify(0.0, 0.0, 1.0, false);
    }

    ~fractal() {
        uninitialise();
    }

    bool initialise() {
        return initialise(cuda_groups_, cuda_threads_);
    }

    bool initialise(const uint32_t cuda_groups, const uint32_t cuda_threads);
    void uninitialise();

    void limits(const uint64_t escape_limit, const uint64_t escape_block = kDefaultEscapeBlock) {
        escape_block_ = escape_block;
        escape_limit_ = escape_limit;

        if (escape_block_ >= escape_limit_) {
            escape_block_ = escape_limit_ / 2;
        }
    }

    void colour(const uint8_t method, std::vector<std::tuple<double, double, double>> &palette) {
        colour_method_ = method;
        palette_ = palette;
    }

    void grid(const uint32_t x, const uint32_t y, const uint32_t levels, const uint32_t steps) {
        grid_x_ = x;
        grid_y_ = y;
        grid_levels_ = levels;
        grid_steps_ = steps;
    }

    void resize(const uint32_t image_width, const uint32_t image_height);
    void specify(const T &re, const T &im, const T &scale, bool perturbation);
    void specify_julia(const T &re_c, const T &im_c);

    bool generate(std::function<bool()> callback = []() {});

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
    bool generate(kernel_params<T> &params, std::function<bool()> callback);
    bool generate_perturbation_reference(kernel_params<T> &params, std::function<bool()> callback);

private:
    uint32_t cuda_groups_;
    uint32_t cuda_threads_;
    uint64_t escape_block_;
    uint64_t escape_limit_;

    uint32_t image_width_;
    uint32_t image_height_;
    uint32_t image_viewport_;

    uint8_t colour_method_;
    std::vector<std::tuple<double, double, double>> palette_;

    T re_;
    T im_;
    T scale_;

    bool perturbation_;
    uint32_t grid_x_;
    uint32_t grid_y_;
    uint32_t grid_levels_;
    uint32_t grid_steps_;

    bool julia_;
    T re_c_;
    T im_c_;

    uint32_t *image_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
