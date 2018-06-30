//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "fixed_point.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct s_d {
    uint64_t escape;
    double re_;
    double im_;

    double __host__ __device__ re() { return re_; }
    double __host__ __device__ im() { return im_; }
};

struct s_d_param {
    uint64_t image_width_;
    uint64_t image_height_;
    double re_;
    double im_;
    double scale_;
};

template<uint32_t I, uint32_t F>
struct s_fp {
    uint64_t escape;
    fixed_point<I, F> re_;
    fixed_point<I, F> im_;

    double __host__ __device__ re() { return re_.get_double(); }
    double __host__ __device__ im() { return im_.get_double(); }
};

template<uint32_t I, uint32_t F>
struct s_fp_param {
    uint64_t image_width_;
    uint64_t image_height_;
    fixed_point<I, F> re_;
    fixed_point<I, F> im_;
    fixed_point<I, F> scale_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
class fractal {
public:
    fractal() : fractal(640, 480, 64, 1024) {
    }

    fractal(const uint64_t image_width, const uint64_t image_height) : fractal(image_width, image_height, 64, 1024) {
    }

    fractal(const uint64_t image_width, const uint64_t image_height, uint64_t cuda_groups, uint64_t cuda_threads) :
            image_(nullptr), device_chunk_d_(nullptr), device_chunk_fp_(nullptr), device_chunk_image_(nullptr),
            cuda_groups_(cuda_groups), cuda_threads_(cuda_threads) {
        resize(image_width, image_height);
        specify(0.0, 0.0, 1.0);
        initialise();
    }

    ~fractal() {
        uninitialise();
        if (image_ != nullptr) {
            delete[] image_;
        }
    }

    bool initialise(); 
    void uninitialise();

    void resize(const uint64_t image_width, const uint64_t image_height);
    void specify(const double re, const double im, const double scale);
    void specify(const fixed_point<I, F> &re, const fixed_point<I, F> &im, const fixed_point<I, F> &scale);

    bool generate(bool fixed_point = false);

    const uint64_t image_width() {
        return image_width_;
    }

    const uint64_t image_height() {
        return image_height_;
    }

    const uint64_t image_size() {
        return image_width_ * image_height_ * sizeof(uint32_t);
    }

    const uint32_t *image() {
        return image_;
    }

private:
    uint64_t cuda_groups_;
    uint64_t cuda_threads_;

    uint64_t image_width_;
    uint64_t image_height_;

    double re_d_;
    fixed_point<I, F> re_fp_;
    double im_d_;
    fixed_point<I, F> im_fp_;
    double scale_d_;
    fixed_point<I, F> scale_fp_;

    uint32_t *image_;
    s_d *device_chunk_d_;
    s_fp<I, F> *device_chunk_fp_;
    uint32_t *device_chunk_image_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
