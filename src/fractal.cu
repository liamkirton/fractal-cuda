////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "fractal.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template class fractal<1, 2>;
template class fractal<2, 2>;
template class fractal<2, 4>;
template class fractal<2, 6>;
template class fractal<2, 8>;
template class fractal<2, 16>;
template class fractal<4, 4>;
template class fractal<4, 6>;
template class fractal<4, 8>;
template class fractal<4, 16>;

__global__ void kernel_mandelbrot(kernel_block<double> *blocks, kernel_params<double> *params);

template<uint32_t I, uint32_t F>
__global__ void kernel_mandelbrot(kernel_block<fixed_point<I, F>> *blocks, kernel_params<fixed_point<I, F>> *params);

template<typename T>
__global__ void kernel_colour(kernel_block<T> *blocks, kernel_params<T> *params, uint32_t *block_image);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host Methods
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
bool fractal<I, F>::initialise(uint64_t cuda_groups, uint64_t cuda_threads) {
    cuda_groups_ = cuda_groups;
    cuda_threads_ = cuda_threads;
    uninitialise();
    cudaError_t cuda_status;
    if ((cuda_status = cudaSetDevice(0)) != cudaSuccess) {
        std::wcout << L"ERROR: cudaSetDevice() Failed. [" << cuda_status << L"]" << std::endl << std::endl;
        return false;
    }
    cudaMalloc(&block_image_, cuda_groups_ * cuda_threads_ * sizeof(uint32_t));
    resize(image_width_, image_height_);
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
void fractal<I, F>::uninitialise() {
    cudaDeviceReset();
    if (block_double_ != nullptr) {
        cudaFree(block_double_);
        block_double_ = nullptr;
    }
    if (block_fixed_point_ != nullptr) {
        cudaFree(block_fixed_point_);
        block_fixed_point_ = nullptr;
    }
    if (block_image_ != nullptr) {
        cudaFree(block_image_);
        block_image_ = nullptr;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
void fractal<I, F>::resize(const uint64_t image_width, const uint64_t image_height) {
    if ((image_width != image_width_) || (image_height != image_height_)) {
        image_width_ = image_width;
        image_height_ = image_height;
        if (image_ != nullptr) {
            delete[] image_;
            image_ = nullptr;
        }
    }
    if (image_ == nullptr) {
        image_ = new uint32_t[image_width_ * image_height_];
    }
    memset(image_, 0, sizeof(uint32_t) * image_width_ * image_height_);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
void fractal<I, F>::specify(double re, double im, double scale) {
    re_d_ = re; re_fp_.set(re);
    im_d_ = im; im_fp_.set(im);
    scale_d_ = 1.0 / scale;
    scale_fp_.set(1.0 / scale);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
void fractal<I, F>::specify(const fixed_point<I, F> &re, const fixed_point<I, F> &im, const fixed_point<I, F> &scale) {
    re_d_ = re.get_double(); re_fp_.set(re);
    im_d_ = im.get_double(); im_fp_.set(im);
    scale_d_ = 1.0 / scale.get_double();
    scale_fp_.set(scale);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
bool fractal<I, F>::generate(bool use_fixed_point) {
    if (image_ == nullptr) {
        resize(image_width_, image_height_);
    }

    kernel_params<fixed_point<I, F>> *params_fixed_point{ nullptr }; 
    kernel_params<double> *params_double{ nullptr };

    if (use_fixed_point) {
        if (block_fixed_point_ == nullptr) {
            cudaMalloc(&block_fixed_point_, cuda_groups_ * cuda_threads_ * sizeof(kernel_block<fixed_point<I, F>>));
            if (block_fixed_point_ == nullptr) {
                return false;
            }
        }
        kernel_params<fixed_point<I, F>> params(image_width_, image_height_, escape_block_, escape_limit_,
            re_fp_, im_fp_, scale_fp_);
        cudaMalloc(&params_fixed_point, sizeof(kernel_params<fixed_point<I, F>>));
        cudaMemcpy(params_fixed_point, &params, sizeof(params), cudaMemcpyHostToDevice);
    }
    else {
        if (block_double_ == nullptr) {
            cudaMalloc(&block_double_, cuda_groups_ * cuda_threads_ * sizeof(kernel_params<double>));
            if (block_double_ == nullptr) {
                return false;
            }
        }
        kernel_params<double> params(image_width_, image_height_, escape_block_, escape_limit_,
            re_d_, im_d_, scale_d_);
        cudaMalloc(&params_double, sizeof(kernel_params<double>));
        cudaMemcpy(params_double, &params, sizeof(params), cudaMemcpyHostToDevice);
    }

    std::wcout << L"[+] Chunks: "
        << 1 + image_width_ * image_height_ / (cuda_groups_ * cuda_threads_)
        << L" " << std::flush;

    cudaError_t cudaError;

    for (uint64_t image_chunk = 0; image_chunk < (image_width_ * image_height_); image_chunk += (cuda_groups_ * cuda_threads_)) {
        uint64_t chunk_size = std::min((image_width_ * image_height_) - image_chunk, cuda_groups_ * cuda_threads_);
        uint64_t chunk_groups = chunk_size / cuda_threads_;
        if (use_fixed_point) {
            cudaMemset(block_fixed_point_, 0, cuda_groups_ * cuda_threads_ * sizeof(kernel_block<fixed_point<I, F>>));
        }
        else {
            cudaMemset(block_double_, 0, cuda_groups_ * cuda_threads_ * sizeof(kernel_block<double>));
        }

        std::wcout << L"+" << std::flush;

        for (uint32_t i = 0; i < (escape_limit_ / escape_block_); ++i) {
            if (use_fixed_point) {
                kernel_mandelbrot<I, F><<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(cuda_threads_)>>>(block_fixed_point_, params_fixed_point);
            }
            else {
                kernel_mandelbrot<<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(cuda_threads_)>>>(block_double_, params_double);
            }
            if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                std::wcout << std::endl << "[!] cudaDeviceSynchronize(): cudaError: " << cudaError << std::endl;
                break;
            }
        }

        if (cudaError != cudaSuccess) {
            break;
        }

        cudaMemset(block_image_, 0, cuda_groups_ * cuda_threads_ * sizeof(uint32_t));
        if (use_fixed_point) {
            kernel_colour<fixed_point<I, F>><<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(cuda_threads_)>>>(block_fixed_point_, params_fixed_point, block_image_);
        }
        else {
            kernel_colour<double><<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(cuda_threads_)>>>(block_double_, params_double, block_image_);
        }

        if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
            std::wcout << std::endl << "[!] cudaDeviceSynchronize(): cudaError: " << cudaError << std::endl;
            break;
        }

        cudaMemcpy(&image_[image_chunk], block_image_, chunk_groups * cuda_threads_ * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    std::wcout << std::endl;
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Kernels
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_mandelbrot(kernel_block<double> *blocks, kernel_params<double> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int pixel_x = (params->image_chunk_ + tid) % params->image_width_;
    const int pixel_y = (params->image_chunk_ + tid) / params->image_width_;

    const double re_c = params->re_ + (-2.0 + pixel_x * 3.0 / params->image_width_) * params->scale_;
    const double im_c = params->im_ + (1.0 - pixel_y * 2.0 / params->image_height_) * params->scale_;

    kernel_block<double> *block = &blocks[tid];
    uint64_t escape = block->escape_;
    double re = block->re_;
    double im = block->im_;
    double abs = 0.0;

    if (params->i_ == 0) {
        escape = params->escape_limit_;
        re = re_c;
        im = im_c;
    }

    if (escape == params->escape_limit_) {
        for (uint64_t i = 0; i < params->escape_block_; ++i) {
            double re_z_i = re;
            re = (re * re) - (im * im) + re_c;
            im = (2.0 * re_z_i * im) + im_c;
            abs = re * re + im * im;
            if (abs > 4.0) {
                escape = i + params->i_ * params->escape_block_;
                break;
            }
        }
    }

    block->escape_ = escape;
    block->re_ = re;
    block->im_ = im;
}

template<uint32_t I, uint32_t F>
__global__ void kernel_mandelbrot(kernel_block<fixed_point<I, F>> *blocks, kernel_params<fixed_point<I, F>> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int pixel_x = (params->image_chunk_ + tid) % params->image_width_;
    const int pixel_y = (params->image_chunk_ + tid) / params->image_width_;

    fixed_point<I, F> re_c(-2.0 + pixel_x * 3.0 / params->image_width_);
    re_c.multiply(params->scale_);
    re_c.add(params->re_);
    fixed_point<I, F> im_c(1.0 - pixel_y * 2.0 / params->image_height_);
    im_c.multiply(params->scale_);
    im_c.add(params->im_);

    kernel_block<fixed_point<I, F>> *block = &blocks[tid];
    uint64_t escape = block->escape_;

    fixed_point<I, F> re_z(block->re_);
    fixed_point<I, F> im_z(block->im_);

    if (params->i_ == 0) {
        escape = params->escape_limit_;
        re_z.set(re_c);
        im_z.set(im_c);
    }

    fixed_point<I, F> re_prod(re_z);
    re_prod.multiply(re_z);

    fixed_point<I, F> im_prod(im_z);
    im_prod.multiply(im_z);

    fixed_point<I, F> re_imed;
    fixed_point<I, F> im_imed;

    if (escape == params->escape_limit_) {
        for (uint32_t i = 0; i < params->escape_block_; ++i) {
            re_imed.set(im_prod);
            re_imed.negate();
            re_imed.add(re_prod);
            re_imed.add(re_c);

            im_imed.set(2);
            im_imed.multiply(re_z);
            im_imed.multiply(im_z);
            im_imed.add(im_c);

            re_z.set(re_imed);
            im_z.set(im_imed);

            re_prod.set(re_z);
            re_prod.multiply(re_z);

            im_prod.set(im_z);
            im_prod.multiply(im_z);

            if (re_prod.get_integer() + im_prod.get_integer() > 4) {
                escape = i + params->i_ * params->escape_block_;
                break;
            }
        }
    }

    block->escape_ = escape;
    block->re_.set(re_z);
    block->im_.set(im_z);
}

template<typename T>
__global__ void kernel_colour(kernel_block<T> *blocks, kernel_params<T> *params, uint32_t *block_image) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    kernel_block<T> *block = &blocks[tid];
    uint32_t escape = block->escape_;
    double re_z = static_cast<double>(block->re_);
    double im_z = static_cast<double>(block->im_);
    double abs_z = sqrtf(re_z * re_z + im_z * im_z);

    double hue = 360.0 * log(1.0 * escape) / log(1.0 * params->escape_limit_) + 1.0 - (log(log(abs_z)) / log(2.0)); 
    //double hue = 360.0 * (log(1.0 * escape) - log(log(abs_z))) / (log(1.0 * escape_limit) + log(2.0));
    double sat = 0.85;
    double val = 1.0;

    hue = fmod(hue, 360.0);
    hue /= 60;

    double hue_fract = hue - floor(hue);
    double p = val * (1.0 - sat);
    double q = val * (1.0 - sat * hue_fract);
    double t = val * (1.0 - sat * (1.0 - hue_fract));

    double r = 0;
    double g = 0;
    double b = 0;

    if (escape < params->escape_limit_) {
        switch ((static_cast<unsigned char>(floor(hue)) + 3) % 6) {
        case 0:
            r = val; g = t; b = p;
            break;
        case 1:
            r = q; g = val; b = p;
            break;
        case 2:
            r = p; g = val; b = t;
            break;
        case 3:
            r = p; g = q; b = val;
            break;
        case 4:
            r = t; g = p; b = val;
            break;
        case 5:
            r = val; g = p; b = q;
            break;
        default:
            break;
        }
        r = floor(r * 255); g = floor(g * 255); b = floor(b * 255);
    }

    block_image[tid] =
        (static_cast<unsigned char>(r)) |
        (static_cast<unsigned char>(g) << 8) |
        (static_cast<unsigned char>(b) << 16) |
        (255 << 24);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
