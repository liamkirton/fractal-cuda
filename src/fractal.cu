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

template class fractal<double>;
template class fractal<fixed_point<1, 2>>;
template class fractal<fixed_point<2, 2>>;
template class fractal<fixed_point<2, 4>>;
template class fractal<fixed_point<2, 8>>;
template class fractal<fixed_point<2, 16>>;
template class fractal<fixed_point<2, 32>>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

template<typename T>
bool fractal<T>::initialise(uint64_t cuda_groups, uint64_t cuda_threads) {
    if (cuda_groups * cuda_threads < preview_image_width * preview_image_height) {
        cuda_groups = preview_image_height;
        cuda_threads = preview_image_width;
    }

    cuda_groups_ = cuda_groups;
    cuda_threads_ = cuda_threads;

    uninitialise();

    cudaError_t cuda_status;
    if ((cuda_status = cudaSetDevice(0)) != cudaSuccess) {
        std::wcout << L"ERROR: cudaSetDevice() Failed. [" << cuda_status << L"]" << std::endl << std::endl;
        return false;
    }

    resize(image_width_, image_height_);
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void fractal<T>::uninitialise() {
    cudaDeviceReset();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void fractal<T>::resize(const uint64_t image_width, const uint64_t image_height) {
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

template<>
void fractal<double>::specify(const double &re, const double &im, const double &scale) {
    re_ = re;
    im_ = im;
    scale_ = scale;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void fractal<T>::specify(const T &re, const T &im, const T &scale) {
    re_.set(re);
    im_.set(im);
    scale_.set(scale);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
bool fractal<T>::generate() {
    resize(image_width_, image_height_);

    kernel_params<T> params_preview(preview_image_width, preview_image_height, escape_block_, escape_limit_, re_, im_, scale_);
    kernel_params<T> params(image_width_, image_height_, escape_block_, escape_limit_, re_, im_, scale_);

    kernel_block<T> *block_device;
    cudaMalloc(&block_device, cuda_groups_ * cuda_threads_ * sizeof(kernel_block<T>));
    if (block_device == nullptr) {
        return false;
    }

    if (!generate(params_preview, block_device, false)) {
        return false;
    }

    kernel_block<T> *preview = new kernel_block<T>[preview_image_width * preview_image_height];
    cudaMemcpy(preview, block_device, preview_image_width * preview_image_height * sizeof(kernel_block<T>), cudaMemcpyDeviceToHost);

    params.escape_range_min_ = 0xffffffffffffffff;
    params.escape_range_max_ = 0;

    for (uint32_t i = 0; i < preview_image_width * preview_image_height; ++i) {
        if (preview[i].escape_ < params.escape_range_min_) {
            params.escape_range_min_ = preview[i].escape_;
        }
        if (preview[i].escape_ > params.escape_range_max_) {
            params.escape_range_max_ = preview[i].escape_;
        }
    }

    if (params.escape_range_min_ == params.escape_range_max_) {
        params.escape_range_max_++;
    }

    std::cout << params.escape_range_min_ << " : " << params.escape_range_max_ << std::endl;

    delete[] preview;

    if (!generate(params, block_device, true)) {
        return false;
    }

    cudaFree(block_device);
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
bool fractal<T>::generate(kernel_params<T> &params, kernel_block<T> *block_device, bool colour) {
    uint32_t *block_image_device;
    kernel_params<T> *params_device{ nullptr };

    cudaMalloc(&block_image_device, cuda_groups_ * cuda_threads_ * sizeof(uint32_t)); 
    if (block_image_device == nullptr) {
        return false;
    }
    cudaMalloc(&params_device, sizeof(kernel_params<T>));
    if (params_device == nullptr) {
        return false;
    }

    cudaError_t cudaError{ cudaSuccess };

    for (uint64_t image_chunk = 0; image_chunk < (params.image_width_ * params.image_height_); image_chunk += (cuda_groups_ * cuda_threads_)) {
        uint64_t chunk_size = std::min((params.image_width_ * params.image_height_) - image_chunk, cuda_groups_ * cuda_threads_);
        uint64_t chunk_groups = chunk_size / cuda_threads_;
        cudaMemset(block_device, 0, cuda_groups_ * cuda_threads_ * sizeof(kernel_block<T>));

        for (uint32_t i = 0; i < (escape_limit_ / escape_block_); ++i) {
            if ((i % 10) == 0) {
                std::wcout << L"\r[+] Chunk: " << image_chunk / (cuda_groups_ * cuda_threads_) << " / "
                    << 1 + params.image_width_ * params.image_height_ / (cuda_groups_ * cuda_threads_)
                    << L", Block: " << i * escape_block_ << " / " << escape_limit_ << std::flush;
            }
            params.image_chunk_ = image_chunk;
            params.escape_i_ = i;
            cudaMemcpy(params_device, &params, sizeof(params), cudaMemcpyHostToDevice);

            kernel_mandelbrot<<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(cuda_threads_)>>>(block_device, params_device);

            if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                std::wcout << std::endl << "[!] cudaDeviceSynchronize(): cudaError: " << cudaError << std::endl;
                break;
            }
        }

        if (cudaError != cudaSuccess) {
            break;
        }

        if (block_image_device != nullptr) {
            cudaMemset(block_image_device, 0, cuda_groups_ * cuda_threads_ * sizeof(uint32_t));
            kernel_colour<T> <<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(cuda_threads_)>>>(block_device, params_device, block_image_device);

            if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                std::wcout << std::endl << "[!] cudaDeviceSynchronize(): cudaError: " << cudaError << std::endl;
                break;
            }

            cudaMemcpy(&image_[image_chunk], block_image_device, chunk_groups * cuda_threads_ * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        }
    }

    cudaFree(block_image_device);
    cudaFree(params_device);

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
    double abs = block->abs_;

    if (params->escape_i_ == 0) {
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
                escape = i + params->escape_i_ * params->escape_block_;
                break;
            }
        }
    }

    block->escape_ = escape;
    block->re_ = re;
    block->im_ = im;
    block->abs_ = abs;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_mandelbrot(kernel_block<fixed_point<I, F>> *blocks, kernel_params<fixed_point<I, F>> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int pixel_x = (params->image_chunk_ + tid) % params->image_width_;
    const int pixel_y = (params->image_chunk_ + tid) / params->image_width_;

    fixed_point<I, F> re_c(-2.0 + pixel_x * 3.0 / params->image_width_);
    fixed_point<I, F> im_c(1.0 - pixel_y * 2.0 / params->image_height_);
    re_c.multiply(params->scale_);
    im_c.multiply(params->scale_);
    re_c.add(params->re_);
    im_c.add(params->im_);

    kernel_block<fixed_point<I, F>> *block = &blocks[tid];
    uint64_t escape = block->escape_;
    fixed_point<I, F> re(block->re_);
    fixed_point<I, F> im(block->im_);
    fixed_point<I, F> abs(block->abs_);

    if (params->escape_i_ == 0) {
        escape = params->escape_limit_;
        re.set(re_c);
        im.set(im_c);
        abs.set(0);
    }

    fixed_point<I, F> re_prod(re);
    re_prod.multiply(re);

    fixed_point<I, F> im_prod(im);
    im_prod.multiply(im);

    fixed_point<I, F> re_imed;
    fixed_point<I, F> im_imed;

    if (escape == params->escape_limit_) {
        for (uint64_t i = 0; i < params->escape_block_; ++i) {
            re_imed.set(im_prod);
            re_imed.negate();
            re_imed.add(re_prod);
            re_imed.add(re_c);

            im_imed.set(2);
            im_imed.multiply(re);
            im_imed.multiply(im);
            im_imed.add(im_c);

            re.set(re_imed);
            im.set(im_imed);

            re_prod.set(re);
            re_prod.multiply(re);

            im_prod.set(im);
            im_prod.multiply(im);

            abs.set(re_prod);
            abs.add(im_prod);

            if (static_cast<double>(abs) > 4.0) {
                escape = i + params->escape_i_ * params->escape_block_;
                break;
            }
        }
    }

    block->escape_ = escape;
    block->re_.set(re);
    block->im_.set(im);
    block->abs_.set(abs);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void kernel_colour(kernel_block<T> *blocks, kernel_params<T> *params, uint32_t *block_image) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    kernel_block<T> *block = &blocks[tid];
    uint64_t escape = block->escape_;

    double r = 0;
    double g = 0;
    double b = 0;

    if (escape < params->escape_limit_) {
        escape -= (params->escape_range_min_ - 1);

        double abs = static_cast<double>(block->abs_);

        //double hue = 360.0 * log(1.0 * escape) / log(1.0 * (params->escape_range_max_ - params->escape_range_min_));// +1.0 - (log(log(abs_z)) / log(2.0));
        //double hue = (360.0 * (log(1.0 * escape) - log(log(abs))) / (log(1.0 * (params->escape_range_max_ - params->escape_range_min_)) + log(2.0)));

        double hue = 360.0 * (log(1.0 * escape) / log(1.0 * (params->escape_range_max_ - params->escape_range_min_)));
        double sat = 0.95 - log(2.0) + log(log(abs));
        double val = 0.95;

        hue = fmod(hue, 360.0);
        hue /= 60;

        double hue_floor = floor(hue);
        double hue_fract = hue - hue_floor;
        double p = val * (1.0 - sat);
        double q = val * (1.0 - sat * hue_fract);
        double t = val * (1.0 - sat * (1.0 - hue_fract));

        switch (static_cast<unsigned char>(hue_floor) % 6) {
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
