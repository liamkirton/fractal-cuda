////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "fractal.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template class fractal<double>;
template class fractal<fixed_point<1, 2>>;
template class fractal<fixed_point<1, 3>>;
template class fractal<fixed_point<1, 4>>;
template class fractal<fixed_point<1, 6>>;
template class fractal<fixed_point<1, 8>>;
template class fractal<fixed_point<1, 12>>;
template class fractal<fixed_point<1, 16>>;
template class fractal<fixed_point<1, 20>>;
template class fractal<fixed_point<1, 24>>;
template class fractal<fixed_point<1, 28>>;
template class fractal<fixed_point<1, 32>>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_init_julia(kernel_block<double> *blocks, kernel_params<double> *params);
__global__ void kernel_init_mandelbrot(kernel_block<double> *blocks, kernel_params<double> *params);
__global__ void kernel_iterate(kernel_block<double> *blocks, kernel_params<double> *params);

template<uint32_t I, uint32_t F> __global__ void kernel_init_julia(kernel_block<fixed_point<I, F>> *blocks, kernel_params<fixed_point<I, F>> *params);
template<uint32_t I, uint32_t F> __global__ void kernel_init_mandelbrot(kernel_block<fixed_point<I, F>> *blocks, kernel_params<fixed_point<I, F>> *params);
template<uint32_t I, uint32_t F> __global__ void kernel_iterate(kernel_block<fixed_point<I, F>> *blocks, kernel_params<fixed_point<I, F>> *params);

template<typename T>
__global__ void kernel_colour(kernel_block<T> *blocks, kernel_params<T> *params, uint32_t *block_image);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host Methods
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
bool fractal<T>::initialise(uint32_t cuda_groups, uint32_t cuda_threads) {
    cuda_groups_ = cuda_groups;
    cuda_threads_ = cuda_threads;

    trial_image_width_ = trial_image_height_ = static_cast<uint32_t>(floor(sqrt(cuda_groups * cuda_threads)));

    uninitialise();

    cudaError_t cuda_status;
    if ((cuda_status = cudaSetDevice(0)) != cudaSuccess) {
        std::wcout << L"ERROR: cudaSetDevice() Failed. [" << cuda_status << L"]" << std::endl << std::endl;
        return false;
    }

    cudaMalloc(&block_device_, cuda_groups_ * cuda_threads_ * sizeof(kernel_block<T>));
    if (block_device_ == nullptr) {
        return false;
    }

    cudaMalloc(&block_device_image_, cuda_groups_ * cuda_threads_ * sizeof(uint32_t));
    if (block_device_image_ == nullptr) {
        return false;
    }

    resize(image_width_, image_height_);
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void fractal<T>::uninitialise() {
    if (block_device_ != nullptr) {
        cudaFree(block_device_);
        block_device_ = nullptr;
    }
    if (block_device_image_ != nullptr) {
        block_device_image_ = nullptr;
    }
    cudaDeviceReset();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void fractal<T>::resize(const uint32_t image_width, const uint32_t image_height) {
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
    re_ = re_max_variance_ = re;
    im_ = im_max_variance_ = im;
    scale_ = scale;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void fractal<T>::specify(const T &re, const T &im, const T &scale) {
    re_.set(re);
    re_max_variance_.set(re);
    im_.set(im);
    im_max_variance_.set(im);
    scale_.set(scale);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
void fractal<double>::specify_julia(const double &re_c, const double &im_c) {
    julia_ = true;
    re_c_ = re_c;
    im_c_ = im_c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void fractal<T>::specify_julia(const T &re_c, const T &im_c) {
    julia_ = true;
    re_c_.set(re_c);
    im_c_.set(im_c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
bool fractal<T>::generate(bool trial, std::function<void(void)> block_callback) {
    resize(image_width_, image_height_);

    kernel_params<T> params_trial(trial_image_width_, trial_image_height_, escape_block_, escape_limit_, 0, re_, im_, scale_, re_c_, im_c_);
    kernel_params<T> params(image_width_, image_height_, escape_block_, escape_limit_, colour_method_, re_, im_, scale_, re_c_, im_c_);

    if ((block_device_ == nullptr) || (block_device_image_ == nullptr)) {
        return false;
    }

    std::cout << "  [+] Re: " << re_ << std::endl
        << "  [+] Im: " << im_ << std::endl
        << "  [+] Sc: " << scale_ << std::endl;

    if (trial) {
        std::cout << "  [+] Trial " << trial_image_width_ << "x" << trial_image_height_ << std::endl;
        if (!generate(params_trial, false, []() {})) {
            return false;
        }

        std::unique_ptr<kernel_block<T>> preview(new kernel_block<T>[trial_image_width_ * trial_image_height_]);
        cudaMemcpy(preview.get(), block_device_, trial_image_width_ * trial_image_height_ * sizeof(kernel_block<T>), cudaMemcpyDeviceToHost);
        if (!process_trial(params_trial, params, preview.get())) {
            return false;
        }
    }

    std::cout << "  [+] Full Image: " << image_width_ << "x" << image_height_ << " (" << image_size() << " bytes)" << std::endl;

    if (!generate(params, true, block_callback)) {
        return false;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
bool fractal<T>::generate(kernel_params<T> &params, bool colour, std::function<void(void)> block_callback) {
    kernel_params<T> *params_device{ nullptr };

    if (colour && (palette_.size() > 0)) {
        cudaMalloc(&params.palette_, 3 * sizeof(double) * palette_.size());

        params.set_hue_ = std::get<0>(palette_.at(0));
        params.set_sat_ = std::get<1>(palette_.at(0));
        params.set_val_ = std::get<2>(palette_.at(0));

        if (params.palette_ != nullptr) {
            params.palette_count_ = palette_.size() - 1;

            double *create_palette = new double[(palette_.size() - 1) * 3];
            for (uint32_t i = 0; i < palette_.size() - 1; ++i) {
                create_palette[i * 3 + 0] = std::get<0>(palette_.at(i + 1));
                create_palette[i * 3 + 1] = std::get<1>(palette_.at(i + 1));
                create_palette[i * 3 + 2] = std::get<2>(palette_.at(i + 1));
            }

            cudaMemcpy(params.palette_, create_palette, 3 * sizeof(double) * (palette_.size() - 1), cudaMemcpyHostToDevice);
            delete[] create_palette;
        }
    }

    cudaMalloc(&params_device, sizeof(kernel_params<T>));
    if (params_device == nullptr) {
        return false;
    }

    cudaError_t cudaError{ cudaSuccess };

    for (uint32_t image_chunk = 0; image_chunk < (params.image_width_ * params.image_height_); image_chunk += (cuda_groups_ * cuda_threads_)) {
        uint32_t chunk_size = std::min((params.image_width_ * params.image_height_) - image_chunk, cuda_groups_ * cuda_threads_);
        uint32_t chunk_groups = chunk_size / cuda_threads_;
        cudaMemset(block_device_, 0, cuda_groups_ * cuda_threads_ * sizeof(kernel_block<T>));

        for (uint32_t i = 0; i < (escape_limit_ / escape_block_); ++i) {
            if ((i % 10) == 0) {
                std::cout << "\r    [+] Chunk: " << image_chunk / (cuda_groups_ * cuda_threads_) << " / "
                    << params.image_width_ * params.image_height_ / (cuda_groups_ * cuda_threads_)
                    << ", Block: " << i * escape_block_ << " / " << escape_limit_ << "           " << std::flush;
            }
            params.image_chunk_ = image_chunk;
            params.escape_i_ = i;
            cudaMemcpy(params_device, &params, sizeof(params), cudaMemcpyHostToDevice);

            if (i == 0) {
                if (julia_) {
                    kernel_init_julia<<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(cuda_threads_)>>>(block_device_, params_device);
                }
                else {
                    kernel_init_mandelbrot<<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(cuda_threads_)>>>(block_device_, params_device);
                }
            }
            kernel_iterate<<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(cuda_threads_)>>>(block_device_, params_device);

            if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                std::cout << std::endl << "[!] cudaDeviceSynchronize(): cudaError: " << cudaError << std::endl;
                break;
            }
        }

        if (cudaError != cudaSuccess) {
            break;
        }

        if (colour) {
            cudaMemset(block_device_image_, 0, cuda_groups_ * cuda_threads_ * sizeof(uint32_t));
            kernel_colour<T><<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(cuda_threads_)>>>(block_device_, params_device, block_device_image_);

            if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                std::cout << std::endl << "[!] cudaDeviceSynchronize(): cudaError: " << cudaError << std::endl;
                break;
            }

            cudaMemcpy(&image_[image_chunk], block_device_image_, chunk_groups * cuda_threads_ * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            block_callback();
        }
    }

    cudaFree(params.palette_);
    cudaFree(params_device);

    std::cout << std::endl;
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
void fractal<double>::pixel_to_coord(uint32_t x, uint32_t image_width, double &re, uint32_t y, uint32_t image_height, double &im) {
    re = re_ + (re_min + x * (re_max - re_min) / image_width) * scale_;
    im = im_ + (im_max - y * (im_max - im_min) / image_height) * scale_;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void fractal<T>::pixel_to_coord(uint32_t x, uint32_t image_width, T &re, uint32_t y, uint32_t image_height, T &im) {
    re.set(re_min + x * (re_max - re_min) / image_width);
    im.set(im_max - y * (im_max - im_min) / image_height);
    re.multiply(scale_);
    im.multiply(scale_);
    re.add(re_);
    im.add(im_);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
bool fractal<T>::process_trial(kernel_params<T> &params_trial, kernel_params<T> &params, kernel_block<T> *preview) {
    //
    // Calculate Escape Range
    //

    params.escape_range_min_ = 0xffffffffffffffff;
    params.escape_range_max_ = 0;

    for (uint32_t i = 0; i < trial_image_width_ * trial_image_height_; ++i) {
        if (preview[i].escape_ < params.escape_range_min_) {
            params.escape_range_min_ = preview[i].escape_;
        }
        if (preview[i].escape_ > params.escape_range_max_) {
            params.escape_range_max_ = preview[i].escape_;
        }
    }

    std::cout << "    [+] Escape Range: " << params.escape_range_min_ << " => " << params.escape_range_max_ << std::endl;

    if (params.escape_range_min_ == params.escape_range_max_) {
        std::cout << "    [+] Complete - Zero Escape Range" << std::endl;
        return false;
    }

    ////
    //// Calculate Variance
    ////

    //std::vector<std::tuple<double, int32_t, int32_t>> variances;

    //int32_t x_min = 0;
    //int32_t x_max = trial_image_width_;
    //int32_t y_min = 0;
    //int32_t y_max = trial_image_height_;

    //int32_t x_centre = trial_image_width_ / 2;
    //int32_t y_centre = trial_image_width_ / 2;

    //double escape_max = static_cast<double>(1.0 + params.escape_range_max_ - params.escape_range_min_);
    //constexpr uint32_t block_radius = 5;

    //bool mid_block_variance = false;

    //for (int32_t y = y_min; y < y_max; ++y) {
    //    for (int32_t x = x_min; x < x_max; ++x) {
    //        uint32_t block_escapes[block_radius * block_radius]{ 0 };
    //        for (int32_t j = 0; j < block_radius * block_radius; ++j) {
    //            int32_t r_ix = y - (block_radius / 2) + (j / block_radius);
    //            if ((r_ix >= 0) && (r_ix < trial_image_height_)) {
    //                int32_t p_ix = x - (block_radius / 2) + (j % block_radius);
    //                if ((p_ix >= 0) && (p_ix < trial_image_width_)) {
    //                    auto &block = preview[r_ix * trial_image_width_ + p_ix];
    //                    if (block.escape_ < params.escape_limit_) {
    //                        double abs = pow(static_cast<double>(block.re_), 2.0) + pow(static_cast<double>(block.im_), 2.0);
    //                        double escape = static_cast<double>(block.escape_) - static_cast<double>(params.escape_range_min_);
    //                        double mu = 1.0 + escape - log2(0.5 * log(static_cast<double>(abs)) / log(default_escape_radius));
    //                        if (mu < 0.0) mu = 0.0;

    //                        double delta = max(1.0, sqrt(pow(x_centre - x, 2) + pow(y_centre - y, 2)));
    //                        block_escapes[j] = mu / (delta + log(escape_max));
    //                    }
    //                }
    //            }
    //        }

    //        double block_mean = 0.0;
    //        for (int32_t j = 0; j < block_radius * block_radius; ++j) {
    //            block_mean += block_escapes[j];
    //        }
    //        block_mean /= (block_radius * block_radius);

    //        double block_variance = 0.0;
    //        for (int32_t j = 0; j < block_radius * block_radius; ++j) {
    //            block_variance += (block_escapes[j] - block_mean) * (block_escapes[j] - block_mean);
    //        }
    //        block_variance /= (block_radius * block_radius);

    //        variances.push_back(std::make_tuple(block_variance, x, y));

    //        if ((x == trial_image_width_ / 2) && (y == trial_image_height_ / 2)) {
    //            mid_block_variance = block_variance != 0;
    //        }
    //    }
    //}

    //std::sort(variances.begin(), variances.end(), [](auto a, auto b) { return std::get<0>(a) > std::get<0>(b); });
    //std::remove_if(variances.begin(), variances.end(), [](auto a) { return std::get<0>(a) == 0.0; });

    //std::random_device random;
    //std::mt19937 gen(random());
    //std::uniform_int_distribution<> dist(0, variances.size() / 20);

    //auto max = variances.at(dist(gen));
    //double max_variance = std::get<0>(max);

    //if (!mid_block_variance && (max_variance != 0.0)) {
    //    uint32_t max_variance_x = static_cast<uint32_t>(std::get<1>(max));
    //    uint32_t max_variance_y = static_cast<uint32_t>(std::get<2>(max));
    //    pixel_to_coord(max_variance_x, trial_image_width_, re_max_variance_, max_variance_y, trial_image_height_, im_max_variance_);
    //    std::cout << "    [+] Max Variance: " << std::get<0>(max) << " At x: " << max_variance_x << ", y: " << max_variance_y
    //        << " => Re: " << std::setprecision(12) << static_cast<double>(re_max_variance_) << ", Im: " << static_cast<double>(im_max_variance_)
    //        << std::endl;
    //}

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Kernels
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_init_julia(kernel_block<double> *blocks, kernel_params<double> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    const double pixel_x = (params->image_chunk_ + tid) % params->image_width_;
    const double pixel_y = (params->image_chunk_ + tid) / params->image_width_;

    const double re_c = params->re_ + (re_min + pixel_x * (re_max - re_min) / params->image_width_) * params->scale_;
    const double im_c = params->im_ + (im_max - pixel_y * (im_max - im_min) / params->image_height_) * params->scale_;

    kernel_block<double> *block = &blocks[tid];
    block->escape_ = params->escape_limit_;
    block->re_c_ = params->re_c_;
    block->im_c_ = params->im_c_;
    block->re_ = re_c;
    block->im_ = im_c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_init_mandelbrot(kernel_block<double> *blocks, kernel_params<double> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    const double pixel_x = (params->image_chunk_ + tid) % params->image_width_;
    const double pixel_y = (params->image_chunk_ + tid) / params->image_width_;

    const double re_c = params->re_ + (re_min + pixel_x * (re_max - re_min) / params->image_width_) * params->scale_;
    const double im_c = params->im_ + (im_max - pixel_y * (im_max - im_min) / params->image_height_) * params->scale_;

    kernel_block<double> *block = &blocks[tid];
    block->escape_ = params->escape_limit_;
    block->re_c_ = re_c; 
    block->im_c_ = im_c; 
    block->re_ = re_c;
    block->im_ = im_c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_iterate(kernel_block<double> *blocks, kernel_params<double> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    kernel_block<double> *block = &blocks[tid];
    double re_c = block->re_c_;
    double im_c = block->im_c_;
    double re = block->re_;
    double im = block->im_;
    double re_prod = re * re;
    double im_prod = im * im;

    const uint64_t escape = block->escape_;
    const uint64_t escape_block = params->escape_block_;
    const uint64_t escape_limit = params->escape_limit_;

    if (escape == escape_limit) {
        for (uint64_t i = 0; i < escape_block; ++i) {
            im = 2.0 * re * im + im_c;
            re = re_prod - im_prod + re_c;
            re_prod = re * re;
            im_prod = im * im;
            if ((re_prod + im_prod) >= default_escape_radius_square) {
                block->escape_ = i + params->escape_i_ * escape_block;
                break;
            }
        }

        block->re_ = re;
        block->im_ = im;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_init_julia(kernel_block<fixed_point<I, F>> *blocks, kernel_params<fixed_point<I, F>> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    const double pixel_x = (params->image_chunk_ + tid) % params->image_width_;
    const double pixel_y = (params->image_chunk_ + tid) / params->image_width_;

    fixed_point<I, F> re_c(re_min + pixel_x * (re_max - re_min) / params->image_width_);
    fixed_point<I, F> im_c(im_max - pixel_y * (im_max - im_min) / params->image_height_);
    re_c.multiply(params->scale_);
    im_c.multiply(params->scale_);
    re_c.add(params->re_);
    im_c.add(params->im_);

    kernel_block<fixed_point<I, F>> *block = &blocks[tid];
    block->escape_ = params->escape_limit_;
    block->re_c_.set(params->re_c_);
    block->im_c_.set(params->im_c_);
    block->re_.set(re_c);
    block->im_.set(im_c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_init_mandelbrot(kernel_block<fixed_point<I, F>> *blocks, kernel_params<fixed_point<I, F>> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    const double pixel_x = (params->image_chunk_ + tid) % params->image_width_;
    const double pixel_y = (params->image_chunk_ + tid) / params->image_width_;

    fixed_point<I, F> re_c(re_min + pixel_x * (re_max - re_min) / params->image_width_);
    fixed_point<I, F> im_c(im_max - pixel_y * (im_max - im_min) / params->image_height_);
    re_c.multiply(params->scale_);
    im_c.multiply(params->scale_);
    re_c.add(params->re_);
    im_c.add(params->im_);

    kernel_block<fixed_point<I, F>> *block = &blocks[tid];
    block->escape_ = params->escape_limit_;
    block->re_c_.set(re_c);
    block->im_c_.set(im_c);
    block->re_.set(re_c);
    block->im_.set(im_c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_iterate(kernel_block<fixed_point<I, F>> *blocks, kernel_params<fixed_point<I, F>> *params) {
    kernel_block<fixed_point<I, F>> *block = &blocks[threadIdx.x + blockIdx.x * blockDim.x];

    fixed_point<I, F> re(block->re_);
    fixed_point<I, F> im(block->im_);

    const uint64_t escape = block->escape_;
    const uint64_t escape_block = params->escape_block_;
    const uint64_t escape_limit = params->escape_limit_;

    fixed_point<I, F> re_prod;
    fixed_point<I, F> im_prod;
    re.multiply(re, re_prod);
    im.multiply(im, im_prod);

    if (escape == escape_limit) {
        for (uint64_t i = 0; i < escape_block; ++i) {
            im.multiply(re);
            im.multiply(2ULL);
            im.add(block->im_c_);

            re.set(im_prod);
            re.negate();
            re.add(re_prod);
            re.add(block->re_c_);

            re.multiply(re, re_prod);
            im.multiply(im, im_prod);

            if ((static_cast<double>(re_prod) + static_cast<double>(im_prod)) >= default_escape_radius_square) {
                block->escape_ = i + params->escape_i_ * escape_block;
                break;
            }
        }

        block->re_.set(re);
        block->im_.set(im);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void kernel_colour(kernel_block<T> *blocks, kernel_params<T> *params, uint32_t *block_image) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    kernel_block<T> *block = &blocks[tid];

    double hue = params->set_hue_;
    double sat = params->set_sat_;
    double val = params->set_val_;

    if (block->escape_ < params->escape_limit_) {
        double abs = pow(static_cast<double>(block->re_), 2.0) + pow(static_cast<double>(block->im_), 2.0);
        double escape = static_cast<double>(block->escape_) - static_cast<double>(params->escape_range_min_);
        double escape_max = static_cast<double>(1.0 + params->escape_range_max_ - params->escape_range_min_);

        double mu = 1.0 + escape - log2(0.5 * log(abs) / log(default_escape_radius));
        if (mu < 1.0) mu = 1.0;

        switch (params->colour_method_) {
        default:
        case 0:
        case 1:
            hue = 0.0;
            sat = 0.0;
            val = 1.0;
            break;
        case 2:
        case 3:
            {
                double t = log(mu) / log(escape_max);
                hue = 0.0;
                sat = 0.0;
                val = 0.0;
                for (uint32_t i = 0; i < params->palette_count_; ++i) {
                    double poly = pow(t, static_cast<double>(i)) * pow(1.0 - t, static_cast<double>(params->palette_count_ - 1 - i));
                    hue += params->palette_[3 * i + 0] * poly;
                    sat += params->palette_[3 * i + 1] * poly;
                    val += params->palette_[3 * i + 2] * poly;
                }
                hue *= 360.0;
            }
            break;
        case 4:
        case 5:
            hue = 360.0 * log(mu) / log(escape_max);
            sat = 0.95;
            val = 0.95;
            break;
        case 6:
        case 7:
            if (mu < 2.71828182846) {
                mu = 2.71828182846;
            }
            hue = 360.0 * log(escape_max) / log(mu);
            sat = 0.95;
            val = 0.95;
            break;
        }

        if (params->colour_method_ % 2 == 1) {
            sat = 0.95 - log(2.0) + log(log(abs));
        }
    }

    hue = fmod(hue, 360.0);
    hue /= 60.0;

    double hue_floor = floor(hue);
    double hue_fract = hue - hue_floor;
    double p = val * (1.0 - sat);
    double q = val * (1.0 - sat * hue_fract);
    double t = val * (1.0 - sat * (1.0 - hue_fract));

    double r = 0.0;
    double g = 0.0;
    double b = 0.0;

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

    block_image[tid] =
        (static_cast<unsigned char>(b)) |
        (static_cast<unsigned char>(g) << 8) |
        (static_cast<unsigned char>(r) << 16) |
        (255 << 24);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
