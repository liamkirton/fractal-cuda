////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <cstdint>
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
template class fractal<fixed_point<1, 1>>; 
template class fractal<fixed_point<1, 2>>;
template class fractal<fixed_point<2, 2>>;
template class fractal<fixed_point<2, 4>>;
template class fractal<fixed_point<2, 8>>;
template class fractal<fixed_point<2, 16>>;
template class fractal<fixed_point<2, 24>>; 
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
    cuda_groups_ = cuda_groups;
    cuda_threads_ = cuda_threads;

    if (trial_image_width_ * trial_image_height_ > cuda_groups_ * cuda_threads_) {
        trial_image_height_ = cuda_groups;
        trial_image_width_ = cuda_threads;
    }

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
    re_ = re_max_variance_ = re;
    im_ = im_max_variance_ = im;
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
bool fractal<T>::generate(bool trial) {
    resize(image_width_, image_height_);

    kernel_params<T> params_trial(trial_image_width_, trial_image_height_, escape_block_, escape_limit_, 0, re_, im_, scale_);
    kernel_params<T> params(image_width_, image_height_, escape_block_, escape_limit_, colour_method_, re_, im_, scale_);

    if ((block_device_ == nullptr) || (block_device_image_ == nullptr)) {
        return false;
    }

    std::cout << "  [+] Re: " << std::setprecision(5) << static_cast<double>(re_) << ", Im: " << static_cast<double>(im_) << ", Scale: " << static_cast<double>(scale_) << std::endl;

    if (trial) {
        std::cout << "  [+] Trial " << trial_image_width_ << "x" << trial_image_height_ << std::endl;
        if (!generate(params_trial, false)) {
            return false;
        }

        kernel_block<T> *preview = new kernel_block<T>[trial_image_width_ * trial_image_height_];
        cudaMemcpy(preview, block_device_, trial_image_width_ * trial_image_height_ * sizeof(kernel_block<T>), cudaMemcpyDeviceToHost);
        process_trial(params_trial, params, preview);
        delete[] preview;
    }

    std::cout << "  [+] Full Image: " << image_width_ << "x" << image_height_ << " (" << image_size() << " bytes)" << std::endl;

    if (!generate(params, true)) {
        return false;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
bool fractal<T>::generate(kernel_params<T> &params, bool colour) {
    kernel_params<T> *params_device{ nullptr };

    if (colour && (palette_.size() > 0)) {
        cudaMalloc(&params.palette_, 3 * sizeof(double) * palette_.size()); 
        if (params.palette_ != nullptr) {
            params.palette_count_ = palette_.size();

            double *create_palette = new double[palette_.size() * 3];
            for (uint32_t i = 0; i < palette_.size(); ++i) {
                create_palette[i * 3 + 0] = std::get<0>(palette_.at(i));
                create_palette[i * 3 + 1] = std::get<1>(palette_.at(i));
                create_palette[i * 3 + 2] = std::get<2>(palette_.at(i));
            }

            cudaMemcpy(params.palette_, create_palette, 3 * sizeof(double) * palette_.size(), cudaMemcpyHostToDevice);
            delete[] create_palette;
        }
    }

    cudaMalloc(&params_device, sizeof(kernel_params<T>));
    if (params_device == nullptr) {
        return false;
    }

    cudaError_t cudaError{ cudaSuccess };

    for (uint64_t image_chunk = 0; image_chunk < (params.image_width_ * params.image_height_); image_chunk += (cuda_groups_ * cuda_threads_)) {
        uint64_t chunk_size = std::min((params.image_width_ * params.image_height_) - image_chunk, cuda_groups_ * cuda_threads_);
        uint64_t chunk_groups = chunk_size / cuda_threads_;
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

            kernel_mandelbrot<<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(cuda_threads_)>>>(block_device_, params_device);

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
            kernel_colour<T> <<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(cuda_threads_)>>>(block_device_, params_device, block_device_image_);

            if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                std::cout << std::endl << "[!] cudaDeviceSynchronize(): cudaError: " << cudaError << std::endl;
                break;
            }

            cudaMemcpy(&image_[image_chunk], block_device_image_, chunk_groups * cuda_threads_ * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        }
    }

    cudaFree(params.palette_);
    cudaFree(params_device);

    std::cout << std::endl;
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
void fractal<double>::pixel_to_coord(uint64_t x, uint64_t image_width, double &re, uint64_t y, uint64_t image_height, double &im) {
    re = re_ + (re_min + x * (re_max - re_min) / image_width) * scale_;
    im = im_ + (im_max - y * (im_max - im_min) / image_height) * scale_;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void fractal<T>::pixel_to_coord(uint64_t x, uint64_t image_width, T &re, uint64_t y, uint64_t image_height, T &im) {
    re.set(re_min + x * (re_max - re_min) / image_width);
    im.set(im_max - y * (im_max - im_min) / image_height);
    re.multiply(scale_);
    im.multiply(scale_);
    re.add(re_);
    im.add(im_);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void fractal<T>::process_trial(kernel_params<T> &params_trial, kernel_params<T> &params, kernel_block<T> *preview) {
    //
    // Calculate Escape Range
    //

    params.escape_range_min_ = 0xffffffffffffffff;
    params.escape_range_max_ = 0;

    for (uint64_t i = 0; i < trial_image_width_ * trial_image_height_; ++i) {
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

    std::cout << "    [+] Escape Range: " << params.escape_range_min_ << " => " << params.escape_range_max_ << std::endl;

    //
    // Calculate Escape Range
    //

    std::vector<std::tuple<double, int32_t, int32_t>> variances;

    int32_t x_min = trial_image_width_ / 4;
    int32_t x_max = 3 * trial_image_width_ / 4;
    int32_t y_min = trial_image_height_ / 4;
    int32_t y_max = 3 * trial_image_height_ / 4;

    double escape_max = static_cast<double>(1.0 + params.escape_range_max_ - params.escape_range_min_);

    for (int32_t y = y_min; y < y_max; ++y) {
        for (int32_t x = x_min; x < x_max; ++x) {
            uint32_t block_escapes[9]{ 0 };
            for (int32_t j = 0; j < 9; ++j) {
                int32_t r_ix = y - 1 + (j / 3);
                if ((r_ix >= 0) && (r_ix < trial_image_height_)) {
                    int32_t p_ix = x - 1 + (j % 3);
                    if ((p_ix >= 0) && (p_ix < trial_image_width_)) {
                        if (preview[r_ix * trial_image_width_ + p_ix].escape_ < params.escape_limit_) {
                            double escape = static_cast<double>(preview[r_ix * trial_image_width_ + p_ix].escape_) - static_cast<double>(params.escape_range_min_);
                            double mu = 1.0 + escape - log2(0.5 * log(static_cast<double>(preview[r_ix * trial_image_width_ + p_ix].abs_)) / log(default_escape_radius));
                            if (mu < 0.0) mu = 0.0;
                            block_escapes[j] = mu / log(escape_max);
                        }
                    }
                }
            }

            double block_mean = 0.0;
            for (int32_t j = 0; j < 9; ++j) {
                block_mean += block_escapes[j];
            }
            block_mean /= 9;

            double block_variance = 0.0;
            for (int32_t j = 0; j < 9; ++j) {
                block_variance += (block_escapes[j] - block_mean) * (block_escapes[j] - block_mean);
            }
            block_variance /= 9;
            block_variance /= (1.0 + (pow(1.0 * trial_image_width_ / 2.0 - x, 2.0) + pow(1.0 * trial_image_height_ / 2.0 - y, 2.0)));

            variances.push_back(std::make_tuple(block_variance, x, y));
        }
    }

    std::sort(variances.begin(), variances.end(), [](auto a, auto b) { return std::get<0>(a) > std::get<0>(b); });
    std::remove_if(variances.begin(), variances.end(), [](auto a) { return std::get<0>(a) == 0.0; });

    std::random_device random;
    std::mt19937 gen(random());
    std::uniform_int_distribution<> dist(0, variances.size() / 20);
    auto max = variances.at(dist(gen));

    double max_variance = std::get<0>(max);

    if (max_variance != 0.0) {
        uint64_t max_variance_x = static_cast<uint64_t>(std::get<1>(max));
        uint64_t max_variance_y = static_cast<uint64_t>(std::get<2>(max));
        pixel_to_coord(max_variance_x, trial_image_width_, re_max_variance_, max_variance_y, trial_image_height_, im_max_variance_);
        std::cout << "    [+] Max Variance: " << std::get<0>(max) << " => x: " << max_variance_x << ", y: " << max_variance_y << std::endl;
    }
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

    const double re_c = params->re_ + (re_min + pixel_x * (re_max - re_min) / params->image_width_) * params->scale_;
    const double im_c = params->im_ + (im_max - pixel_y * (im_max - im_min) / params->image_height_) * params->scale_;

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
            double re_t = re;
            re = (re * re) - (im * im) + re_c;
            im = (2.0 * re_t * im) + im_c;
            abs = re * re + im * im;
            if (abs >= (default_escape_radius * default_escape_radius)) {
                escape = i + params->escape_i_ * params->escape_block_;
                break;
            }
        }
        block->escape_ = escape;
        block->re_ = re;
        block->im_ = im;
        block->abs_ = abs;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_mandelbrot(kernel_block<fixed_point<I, F>> *blocks, kernel_params<fixed_point<I, F>> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int pixel_x = (params->image_chunk_ + tid) % params->image_width_;
    const int pixel_y = (params->image_chunk_ + tid) / params->image_width_;

    fixed_point<I, F> re_c(re_min + pixel_x * (re_max - re_min) / params->image_width_);
    fixed_point<I, F> im_c(im_max - pixel_y * (im_max - im_min) / params->image_height_);
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

            if (static_cast<double>(abs) >= (default_escape_radius * default_escape_radius)) {
                escape = i + params->escape_i_ * params->escape_block_;
                break;
            }
        }

        block->escape_ = escape;
        block->re_.set(re);
        block->im_.set(im);
        block->abs_.set(abs);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void kernel_colour(kernel_block<T> *blocks, kernel_params<T> *params, uint32_t *block_image) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    kernel_block<T> *block = &blocks[tid];

    double r = 0.0;
    double g = 0.0;
    double b = 0.0;

    if (block->escape_ < params->escape_limit_) {
        double abs = static_cast<double>(block->abs_); 
        double escape = static_cast<double>(block->escape_) - static_cast<double>(params->escape_range_min_);
        double escape_max = static_cast<double>(1.0 + params->escape_range_max_ - params->escape_range_min_);

        double hue = 0.0;
        double sat = 0.95;
        double val = 0.95;

        double mu = 1.0 + escape - log2(0.5 * log(abs) / log(default_escape_radius));
        if (mu < 1.0) mu = 1.0;

        switch (params->colour_method_) {
        default:
        case 0:
            sat = 0.0;
            break;
        case 1:
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
        case 2:
            hue = 360.0 * log(mu) / log(escape_max);
            break; 
        case 3:
            if (mu < 2.71828182846) {
                mu = 2.71828182846;
            }
            hue = 360.0 * log(escape_max) / log(mu);
            break;
        }

        hue = fmod(hue, 360.0);
        hue /= 60.0;

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
