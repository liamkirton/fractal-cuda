////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "fractal.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr uint32_t grid_resX = 1;
constexpr uint32_t grid_resY = 1;

template class fractal<double>;
template class fractal<fixed_point<1, 2>>;
#ifndef _DEBUG
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
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_init_julia(kernel_chunk<double> *chunks, kernel_params<double> *params);
__global__ void kernel_init_mandelbrot(kernel_chunk<double> *chunks, kernel_params<double> *params);
__global__ void kernel_init_mandelbrot_perturbation(kernel_chunk_perturbation_reference<double> *ref_chunks, kernel_chunk<double> *chunks, kernel_params<double> *params);
__global__ void kernel_init_mandelbrot_perturbation_reference(kernel_chunk_perturbation_reference<double> *ref_chunks, kernel_params<double> *params);
__global__ void kernel_iterate(kernel_chunk<double> *chunks, kernel_params<double> *params);
__global__ void kernel_iterate_perturbation(kernel_chunk_perturbation_reference<double> *ref_chunks, kernel_chunk_perturbation<double>, kernel_params<double> *params);
__global__ void kernel_iterate_perturbation_reference(kernel_chunk_perturbation_reference<double> *ref_chunks, kernel_params<double> *params);

template<uint32_t I, uint32_t F> __global__ void kernel_init_julia(kernel_chunk<fixed_point<I, F>> *chunks, kernel_params<fixed_point<I, F>> *params);
template<uint32_t I, uint32_t F> __global__ void kernel_init_mandelbrot(kernel_chunk<fixed_point<I, F>> *chunks, kernel_params<fixed_point<I, F>> *params);
template<uint32_t I, uint32_t F> __global__ void kernel_init_mandelbrot_perturbation(kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunks, kernel_chunk<fixed_point<I, F>> *chunks, kernel_params<fixed_point<I, F>> *params);
template<uint32_t I, uint32_t F> __global__ void kernel_init_mandelbrot_perturbation_reference(kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunks, kernel_params<fixed_point<I, F>> *params);
template<uint32_t I, uint32_t F> __global__ void kernel_iterate(kernel_chunk<fixed_point<I, F>> *chunks, kernel_params<fixed_point<I, F>> *params);
template<uint32_t I, uint32_t F> __global__ void kernel_iterate_perturbation(kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunks, kernel_chunk_perturbation<fixed_point<I, F>> *chunks, kernel_params<fixed_point<I, F>> *params);
template<uint32_t I, uint32_t F> __global__ void kernel_iterate_perturbation_reference(kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunks, kernel_params<fixed_point<I, F>> *params);

template<typename S, typename T> __global__ void kernel_colour(S *chunks, kernel_params<T> *params, uint32_t *block_image);

template<typename T>
__global__ void kernel_reduce(kernel_chunk<T> *chunks, kernel_params<T> *params, uint32_t chunk_count);

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

    //cudaMalloc(&chunk_buffer_device_, cuda_groups_ * cuda_threads_ * sizeof(kernel_chunk<T>));
    //if (chunk_buffer_device_ == nullptr) {
    //    return false;
    //}

    cudaMalloc(&image_device_, cuda_groups_ * cuda_threads_ * sizeof(uint32_t));
    if (image_device_ == nullptr) {
        return false;
    }

    resize(image_width_, image_height_);

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void fractal<T>::uninitialise() {
    if (chunk_buffer_ != nullptr) {
        delete[] chunk_buffer_;
        chunk_buffer_ = nullptr;
    }
    if (chunk_buffer_device_ != nullptr) {
        cudaFree(chunk_buffer_device_);
        chunk_buffer_device_ = nullptr;
    }
    if (image_ != nullptr) {
        delete[] image_;
        image_ = nullptr;
    }
    if (image_device_ != nullptr) {
        cudaFree(image_device_);
        image_device_ = nullptr;
    }
    cudaDeviceReset();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void fractal<T>::resize(const uint32_t image_width, const uint32_t image_height) {
    if ((image_width != image_width_) || (image_height != image_height_)) {
        image_width_ = image_width;
        image_height_ = image_height;
        if (chunk_buffer_ != nullptr) {
            delete[] chunk_buffer_;
            chunk_buffer_ = nullptr;
        }
        if (image_ != nullptr) {
            delete[] image_;
            image_ = nullptr;
        }
    }
    if (chunk_buffer_ == nullptr) {
        chunk_buffer_ = new kernel_chunk<T>[image_width_ * image_height_];
    }
    if (image_ == nullptr) {
        image_ = new uint32_t[image_width_ * image_height_];
    }
    memset(chunk_buffer_, 0, sizeof(kernel_chunk<T>) * image_width_ * image_height_);
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
bool fractal<T>::generate(bool trial, bool interactive, std::function<bool(bool)> callback) {
    resize(image_width_, image_height_);

    //kernel_params<T> params_trial(trial_image_width_, trial_image_height_, escape_block_, escape_limit_, colour_method_, re_, im_, scale_, re_c_, im_c_);
    kernel_params<T> params(image_width_, image_height_, escape_block_, escape_limit_, colour_method_, re_, im_, scale_, re_c_, im_c_);
    params.escape_block_ = (params.escape_block_ > kernel_chunk_perturbation_reference<T>::re_d_) ? 2048 : params.escape_block_;

    if (/*(chunk_buffer_device_ == nullptr) || */(image_device_ == nullptr)) {
        return false;
    }

    std::cout << "  [+] Re: " << std::setprecision(24) << re_ << std::endl
        << "  [+] Im: " << std::setprecision(24) << im_ << std::endl
        << "  [+] Sc: " << std::setprecision(24) << scale_ << std::endl;

    /*if (trial) {
        std::cout << "  [+] Trial: " << trial_image_width_ << "x" << trial_image_height_ << " (" << sizeof(uint32_t) * trial_image_width_ * trial_image_height_ << ")" << std::endl;

        params_trial.escape_range_min_ = 0xffffffffffffffff;
        params_trial.escape_range_max_ = 0;

        if (!dev_perturbation(params_trial, interactive, [&](bool render) {
            if (interactive) {
                uint32_t *trial_image = new uint32_t[trial_image_width_ * trial_image_height_];
                memcpy(trial_image, image_, sizeof(uint32_t) * trial_image_width_ * trial_image_height_);
                for (uint32_t x = 0; x < image_width_; ++x) {
                    for (uint32_t y = 0; y < image_height_; ++y) {
                        image_[x + y * image_width_] = trial_image[x * trial_image_width_ / image_width_ + y * trial_image_height_ / image_height_ * trial_image_width_];
                    }
                }
                delete[] trial_image;
                return callback(render);
            }
            return true;
        })) {
            return false;
        }

        std::unique_ptr<kernel_chunk<T>> preview(new kernel_chunk<T>[trial_image_width_ * trial_image_height_]);
        cudaMemcpy(preview.get(), chunk_buffer_device_, trial_image_width_ * trial_image_height_ * sizeof(kernel_chunk<T>), cudaMemcpyDeviceToHost);
        if (!process_trial(params_trial, params, preview.get())) {
            return false;
        }
    }*/

    std::cout << "  [+] Full Image: " << image_width_ << "x" << image_height_ << " (" << image_size() << " bytes)" << std::endl;

    if (!dev_perturbation(params, interactive, callback)) {
        return false;
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
bool fractal<T>::dev_perturbation(kernel_params<T> &params, bool interactive, std::function<bool(bool)> callback) {
    kernel_params<T> *params_device{ nullptr };
    cudaMalloc(&params_device, sizeof(kernel_params<T>));
    if (params_device == nullptr) {
        return false;
    }

    kernel_chunk_perturbation_reference<T> *chunk_reference_buffer_device_{ nullptr };
    cudaMalloc(&chunk_reference_buffer_device_, grid_resX * grid_resY * sizeof(kernel_chunk_perturbation_reference<T>));
    if (chunk_reference_buffer_device_ == nullptr) {
        return false;
    }

    kernel_chunk_perturbation<T> *chunk_buffer_device_{ nullptr };
    cudaMalloc(&chunk_buffer_device_, cuda_groups_ * cuda_threads_ * sizeof(kernel_chunk_perturbation<T>));
    if (chunk_buffer_device_ == nullptr) {
        return false;
    }

    cudaError_t cudaError{ cudaSuccess };

    uint32_t chunk_count = (params.image_width_ * params.image_height_) / (cuda_groups_ * cuda_threads_);
    if (((params.image_width_ * params.image_height_) % (cuda_groups_ * cuda_threads_)) != 0) {
        chunk_count++;
    }

    if ((cudaError = cudaMemset(chunk_reference_buffer_device_, 0, grid_resX * grid_resY * sizeof(kernel_chunk_perturbation_reference<T>))) != cudaSuccess) {
        std::cout << std::endl << "[!] ERROR: cudaMemset(): " << cudaError << std::endl;
        return false;
    }

    if ((cudaError = cudaMemset(chunk_buffer_device_, 0, cuda_groups_ * cuda_threads_ * sizeof(kernel_chunk_perturbation<T>))) != cudaSuccess) {
        std::cout << std::endl << "[!] ERROR: cudaMemset(): " << cudaError << std::endl;
        return false;
    }

    uint32_t escape_count = static_cast<uint32_t>(escape_limit_ / escape_block_);
    for (uint64_t chunk_i = 0; chunk_i < chunk_count; ++chunk_i) {
        uint32_t chunk_offset = chunk_i * cuda_groups_ * cuda_threads_;
        uint32_t chunk_size = std::min((params.image_width_ * params.image_height_) - chunk_offset, cuda_groups_ * cuda_threads_);
        uint32_t chunk_cuda_groups = chunk_size / cuda_threads_;

        for (uint64_t escape_i = 0; escape_i < escape_count; ++escape_i) {
            std::cout << "                                \r    [+] Chunk: " << chunk_i + 1 << " / "
                << chunk_count
                << ", Block: " << escape_i * escape_block_ << " / " << escape_limit_ << std::flush;

            params.escape_i_ = escape_i;
            params.chunk_offset_ = chunk_offset;

            if ((cudaError = cudaMemcpy(params_device, &params, sizeof(params), cudaMemcpyHostToDevice)) != cudaSuccess) {
                std::cout << std::endl << "[!] ERROR: cudaMemcpy(cudaMemcpyHostToDevice): " << cudaError << std::endl;
                break;
            }

            //if ((cudaError = cudaMemcpy(chunk_buffer_device_,
            //    &chunk_buffer_[chunk_offset],
            //    chunk_size * sizeof(kernel_chunk<T>),
            //    cudaMemcpyHostToDevice)) != cudaSuccess) {
            //    std::cout << std::endl << "[!] ERROR: cudaMemcpy(cudaMemcpyHostToDevice): " << cudaError << std::endl;
            //    break;
            //}

            if (escape_i == 0) {
                kernel_init_mandelbrot_perturbation_reference << <grid_resY, grid_resX >> > (chunk_reference_buffer_device_, params_device);
                if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                    std::cout << std::endl << "[!] ERROR: cudaDeviceSynchronize() [ kernel_init_mandelbrot_perturbation_reference ]: " << cudaError << std::endl;
                    break;
                }

                kernel_init_mandelbrot_perturbation << <static_cast<uint32_t>(chunk_cuda_groups), static_cast<uint32_t>(cuda_threads_) >> > (chunk_reference_buffer_device_, chunk_buffer_device_, params_device);
                if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                    std::cout << std::endl << "[!] ERROR: cudaDeviceSynchronize() [ kernel_init_mandelbrot_perturbation ]: " << cudaError << std::endl;
                    break;
                }
            }

            kernel_iterate_perturbation_reference<<<grid_resY, grid_resX>>>(chunk_reference_buffer_device_, params_device);
            if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                std::cout << std::endl << "[!] ERROR: cudaDeviceSynchronize() [ kernel_iterate_perturbation_reference ]: " << cudaError << std::endl;
                break;
            }

            kernel_iterate_perturbation<<<static_cast<uint32_t>(chunk_cuda_groups), static_cast<uint32_t>(cuda_threads_)>>>(chunk_reference_buffer_device_, chunk_buffer_device_, params_device);
            if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                std::cout << std::endl << "[!] ERROR: cudaDeviceSynchronize() [ kernel_iterate_perturbation ]: " << cudaError << std::endl;
                break;
            }

            if (!callback(params.escape_range_min_ < ((escape_i + 1) * params.escape_block_))) {
                std::cout << std::endl << "    [+] Aborted.";
                break;
            }
        }

        if ((cudaError = cudaMemset(image_device_, 0, cuda_groups_ * cuda_threads_ * sizeof(uint32_t))) != cudaSuccess) {
            std::cout << std::endl << "[!] ERROR: cudaMemset(): " << cudaError << std::endl;
            break;
        }

        kernel_colour<kernel_chunk_perturbation<T>, T><<<static_cast<uint32_t>(chunk_cuda_groups), static_cast<uint32_t>(cuda_threads_)>>>(chunk_buffer_device_, params_device, image_device_);
        if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
            std::cout << std::endl << "[!] ERROR: cudaDeviceSynchronize() [ kernel_colour ]: " << cudaError << std::endl;
            break;
        }

        if ((cudaError = cudaMemcpy(&image_[chunk_offset],
            image_device_,
            chunk_size * sizeof(uint32_t),
            cudaMemcpyDeviceToHost)) != cudaSuccess) {
            std::cout << std::endl << "[!] ERROR: cudaMemcpy(cudaMemcpyDeviceToHost): " << cudaError << std::endl;
            break;
        }

    }

    cudaFree(params.palette_);
    cudaFree(params_device);

    std::cout << std::endl;
    return cudaError == cudaSuccess;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
bool fractal<T>::generate(kernel_params<T> &params, bool interactive, std::function<bool(bool)> callback) {
    if (palette_.size() > 0) {
        cudaMalloc(&params.palette_, 3 * sizeof(double) * palette_.size());

        params.set_hue_ = std::get<0>(palette_.at(0));
        params.set_sat_ = std::get<1>(palette_.at(0));
        params.set_val_ = std::get<2>(palette_.at(0));

        if (params.palette_ != nullptr) {
            params.palette_count_ = static_cast<uint32_t>(palette_.size()) - 1;

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

    kernel_params<T> *params_device{ nullptr };
    cudaMalloc(&params_device, sizeof(kernel_params<T>));
    if (params_device == nullptr) {
        return false;
    }

    kernel_reduce_params *reduce_device{ nullptr };
    cudaMalloc(&reduce_device, cuda_threads_ * sizeof(kernel_reduce_params));
    if (reduce_device == nullptr) {
        return false;
    }

    bool chunk_dirty = true;
    std::map<uint32_t, bool> chunk_status;

    uint32_t chunk_i = 0;
    uint32_t chunk_p = 0;
    uint32_t chunk_count = (params.image_width_ * params.image_height_) / (cuda_groups_ * cuda_threads_);
    if (((params.image_width_ * params.image_height_) % (cuda_groups_ * cuda_threads_)) != 0) {
        chunk_count++;
    }
    uint32_t chunk_count_complete = 0;

    uint32_t escape_i = 0;
    uint32_t escape_count = static_cast<uint32_t>(escape_limit_ / escape_block_);
    if ((escape_limit_ % escape_block_) != 0) {
        escape_count++;
    }

    for (uint32_t i = 0; i < chunk_count; ++i) {
        chunk_status[i] = false;
    }

    auto complete = [&]() {
        return !std::any_of(chunk_status.begin(), chunk_status.end(), [](const std::pair<uint32_t, bool> &v) {
            return !v.second;
        }) || (escape_i >= escape_count);
    };

    auto next = [&]() {
        if (complete()) {
            if (chunk_p++ < chunk_count) {
                chunk_i = (chunk_i + 1) % chunk_count;
                chunk_dirty = true;
                return true;
            }
            return false;
        }

        if (interactive) {
            chunk_i = (chunk_i + 1) % chunk_count;
            if ((chunk_i == 0) && (escape_i++ >= escape_count)) {
                return false;
            }
            chunk_dirty = true;
        }
        else {
            escape_i = chunk_status[chunk_i] ? 0 : (escape_i + 1) % escape_count;
            if (escape_i == 0) {
                if (++chunk_i >= chunk_count) {
                    return false;
                }
                chunk_dirty = true;
            }
        }

        return true;
    };

    cudaError_t cudaError{ cudaSuccess };

    while (true) {
        uint32_t chunk_offset = chunk_i * cuda_groups_ * cuda_threads_;
        uint32_t chunk_size = std::min((params.image_width_ * params.image_height_) - chunk_offset, cuda_groups_ * cuda_threads_);
        uint32_t chunk_cuda_groups = chunk_size / cuda_threads_;

        std::cout << "                                \r    [+] Chunk: " << chunk_i + 1 << " / "
            << chunk_count << " (Done: " << chunk_count_complete << ")"
            << ", Block: " << escape_i * escape_block_ << " / " << escape_limit_ << std::flush;

        if (!chunk_status[chunk_i] || complete()) {
            params.escape_i_ = escape_i;
            params.chunk_offset_ = chunk_offset;

            if ((cudaError = cudaMemcpy(params_device, &params, sizeof(params), cudaMemcpyHostToDevice)) != cudaSuccess) {
                std::cout << std::endl << "[!] ERROR: cudaMemcpy(cudaMemcpyHostToDevice): " << cudaError << std::endl;
                break;
            }

            if (chunk_dirty) {
                if ((cudaError = cudaMemcpy(chunk_buffer_device_,
                        &chunk_buffer_[chunk_offset],
                        chunk_size * sizeof(kernel_chunk<T>),
                        cudaMemcpyHostToDevice)) != cudaSuccess) {
                    std::cout << std::endl << "[!] ERROR: cudaMemcpy(cudaMemcpyHostToDevice): " << cudaError << std::endl;
                    break;
                }
                chunk_dirty = false;
            }

            if (!chunk_status[chunk_i] && ((params.escape_i_ * params.escape_block_) < params.escape_limit_)) {
                if (escape_i == 0) {
                    if (julia_) {
                        kernel_init_julia<<<static_cast<uint32_t>(chunk_cuda_groups), static_cast<uint32_t>(cuda_threads_)>>>(chunk_buffer_device_, params_device);
                    }
                    else {
                        kernel_init_mandelbrot<<<static_cast<uint32_t>(chunk_cuda_groups), static_cast<uint32_t>(cuda_threads_)>>>(chunk_buffer_device_, params_device);
                    }
                }

                kernel_iterate<<<static_cast<uint32_t>(chunk_cuda_groups), static_cast<uint32_t>(cuda_threads_)>>>(chunk_buffer_device_, params_device);

                if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                    std::cout << std::endl << "[!] ERROR: cudaDeviceSynchronize() [ kernel_iterate ]: " << cudaError << std::endl;
                    break;
                }

                if ((cudaError = cudaMemset(reduce_device, 0, sizeof(kernel_reduce_params) * cuda_threads_)) != cudaSuccess) {
                    std::cout << std::endl << "[!] ERROR: cudaDeviceSynchronize() [ kernel_iterate ]: " << cudaError << std::endl;
                    break;
                }

                kernel_reduce<<<1, static_cast<uint32_t>(cuda_threads_)>>>(chunk_buffer_device_, params_device, reduce_device, static_cast<uint32_t>(chunk_cuda_groups));

                if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                    std::cout << std::endl << "[!] ERROR: cudaDeviceSynchronize() [ kernel_reduce ]: " << cudaError << std::endl;
                    break;
                }

                if ((cudaError = cudaMemcpy(&chunk_buffer_[chunk_offset],
                        chunk_buffer_device_,
                        chunk_size * sizeof(kernel_chunk<T>),
                        cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    std::cout << std::endl << "[!] ERROR: cudaMemcpy(cudaMemcpyDeviceToHost): " << cudaError << std::endl;
                    break;
                }

                kernel_reduce_params reduce{ 0 };
                if ((cudaError = cudaMemcpy(&reduce,
                        reduce_device,
                        sizeof(kernel_reduce_params),
                        cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    std::cout << std::endl << "[!] ERROR: cudaMemcpy(cudaMemcpyDeviceToHost): " << cudaError << std::endl;
                    break;
                }

                bool params_update = false;

                if (reduce.escape_reduce_ == (chunk_cuda_groups * cuda_threads_)) {
                    chunk_status[chunk_i] = true;
                    chunk_count_complete++;
                    std::cout << " +C";
                }
                if (reduce.escape_reduce_min_ < params.escape_range_min_) {
                    params.escape_range_min_ = reduce.escape_reduce_min_;
                    std::cout << " < " << params.escape_range_min_;
                    params_update = true;
                }
                if (reduce.escape_reduce_max_ > params.escape_range_max_) {
                    params.escape_range_max_ = reduce.escape_reduce_max_;
                    std::cout << " > " << params.escape_range_max_;
                    params_update = true;
                }

                if (params_update) {
                    if ((cudaError = cudaMemcpy(params_device, &params, sizeof(params), cudaMemcpyHostToDevice)) != cudaSuccess) {
                        std::cout << std::endl << "[!] ERROR: cudaMemcpy(cudaMemcpyHostToDevice): " << cudaError << std::endl;
                        break;
                    }
                }
            }

            if ((cudaError = cudaMemset(image_device_, 0, cuda_groups_ * cuda_threads_ * sizeof(uint32_t))) != cudaSuccess) {
                std::cout << std::endl << "[!] ERROR: cudaMemset(): " << cudaError << std::endl;
                break;
            }

            kernel_colour<kernel_chunk<T>, T><<<static_cast<uint32_t>(chunk_cuda_groups), static_cast<uint32_t>(cuda_threads_)>>>(chunk_buffer_device_, params_device, image_device_);
            if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                std::cout << std::endl << "[!] ERROR: cudaDeviceSynchronize() [ kernel_colour ]: " << cudaError << std::endl;
                break;
            }

            if ((cudaError = cudaMemcpy(&image_[chunk_offset],
                    image_device_,
                    chunk_size * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost)) != cudaSuccess) {
                std::cout << std::endl << "[!] ERROR: cudaMemcpy(cudaMemcpyDeviceToHost): " << cudaError << std::endl;
                break;
            }

            if (!callback(params.escape_range_min_ < ((escape_i + 1) * params.escape_block_))) {
                std::cout << std::endl << "    [+] Aborted.";
                break;
            }
        }

        if (!next()) {
            break;
        }
    }

    cudaFree(params.palette_);
    cudaFree(params_device);
    cudaFree(reduce_device);

    std::cout << std::endl;
    return cudaError == cudaSuccess;
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
bool fractal<T>::process_trial(kernel_params<T> &params_trial, kernel_params<T> &params, kernel_chunk<T> *preview) {
    std::cout << "    [+] Escape Range: " << params_trial.escape_range_min_ << " => " << params_trial.escape_range_max_ << std::endl;

    if (params_trial.escape_range_min_ == params_trial.escape_range_max_) {
        std::cout << "    [+] Complete - Zero Escape Range" << std::endl;
        return false;
    }

    params.escape_range_min_ = params_trial.escape_range_min_;
    params.escape_range_max_ = params_trial.escape_range_max_;

    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Kernels
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_init_julia(kernel_chunk<double> *chunks, kernel_params<double> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    const double pixel_x = (params->chunk_offset_ + tid) % params->image_width_;
    const double pixel_y = (params->chunk_offset_ + tid) / params->image_width_;

    const double re_c = params->re_ + (re_min + pixel_x * (re_max - re_min) / params->image_width_) * params->scale_;
    const double im_c = params->im_ + (im_max - pixel_y * (im_max - im_min) / params->image_height_) * params->scale_;

    kernel_chunk<double> *chunk = &chunks[tid];
    chunk->escape_ = params->escape_limit_;
    chunk->re_c_ = params->re_c_;
    chunk->im_c_ = params->im_c_;
    chunk->re_ = re_c;
    chunk->im_ = im_c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_init_julia(kernel_chunk<fixed_point<I, F>> *chunks, kernel_params<fixed_point<I, F>> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    const double pixel_x = (params->chunk_offset_ + tid) % params->image_width_;
    const double pixel_y = (params->chunk_offset_ + tid) / params->image_width_;

    fixed_point<I, F> re_c(re_min + pixel_x * (re_max - re_min) / params->image_width_);
    fixed_point<I, F> im_c(im_max - pixel_y * (im_max - im_min) / params->image_height_);
    re_c.multiply(params->scale_);
    im_c.multiply(params->scale_);
    re_c.add(params->re_);
    im_c.add(params->im_);

    kernel_chunk<fixed_point<I, F>> *chunk = &chunks[tid];
    chunk->escape_ = params->escape_limit_;
    chunk->re_c_.set(params->re_c_);
    chunk->im_c_.set(params->im_c_);
    chunk->re_.set(re_c);
    chunk->im_.set(im_c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_init_mandelbrot(kernel_chunk<double> *chunks, kernel_params<double> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    const double pixel_x = (params->chunk_offset_ + tid) % params->image_width_;
    const double pixel_y = (params->chunk_offset_ + tid) / params->image_width_;

    const double re_c = params->re_ + (re_min + pixel_x * (re_max - re_min) / params->image_width_) * params->scale_;
    const double im_c = params->im_ + (im_max - pixel_y * (im_max - im_min) / params->image_height_) * params->scale_;

    kernel_chunk<double> *chunk = &chunks[tid];
    chunk->escape_ = params->escape_limit_;
    chunk->re_c_ = re_c;
    chunk->im_c_ = im_c;
    chunk->re_ = re_c;
    chunk->im_ = im_c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_init_mandelbrot(kernel_chunk<fixed_point<I, F>> *chunks, kernel_params<fixed_point<I, F>> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    const double pixel_x = (params->chunk_offset_ + tid) % params->image_width_;
    const double pixel_y = (params->chunk_offset_ + tid) / params->image_width_;

    fixed_point<I, F> re_c(re_min + pixel_x * (re_max - re_min) / params->image_width_);
    fixed_point<I, F> im_c(im_max - pixel_y * (im_max - im_min) / params->image_height_);
    re_c.multiply(params->scale_);
    im_c.multiply(params->scale_);
    re_c.add(params->re_);
    im_c.add(params->im_);

    kernel_chunk<fixed_point<I, F>> *chunk = &chunks[tid];
    chunk->escape_ = params->escape_limit_;
    chunk->re_c_.set(re_c);
    chunk->im_c_.set(im_c);
    chunk->re_.set(re_c);
    chunk->im_.set(im_c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_init_mandelbrot_perturbation(kernel_chunk_perturbation_reference<double> *ref_chunks, kernel_chunk_perturbation<double> *chunks, kernel_params<double> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_init_mandelbrot_perturbation(kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunks, kernel_chunk_perturbation<fixed_point<I, F>> *chunks, kernel_params<fixed_point<I, F>> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    const double pixel_x = (params->chunk_offset_ + tid) % params->image_width_;
    const double pixel_y = (params->chunk_offset_ + tid) / params->image_width_;

    const uint32_t resolution_x = params->image_width_ / grid_resX;
    const uint32_t resolution_y = params->image_height_ / grid_resY;

    const uint32_t ref_pixel_x = pixel_x / resolution_x;
    const uint32_t ref_pixel_y = pixel_y / resolution_y;

    kernel_chunk_perturbation<fixed_point<I, F>> *chunk = &chunks[tid];
    chunk->escape_ = params->escape_limit_; 
    chunk->rid_ = ref_pixel_x + grid_resX * ref_pixel_y;

    kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunk = &ref_chunks[chunk->rid_];

    fixed_point<I, F> re_pert(re_min + pixel_x * (re_max - re_min) / params->image_width_);
    fixed_point<I, F> im_pert(im_max - pixel_y * (im_max - im_min) / params->image_height_);
    re_pert.multiply(params->scale_);
    im_pert.multiply(params->scale_);
    re_pert.add(params->re_);
    im_pert.add(params->im_);

    fixed_point<I, F> re_delta(ref_chunk->re_c_);
    fixed_point<I, F> im_delta(ref_chunk->im_c_);
    re_delta.negate();
    im_delta.negate();
    re_delta.add(re_pert);
    im_delta.add(im_pert);

    chunk->re_delta_0_ = chunk->re_delta_n_ = static_cast<double>(re_delta);
    chunk->im_delta_0_ = chunk->im_delta_n_ = static_cast<double>(im_delta);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_init_mandelbrot_perturbation_reference(kernel_chunk_perturbation_reference<double> *ref_chunks, kernel_params<double> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_init_mandelbrot_perturbation_reference(kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunks, kernel_params<fixed_point<I, F>> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    const uint32_t resolution_x = params->image_width_ / grid_resX;
    const uint32_t resolution_y = params->image_height_ / grid_resY;

    const double pixel_x = (resolution_x / 2) + threadIdx.x * resolution_x;
    const double pixel_y = (resolution_y / 2) + blockIdx.x * resolution_y;

    fixed_point<I, F> re_c(re_min + pixel_x * (re_max - re_min) / params->image_width_);
    fixed_point<I, F> im_c(im_max - pixel_y * (im_max - im_min) / params->image_height_);
    re_c.multiply(params->scale_);
    im_c.multiply(params->scale_);
    re_c.add(params->re_);
    im_c.add(params->im_);

    kernel_chunk_perturbation_reference<fixed_point<I, F>> *chunk = &ref_chunks[tid];
    chunk->escape_ = params->escape_limit_;
    chunk->index_ = 0;
    chunk->re_c_ = re_c;
    chunk->im_c_ = im_c;
    chunk->re_.set(re_c);
    chunk->im_.set(im_c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_iterate(kernel_chunk<double> *chunks, kernel_params<double> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    kernel_chunk<double> *chunk = &chunks[tid];
    double re_c = chunk->re_c_;
    double im_c = chunk->im_c_;
    double re = chunk->re_;
    double im = chunk->im_;
    double re_prod = re * re;
    double im_prod = im * im;

    const uint64_t escape = chunk->escape_;
    const uint64_t escape_block = params->escape_block_;
    const uint64_t escape_limit = params->escape_limit_;

    if (escape == escape_limit) {
        for (uint64_t i = 0; i < escape_block; ++i) {
            im = 2.0 * re * im + im_c;
            re = re_prod - im_prod + re_c;
            re_prod = re * re;
            im_prod = im * im;
            if ((re_prod + im_prod) >= default_escape_radius_square) {
                chunk->escape_ = i + params->escape_i_ * escape_block;
                break;
            }
        }

        chunk->re_ = re;
        chunk->im_ = im;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_iterate(kernel_chunk<fixed_point<I, F>> *chunks, kernel_params<fixed_point<I, F>> *params) {
    kernel_chunk<fixed_point<I, F>> *chunk = &chunks[threadIdx.x + blockIdx.x * blockDim.x];

    fixed_point<I, F> re(chunk->re_);
    fixed_point<I, F> im(chunk->im_);

    const uint64_t escape = chunk->escape_;
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
            im.add(chunk->im_c_);

            re.set(im_prod);
            re.negate();
            re.add(re_prod);
            re.add(chunk->re_c_);

            re.multiply(re, re_prod);
            im.multiply(im, im_prod);

            if ((static_cast<double>(re_prod) + static_cast<double>(im_prod)) >= default_escape_radius_square) {
                chunk->escape_ = i + params->escape_i_ * escape_block;
                break;
            }
        }

        chunk->re_.set(re);
        chunk->im_.set(im);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_iterate_perturbation(kernel_chunk_perturbation_reference<double> *ref_chunks, kernel_chunk_perturbation<double> *chunks, kernel_params<double> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_iterate_perturbation(kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunks, kernel_chunk_perturbation<fixed_point<I, F>> *chunks, kernel_params<fixed_point<I, F>> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    kernel_chunk_perturbation<fixed_point<I, F>> *chunk = &chunks[tid];
    kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunk = &ref_chunks[chunk->rid_]; 

    const uint64_t escape = chunk->escape_;
    const uint64_t escape_block = params->escape_block_;
    const uint64_t escape_limit = params->escape_limit_;

    if (escape == escape_limit) {
        double re_delta_0 = chunk->re_delta_0_;
        double im_delta_0 = chunk->im_delta_0_;
        double re_delta_n = chunk->re_delta_n_;
        double im_delta_n = chunk->im_delta_n_;

        uint64_t ref_index = ref_chunk->index_;

        for (uint64_t i = 0; (i < escape_block) && (i < ref_index); ++i) {
            double re_ref = ref_chunk->re_d_[i];
            double im_ref = ref_chunk->im_d_[i];

            double re_y = re_ref + re_delta_n;
            double im_y = im_ref + im_delta_n;

            if ((re_y * re_y + im_y * im_y) >= default_escape_radius_square) {
                chunk->escape_ = i + params->escape_i_ * escape_block;
                chunk->re_ = re_y;
                chunk->im_ = im_y;
                break;
            }

            double re_delta_n1 = 2 * (re_ref * re_delta_n - im_ref * im_delta_n);
            re_delta_n1 += (re_delta_n * re_delta_n - im_delta_n * im_delta_n);
            re_delta_n1 += re_delta_0;

            double im_delta_n1 = 2 * (im_ref * re_delta_n + re_ref * im_delta_n);
            im_delta_n1 += (2 * re_delta_n * im_delta_n);
            im_delta_n1 += im_delta_0;

            re_delta_n = re_delta_n1;
            im_delta_n = im_delta_n1;
        }

        chunk->re_delta_n_ = re_delta_n;
        chunk->im_delta_n_ = im_delta_n;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_iterate_perturbation_reference(kernel_chunk_perturbation_reference<double> *ref_chunks, kernel_params<double> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_iterate_perturbation_reference(kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunks, kernel_params<fixed_point<I, F>> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    kernel_chunk_perturbation_reference<fixed_point<I, F>> *chunk = &ref_chunks[tid];

    fixed_point<I, F> re(chunk->re_);
    fixed_point<I, F> im(chunk->im_);

    const uint64_t escape = chunk->escape_;
    const uint64_t escape_block = params->escape_block_;
    const uint64_t escape_limit = params->escape_limit_;

    fixed_point<I, F> re_prod;
    fixed_point<I, F> im_prod;
    re.multiply(re, re_prod);
    im.multiply(im, im_prod);

    uint64_t index = 0;

    if (escape == escape_limit) {
        for (uint64_t i = 0; i < escape_block; ++i) {
            im.multiply(re);
            im.multiply(2ULL);
            im.add(chunk->im_c_);

            re.set(im_prod);
            re.negate();
            re.add(re_prod);
            re.add(chunk->re_c_);

            re.multiply(re, re_prod);
            im.multiply(im, im_prod);

            chunk->re_d_[index] = static_cast<double>(re);
            chunk->im_d_[index] = static_cast<double>(im);
            index++;

            if ((static_cast<double>(re_prod) + static_cast<double>(im_prod)) >= default_escape_radius_square) {
                chunk->escape_ = i + params->escape_i_ * escape_block;
                break;
            }
        }

        chunk->index_ = index;
        chunk->re_.set(re);
        chunk->im_.set(im);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename S, typename T>
__global__ void kernel_colour(S *chunks, kernel_params<T> *params, uint32_t *image) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    S *chunk = &chunks[tid];

    double hue = params->set_hue_;
    double sat = params->set_sat_;
    double val = params->set_val_;

    if (chunk->escape_ < params->escape_limit_) {
        double abs = pow(static_cast<double>(chunk->re_), 2.0) + pow(static_cast<double>(chunk->im_), 2.0);
        double escape = static_cast<double>(chunk->escape_) - static_cast<double>(params->escape_range_min_);
        double escape_max = static_cast<double>(1.0 + params->escape_range_max_ - params->escape_range_min_);

        double mu = 1.0 + escape - log2(0.5 * log(abs) / log(default_escape_radius));
        if (mu < 1.0) mu = 1.0;

        switch (13/*params->colour_method_*/) {
        default:
        case 0:
        case 1:
            hue = 0.0;
            sat = 0.95;
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
        case 6:
        case 7:
            {
                if (mu < 2.71828182846) {
                    mu = 2.71828182846;
                }

                double t = log(escape_max) / log(1.0 + mu);
                if (params->colour_method_ < 6) {
                    t = fmod(t, 2.0);
                    if (t > 1.0) {
                        t = 2.0 - t;
                    }
                }
                else {
                    t = fabs(sin(t));
                }

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

        case 8:
        case 9:
            hue = 360.0 * log(mu) / log(escape_max);
            sat = 0.95;
            val = 1.0;
            break;

        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
            if (mu < 2.71828182846) {
                mu = 2.71828182846;
            }
            hue = log(escape_max) / log(1.0 + mu);
            if (params->colour_method_ <= 11) {

            }
            else if (params->colour_method_ <= 13) {
                hue = fmod(hue, 2.0);
                if (hue > 1.0) {
                    hue = 2.0 - hue;
                }
            }
            else {
                hue = fabs(sin(hue));
            }
            hue *= 360;
            sat = 0.95;
            val = 1.0;
            break;
        }

        if (params->colour_method_ % 2 == 1) {
            sat = 0.95 - log(2.0) + log(0.5 * log(abs));
        }
    }

    hue = fmod(hue, 360.0);
    hue /= 60.0;

    double hue_floor = floor(hue);
    double hue_fract = hue - hue_floor;
    double p = val * (1.0 - sat);
    double q = val * (1.0 - sat * hue_fract);
    double t = val * (1.0 - sat * (1.0 - hue_fract));

    double r;
    double g;
    double b;

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
    }

    image[tid] =
        (static_cast<unsigned char>(r * 255.0)) |
        (static_cast<unsigned char>(g * 255.0) << 8) |
        (static_cast<unsigned char>(b * 255.0) << 16) |
        (0xff << 24);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void kernel_reduce(kernel_chunk<T> *chunks, kernel_params<T> *params, kernel_reduce_params *reduce_params, uint32_t chunk_count) {
    kernel_reduce_params *reduce = &reduce_params[threadIdx.x];

    uint64_t escape_reduce = 0;
    uint64_t escape_reduce_min = 0xffffffffffffffff;
    uint64_t escape_reduce_max = 0;

    for (uint32_t i = 0; i < chunk_count; ++i) {
        kernel_chunk<T> *c = &chunks[threadIdx.x + i * blockDim.x];
        if (c->escape_ < params->escape_limit_) {
            escape_reduce++;
        }
        if (c->escape_ < escape_reduce_min) {
            escape_reduce_min = c->escape_;
        }
        if (c->escape_ > escape_reduce_max) {
            escape_reduce_max = c->escape_;
        }
    }

    reduce->escape_reduce_ = escape_reduce;
    reduce->escape_reduce_min_ = escape_reduce_min;
    reduce->escape_reduce_max_ = escape_reduce_max;

    __syncthreads();

    if (threadIdx.x == 0) {
        escape_reduce = 0;
        for (uint32_t i = 0; i < blockDim.x; ++i) {
            escape_reduce += reduce_params[i].escape_reduce_;
            if (reduce_params[i].escape_reduce_min_ < reduce->escape_reduce_min_) {
                reduce->escape_reduce_min_ = reduce_params[i].escape_reduce_min_;
            }
            if (reduce_params[i].escape_reduce_max_ > reduce->escape_reduce_max_) {
                reduce->escape_reduce_max_ = reduce_params[i].escape_reduce_max_;
            }
        }
        reduce->escape_reduce_ = escape_reduce;
    }

    __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
