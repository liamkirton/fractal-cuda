////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Fractal
// (C)2018-22 Liam Kirton <liam@int3.ws>
//
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
#include <thread>
#include <tuple>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "fractal.h"

#include "console.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Init Kernels

__global__ void kernel_init_julia(kernel_chunk<double> *chunks, kernel_params<double> *params);

__global__ void kernel_init_mandelbrot(kernel_chunk<double> *chunks, kernel_params<double> *params);

__global__ void kernel_init_mandelbrot_perturbation(kernel_chunk_perturbation_reference<double> *ref_chunks,
    kernel_chunk<double> *chunks,
    kernel_params<double> *params);

__global__ void kernel_init_mandelbrot_perturbation_reference(kernel_chunk_perturbation_reference<double> *ref_chunks,
    kernel_params<double> *params);

template<uint32_t I, uint32_t F> __global__ void kernel_init_julia(kernel_chunk<fixed_point<I, F>>* chunks,
    kernel_params<fixed_point<I, F>>* params);

template<uint32_t I, uint32_t F> __global__ void kernel_init_mandelbrot(kernel_chunk<fixed_point<I, F>>* chunks,
    kernel_params<fixed_point<I, F>>* params);

template<uint32_t I, uint32_t F> __global__ void kernel_init_mandelbrot_perturbation(
    kernel_chunk_perturbation_reference<fixed_point<I, F>>* ref_chunks,
    kernel_chunk<fixed_point<I, F>>* chunks, kernel_params<fixed_point<I, F>>* params);

// Iterate Kernels

__global__ void kernel_iterate(kernel_chunk<double> *chunks, kernel_params<double> *params);

__global__ void kernel_iterate_perturbation(kernel_chunk_perturbation_reference<double> *ref_chunks,
    kernel_chunk_perturbation *chunks, kernel_params<double> *params);

__global__ void kernel_iterate_perturbation_reference(kernel_chunk_perturbation_reference<double> *ref_chunks,
    kernel_params<double> *params);

template<uint32_t I, uint32_t F> __global__ void kernel_init_mandelbrot_perturbation_reference(
    kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunks,
    kernel_params<fixed_point<I, F>> *params);

template<uint32_t I, uint32_t F> __global__ void kernel_iterate(kernel_chunk<fixed_point<I, F>> *chunks,
    kernel_params<fixed_point<I, F>> *params);

template<uint32_t I, uint32_t F> __global__ void kernel_iterate_perturbation(
    kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunks,
    kernel_chunk_perturbation *chunks, kernel_params<fixed_point<I, F>> *params);

template<uint32_t I, uint32_t F> __global__ void kernel_iterate_perturbation_reference(
    kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunks,
    kernel_params<fixed_point<I, F>> *params);

// Other Kernels

template<typename S, typename T> __global__ void kernel_colour(S *chunks,
    kernel_params<T> *params,
    uint32_t *block_image);

template<typename S, typename T> __global__ void kernel_reduce(S *chunks,
    kernel_params<T> *params,
    uint32_t chunk_count);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host Methods
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
bool fractal<T>::initialise(uint32_t cuda_groups, uint32_t cuda_threads) {
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
    if (image_ != nullptr) {
        delete[] image_;
        image_ = nullptr;
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
void fractal<double>::specify(const double &re, const double &im, const double &scale, bool perturbation) {
    perturbation_ = perturbation;

    re_ = re;
    im_ = im;
    scale_ = scale;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void fractal<T>::specify(const T &re, const T &im, const T &scale, bool perturbation) {
    perturbation_ = perturbation;

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
bool fractal<T>::generate(std::function<bool()> callback) {
    resize(image_width_, image_height_);

    kernel_params<T> params(image_width_,
        image_height_,
        escape_block_,
        escape_limit_,
        colour_method_,
        re_,
        im_,
        scale_,
        re_c_,
        im_c_,
        grid_x_,
        grid_y_);

    if (perturbation_) {
        escape_block_ = params.escape_block_ = (escape_block_ > kKernelChunkPerturbationReferenceBlock) ?
            kKernelChunkPerturbationReferenceBlock : escape_block_;

        if (!generate_perturbation_reference(params, callback)) {
            return false;
        }
    }

    return generate(params, callback);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
bool fractal<T>::generate(kernel_params<T> &params, std::function<bool()> callback) {
    std::cout << "  [+] Re: " << std::setprecision(24) << re_ << std::endl
        << "  [+] Im: " << std::setprecision(24) << im_ << std::endl
        << "  [+] Sc: " << std::setprecision(24) << scale_ << std::endl;

    cudaError_t cudaError{ cudaErrorMemoryAllocation };

    //
    // Host & Device Memory Allocation
    //

    kernel_params<T> *params_device{ nullptr };

    kernel_chunk<T> *chunk_buffer{ nullptr };
    kernel_chunk<T> *chunk_buffer_device{ nullptr };

    kernel_chunk_perturbation *chunk_perturbation_buffer{ nullptr };
    kernel_chunk_perturbation *chunk_perturbation_buffer_device{ nullptr };
    kernel_chunk_perturbation_reference<T> *chunk_perturbation_reference_buffer_device{ nullptr };

    kernel_reduce_params *reduce_device{ nullptr };
    uint32_t *image_device{ nullptr };

    auto cleanup = [&]() {
        if (params_device != nullptr) {
            cudaFree(params_device);
        }
        if (chunk_buffer != nullptr) {
            delete[] chunk_buffer;
        }
        if (chunk_buffer_device != nullptr) {
            cudaFree(chunk_buffer_device);
        }
        if (chunk_perturbation_buffer != nullptr) {
            delete[] chunk_perturbation_buffer;
        }
        if (chunk_perturbation_buffer_device != nullptr) {
            cudaFree(chunk_perturbation_buffer_device);
        }
        if (chunk_perturbation_reference_buffer_device != nullptr) {
            cudaFree(chunk_perturbation_reference_buffer_device);
        }
        if (reduce_device != nullptr) {
            cudaFree(reduce_device);
        }
        if (image_device != nullptr) {
            cudaFree(image_device);
        }
        return cudaError == cudaSuccess;
    };

    cudaMalloc(&params_device, sizeof(kernel_params<T>));
    if (params_device == nullptr) {
        return cleanup();
    }

    if (perturbation_) {
        chunk_perturbation_buffer = new kernel_chunk_perturbation[image_width_ * image_height_];
        std::memset(chunk_perturbation_buffer, 0,
            sizeof(kernel_chunk_perturbation) * image_width_ * image_height_);

        cudaMalloc(&chunk_perturbation_buffer_device,
            cuda_groups_ * cuda_threads_ * sizeof(kernel_chunk_perturbation));
        if ((chunk_perturbation_buffer_device == nullptr) ||
                ((cudaError = cudaMemset(chunk_perturbation_buffer_device, 0,
                    cuda_groups_ * cuda_threads_ * sizeof(kernel_chunk_perturbation))) != cudaSuccess)) {
            return cleanup();
        }

        cudaMalloc(&chunk_perturbation_reference_buffer_device, sizeof(kernel_chunk_perturbation_reference<T>));
        if ((chunk_perturbation_reference_buffer_device == nullptr) ||
                ((cudaError = cudaMemset(chunk_perturbation_reference_buffer_device, 0,
                    sizeof(kernel_chunk_perturbation_reference<T>))) != cudaSuccess)) {
            std::cout << std::endl << "[!] ERROR: cudaMemset(): " << cudaError << std::endl;
            return cleanup();
        }
    }
    else {
        chunk_buffer = new kernel_chunk<T>[image_width_ * image_height_];
        std::memset(chunk_buffer, 0, sizeof(kernel_chunk<T>) * image_width_ * image_height_);

        cudaMalloc(&chunk_buffer_device, cuda_groups_ * cuda_threads_ * sizeof(kernel_chunk<T>));
        if ((chunk_buffer_device == nullptr) ||
            ((cudaError = cudaMemset(chunk_buffer_device, 0,
                cuda_groups_ * cuda_threads_ * sizeof(kernel_chunk<T>))) != cudaSuccess)) {
            return cleanup();
        }
    }

    cudaMalloc(&reduce_device, cuda_threads_ * sizeof(kernel_reduce_params));
    if (reduce_device == nullptr) {
        return cleanup();
    }

    cudaMalloc(&image_device, cuda_groups_ * cuda_threads_ * sizeof(uint32_t));
    if (image_device == nullptr) {
        return cleanup();
    }

    //
    // Setup Palette
    //

    if (palette_.size() > 0) {
        cudaMalloc(&params.palette_device_, 3 * sizeof(double) * palette_.size());

        params.set_hue_ = std::get<0>(palette_.at(0));
        params.set_sat_ = std::get<1>(palette_.at(0));
        params.set_val_ = std::get<2>(palette_.at(0));

        if (params.palette_device_ != nullptr) {
            params.palette_count_ = static_cast<uint32_t>(palette_.size()) - 1;

            double *create_palette = new double[(palette_.size() - 1) * 3];
            for (uint32_t i = 0; i < palette_.size() - 1; ++i) {
                create_palette[i * 3 + 0] = std::get<0>(palette_.at(i + 1));
                create_palette[i * 3 + 1] = std::get<1>(palette_.at(i + 1));
                create_palette[i * 3 + 2] = std::get<2>(palette_.at(i + 1));
            }

            cudaMemcpy(params.palette_device_, create_palette,
                3 * sizeof(double) * (palette_.size() - 1), cudaMemcpyHostToDevice);
            delete[] create_palette;
        }
    }

    //
    // Setup Chunks
    //

    uint32_t chunk_count = (params.image_width_ * params.image_height_) / (cuda_groups_ * cuda_threads_);
    if (((params.image_width_ * params.image_height_) % (cuda_groups_ * cuda_threads_)) != 0) {
        chunk_count++;
    }

    std::map<uint32_t, std::tuple<bool, uint64_t, uint64_t>> chunk_status;
    for (uint32_t i = 0; i < chunk_count; ++i) {
        chunk_status[i] = std::make_tuple(false, 0, escape_limit_);
    }

    uint32_t escape_count = static_cast<uint32_t>(escape_limit_ / escape_block_);
    if ((escape_limit_ % escape_block_) != 0) {
        escape_count++;
    }

    //
    // Generation Loop
    //

    params.escape_range_min_ = 0;
    params.escape_range_max_ = escape_limit_;

    for (uint64_t escape_i = 0; escape_i <= escape_count; ++escape_i) {
        for (uint32_t chunk_i = 0; chunk_i < chunk_count; ++chunk_i) {
            uint32_t chunk_offset = chunk_i * cuda_groups_ * cuda_threads_;
            uint32_t chunk_size = std::min((params.image_width_ * params.image_height_) - chunk_offset,
                cuda_groups_ * cuda_threads_);
            uint32_t chunk_cuda_groups = chunk_size / cuda_threads_;

            std::cout << "\r    [+] Chunk: "
                << chunk_i + 1 << " / " << chunk_count
                << ", Block: " << escape_i * escape_block_ << " / " << escape_limit_
                << (std::get<0>(chunk_status[chunk_i]) ? ", Complete" : "")
                << std::flush;

            params.chunk_offset_ = chunk_offset;
            params.escape_i_ = escape_i;

            if ((cudaError = cudaMemcpy(params_device,
                    &params,
                    sizeof(params), cudaMemcpyHostToDevice)) != cudaSuccess) {
                std::cout << std::endl << "[!] ERROR: cudaMemcpy(cudaMemcpyHostToDevice): " << cudaError << std::endl;
                return cleanup();
            }

            //
            // Perturbation: Chunk 0 => Iterate Reference Point
            //

            if (perturbation_ && (chunk_i == 0)) {
                if (escape_i == 0) {
                    kernel_init_mandelbrot_perturbation_reference<<<1, 1>>>(chunk_perturbation_reference_buffer_device,
                        params_device);

                    if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                        std::cout << std::endl
                            << "[!] ERROR: cudaDeviceSynchronize() [ kernel_init_mandelbrot_perturbation_reference ]: "
                            << cudaError << std::endl;
                        return cleanup();
                    }
                }

                kernel_iterate_perturbation_reference<<<1, 1>>>(chunk_perturbation_reference_buffer_device,
                    params_device);

                if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                    std::cout << std::endl
                        << "[!] ERROR: cudaDeviceSynchronize() [ kernel_iterate_perturbation_reference ]: "
                        << cudaError << std::endl;
                    return cleanup();
                }
            }

            //
            // Setup escape_i : chunk_i Iteration Calculation
            //

            if (perturbation_) {
                if (escape_i == 0) {
                    kernel_init_mandelbrot_perturbation<<<chunk_cuda_groups, cuda_threads_>>>(
                        chunk_perturbation_reference_buffer_device,
                        chunk_perturbation_buffer_device,
                        params_device);

                    if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                        std::cout << std::endl
                            << "[!] ERROR: cudaDeviceSynchronize() [ kernel_init_mandelbrot_perturbation ]: "
                            << cudaError << std::endl;
                        break;
                    }
                }
                else if ((cudaError = cudaMemcpy(chunk_perturbation_buffer_device,
                        &chunk_perturbation_buffer[chunk_offset],
                        chunk_size * sizeof(kernel_chunk_perturbation),
                        cudaMemcpyHostToDevice)) != cudaSuccess) {
                    std::cout << std::endl
                        << "[!] ERROR: cudaMemcpy(cudaMemcpyHostToDevice): "
                        << cudaError << std::endl;
                    break;
                }
            }
            else {
                if (escape_i == 0) {
                    if (julia_) {
                        kernel_init_julia<<<chunk_cuda_groups, cuda_threads_>>>(chunk_buffer_device, params_device);
                    }
                    else {
                        kernel_init_mandelbrot<<<chunk_cuda_groups, cuda_threads_>>>(chunk_buffer_device,
                            params_device);
                    }
                }
                else if ((cudaError = cudaMemcpy(chunk_buffer_device,
                        &chunk_buffer[chunk_offset],
                        chunk_size * sizeof(kernel_chunk<T>),
                        cudaMemcpyHostToDevice)) != cudaSuccess) {
                    std::cout << std::endl
                        << "[!] ERROR: cudaMemcpy(cudaMemcpyHostToDevice): "
                        << cudaError << std::endl;
                    return cleanup();
                }
            }

            if ((chunk_i < chunk_count) && !std::get<0>(chunk_status[chunk_i])) {
                //
                // Core Chunk Iteration
                //

                if (perturbation_) {
                    kernel_iterate_perturbation<<<chunk_cuda_groups, cuda_threads_>>>(
                        chunk_perturbation_reference_buffer_device,
                        chunk_perturbation_buffer_device,
                        params_device);
                }
                else {
                    kernel_iterate<<<chunk_cuda_groups, cuda_threads_>>>(chunk_buffer_device, params_device);
                }

                if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                    std::cout << std::endl
                        << "[!] ERROR: cudaDeviceSynchronize() [ kernel_iterate ]: "
                        << cudaError << std::endl;
                    return cleanup();
                }

                if (perturbation_) {
                    if ((cudaError = cudaMemcpy(&chunk_perturbation_buffer[chunk_offset],
                            chunk_perturbation_buffer_device,
                            chunk_size * sizeof(kernel_chunk_perturbation),
                            cudaMemcpyDeviceToHost)) != cudaSuccess) {
                        std::cout << std::endl
                            << "[!] ERROR: cudaMemcpy(cudaMemcpyDeviceToHost): "
                            << cudaError << std::endl;
                        return cleanup();
                    }
                }
                else {
                    if ((cudaError = cudaMemcpy(&chunk_buffer[chunk_offset],
                            chunk_buffer_device,
                            chunk_size * sizeof(kernel_chunk<T>),
                            cudaMemcpyDeviceToHost)) != cudaSuccess) {
                        std::cout << std::endl
                            << "[!] ERROR: cudaMemcpy(cudaMemcpyDeviceToHost): "
                            << cudaError << std::endl;
                        return cleanup();
                    }
                }

                //
                // Reduce Chunk
                //

                if ((cudaError = cudaMemset(reduce_device, 0,
                        sizeof(kernel_reduce_params) * cuda_threads_)) != cudaSuccess) {
                    std::cout << std::endl
                        << "[!] ERROR: cudaDeviceSynchronize() [ kernel_iterate ]: "
                        << cudaError << std::endl;
                    return cleanup();
                }

                if (perturbation_) {
                    kernel_reduce<<<1, cuda_threads_>>>(chunk_perturbation_buffer_device,
                        params_device,
                        reduce_device,
                        chunk_cuda_groups);
                }
                else {
                    kernel_reduce<<<1, cuda_threads_>>>(chunk_buffer_device,
                        params_device,
                        reduce_device,
                        chunk_cuda_groups);
                }

                if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                    std::cout << std::endl
                        << "[!] ERROR: cudaDeviceSynchronize() [ kernel_reduce ]: "
                        << cudaError << std::endl;
                    return cleanup();
                }

                kernel_reduce_params reduce{ 0 };
                if ((cudaError = cudaMemcpy(&reduce,
                        reduce_device,
                        sizeof(kernel_reduce_params),
                        cudaMemcpyDeviceToHost)) != cudaSuccess) {
                    std::cout << std::endl
                        << "[!] ERROR: cudaMemcpy(cudaMemcpyDeviceToHost): "
                        << cudaError << std::endl;
                    return cleanup();
                }

                if (reduce.escape_reduce_ == (chunk_cuda_groups * cuda_threads_)) {
                    std::get<0>(chunk_status[chunk_i]) = true;
                }
                std::get<1>(chunk_status[chunk_i]) = reduce.escape_reduce_min_;
                std::get<2>(chunk_status[chunk_i]) = reduce.escape_reduce_max_;

                if ((cudaError = cudaMemcpy(params_device, &params, sizeof(params),
                        cudaMemcpyHostToDevice)) != cudaSuccess) {
                    std::cout << std::endl
                        << "[!] ERROR: cudaMemcpy(cudaMemcpyHostToDevice): "
                        << cudaError << std::endl;
                    return cleanup();
                }
            }

            //
            // Recalculate Escape Range
            //

            params.escape_range_min_ = escape_limit_;
            params.escape_range_max_ = 0;

            for (auto& v : chunk_status) {
                auto& escape_range_min = std::get<1>(v.second);
                auto& escape_range_max = std::get<2>(v.second);

                if ((escape_range_min > 0) && (escape_range_min < params.escape_range_min_)) {
                    params.escape_range_min_ = escape_range_min;
                }
                if ((escape_range_max < escape_limit_) && (escape_range_max > params.escape_range_max_)) {
                    params.escape_range_max_ = escape_range_max;
                }
            }

            if (params.escape_range_min_ == escape_limit_) {
                params.escape_range_min_ = 0;
            }
            if (params.escape_range_max_ == 0) {
                params.escape_range_max_ = escape_limit_;
            }

            std::cout << "; Range: " << params.escape_range_min_ << " : "
                << params.escape_range_max_ << std::flush;

            //
            // Colour Chunk
            //

            if ((cudaError = cudaMemset(image_device, 0,
                    cuda_groups_ * cuda_threads_ * sizeof(uint32_t))) != cudaSuccess) {
                std::cout << std::endl << "[!] ERROR: cudaMemset(): " << cudaError << std::endl;
                return cleanup();
            }

            if (perturbation_) {
                kernel_colour<kernel_chunk_perturbation, T><<<chunk_cuda_groups, cuda_threads_>>>(
                    chunk_perturbation_buffer_device,
                    params_device,
                    image_device);
            }
            else {
                kernel_colour<kernel_chunk<T>, T><<<chunk_cuda_groups, cuda_threads_>>>(chunk_buffer_device,
                    params_device,
                    image_device);
            }

            if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                std::cout << std::endl
                    << "[!] ERROR: cudaDeviceSynchronize() [ kernel_colour ]: "
                    << cudaError << std::endl;
                return cleanup();
            }

            if ((cudaError = cudaMemcpy(&image_[chunk_offset],
                    image_device,
                    chunk_size * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost)) != cudaSuccess) {
                std::cout << std::endl
                    << "[!] ERROR: cudaMemcpy(cudaMemcpyDeviceToHost): "
                    << cudaError << std::endl;
                return cleanup();
            }

            std::cout << fill_console_line();

            if (!callback()) {
                std::cout << std::endl << "    [+] Aborted.";
                return cleanup();
            }
            else if ((escape_i < escape_count) &&
                     std::all_of(chunk_status.begin(),
                         chunk_status.end(),
                         [](auto &c) { return std::get<0>(c.second); })) {
                escape_i = (escape_count - 1);
            }
        }
    }

    std::cout << std::endl;
    return cleanup();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
bool fractal<T>::generate_perturbation_reference(kernel_params<T> &params, std::function<bool()> callback) {
    cudaError_t cudaError{ cudaErrorMemoryAllocation };

    //
    // Host & Device Memory Allocation
    //

    kernel_params<T> *params_device{ nullptr };

    kernel_chunk_perturbation_reference<T> *chunk_perturbation_reference_buffer{ nullptr };
    kernel_chunk_perturbation_reference<T> *chunk_perturbation_reference_buffer_device{ nullptr };

    auto cleanup = [&]() {
        if (params_device != nullptr) {
            cudaFree(params_device);
        }
        if (chunk_perturbation_reference_buffer != nullptr) {
            delete[] chunk_perturbation_reference_buffer;
        }
        if (chunk_perturbation_reference_buffer_device != nullptr) {
            cudaFree(chunk_perturbation_reference_buffer_device);
        }
        return cudaError == cudaSuccess;
    };

    cudaMalloc(&params_device, sizeof(kernel_params<T>));
    if (params_device == nullptr) {
        return cleanup();
    }

    chunk_perturbation_reference_buffer = new kernel_chunk_perturbation_reference<T>[grid_x_ * grid_y_];
    std::memset(chunk_perturbation_reference_buffer, 0,
        grid_x_ * grid_y_ * sizeof(kernel_chunk_perturbation_reference<T>));

    cudaMalloc(&chunk_perturbation_reference_buffer_device,
        grid_x_ * grid_y_ * sizeof(kernel_chunk_perturbation_reference<T>));
    if ((chunk_perturbation_reference_buffer_device == nullptr) ||
            ((cudaError = cudaMemset(chunk_perturbation_reference_buffer_device, 0,
                grid_x_ * grid_y_ * sizeof(kernel_chunk_perturbation_reference<T>))) != cudaSuccess)) {
        std::cout << std::endl << "[!] ERROR: cudaMemset(): " << cudaError << std::endl;
        return cleanup();
    }

    //
    // Reference Loop
    //

    uint32_t escape_count = static_cast<uint32_t>(escape_limit_ / kKernelChunkPerturbationReferenceBlock);
    if ((escape_limit_ % escape_block_) != 0) {
        escape_count++;
    }

    kernel_params<T> params_reference = params;

    for (uint32_t level_i = 0; level_i < grid_levels_; ++level_i) {
        std::cout << "\r    [+] Ref Level: " << level_i + 1 << std::flush;

        params_reference.chunk_offset_ = 0;
        params_reference.escape_i_ = 0;

        if ((cudaError = cudaMemcpy(params_device, &params_reference, sizeof(params_reference),
                cudaMemcpyHostToDevice)) != cudaSuccess) {
            std::cout << std::endl << "[!] ERROR: cudaMemcpy(cudaMemcpyHostToDevice): " << cudaError << std::endl;
            return cleanup();
        }

        //
        // Initialise References
        //

        if ((cudaError = cudaMemset(chunk_perturbation_reference_buffer_device, 0,
                grid_x_ * grid_y_ * sizeof(kernel_chunk_perturbation_reference<T>))) != cudaSuccess) {
            std::cout << std::endl << "[!] ERROR: cudaMemset(): " << cudaError << std::endl;
            return cleanup();
        }

        kernel_init_mandelbrot_perturbation_reference<<<grid_y_, grid_x_>>>(chunk_perturbation_reference_buffer_device,
            params_device);

        if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
            std::cout << std::endl
                << "[!] ERROR: cudaDeviceSynchronize() [ kernel_init_mandelbrot_perturbation_reference ]: "
                << cudaError << std::endl;
            return cleanup();
        }

        //
        // Iterate References
        //

        for (uint32_t escape_i = 0; escape_i < escape_count; ++escape_i) {
            params_reference.escape_i_ = escape_i;

            if ((cudaError = cudaMemcpy(params_device, &params_reference, sizeof(params_reference),
                cudaMemcpyHostToDevice)) != cudaSuccess) {
                std::cout << std::endl << "[!] ERROR: cudaMemcpy(cudaMemcpyHostToDevice): " << cudaError << std::endl;
                return cleanup();
            }

            kernel_iterate_perturbation_reference<<<grid_y_, grid_x_>>>(chunk_perturbation_reference_buffer_device,
                params_device);

            if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                std::cout << std::endl
                    << "[!] ERROR: cudaDeviceSynchronize() [ kernel_iterate_perturbation_reference ]: "
                    << cudaError << std::endl;
                return cleanup();
            }
        }

        if ((cudaError = cudaMemcpy(chunk_perturbation_reference_buffer,
                chunk_perturbation_reference_buffer_device,
                grid_x_ * grid_y_ * sizeof(kernel_chunk_perturbation_reference<T>),
                cudaMemcpyDeviceToHost)) != cudaSuccess) {
            std::cout << std::endl << "[!] ERROR: cudaMemcpy(cudaMemcpyDeviceToHost): " << cudaError << std::endl;
            return cleanup();
        }

        //
        // Calculate Best Reference
        //

        double abs_min = DBL_MAX;
        uint64_t escape_max = 0;

        for (uint32_t i = 0; i < grid_x_ * grid_y_; ++i) {
            kernel_chunk_perturbation_reference<T> *ref_chunk = &chunk_perturbation_reference_buffer[i];

            if (ref_chunk->index_ > 0) {
                double ref_abs = ref_chunk->re_d_[ref_chunk->index_ - 1] * ref_chunk->re_d_[ref_chunk->index_ - 1] +
                    ref_chunk->im_d_[ref_chunk->index_ - 1] * ref_chunk->im_d_[ref_chunk->index_ - 1];

                uint64_t ref_escape = ref_chunk->escape_;

                if ((ref_escape > escape_max) || ((ref_escape == escape_max) && (ref_abs < abs_min))) {
                    abs_min = ref_abs;
                    escape_max = ref_escape;

                    params_reference.re_ref_ = ref_chunk->re_c_;
                    params_reference.im_ref_ = ref_chunk->im_c_;
                }
            }
        }

        params_reference.re_ = params_reference.re_ref_;
        params_reference.im_ = params_reference.im_ref_;
        params_reference.scale_ *= 2.0 / (grid_x_ * grid_y_);

        std::cout << ", Esc: " << escape_max << ", Abs: " << abs_min
            << ", Re: " << params_reference.re_ref_ << ", Im: " << params_reference.im_ref_ << std::flush;
        std::cout << fill_console_line() << std::flush;
    }

    params.re_ref_ = params_reference.re_ref_;
    params.im_ref_ = params_reference.im_ref_;
    params.have_ref_ = true;

    std::cout << std::endl;
    return cleanup();
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

    const double re_c = params->re_ + (kReMin + pixel_x * (kReMax - kReMin) / params->image_width_) * params->scale_;
    const double im_c = params->im_ + (kImMax - pixel_y * (kImMax - kImMin) / params->image_height_) * params->scale_;

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

    fixed_point<I, F> re_c(kReMin + pixel_x * (kReMax - kReMin) / params->image_width_);
    fixed_point<I, F> im_c(kImMax - pixel_y * (kImMax - kImMin) / params->image_height_);
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

    const double pixel_shift_x = (params->image_viewport_ - params->image_width_) / 2;
    const double pixel_shift_y = (params->image_viewport_ - params->image_height_) / 2;

    const double re_c = params->re_ +
        (kReMin + (pixel_x + pixel_shift_x) * (kReMax - kReMin) / params->image_viewport_) * params->scale_;

    const double im_c = params->im_ +
        (kImMax - (pixel_y + pixel_shift_y) * (kImMax - kImMin) / params->image_viewport_) * params->scale_;

    kernel_chunk<double> *chunk = &chunks[tid];
    chunk->escape_ = params->escape_limit_;
    chunk->re_c_ = re_c;
    chunk->im_c_ = im_c;
    chunk->re_ = re_c;
    chunk->im_ = im_c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_init_mandelbrot(kernel_chunk<fixed_point<I, F>> *chunks,
        kernel_params<fixed_point<I, F>> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    const double pixel_x = (params->chunk_offset_ + tid) % params->image_width_;
    const double pixel_y = (params->chunk_offset_ + tid) / params->image_width_;

    const double pixel_shift_x = (params->image_viewport_ - params->image_width_) / 2;
    const double pixel_shift_y = (params->image_viewport_ - params->image_height_) / 2;

    fixed_point<I, F> re_c(kReMin + (pixel_x + pixel_shift_x) * (kReMax - kReMin) / params->image_viewport_);
    fixed_point<I, F> im_c(kImMax - (pixel_y + pixel_shift_y) * (kImMax - kImMin) / params->image_viewport_);
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

__global__ void kernel_init_mandelbrot_perturbation(kernel_chunk_perturbation_reference<double> *ref_chunks,
        kernel_chunk_perturbation *chunks, kernel_params<double> *params) {

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_init_mandelbrot_perturbation(kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunks,
        kernel_chunk_perturbation *chunks, kernel_params<fixed_point<I, F>> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    const double pixel_x = (params->chunk_offset_ + tid) % params->image_width_;
    const double pixel_y = (params->chunk_offset_ + tid) / params->image_width_;
    uint32_t ref_ix = 0;

    if (!params->have_ref_) {
        const uint32_t resolution_x = params->image_width_ / params->grid_x_;
        const uint32_t resolution_y = params->image_height_ / params->grid_y_;
        const uint32_t ref_pixel_x = pixel_x / resolution_x;
        const uint32_t ref_pixel_y = pixel_y / resolution_y;
        ref_ix = ref_pixel_y * params->grid_x_ + ref_pixel_x;
    }

    const kernel_chunk_perturbation_reference<fixed_point<I, F>> *const ref_chunk = &ref_chunks[ref_ix];

    fixed_point<I, F> re_pert(kReMin + pixel_x * (kReMax - kReMin) / params->image_width_);
    fixed_point<I, F> im_pert(kImMax - pixel_y * (kImMax - kImMin) / params->image_height_);
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

    kernel_chunk_perturbation *const chunk = &chunks[tid];
    chunk->escape_ = params->escape_limit_;
    chunk->ref_ = ref_ix;
    chunk->re_delta_0_ = chunk->re_delta_n_ = static_cast<double>(re_delta);
    chunk->im_delta_0_ = chunk->im_delta_n_ = static_cast<double>(im_delta);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_init_mandelbrot_perturbation_reference(kernel_chunk_perturbation_reference<double> *ref_chunks,
        kernel_params<double> *params) {

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_init_mandelbrot_perturbation_reference(
        kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunks,
        kernel_params<fixed_point<I, F>> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    fixed_point<I, F> re_c;
    fixed_point<I, F> im_c;

    if (!params->have_ref_) {
        const uint32_t resolution_x = params->image_width_ / params->grid_x_;
        const uint32_t resolution_y = params->image_height_ / params->grid_y_;
        const double pixel_x = (resolution_x / 2) + threadIdx.x * resolution_x;
        const double pixel_y = (resolution_y / 2) + blockIdx.x * resolution_y;

        re_c.set(kReMin + pixel_x * (kReMax - kReMin) / params->image_width_);
        im_c.set(kImMax - pixel_y * (kImMax - kImMin) / params->image_height_);
        re_c.multiply(params->scale_);
        im_c.multiply(params->scale_);
        re_c.add(params->re_);
        im_c.add(params->im_);
    }
    else {
        re_c.set(params->re_ref_);
        im_c.set(params->im_ref_);
    }

    kernel_chunk_perturbation_reference<fixed_point<I, F>> *const chunk = &ref_chunks[tid];
    chunk->escape_ = params->escape_limit_;
    chunk->index_ = 0;
    chunk->re_c_.set(re_c);
    chunk->im_c_.set(im_c);
    chunk->re_.set(re_c);
    chunk->im_.set(im_c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_iterate(kernel_chunk<double> *chunks, kernel_params<double> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    kernel_chunk<double> *const chunk = &chunks[tid];
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
            if ((re_prod + im_prod) >= kDefaultEscapeRadiusSquared) {
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
    kernel_chunk<fixed_point<I, F>> *const chunk = &chunks[threadIdx.x + blockIdx.x * blockDim.x];

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

            if ((static_cast<double>(re_prod) + static_cast<double>(im_prod)) >= kDefaultEscapeRadiusSquared) {
                chunk->escape_ = i + params->escape_i_ * escape_block;
                break;
            }
        }

        chunk->re_.set(re);
        chunk->im_.set(im);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void kernel_iterate_perturbation(kernel_chunk_perturbation_reference<double> *ref_chunks,
        kernel_chunk_perturbation *chunks, kernel_params<double> *params) {

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_iterate_perturbation(kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunks,
        kernel_chunk_perturbation *chunks, kernel_params<fixed_point<I, F>> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    kernel_chunk_perturbation *const chunk = &chunks[tid];
    kernel_chunk_perturbation_reference<fixed_point<I, F>> *const ref_chunk = &ref_chunks[chunk->ref_];

    const uint64_t escape = chunk->escape_;
    const uint64_t escape_block = params->escape_block_;
    const uint64_t escape_limit = params->escape_limit_;

    if (escape == escape_limit) {
        const double re_delta_0 = chunk->re_delta_0_;
        const double im_delta_0 = chunk->im_delta_0_;
        double re_delta_n = chunk->re_delta_n_;
        double im_delta_n = chunk->im_delta_n_;

        const uint64_t ref_index = ref_chunk->index_;

        for (uint64_t i = 0; (i < escape_block) && (i < ref_index); ++i) {
            const double re_ref = ref_chunk->re_d_[i];
            const double im_ref = ref_chunk->im_d_[i];

            const double re_y = re_ref + re_delta_n;
            const double im_y = im_ref + im_delta_n;

            if ((re_y * re_y + im_y * im_y) >= kDefaultEscapeRadiusSquared) {
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

__global__ void kernel_iterate_perturbation_reference(kernel_chunk_perturbation_reference<double> *ref_chunks,
        kernel_params<double> *params) {

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
__global__ void kernel_iterate_perturbation_reference(
        kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunks,
        kernel_params<fixed_point<I, F>> *params) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    kernel_chunk_perturbation_reference<fixed_point<I, F>> *ref_chunk = &ref_chunks[tid];

    fixed_point<I, F> re(ref_chunk->re_);
    fixed_point<I, F> im(ref_chunk->im_);

    const uint64_t escape = ref_chunk->escape_;
    const uint64_t escape_block = params->escape_block_;
    const uint64_t escape_limit = params->escape_limit_;

    if (escape == escape_limit) {
        uint64_t index = ref_chunk->index_ = 0;

        fixed_point<I, F> re_prod;
        fixed_point<I, F> im_prod;
        re.multiply(re, re_prod);
        im.multiply(im, im_prod);

        for (uint64_t i = 0; i < escape_block; ++i) {
            im.multiply(re);
            im.multiply(2ULL);
            im.add(ref_chunk->im_c_);

            re.set(im_prod);
            re.negate();
            re.add(re_prod);
            re.add(ref_chunk->re_c_);

            re.multiply(re, re_prod);
            im.multiply(im, im_prod);

            ref_chunk->re_d_[index] = static_cast<double>(re);
            ref_chunk->im_d_[index] = static_cast<double>(im);
            index++;

            if ((static_cast<double>(re_prod) + static_cast<double>(im_prod)) >= kDefaultEscapeRadiusSquared) {
                ref_chunk->escape_ = i + params->escape_i_ * escape_block;
                break;
            }
        }

        ref_chunk->index_ = index;
        ref_chunk->re_.set(re);
        ref_chunk->im_.set(im);
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

        double mu = 1.0 + escape - log2(0.5 * log(abs) / log(kDefaultEscapeRadius));
        if (mu < 1.0) mu = 1.0;

        switch (params->colour_method_) {
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
                    double poly = pow(t, static_cast<double>(i)) *
                        pow(1.0 - t, static_cast<double>(params->palette_count_ - 1 - i));
                    hue += params->palette_device_[3 * i + 0] * poly;
                    sat += params->palette_device_[3 * i + 1] * poly;
                    val += params->palette_device_[3 * i + 2] * poly;
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
                    double poly = pow(t, static_cast<double>(i)) *
                        pow(1.0 - t, static_cast<double>(params->palette_count_ - 1 - i));
                    hue += params->palette_device_[3 * i + 0] * poly;
                    sat += params->palette_device_[3 * i + 1] * poly;
                    val += params->palette_device_[3 * i + 2] * poly;
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
            if ((params->colour_method_ >= 12) && (params->colour_method_ <= 13)) {
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

template<typename S, typename T>
__global__ void kernel_reduce(S *chunks, kernel_params<T> *params, kernel_reduce_params *reduce_params,
        uint32_t chunk_count) {
    kernel_reduce_params *reduce = &reduce_params[threadIdx.x];

    uint64_t escape_reduce = 0;
    uint64_t escape_reduce_min = params->escape_limit_;
    uint64_t escape_reduce_max = 0;

    for (uint32_t i = 0; i < chunk_count; ++i) {
        S *c = &chunks[threadIdx.x + i * blockDim.x];
        if (c->escape_ < params->escape_limit_) {
            escape_reduce++;
        }
        if (c->escape_ < escape_reduce_min) {
            escape_reduce_min = c->escape_;
        }
        if ((c->escape_ < params->escape_limit_) && (c->escape_ > escape_reduce_max)) {
            escape_reduce_max = c->escape_;
        }
    }

    reduce->escape_reduce_ = escape_reduce;
    reduce->escape_reduce_min_ = (escape_reduce_min != params->escape_limit_) ? escape_reduce_min : 0;
    reduce->escape_reduce_max_ = (escape_reduce_max != 0) ? escape_reduce_max : params->escape_limit_;

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

template class fractal<double>;
template class fractal<fixed_point<1, 2>>;
template class fractal<fixed_point<1, 3>>;
#ifndef _DEBUG
template class fractal<fixed_point<1, 4>>;
template class fractal<fixed_point<1, 6>>;
template class fractal<fixed_point<1, 8>>;
template class fractal<fixed_point<1, 12>>;
template class fractal<fixed_point<1, 16>>;
template class fractal<fixed_point<1, 20>>;
template class fractal<fixed_point<1, 24>>;
template class fractal<fixed_point<1, 28>>;
template class fractal<fixed_point<1, 32>>;
template class fractal<fixed_point<1, 48>>;
template class fractal<fixed_point<1, 56>>;
template class fractal<fixed_point<1, 64>>;
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
