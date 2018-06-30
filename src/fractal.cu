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

constexpr uint32_t escape_block = 128;
constexpr uint32_t escape_limit = 128;

__global__ void mandelbrot_kernel_d(s_d *chunk_buffer, s_d_param *t, const uint64_t image_chunk, const uint32_t escape_i);

template<uint32_t I, uint32_t F>
__global__ void mandelbrot_kernel_fp(s_fp<I, F> *chunk_buffer, s_fp_param<I, F> *t, const uint64_t image_chunk, const uint32_t escape_i);

template<typename T>
__global__ void mandelbrot_kernel_colour(T *chunk_buffer, uint32_t *image_chunk_buffer, const uint64_t image_width, const uint64_t image_height, const uint64_t image_chunk);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Host Methods
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
bool fractal<I, F>::initialise() {
    uninitialise();
    cudaError_t cuda_status;
    if ((cuda_status = cudaSetDevice(0)) != cudaSuccess) {
        std::wcout << L"ERROR: cudaSetDevice() Failed. [" << cuda_status << L"]" << std::endl << std::endl;
        return false;
    }
    cudaMalloc(&device_chunk_image_, cuda_groups_ * cuda_threads_ * sizeof(uint32_t));
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
void fractal<I, F>::uninitialise() {
    cudaDeviceReset();
    if (device_chunk_d_ != nullptr) {
        cudaFree(device_chunk_d_);
        device_chunk_d_ = nullptr;
    }
    if (device_chunk_fp_ != nullptr) {
        cudaFree(device_chunk_fp_);
        device_chunk_fp_ = nullptr;
    }
    if (device_chunk_image_ != nullptr) {
        cudaFree(device_chunk_image_);
        device_chunk_image_ = nullptr;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
void fractal<I, F>::resize(const uint64_t image_width, const uint64_t image_height) {
    image_width_ = image_width;
    image_height_ = image_height;
    if (image_ != nullptr) {
        delete[] image_;
    }
    image_ = new uint32_t[image_width_ * image_height_];
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
bool fractal<I, F>::generate(bool fixed_point) {
    cudaError_t cudaError;

    s_d_param *dp{ nullptr };
    s_fp_param<I, F> *dfp{ nullptr };

    if (fixed_point) {
        if (device_chunk_fp_ == nullptr) {
            cudaMalloc(&device_chunk_fp_, cuda_groups_ * cuda_threads_ * sizeof(s_fp<I, F>));
            if (device_chunk_fp_ == nullptr) {
                return false;
            }
        }
        s_fp_param<I, F> t;
        t.image_width_ = image_width_;
        t.image_height_ = image_height_;
        t.re_.set(re_fp_);
        t.im_.set(im_fp_);
        t.scale_.set(scale_fp_);
        cudaMalloc(&dfp, sizeof(s_fp_param<I, F>));
        cudaMemcpy(dfp, &t, sizeof(t), cudaMemcpyHostToDevice);
    }
    else {
        if (device_chunk_d_ == nullptr) {
            cudaMalloc(&device_chunk_d_, cuda_groups_ * cuda_threads_ * sizeof(s_d));
            if (device_chunk_d_ == nullptr) {
                return false;
            }
        }
        s_d_param t;
        t.image_width_ = image_width_;
        t.image_height_ = image_height_;
        t.re_ = re_d_;
        t.im_ = im_d_;
        t.scale_ = scale_d_;
        cudaMalloc(&dp, sizeof(s_d_param));
        cudaMemcpy(dp, &t, sizeof(t), cudaMemcpyHostToDevice);
    }

    std::wcout << L"[+] Chunks: "
        << 1 + image_width_ * image_height_ / (cuda_groups_ * cuda_threads_)
        << L" " << std::flush;

    for (uint64_t image_chunk = 0; image_chunk < (image_width_ * image_height_); image_chunk += (cuda_groups_ * cuda_threads_)) {
        uint64_t chunk_size = std::min((image_width_ * image_height_) - image_chunk, cuda_groups_ * cuda_threads_);
        uint64_t chunk_groups = chunk_size / cuda_threads_;
        if (fixed_point) {
            cudaMemset(device_chunk_fp_, 0, cuda_groups_ * cuda_threads_ * sizeof(s_fp<I, F>));
        }
        else {
            cudaMemset(device_chunk_d_, 0, cuda_groups_ * cuda_threads_ * sizeof(s_d));
        }

        std::wcout << L"+" << std::flush;

        for (uint32_t i = 0; i < (escape_limit / escape_block); ++i) {
            if (fixed_point) {
                mandelbrot_kernel_fp<I, F><<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(cuda_threads_)>>>(device_chunk_fp_, dfp, image_chunk, i);
            }
            else {
                mandelbrot_kernel_d<<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(cuda_threads_) >>> (device_chunk_d_, dp, image_chunk, i);
            }
            if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
                std::wcout << std::endl << "[!] cudaDeviceSynchronize(): cudaError: " << cudaError << std::endl;
                break;
            }
        }

        if (cudaError != cudaSuccess) {
            break;
        }

        cudaMemset(device_chunk_image_, 0, cuda_groups_ * cuda_threads_ * sizeof(uint32_t));
        if (fixed_point) {
            mandelbrot_kernel_colour<s_fp<I, F>><<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(cuda_threads_)>>>(device_chunk_fp_, device_chunk_image_, image_width_, image_height_, image_chunk);
        }
        else {
            mandelbrot_kernel_colour<s_d><<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(cuda_threads_)>>>(device_chunk_d_, device_chunk_image_, image_width_, image_height_, image_chunk);
        }

        if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
            std::wcout << std::endl << "[!] cudaDeviceSynchronize(): cudaError: " << cudaError << std::endl;
            break;
        }

        cudaMemcpy(&image_[image_chunk], device_chunk_image_, chunk_groups * cuda_threads_ * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    std::wcout << std::endl;
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Kernels
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void mandelbrot_kernel_d(s_d *chunk_buffer, s_d_param *t, const uint64_t image_chunk, const uint32_t escape_i) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int pixel_x = (image_chunk + tid) % t->image_width_;
    const int pixel_y = (image_chunk + tid) / t->image_width_;

    const double re_c = t->re_ + (-2.0 + pixel_x * 3.0 / t->image_width_) * t->scale_;
    const double im_c = t->im_ + (1.0 - pixel_y * 2.0 / t->image_height_) * t->scale_;

    s_d *b = &chunk_buffer[tid];
    uint32_t escape = b->escape;
    double re_z = b->re_;
    double im_z = b->im_;
    double abs_z = 0.0;

    if (escape_i == 0) {
        escape = escape_limit;
        re_z = re_c;
        im_z = im_c;
    }

    if (escape == escape_limit) {
        for (uint32_t i = 0; i < escape_block; ++i) {
            double re_z_i = re_z;
            re_z = (re_z * re_z) - (im_z * im_z) + re_c;
            im_z = (2.0 * re_z_i * im_z) + im_c;
            abs_z = re_z * re_z + im_z * im_z;
            if (abs_z > 4.0) {
                escape = i + escape_i * escape_block;
                break;
            }
        }
    }

    b->escape = escape;
    b->re_ = re_z;
    b->im_ = im_z;
}

template<uint32_t I, uint32_t F>
__global__ void mandelbrot_kernel_fp(s_fp<I, F> *chunk_buffer, s_fp_param<I, F> *t, const uint64_t image_chunk, const uint32_t escape_i) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int pixel_x = (image_chunk + tid) % t->image_width_;
    const int pixel_y = (image_chunk + tid) / t->image_width_;

    fixed_point<I, F> re_c(-2.0 + pixel_x * 3.0 / t->image_width_);
    re_c.multiply(t->scale_);
    re_c.add(t->re_);
    fixed_point<I, F> im_c(1.0 - pixel_y * 2.0 / t->image_height_);
    im_c.multiply(t->scale_);
    im_c.add(t->im_);

    s_fp<I, F> *b = &chunk_buffer[tid];
    uint32_t escape = b->escape;

    fixed_point<I, F> re_z(b->re_);
    fixed_point<I, F> im_z(b->im_);

    if (escape_i == 0) {
        escape = escape_limit;
        re_z.set(re_c);
        im_z.set(im_c);
    }

    fixed_point<I, F> re_prod(re_z);
    re_prod.multiply(re_z);

    fixed_point<I, F> im_prod(im_z);
    im_prod.multiply(im_z);

    fixed_point<I, F> re_imed;
    fixed_point<I, F> im_imed;

    if (escape == escape_limit) {
        for (uint32_t i = 0; i < escape_block; ++i) {
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
                escape = i + escape_i * escape_block;
                break;
            }
        }
    }

    b->escape = escape;
    b->re_.set(re_z);
    b->im_.set(im_z);
}

template<typename T>
__global__ void mandelbrot_kernel_colour(T *chunk_buffer, uint32_t *image_chunk_buffer, const uint64_t image_width, const uint64_t image_height, const uint64_t image_chunk) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t escape = chunk_buffer[tid].escape;
    double re_z = chunk_buffer[tid].re();
    double im_z = chunk_buffer[tid].im();
    double abs_z = sqrtf(re_z * re_z + im_z * im_z);

    double hue = 360.0 * log(1.0 * escape) / log(1.0 * escape_limit) + 1.0 - (log(log(abs_z)) / log(2.0)); 
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

    if (escape < escape_limit) {
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

    image_chunk_buffer[tid] =
        (static_cast<unsigned char>(r)) |
        (static_cast<unsigned char>(g) << 8) |
        (static_cast<unsigned char>(b) << 16) |
        (255 << 24);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//constexpr uint32_t escape_block = 4096;
//constexpr uint32_t escape_limit = 4096;
//
//__global__ void mandelbrot_kernel(uint64_t *chunk_buffer, const uint64_t image_width, const uint64_t image_height, const double image_re, const double image_im, const double image_scale, const uint64_t image_chunk, const uint32_t escape_i) {
//    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
//    const int pixel_x = (image_chunk + tid) % image_width;
//    const int pixel_y = (image_chunk + tid) / image_width;
//
//    const double re_c = image_re + (-2.0 + pixel_x * 3.0 / image_width) / image_scale;
//    const double im_c = image_im + (1.0 - pixel_y * 2.0 / image_height) / image_scale;
//
//    uint32_t escape = static_cast<uint32_t>(chunk_buffer[tid * 3]);
//    double re_z = reinterpret_cast<double *>(chunk_buffer)[tid * 3 + 1];
//    double im_z = reinterpret_cast<double *>(chunk_buffer)[tid * 3 + 2];
//    double abs_z = 0.0;
//
//    if (escape_i == 0) {
//        escape = escape_limit;
//        re_z = re_c;
//        im_z = im_c;
//    }
//
//    if (escape == escape_limit) {
//        for (uint32_t i = 0; i < escape_block; ++i) {
//            double re_z_i = re_z;
//            re_z = (re_z * re_z) - (im_z * im_z) + re_c;
//            im_z = (2.0 * re_z_i * im_z) + im_c;
//            abs_z = re_z * re_z + im_z * im_z;
//            if (abs_z > 4.0) {
//                escape = i + escape_i * escape_block;
//                break;
//            }
//        }
//    }
//
//    chunk_buffer[tid * 3] = escape;
//    reinterpret_cast<double *>(chunk_buffer)[tid * 3 + 1] = re_z;
//    reinterpret_cast<double *>(chunk_buffer)[tid * 3 + 2] = im_z;
//}
//
//template<uint32_t I, uint32_t F>
//__global__ void mandelbrot_kernel_fp(uint64_t *chunk_buffer, const uint64_t image_width, const uint64_t image_height, const double image_re, const double image_im, const double image_scale, const uint64_t image_chunk, const uint32_t escape_i) {
//    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
//    const int pixel_x = (image_chunk + tid) % image_width;
//    const int pixel_y = (image_chunk + tid) / image_width;
//
//    fixed_point<I, F> i_scale(0.5);
//    for (uint32_t i = 0; i < 12; ++i) {
//        i_scale.multiply(0.5);
//    }
//
//    fixed_point<I, F> re_c(-2.0 + pixel_x * 3.0 / image_width);
//    re_c.multiply(i_scale);
//    re_c.add(image_re);
//    fixed_point<I, F> im_c(1.0 - pixel_y * 2.0 / image_height);
//    im_c.multiply(i_scale);
//    im_c.add(image_im);
//
//    uint32_t escape = static_cast<uint32_t>(chunk_buffer[tid * 3]);
//
//    fixed_point<I, F> re_z; // = prev
//    fixed_point<I, F> im_z; // = prev
//
//    if (escape_i == 0) {
//        escape = escape_limit;
//        re_z.set(re_c);
//        im_z.set(im_c);
//    }
//
//    fixed_point<I, F> re_prod(re_z);
//    re_prod.multiply(re_z);
//
//    fixed_point<I, F> im_prod(im_z);
//    im_prod.multiply(im_z);
//
//    fixed_point<I, F> re_imed;
//    fixed_point<I, F> im_imed;
//
//    if (escape == escape_limit) {
//        for (uint32_t i = 0; i < escape_block; ++i) {
//            re_imed.set(im_prod);
//            re_imed.negate();
//            re_imed.add(re_prod);
//            re_imed.add(re_c);
//
//            im_imed.set(2);
//            im_imed.multiply(re_z);
//            im_imed.multiply(im_z);
//            im_imed.add(im_c);
//
//            re_z.set(re_imed);
//            im_z.set(im_imed);
//
//            re_prod.set(re_z);
//            re_prod.multiply(re_z);
//
//            im_prod.set(im_z);
//            im_prod.multiply(im_z);
//
//            if (re_prod.get_integer() + im_prod.get_integer() > 4) {
//                escape = i + escape_i * escape_block;
//                break;
//            }
//        }
//    }
//
//    chunk_buffer[tid * 3] = escape;
//    reinterpret_cast<double *>(chunk_buffer)[tid * 3 + 1] = static_cast<double>(1.0 * re_z.get_integer());
//    reinterpret_cast<double *>(chunk_buffer)[tid * 3 + 2] = static_cast<double>(1.0 * im_z.get_integer());
//}
//
//__global__ void mandelbrot_kernel_colour(uint64_t *chunk_buffer, uint32_t *image_chunk_buffer, const uint64_t image_width, const uint64_t image_height, const uint64_t image_chunk) {
//    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
//
//    uint32_t escape = chunk_buffer[tid * 3];
//    double re_z = reinterpret_cast<double *>(chunk_buffer)[tid * 3 + 1];
//    double im_z = reinterpret_cast<double *>(chunk_buffer)[tid * 3 + 2];
//    double abs_z = sqrtf(re_z * re_z + im_z * im_z);
//
//    double hue = 360.0 * log(1.0 * escape) / log(1.0 * escape_limit) + 1.0 - (log(log(abs_z)) / log(2.0)); 
//    //double hue = 360.0 * (log(1.0 * escape) - log(log(abs_z))) / (log(1.0 * escape_limit) + log(2.0));
//    double sat = 0.85;
//    double val = 1.0;
//
//    hue = fmod(hue, 360.0);
//    hue /= 60;
//
//    double hue_fract = hue - floor(hue);
//    double p = val * (1.0 - sat);
//    double q = val * (1.0 - sat * hue_fract);
//    double t = val * (1.0 - sat * (1.0 - hue_fract));
//
//    double r = 0;
//    double g = 0;
//    double b = 0;
//
//    if (escape < escape_limit) {
//        switch ((static_cast<unsigned char>(floor(hue)) + 3) % 6) {
//        case 0:
//            r = val; g = t; b = p;
//            break;
//        case 1:
//            r = q; g = val; b = p;
//            break;
//        case 2:
//            r = p; g = val; b = t;
//            break;
//        case 3:
//            r = p; g = q; b = val;
//            break;
//        case 4:
//            r = t; g = p; b = val;
//            break;
//        case 5:
//            r = val; g = p; b = q;
//            break;
//        default:
//            break;
//        }
//        r = floor(r * 255); g = floor(g * 255); b = floor(b * 255);
//    }
//
//    image_chunk_buffer[tid] =
//        (static_cast<unsigned char>(r)) |
//        (static_cast<unsigned char>(g) << 8) |
//        (static_cast<unsigned char>(b) << 16) |
//        (255 << 24);
//}
//
//uint64_t *chunk_buffer{ nullptr };
//uint32_t *chunk_buffer_image{ nullptr };
//
//int init(uint64_t image_width, uint64_t image_height) {
//    cudaError_t cuda_status;
//    if ((cuda_status = cudaSetDevice(0)) != cudaSuccess) {
//        std::wcout << L"ERROR: cudaSetDevice() Failed. [" << cuda_status << L"]" << std::endl << std::endl;
//        return -1;
//    }
//
//    cudaMalloc(&chunk_buffer, groups * threads * 3 * sizeof(uint64_t));
//    cudaMalloc(&chunk_buffer_image, groups * threads * sizeof(uint32_t));
//
//    return 0;
//}
//
//int uninit(uint64_t image_width, uint64_t image_height) {
//    cudaFree(chunk_buffer);
//    cudaFree(chunk_buffer_image);
//    chunk_buffer = nullptr;
//    chunk_buffer_image = nullptr;
//
//    cudaError_t cuda_status;
//    if ((cuda_status = cudaDeviceReset()) != cudaSuccess) {
//        std::wcout << L"ERROR: cudaDeviceReset() Failed. [" << cuda_status << L"]" << std::endl << std::endl;
//        return -1;
//    }
//
//    return 0;
//}
//
//int mandelbrot(uint32_t *image, const uint64_t image_width, const uint64_t image_height, const double image_center_re, const double image_center_im, const double image_scale) {
//    cudaError_t cudaError;
//
//    std::wcout << L"[+] Chunks: "
//        << 1 + image_width * image_height / (groups * threads)
//        << L" " << std::flush;
//
//    for (uint64_t image_chunk = 0; image_chunk < (image_width * image_height); image_chunk += (groups * threads)) {
//        uint64_t chunk_size = std::min((image_width * image_height) - image_chunk, groups * threads);
//        uint64_t chunk_groups = chunk_size / threads;
//        cudaMemset(chunk_buffer, 0, groups * threads * 5 * sizeof(uint64_t));
//        cudaMemset(chunk_buffer_image, 0, groups * threads * sizeof(uint32_t));
//
//        std::wcout << L"+" << std::flush;
//
//        for (uint32_t i = 0; i < (escape_limit / escape_block); ++i) {
//            mandelbrot_kernel_fp<2, 4><<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(threads)>>>(chunk_buffer, image_width, image_height, image_center_re, image_center_im, image_scale, image_chunk, i);
//            //mandelbrot_kernel<<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(threads)>>>(chunk_buffer, image_width, image_height, image_center_re, image_center_im, image_scale, image_chunk, i);
//            if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
//                std::wcout << std::endl << "[!] cudaDeviceSynchronize(): cudaError: " << cudaError << std::endl;
//                return -1;
//            }
//        }
//
//        mandelbrot_kernel_colour<<<static_cast<uint32_t>(chunk_groups), static_cast<uint32_t>(threads)>>>(chunk_buffer, chunk_buffer_image, image_width, image_height, image_chunk);
//        if ((cudaError = cudaDeviceSynchronize()) != cudaSuccess) {
//            std::wcout << std::endl << "[!] cudaDeviceSynchronize(): cudaError: " << cudaError << std::endl;
//            return -1;
//        }
//
//        cudaMemcpy(&image[image_chunk], chunk_buffer_image, chunk_groups * threads * sizeof(uint32_t), cudaMemcpyDeviceToHost);
//    }
//
//    std::wcout << std::endl;
//
//    return 0;
//}