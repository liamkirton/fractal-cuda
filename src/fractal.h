#pragma once

int init(uint64_t image_width, uint64_t image_height);
int uninit(uint64_t image_width, uint64_t image_height);

int mandelbrot(uint32_t *image, uint64_t image_width, uint64_t image_height, const double image_center_re, const double image_center_im, double image_scale);
int write_png(uint32_t *image, uint64_t image_width, uint64_t image_height, std::string &name);