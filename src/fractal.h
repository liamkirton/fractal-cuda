#pragma once

int init(uint64_t image_width, uint64_t image_height);
int uninit(uint64_t image_width, uint64_t image_height);

int mandelbrot(uint64_t image_width, uint64_t image_height, double image_scale, uint32_t *image);
int write_png(uint64_t image_width, uint64_t image_height, uint32_t *image, std::string &name);