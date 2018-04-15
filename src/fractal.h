#pragma once

int mandelbrot(uint64_t image_width, uint64_t image_height, uint32_t *image);
int write_png(uint64_t image_width, uint64_t image_height, uint32_t *image);