# CUDA Fractal Generator

![alt text](doc/img/mandelbrot.png)

## Prerequisites

* Visual Studio 2019
* CUDA Toolkit 11.4

## Building Dependencies

1. Initialise git submodules
2. Open third_party/zlib/contrib/vstudio/vc14/zlibvc.sln
3. Build "Release" for "x64"
4. Open third_party/libpng/projects/vstudio.sln
5. Build "Release" for "x64"

## Building

1. Open fractal.sln
2. Build "Release" for "x64"

## Basic Usage

To run interactively:

    .\bin\x64\Release\fractal.exe -interactive

Interactive controls:

* Left Click / Page Down = Zoom In
* Right Click / Page Up = Zoom Out
* Middle Click = Centre
* Ctrl + <Click> = Zoom Faster
* Shift + Left Click = Save Image

To render one image with specific coordinates:

    .\bin\x64\Release\fractal.exe -re 0.16008284883560522 -im 0.5725060571509984 -scale 0.00000244

To render using a YAML configuration:

    .\bin\x64\Release\fractal.exe config\examples\5.yaml

You can specify multiple YAML configurations:

    .\bin\x64\Release\fractal.exe config\examples\5.yaml config\4k.yaml

## More information

Explore the configuration examples under config\\*.yaml