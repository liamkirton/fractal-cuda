//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class image {
public:
    image() : image_width_(0), image_height_(0), image_(nullptr) {}

    image(uint32_t image_width, uint32_t image_height, uint32_t *image) : image_width_(image_width),
            image_height_(image_height) {
        image_ = new uint32_t[image_width_ * image_height_];
        memcpy(image_, image, sizeof(uint32_t) * image_width * image_height);
    }

    ~image() {
        if (image_ != nullptr) {
            delete[] image_;
        }
    }

    uint32_t image_width() {
        return image_width_;
    }

    uint32_t image_height() {
        return image_height_;
    }

    uint32_t image_size() {
        return image_width_ * image_height_ * sizeof(uint32_t);
    }

    const uint32_t *image_buffer(bool release = false) {
        const uint32_t *buffer = image_;
        if (release) {
            image_ = nullptr;
        }
        return buffer;
    }

private:
    uint32_t image_width_;
    uint32_t image_height_;
    uint32_t *image_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
