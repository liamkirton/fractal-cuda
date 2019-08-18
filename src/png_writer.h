//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Fractal
// (C)2018-19 Liam Kirton <liam@int3.ws>
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class png_writer {
public:
    png_writer(YAML::Node &run_config);
    ~png_writer();

    virtual void write(image &i, std::string &suffix, uint64_t ix = 0) {
        write(i.image_width(), i.image_height(), i.image_buffer(true), suffix, ix);
    }

    void write(uint32_t image_width, uint32_t image_height, const uint32_t *image_buffer, std::string &suffix, uint64_t ix = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::make_tuple(image_width, image_height, image_buffer, suffix));
    }

private:
    void write(std::tuple<uint64_t, uint64_t, const uint32_t *, std::string> &image);

    std::string directory_;
    std::string prefix_;

    HANDLE exit_event_;
    std::mutex mutex_;
    std::queue<std::tuple<uint64_t, uint64_t, const uint32_t *, std::string>> queue_;
    std::vector<std::thread> threads_;

    bool oversample_;
    uint32_t oversample_multiplier_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
