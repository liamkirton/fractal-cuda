//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class png_writer : public writer {
public:
    png_writer(YAML::Node &run_config);
    ~png_writer();

    virtual void write(uint32_t image_width, uint32_t image_height, const uint32_t *image, std::string &suffix, uint64_t ix = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::make_tuple(image_width, image_height, image, suffix));
    }

private:
    void write(std::tuple<uint64_t, uint64_t, const uint32_t *, std::string> &image);

    std::string directory_;
    std::string prefix_;

    HANDLE exit_event_;
    std::mutex mutex_;
    std::queue<std::tuple<uint64_t, uint64_t, const uint32_t *, std::string>> queue_;
    std::vector<std::thread> threads_;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
