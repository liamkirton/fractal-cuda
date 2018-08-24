//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class png {
public:
    png(YAML::Node &run_config);
    ~png();

    template<typename T>
    void write(fractal<T> &f, uint64_t ix = 0) {
        std::stringstream suffix;
        suffix << std::setfill('0')
            << "ix=" << ix << "_"
            << "re=" << std::setprecision(12) << f.re() << "_"
            << "im=" << std::setprecision(12) << f.im() << "_"
            << "scale=" << std::setprecision(12) << f.scale();

        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::make_tuple(f.image_width(), f.image_height(), f.image(true), suffix.str()));
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
