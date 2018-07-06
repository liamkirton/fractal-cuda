#pragma once

class timer {
public:
    timer(bool auto_start=true) : freq_{ 0 }, t_{ 0 } {
        QueryPerformanceFrequency(&freq_);
        if (auto_start) {
            start();
        }
    }

    void inline start() {
        memset(&t_, 0, sizeof(t_));
        QueryPerformanceCounter(&t_[0]);
    }

    void inline stop() {
        QueryPerformanceCounter(&t_[1]);
    }

    void print() {
        std::cout << "[+] Time: " << (1.0 * (t_[1].QuadPart - t_[0].QuadPart)) / freq_.QuadPart << std::endl;
    }

private:
    LARGE_INTEGER freq_;
    LARGE_INTEGER t_[2];
};