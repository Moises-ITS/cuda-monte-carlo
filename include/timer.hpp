#pragma once

#include <chrono>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

// Wall-clock timer for CPU code. steady_clock because system_clock can jump.
class CpuTimer {
public:
    CpuTimer() : start_(Clock::now()) {}
    void reset() { start_ = Clock::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;
};

// GPU timer built on CUDA events. Kernel launches are asynchronous, so a host
// clock around a launch measures only the launch overhead; events are
// timestamped by the GPU itself when the stream reaches them.
class CudaEventTimer {
public:
    CudaEventTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaEventTimer() {
        // No CUDA_CHECK: never exit() from a destructor.
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void start(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(start_, stream)); }

    // Records the stop event, waits for it, and returns elapsed milliseconds.
    // The synchronize is also where asynchronous kernel faults show up.
    float stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};
