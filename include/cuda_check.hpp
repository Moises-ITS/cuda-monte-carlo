#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// Every CUDA runtime call returns an error code; ignoring it is how you end up
// benchmarking a kernel that never ran. Abort loudly with file:line instead.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s (%s) at %s:%d\n",               \
                         cudaGetErrorName(err_), cudaGetErrorString(err_),      \
                         __FILE__, __LINE__);                                   \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// Kernel launches don't return an error. This catches *launch* failures (bad
// grid/block config, too much shared memory) immediately. Faults that happen
// while the kernel runs only surface at the next synchronizing call, which is
// why the timers below CUDA_CHECK their synchronize.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())
