#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <stdint.h>

// Export a function so that is accessible through a C ABI
#define EXPORT extern "C" __attribute__((visibility("default")))

// Round up a / b to the next integer
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// Rank of the current thread in the block
#define THREAD_RANK (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z))
// Number of threads in the block
#define BLOCK_SIZE (blockDim.x * blockDim.y * blockDim.z)

// Grid stride loop: all threads in the (1D) grid share [0, n)
// global thread g takes g, g + gridDim.x * blockDim.x, ...
#define GRID_STRIDE(i, n) for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < (n); i += (int64_t)gridDim.x * blockDim.x)

// Block stride loop: all threads in the block share [0, n)
// thread t takes t, t + BLOCK_SIZE, ...
#define BLOCK_STRIDE(i, n) for (int64_t i = THREAD_RANK; i < (n); i += BLOCK_SIZE)
// Warp stride loop: all warps in the block share [0, n)
// warp w takes w, w + BLOCK_SIZE / 32, ...
#define WARP_STRIDE(i, n) for (int64_t i = THREAD_RANK / 32; i < (n); i += BLOCK_SIZE / 32)
// Lane stride loop: all threads in the warp share [0, n)
// lane l takes l, l + 32, ...
#define LANE_STRIDE(i, n) for (int64_t i = THREAD_RANK % 32; i < (n); i += 32)

#endif
