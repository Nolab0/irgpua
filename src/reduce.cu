#include "fix_gpu.cuh"
#include "image.hh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cuda/atomic>
#include <cub/device/device_reduce.cuh>
#include <cub/util_allocator.cuh>

__global__
void kernel_reduce(int* buffer, int* total, int size) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    if (i < size) {
        int val = buffer[i];
        if (i + blockDim.x < size) {
            val += buffer[i + blockDim.x];
        }
        sdata[tid] = val;
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(&total[0], sdata[0]);
}

int reduce_gpu(Image& image)
{
    const int image_size = image.width * image.height;
    constexpr int blocksize = 256;
    const int gridsize = (image_size + blocksize - 1) / blocksize;

    int* image_gpu;
    cudaMalloc(&image_gpu, sizeof(int) * image_size);
    cudaMemcpy(image_gpu, image.buffer, sizeof(int) * image_size, cudaMemcpyHostToDevice);

    int* total;
    cudaMalloc(&total, sizeof(int));
    cudaMemset(total, 0, sizeof(int));

	kernel_reduce<<<gridsize, blocksize, sizeof(int) * blocksize>>>(image_gpu, total, image_size);

    int *total_local = (int*)malloc(sizeof(int));
    cudaMemcpy(total_local, total, sizeof(int), cudaMemcpyDeviceToHost);

    return *total_local;
}

int reduce_gpu_industrial(Image& image) {
    const int image_size = image.width * image.height;

    int* image_gpu;
    cudaMalloc(&image_gpu, sizeof(int) * image_size);
    cudaMemcpy(image_gpu, image.buffer, sizeof(int) * image_size, cudaMemcpyHostToDevice);

    int* total;
    cudaMalloc(&total, sizeof(int));
    cudaMemset(total, 0, sizeof(int));
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, image_gpu, total, image_size);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, image_gpu, total, image_size);

    cudaFree(d_temp_storage);
    cudaFree(image_gpu);

    int *total_local = (int*)malloc(sizeof(int));
    cudaMemcpy(total_local, total, sizeof(int), cudaMemcpyDeviceToHost);

    return *total_local;
}