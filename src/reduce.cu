#include "fix_gpu.cuh"
#include "image.hh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cuda/atomic>

__global__
void kernel_reduce(int* buffer, int* total, int size)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = buffer[i];
    __syncthreads();


    for (int s = 1; tid + s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0)
            sdata[tid] += sdata[tid + s];

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