#include "fix_gpu.cuh"
#include "image.hh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cuda/atomic>

__global__
void compact_scan(int* image, int size, int *blockNb, cuda::std::atomic<char>* flags, int* predicate){

    __shared__ int blockId;
    if (threadIdx.x == 0)
        blockId = atomicAdd(blockNb, 1);
    __syncthreads();

    extern __shared__ int sdata[];

    constexpr int garbage_val = -27;

    int tid = threadIdx.x;
    int i = blockId * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    if (image[i] != garbage_val)
        sdata[tid] = 1;
    else
        sdata[tid] = 0;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        int data;
        if (tid + s < blockDim.x){
            data = sdata[tid];
        }
        __syncthreads();
        if (tid + s < blockDim.x){
            sdata[tid + s] += data;
        }
        __syncthreads();
    }
    __syncthreads();

    flags[blockId].store('A');

    while(blockId > 0 && flags[blockId - 1].load() != 'P')
        continue;

    __syncthreads();

    predicate[tid + blockId * blockDim.x] = sdata[tid];

    __syncthreads();

    if (blockId != 0)
        predicate[tid + blockId * blockDim.x] += predicate[blockId * blockDim.x - 1];

    flags[blockId].store('P');
}

__global__
void compact_scatter(int* image, int* predicate, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    if (image[i] != -27)
        image[predicate[i] - 1] = image[i];
}

__global__
void map_fixer(int* image, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    if (i % 4 == 0)
        image[i] = image[i] + 1 <= 255 ? image[i] + 1 : 255;
    else if (i % 4 == 1)
        image[i] = image[i] - 5 >= 0 ? image[i] - 5 : 0;
    else if (i % 4 == 2)
        image[i] = image[i] + 3 <= 255 ? image[i] + 3 : 255;
    else if (i % 4 == 3)
        image[i] = image[i] - 8 >= 0 ? image[i] - 8 : 0;
}

__global__
void create_histogram(int* image, int* histogram, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        atomicAdd(&histogram[image[i]], 1);
}

__global__
void scan_hist(int* hist){
    extern __shared__ int sdata[];

    int tid = threadIdx.x;

    sdata[tid] = hist[tid];

    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        int data;
        if (tid + s < blockDim.x){
            data = sdata[tid];
        }
        __syncthreads();
        if (tid + s < blockDim.x){
            sdata[tid + s] += data;
        }
        __syncthreads();
    }
    __syncthreads();

    hist[tid] = sdata[tid];
}

__global__
void apply_equalization(int* image, int* histogram, int size, int cdf_min){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    image[i] = roundf(((histogram[image[i]] - cdf_min) / static_cast<float>(size - cdf_min)) * 255.0f);
}

void fix_image_gpu(Image& to_fix)
{
    const int image_size = to_fix.width * to_fix.height;
    constexpr int blocksize = 256;
    const int gridsize = (to_fix.size() + blocksize - 1) / blocksize;

    int* blockNb;
    cudaMalloc(&blockNb, sizeof(int));
    cudaMemset(blockNb, 0, sizeof(int));

    int *predicate;
    cudaMalloc(&predicate, sizeof(int) * to_fix.size());
    cudaMemset(predicate, 0, sizeof(int) * to_fix.size());

    int *histogram;
    cudaMalloc(&histogram, sizeof(int) * 256);
    cudaMemset(histogram, 0, sizeof(int) * 256);

    cuda::std::atomic<char>* flags;
    cudaMalloc(&flags, sizeof(cuda::std::atomic<char>) * gridsize);
    cudaMemset(flags, 'X', sizeof(cuda::std::atomic<char>) * gridsize);

    int* image_gpu;
    cudaMalloc(&image_gpu, sizeof(int) * to_fix.size());
    cudaMemcpy(image_gpu, to_fix.buffer, sizeof(int) * to_fix.size(), cudaMemcpyHostToDevice);

    compact_scan<<<gridsize, blocksize, sizeof(int) * blocksize + sizeof(int)>>>(image_gpu, to_fix.size(), blockNb, flags, predicate);
    compact_scatter<<<gridsize, blocksize>>>(image_gpu, predicate, to_fix.size());
    map_fixer<<<gridsize, blocksize>>>(image_gpu, image_size);
    create_histogram<<<gridsize, blocksize>>>(image_gpu, histogram, image_size);
    scan_hist<<<1, blocksize, sizeof(int) * 256 + sizeof(int)>>>(histogram);

    int* final_hist = (int*)calloc(256, sizeof(int));
    cudaMemcpy(final_hist, histogram, sizeof(int) * 256, cudaMemcpyDeviceToHost);

    auto first_none_zero = std::find_if(final_hist, final_hist + 256, [](auto v) { return v != 0; });
    const int cdf_min = *first_none_zero;

    cudaMemcpy(histogram, final_hist, sizeof(int) * 256, cudaMemcpyHostToDevice);
    apply_equalization<<<gridsize, blocksize>>>(image_gpu, histogram, image_size, cdf_min);
    
    cudaMemcpy(to_fix.buffer, image_gpu, image_size * sizeof(int), cudaMemcpyDeviceToHost);
}