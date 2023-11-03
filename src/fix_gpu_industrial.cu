#include "fix_gpu_industrial.cuh"
#include "image.hh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cuda/atomic>
#include <cub/cub.cuh>

__global__
void build_predicate(int* image, int size, int* predicate){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
     __syncthreads();
    if (image[i] != -27)
        predicate[i] = 1;
}

__global__
void compact_scatter_industrial(int* image, int* predicate, int size, int* output){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int val;
    if (i < size)
        val = image[i];
    __syncthreads();
    if (i < size && val != -27)
        output[predicate[i]] = val;
}

__global__
void map_fixer_industrial(int* image, int size){
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
void create_histogram_industrial(int* image, int* histogram, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        atomicAdd(&histogram[image[i]], 1);
}

__global__
void scan_hist_industrial(int* hist){
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
void apply_equalization_industrial(int* image, int* histogram, int size, int cdf_min){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    image[i] = roundf(((histogram[image[i]] - cdf_min) / static_cast<float>(size - cdf_min)) * 255.0f);
}

void fix_image_gpu_industrial(Image& to_fix)
{
    const int image_size = to_fix.width * to_fix.height;
    constexpr int blocksize = 256;
    const int gridsize = (to_fix.size() + blocksize - 1) / blocksize;

    int *predicate;
    cudaMalloc(&predicate, sizeof(int) * to_fix.size());
    cudaMemset(predicate, 0, sizeof(int) * to_fix.size());

    void *d_temp_storage_compact = nullptr;
    size_t temp_storage_bytes_compact = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage_compact, temp_storage_bytes_compact, predicate, predicate, image_size);

    cudaMalloc(&d_temp_storage_compact, temp_storage_bytes_compact);

    int *histogram;
    cudaMalloc(&histogram, sizeof(int) * 256);
    cudaMemset(histogram, 0, sizeof(int) * 256);

    void *d_temp_storage_hist = nullptr;
    size_t temp_storage_bytes_hist = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage_hist, temp_storage_bytes_hist, histogram, histogram, image_size);

    cudaMalloc(&d_temp_storage_hist, temp_storage_bytes_hist);

    int* image_gpu;
    cudaMalloc(&image_gpu, sizeof(int) * to_fix.size());
    cudaMemcpy(image_gpu, to_fix.buffer, sizeof(int) * to_fix.size(), cudaMemcpyHostToDevice);

    int* clean_image;
    cudaMalloc(&clean_image, sizeof(int) * image_size);
    cudaMemset(clean_image, 0, sizeof(int) * image_size);

    build_predicate<<<gridsize, blocksize>>>(image_gpu, to_fix.size(), predicate);
    cub::DeviceScan::ExclusiveSum(d_temp_storage_compact, temp_storage_bytes_compact, predicate, predicate, to_fix.size());
    compact_scatter_industrial<<<gridsize, blocksize>>>(image_gpu, predicate, to_fix.size(), clean_image);

    map_fixer_industrial<<<gridsize, blocksize>>>(clean_image, image_size);
    create_histogram_industrial<<<gridsize, blocksize>>>(clean_image, histogram, image_size);
    cub::DeviceScan::InclusiveSum(d_temp_storage_hist, temp_storage_bytes_hist, histogram, histogram, 256);

    int* final_hist = (int*)calloc(256, sizeof(int));
    cudaMemcpy(final_hist, histogram, sizeof(int) * 256, cudaMemcpyDeviceToHost);

    auto first_none_zero = std::find_if(final_hist, final_hist + 256, [](auto v) { return v != 0; });
    const int cdf_min = *first_none_zero;

    cudaMemcpy(histogram, final_hist, sizeof(int) * 256, cudaMemcpyHostToDevice);
    apply_equalization_industrial<<<gridsize, blocksize>>>(clean_image, histogram, image_size, cdf_min);
    
    cudaMemcpy(to_fix.buffer, clean_image, image_size * sizeof(int), cudaMemcpyDeviceToHost);
}