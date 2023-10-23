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

    predicate[tid + blockId * blockDim.x] = sdata[tid];

    if (blockId != 0)
        predicate[tid + blockId * blockDim.x] += predicate[blockId * blockDim.x - 1];

    flags[blockId].store('P');
}

__global__
void compact_scatter(int* image, int* predicate){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (image[i] != -27)
        image[predicate[i] - 1] = image[i];
}

void fix_image_gpu(Image& to_fix)
{
    int image_size = 1024;

    constexpr int blocksize = 256;
    const int gridsize = (image_size + blocksize - 1) / blocksize;

    int* blockNb;
    cudaMalloc(&blockNb, sizeof(int));
    cudaMemset(blockNb, 0, sizeof(int));

    int *predicate;
    cudaMalloc(&predicate, sizeof(int) * image_size);
    cudaMemset(predicate, 0, sizeof(int) * image_size);


    cuda::std::atomic<char>* flags;
    cudaMalloc(&flags, sizeof(cuda::std::atomic<char>) * gridsize);
    cudaMemset(flags, 'X', sizeof(cuda::std::atomic<char>) * gridsize);

    int* test = (int*)malloc(sizeof(int) * image_size);
    for (int i = 0; i < image_size; i++){
        if (i % 3 == 0)
            test[i] = -27;
        else
            test[i] = i;
    }

    int* image_gpu;

    cudaMalloc(&image_gpu, sizeof(int) * image_size);
    cudaMemcpy(image_gpu, test, sizeof(int) * image_size, cudaMemcpyHostToDevice);

    compact_scan<<<gridsize, blocksize, sizeof(int) * blocksize + sizeof(int)>>>(image_gpu, image_size, blockNb, flags, predicate);
    compact_scatter<<<gridsize, blocksize>>>(image_gpu, predicate);

    int* predic2 = (int*)calloc(image_size, sizeof(int));
    cudaMemcpy(predic2, image_gpu, sizeof(int) * image_size, cudaMemcpyDeviceToHost);

    std::cout << std::endl;
    for (int i = 0; i < image_size; i++){
        std::cout << predic2[i] << " ";
    }

}