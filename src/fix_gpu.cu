#include "fix_gpu.cuh"
#include "image.hh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cuda/atomic>

__global__
void compact(int* image, int size, int* output, int *blockNb, cuda::std::atomic<char>* flags){

    if (threadIdx.x == 0)
        atomicAdd(blockNb, 1);

    __syncthreads();

    extern __shared__ int sdata[];

    constexpr int garbage_val = -27;

    int blockId = *blockNb - 1;

    int tid = threadIdx.x;
    int i = blockId * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    if (image[i] != garbage_val)
        sdata[tid] = 1;
    else
        sdata[tid] = 0;
    __syncthreads();

    int val = sdata[tid];

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

    if (val != 0){
        output[(sdata[tid] - val) + blockId * blockDim.x - blockId * (blockDim.x / 3 + 1)] = image[i];
    }

    flags[blockId].store('P');
}

void fix_image_gpu(Image& to_fix)
{
    int image_size = 512;//to_fix.width * to_fix.height;

    constexpr int blocksize = 256;
    const int gridsize = (image_size + blocksize - 1) / blocksize;

    int* blockNb;
    cudaMalloc(&blockNb, sizeof(int));
    cudaMemset(blockNb, 0, sizeof(int));
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

    int* output;

    cudaMalloc(&output, sizeof(int) * image_size);
    cudaMemset(output, 0, sizeof(int) * image_size);

    compact<<<gridsize, blocksize, sizeof(int) * blocksize>>>(image_gpu, image_size, output, blockNb, flags);

    int* predic = (int*)calloc(image_size, sizeof(int));
    cudaMemcpy(predic, output, sizeof(int) * image_size, cudaMemcpyDeviceToHost);

    std::cout << std::endl;
    for (int i = 0; i < image_size - image_size / 3 - 1; i++){
        std::cout << predic[i] << " ";
    }

    // #1 Compact

    // Build predicate vector

    image_size = to_fix.width * to_fix.height;

    std::vector<int> predicate(to_fix.size(), 0);

    constexpr int garbage_val = -27;
    for (int i = 0; i < to_fix.size(); ++i)
        if (to_fix.buffer[i] != garbage_val)
            predicate[i] = 1;

    // Compute the exclusive sum of the predicate

    std::exclusive_scan(predicate.begin(), predicate.end(), predicate.begin(), 0);

    // Scatter to the corresponding addresses

    for (std::size_t i = 0; i < predicate.size(); ++i)
        if (to_fix.buffer[i] != garbage_val)
            to_fix.buffer[predicate[i]] = to_fix.buffer[i];


    // #2 Apply map to fix pixels

    for (int i = 0; i < image_size; ++i)
    {
        if (i % 4 == 0)
            to_fix.buffer[i] += 1;
        else if (i % 4 == 1)
            to_fix.buffer[i] -= 5;
        else if (i % 4 == 2)
            to_fix.buffer[i] += 3;
        else if (i % 4 == 3)
            to_fix.buffer[i] -= 8;
    }

    // #3 Histogram equalization

    // Histogram

    std::array<int, 256> histo;
    histo.fill(0);
    for (int i = 0; i < image_size; ++i)
        ++histo[to_fix.buffer[i]];

    // Compute the inclusive sum scan of the histogram

    std::inclusive_scan(histo.begin(), histo.end(), histo.begin());

    // Find the first non-zero value in the cumulative histogram

    auto first_none_zero = std::find_if(histo.begin(), histo.end(), [](auto v) { return v != 0; });

    const int cdf_min = *first_none_zero;

    // Apply the map transformation of the histogram equalization

    std::transform(to_fix.buffer, to_fix.buffer + image_size, to_fix.buffer,
        [image_size, cdf_min, &histo](int pixel)
            {
                return std::roundf(((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
            }
    );
}