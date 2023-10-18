#include "fix_gpu.cuh"
#include "image.hh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>

__global__
void compact(int* image, int size, int* predicate){

    extern __shared__ int sdata[];

    constexpr int garbage_val = -27;
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size)
        return;

    if (image[i] != garbage_val)
        sdata[i] = 1;
    else
        sdata[i] = 0;
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

    if (val != garbage_val){
        printf("%d ", image[i]);
        image[sdata[tid] - val] = image[i];
    }
}

void fix_image_gpu(Image& to_fix)
{
    const int image_size = to_fix.width * to_fix.height;

    constexpr int blocksize = 256;
    const int gridsize = (image_size + blocksize - 1) / blocksize;

    const int size = 10;
    const int finalSize = 6;
    int* test = (int*)malloc(sizeof(int) * size);
    for (int i = 0; i < size; i++){
        if (i % 3 == 0)
            test[i] = -27;
        else    
            test[i] = i;
    }

    int* image_gpu;

    cudaMalloc(&image_gpu, sizeof(int) * size);
    cudaMemcpy(image_gpu, test, sizeof(int) * size, cudaMemcpyHostToDevice);

    int* predicate_gpu;

    cudaMalloc(&predicate_gpu, sizeof(int) * size);
    cudaMemset(predicate_gpu, 0, sizeof(int) * size);

    compact<<<1, blocksize, sizeof(int) * blocksize>>>(image_gpu, size, predicate_gpu);

    int* predic = (int*)calloc(size, sizeof(int));
    cudaMemcpy(predic, image_gpu, sizeof(int) * size, cudaMemcpyDeviceToHost);

    /*for (int i = 0; i < size; i++){
        std::cout << predic[i] << " ";
    }*/

    // #1 Compact

    // Build predicate vector

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