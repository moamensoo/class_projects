#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void gaussianBlurKernelTiled(const uint8_t *d_input, uint8_t *d_output,
                                          int width, int height, int channels,
                                          const float *d_filter, int filterSize) {
    int half = filterSize / 2;
    int tileWidth = blockDim.x + 2 * half;
    extern __shared__ uint8_t shmem[];
    int tileOriginX = blockIdx.x * blockDim.x - half;
    int tileOriginY = blockIdx.y * blockDim.y - half;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int tpt = blockDim.x * blockDim.y;
    int totalElements = tileWidth * (blockDim.y + 2 * half);
    for (int idx = tid; idx < totalElements; idx += tpt) {
        int sh_x = idx % tileWidth;
        int sh_y = idx / tileWidth;
        int global_x = tileOriginX + sh_x;
        int global_y = tileOriginY + sh_y;
        for (int c = 0; c < channels; c++) {
            if(global_x >= 0 && global_x < width && global_y >= 0 && global_y < height)
                shmem[(sh_y * tileWidth + sh_x)*channels + c] = d_input[(global_y * width + global_x)*channels + c];
            else
                shmem[(sh_y * tileWidth + sh_x)*channels + c] = 0;
        }
    }
    __syncthreads();
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < width && y < height){
        int local_x = threadIdx.x + half;
        int local_y = threadIdx.y + half;
        for(int c = 0; c < channels; c++){
            float sum = 0.0f;
            for(int i = -half; i <= half; i++){
                for(int j = -half; j <= half; j++){
                    int sh_x = local_x + j, sh_y = local_y + i;
                    float fVal = d_filter[(i+half)*filterSize + (j+half)];
                    uint8_t pixel = shmem[(sh_y * tileWidth + sh_x)*channels + c];
                    sum += fVal * pixel;
                }
            }
            d_output[(y * width + x)*channels + c] = (uint8_t)sum;
        }
    }
}

float* computeGaussianFilter(int filterSize, float sigma) {
    int half = filterSize / 2;
    float *filter = (float*)malloc(filterSize * filterSize * sizeof(float));
    float total = 0.0f;
    for(int i = -half; i <= half; i++){
        for(int j = -half; j <= half; j++){
            float w = expf(-(i*i+j*j)/(2.0f * sigma * sigma));
            filter[(i+half)*filterSize + (j+half)] = w;
            total += w;
        }
    }
    for(int i = 0; i < filterSize*filterSize; i++){
        filter[i] /= total;
    }
    return filter;
}

int main(int argc, char **argv) {
    int width, height, channels;
    uint8_t *h_input = stbi_load("input.jpg", &width, &height, &channels, 0);
    if(!h_input){
        fprintf(stderr, "Error: %s\n", stbi_failure_reason());
        return 1;
    }
    size_t imageSize = width * height * channels * sizeof(uint8_t);
    float sigma = (argc >= 2) ? atof(argv[1]) : 2.0f;
    int filterSize = (int)(2*3.14159265359*sigma);
    if(!(filterSize & 1))
        filterSize++;
    float *h_filter = computeGaussianFilter(filterSize, sigma);
    size_t filterBytes = filterSize * filterSize * sizeof(float);
    uint8_t *d_input, *d_output;
    float *d_filter;
    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, imageSize);
    cudaMalloc((void**)&d_filter, filterBytes);
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filterBytes, cudaMemcpyHostToDevice);
    int tileSize = 32;
    dim3 blockDim(tileSize, tileSize);
    dim3 gridDim((width+tileSize-1)/tileSize, (height+tileSize-1)/tileSize);
    int half = filterSize / 2;
    int tileW = tileSize + 2 * half;
    size_t sharedMemSize = tileW * (tileSize + 2 * half) * channels * sizeof(uint8_t);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gaussianBlurKernelTiled<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_output, width, height, channels, d_filter, filterSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float execTime;
    cudaEventElapsedTime(&execTime, start, stop);
    uint8_t *h_output = (uint8_t*)malloc(imageSize);
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);
    int numBlocksX = (width + tileSize - 1) / tileSize;
    int numBlocksY = (height + tileSize - 1) / tileSize;
    int numBlocks = numBlocksX * numBlocksY;
    long long GMA = numBlocks * ( (long long)(tileW * tileW * channels) + (long long)(tileSize * tileSize * channels) )
                  + (long long)width * height * channels;
    printf("Tiled GPU:\nExecution Time: %.2f ms\nGlobal Memory Accesses: %lld bytes\n", execTime, GMA);
    stbi_write_jpg("output_tiled.jpg", width, height, channels, h_output, 100);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
    free(h_input);
    free(h_filter);
    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
