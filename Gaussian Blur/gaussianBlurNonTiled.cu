#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void gaussianBlurKernelNonTiled(const uint8_t *d_input, uint8_t *d_output,
                                             int width, int height, int channels,
                                             const float *d_filter, int filterSize) {
    int half = filterSize / 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width || y >= height)
        return;
    for(int c = 0; c < channels; c++){
        float sum = 0.0f;
        for(int i = -half; i <= half; i++){
            for(int j = -half; j <= half; j++){
                int curX = x + j, curY = y + i;
                uint8_t pixel = 0;
                if(curX >= 0 && curX < width && curY >= 0 && curY < height)
                    pixel = d_input[(curY * width + curX)*channels + c];
                float fVal = d_filter[(i + half)*filterSize + (j + half)];
                sum += fVal * pixel;
            }
        }
        d_output[(y * width + x)*channels + c] = (uint8_t)sum;
    }
}

float* computeGaussianFilter(int filterSize, float sigma) {
    int half = filterSize / 2;
    float *filter = (float*)malloc(filterSize * filterSize * sizeof(float));
    float total = 0.0f;
    for(int i = -half; i <= half; i++){
        for(int j = -half; j <= half; j++){
            float w = expf(-(i*i + j*j)/(2.0f * sigma * sigma));
            filter[(i+half)*filterSize + (j+half)] = w;
            total += w;
        }
    }
    for(int i = 0; i < filterSize * filterSize; i++){
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
    int filterSize = (int)(2 * 3.14159265359 * sigma);
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
    dim3 blockDim(16, 16);
    dim3 gridDim((width+15)/16, (height+15)/16);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gaussianBlurKernelNonTiled<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels, d_filter, filterSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float execTime;
    cudaEventElapsedTime(&execTime, start, stop);
    uint8_t *h_output = (uint8_t*)malloc(imageSize);
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);
    long long GMA = (long long)width * height * channels * (filterSize * filterSize + 1);
    printf("Non-Tiled GPU:\nExecution Time: %.2f ms\nGlobal Memory Accesses: %lld bytes\n", execTime, GMA);
    stbi_write_j
