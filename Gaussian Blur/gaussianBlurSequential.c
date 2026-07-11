#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

void gaussianFilter(uint8_t *image, uint8_t *output, int width, int height, int channels, int filterSize, float sigma) {
    int i, j, k, l;
    float sum, val, weight, total = 0.0f;
    int half = filterSize / 2;
    float *filter = (float *)malloc(filterSize * filterSize * sizeof(float));
    float *gf = (float *)malloc(filterSize * filterSize * sizeof(float));
    for(i = -half; i <= half; i++){
        for(j = -half; j <= half; j++){
            weight = expf(-(i*i + j*j) / (2.0f * sigma * sigma));
            filter[(i+half)*filterSize + (j+half)] = weight;
            total += weight;
        }
    }
    for(i = 0; i < filterSize*filterSize; i++){
        gf[i] = filter[i] / total;
    }
    for(i = 0; i < height; i++){
        for(j = 0; j < width; j++){
            for(int c = 0; c < channels; c++){
                sum = 0.0f;
                for(k = -half; k <= half; k++){
                    for(l = -half; l <= half; l++){
                        int row = i + k, col = j + l;
                        if(row >= 0 && row < height && col >= 0 && col < width)
                            val = image[(row * width + col)*channels + c];
                        else
                            val = 0.0f;
                        sum += val * gf[(k+half)*filterSize + (l+half)];
                    }
                }
                output[(i * width + j)*channels + c] = (uint8_t)sum;
            }
        }
    }
    free(filter);
    free(gf);
}

int main(int argc, char **argv) {
    int width, height, channels;
    uint8_t *image = stbi_load("input.jpg", &width, &height, &channels, 0);
    if(!image){
        fprintf(stderr, "Error loading image: %s\n", stbi_failure_reason());
        return 1;
    }
    uint8_t *output = (uint8_t *)malloc(width * height * channels * sizeof(uint8_t));
    float sigma = (argc >= 2) ? atof(argv[1]) : 2.0f;
    int filterSize = (int)(2 * 3.14159265359 * sigma);
    if(!(filterSize & 1))
        filterSize++; 
    clock_t start = clock();
    gaussianFilter(image, output, width, height, channels, filterSize, sigma);
    clock_t end = clock();
    double execTime = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    long long GMA = (long long)width * height * channels * (filterSize * filterSize + 1);
    printf("Sequential:\nExecution Time: %.2f ms\nGlobal Memory Accesses: %lld bytes\n", execTime, GMA);
    stbi_write_jpg("output_seq.jpg", width, height, channels, output, 100);
    stbi_image_free(image);
    free(output);
    return 0;
}
