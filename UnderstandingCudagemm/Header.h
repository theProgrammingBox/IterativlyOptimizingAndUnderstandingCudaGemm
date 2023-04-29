#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>


void PrintGPUMatrix(float* d_arr, uint32_t rows, uint32_t cols, const char* label) {
    float* h_arr = (float*)malloc(rows * cols * sizeof(float));
    cudaMemcpy(h_arr, d_arr, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    printf("%s:\n", label);
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++)
            printf("%8.3f ", h_arr[i * cols + j]);
        printf("\n");
    }
    printf("\n");

    free(h_arr);
}


void PrintLargestError(float* d_A, float* d_B, int rows, int cols) {
    int size = rows * cols;
    float* h_A = (float*)malloc(size * sizeof(float));
    float* h_B = (float*)malloc(size * sizeof(float));

    cudaMemcpy(h_A, d_A, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, size * sizeof(float), cudaMemcpyDeviceToHost);

    float max_error = 0;
    for (int i = 0; i < size; i++) {
        float error = fabs(h_A[i] - h_B[i]);
        max_error = fmaxf(max_error, error);
        /*if (error > 0.0001f)
            printf("error at (%d, %d): A=%f, B=%f\n", i / cols, i % cols, h_A[i], h_B[i]);*/
    }

    printf("Largest error: %f\n", max_error);

    free(h_A);
    free(h_B);
}

__global__ void matrixMul_naive(uint32_t hA, uint32_t wA, uint32_t wB, const float* A, const float* B, float* C)
{
    const uint32_t x = (blockIdx.x << 5) + threadIdx.x;
    const uint32_t y = (blockIdx.y << 5) + threadIdx.y;

    if (x >= wB || y >= hA)
        return;

    float tmp = 0.0;
    for (int k = 0; k < wA; k++)
        tmp += A[y * wA + k] * B[k * wB + x];
    C[y * wB + x] = tmp;
}

void matrixMulNaive(uint32_t hA, uint32_t wA, uint32_t wB, const float* A, const float* B, float* C)
{
    matrixMul_naive <<<dim3((wB >> 5) + (wB & 0x1f), (hA >> 5) + (hA & 0x1f)), dim3(32, 32)>>> (hA, wA, wB, A, B, C);
}

__global__ void matrixMul_tiling(uint32_t hA, uint32_t wA, uint32_t wB, const float* A, const float* B, float* C)
{
    const uint32_t blockedX = blockIdx.x << 5;
    const uint32_t blockedY = blockIdx.y << 5;

    const uint32_t x = blockedX + threadIdx.x;
    const uint32_t y = blockedY + threadIdx.y;

    if (x >= wB || y >= hA)
        return;

    __shared__ float sA[1024];
    __shared__ float sB[1024];

    const uint32_t scaledThreadY = threadIdx.y << 5;

    const uint32_t aStart = wA * blockedY;
    const uint32_t aStep = 32;
    const uint32_t aEnd = aStart + wA;

    const uint32_t bStart = blockedX;
    const uint32_t bStep = wB << 5;

    float sum = 0.0;
    for (uint32_t a = aStart, b = bStart; a < aEnd; a += aStep, b += bStep)
    {
        sA[scaledThreadY + threadIdx.x] = A[a + wA * threadIdx.y + threadIdx.x];
        sB[scaledThreadY + threadIdx.x] = B[b + wB * threadIdx.y + threadIdx.x];
        __syncthreads();

        for (uint32_t k = 0; k < 32; k++)
            sum += sA[scaledThreadY + k] * sB[(k << 5) + threadIdx.x];
        __syncthreads();
    }

    C[wB * y + x] = sum;
}

void matrixMulTiling(uint32_t hA, uint32_t wA, uint32_t wB, const float* A, const float* B, float* C)
{
    matrixMul_tiling <<<dim3((wB >> 5) + (wB & 0x1f), (hA >> 5) + (hA & 0x1f)), dim3(32, 32)>>> (hA, wA, wB, A, B, C);
}