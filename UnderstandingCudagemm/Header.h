#pragma once
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

/*
Important lesseons
0. asserts can help compilers optimize
*/

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
    matrixMul_naive <<<dim3((wB >> 5) + bool(wB & 0x1f), (hA >> 5) + bool(hA & 0x1f)), dim3(32, 32)>>> (hA, wA, wB, A, B, C);
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
    matrixMul_tiling <<<dim3((wB >> 5) + bool(wB & 0x1f), (hA >> 5) + bool(hA & 0x1f)), dim3(32, 32)>>> (hA, wA, wB, A, B, C);
}

__global__ void matrixMul_tiling2(uint32_t hA, uint32_t wA, uint32_t wB, const float* A, const float* B, float* C)
{
    const uint32_t blockedX = blockIdx.x << 5;
    const uint32_t blockedY = blockIdx.y << 5;
    const uint32_t x = blockedX + threadIdx.x;
    const uint32_t y = blockedY + threadIdx.y;

    if (x >= wB || y >= hA)
        return;

    __shared__ float sA[1024];
    __shared__ float sB[1024];

    const uint32_t scaledY32 = threadIdx.y << 5;
    const uint32_t scaledwB32 = wB << 5;

    float* sharedA = sA + scaledY32 + threadIdx.x;
    float* sharedB = sB + scaledY32 + threadIdx.x;

    A += threadIdx.x + threadIdx.y * wA + blockedY * wA;
    B += threadIdx.x + threadIdx.y * wB + blockedX;
    C += y * wB + x;

    float sum = 0.0;
    for (uint32_t blockIdx = 0; blockIdx < wA; blockIdx += 32, A += 32, B += scaledwB32)
    {
		*sharedA = *A;
		*sharedB = *B;
		__syncthreads();

		for (uint32_t k = 0; k < 32; ++k)
			sum += sA[scaledY32 + k] * sB[(k << 5) + threadIdx.x];
		__syncthreads();
	}

    *C = sum;
}

void matrixMulTiling2(uint32_t hA, uint32_t wA, uint32_t wB, const float* A, const float* B, float* C)
{
	matrixMul_tiling2 <<<dim3((wB >> 5) + bool(wB & 0x1f), (hA >> 5) + bool(hA & 0x1f)), dim3(32, 32)>>> (hA, wA, wB, A, B, C);
}

__global__ void sgemm1DBlocktiling(int M, int N, int K, const float* A, const float* B, float* C) {

    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;

    const uint blockedX = blockIdx.x * BN;
    const uint blockedY = blockIdx.y * BM;

    __shared__ float As[512];
    __shared__ float Bs[512];
    float threadResults[TM] = { 0.0 };

    const uint threadxForA = threadIdx.x % BK;  // 0 - 7
    const uint threadyForA = threadIdx.x / BK;  // 0 - 63

    const uint threadxForB = threadIdx.x % BN;  // 0 - 63
    const uint threadyForB = threadIdx.x / BN;  // 0 - 7

    float* sharedA = As + threadyForA * BK + threadxForA;
    float* sharedB = Bs + threadyForB * BN + threadxForB;

    A += threadxForA + threadyForA * K + blockedY * K;
    B += threadxForB + threadyForB * N + blockedX;
    C += blockedY * N + blockedX;

    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK, A += BK, B += BK * N)
    {
        *sharedA = *A;
        *sharedB = *B;
        __syncthreads();

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            float tmpB = Bs[dotIdx * BN + threadxForB];
            for (uint resIdx = 0; resIdx < TM; ++resIdx)
                threadResults[resIdx] += As[(threadyForB * TM + resIdx) * BK + dotIdx] * tmpB;
        }
        __syncthreads();
    }

    for (uint resIdx = 0; resIdx < TM; ++resIdx)
        C[(threadyForB * TM + resIdx) * N + threadxForB] = threadResults[resIdx];
}

void matrixMul1DBlocktiling(uint32_t hA, uint32_t wA, uint32_t wB, const float* A, const float* B, float* C)
{
    sgemm1DBlocktiling<<<dim3((wB >> 6) + bool(wB & 0x3f), (hA >> 6) + bool(hA & 0x3f)), 512 >> > (hA, wB, wA, A, B, C);
}

__global__ void sgemm1DBlocktiling2(int M, int N, int K, const float* A, const float* B, float* C)
{
    const uint blockedX = blockIdx.x << 6;
    const uint blockedY = blockIdx.y << 6;

    __shared__ float As[1024];
    __shared__ float Bs[1024];
    float threadResults[4] = { 0.0 };

    const uint threadxForA = threadIdx.x & 0xf;  // 0 - 7
    const uint threadyForA = threadIdx.x >> 4;  // 0 - 63

    const uint threadxForB = threadIdx.x & 0x3F;  // 0 - 63
    const uint threadyForB = threadIdx.x >> 6;  // 0 - 7

    float* sharedA = As + threadIdx.x;
    float* sharedB = Bs + threadIdx.x;

    // can use | instead of +
    A += threadxForA + (threadyForA + blockedY) * K;
    B += threadxForB + threadyForB * N + blockedX;
    C += (blockedY + (threadyForB << 2)) * N + blockedX + threadxForB;

    for (uint bkIdx = 0; bkIdx < K; bkIdx += 16, A += 16, B += N << 4)
    {
        *sharedA = *A;
        *sharedB = *B;
        __syncthreads();

        for (uint dotIdx = 0; dotIdx < 16; ++dotIdx)
        {
            float tmpB = Bs[(dotIdx << 6) + threadxForB];
            for (uint resIdx = 0; resIdx < 4; ++resIdx)
                threadResults[resIdx] += As[(((threadyForB << 2) + resIdx) << 4) + dotIdx] * tmpB;
        }
        __syncthreads();
    }

    for (uint resIdx = 0; resIdx < 4; ++resIdx)
        C[resIdx * N] = threadResults[resIdx];
}

void matrixMul1DBlocktiling2(uint32_t hA, uint32_t wA, uint32_t wB, const float* A, const float* B, float* C)
{
	sgemm1DBlocktiling2<<<dim3((wB >> 6) + bool(wB & 0x3f), (hA >> 6) + bool(hA & 0x3f)), 1024>>> (hA, wB, wA, A, B, C);
}