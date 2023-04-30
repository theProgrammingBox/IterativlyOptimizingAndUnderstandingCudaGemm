#include "Header.cuh"

int main() {
    const int M = 2048, N = 1024, K = 512;
    const float alpha = 1.0f, beta = 0.0f;

    // Initialize cuBLAS and cuRAND handles
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    curandGenerator_t curand_gen;
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    
    // Timing events
    cudaEvent_t start, stop;
    float elapsedTime;

    // Allocate memory for A, B, and C on the device
    float* d_A;
    float* d_B;
    float* d_C;
    float* d_C2;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));
    cudaMalloc((void**)&d_C2, M * N * sizeof(float));

    // Fill A and B with random numbers
    curandGenerateUniform(curand_gen, d_A, M * K);
    curandGenerateUniform(curand_gen, d_B, K * N);


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    matrixMul1DBlocktiling(M, K, N, d_A, d_B, d_C);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Naive kernel time: %f ms\n", elapsedTime);


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C2, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("cuBLAS kernel time: %f ms\n", elapsedTime);


    // Calculate largest error
    PrintLargestError(d_C, d_C2, M, N);

    // Print matrices if they are small enough
    if (M <= 32 && N <= 32 && K <= 32)
    {
        PrintGPUMatrix(d_A, M, K, "A");
        PrintGPUMatrix(d_B, K, N, "B");
        PrintGPUMatrix(d_C, M, N, "C");
        PrintGPUMatrix(d_C2, M, N, "C2");
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C2);
    cublasDestroy(cublas_handle);
    curandDestroyGenerator(curand_gen);

    return 0;
}
