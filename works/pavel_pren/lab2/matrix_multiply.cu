#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define TILE_SIZE 16

void matrixMultiplyCPU(float* A, float* B, float* C, int N) {
    // просто считаем матрицы на ЦП
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void matrixMultiplyBase(float* A, float* B, float* C, int N) {
    // просто считаем матрицы на ГПУ
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}


__global__ void matrixMultiplyTiled(float* A, float* B, float* C, int N) {
    // считаем на ГПУ с тайлингом и shared memory
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + tx;
        if (row < N && aCol < N) {
            tileA[ty][tx] = A[row * N + aCol];
        } else {
            tileA[ty][tx] = 0.0f;
        }
        
        int bRow = t * TILE_SIZE + ty;
        if (bRow < N && col < N) {
            tileB[ty][tx] = B[bRow * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void initMatrix(float* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = (float)(rand() % 10);
    }
}

bool verifyResult(float* cpuResult, float* gpuResult, int N, float tolerance = 0.01f) {
    for (int i = 0; i < N * N; i++) {
        if (fabs(cpuResult[i] - gpuResult[i]) > tolerance) {
            printf("Error at position %d: CPU = %f, GPU = %f\n", i, cpuResult[i], gpuResult[i]);
            return false;
        }
    }
    return true;
}

double getTime() {
    return (double)clock() / CLOCKS_PER_SEC;
}

void runBenchmark(int N, float* h_A, float* h_B, float* h_C_cpu, float* h_C_gpu_basic, 
                  float* h_C_gpu_tiled, float* d_A, float* d_B, float* d_C,
                  cudaEvent_t start, cudaEvent_t stop) {
    printf("\n%dx%d matrices\n", N, N);
    fflush(stdout);
    
    size_t bytes = N * N * sizeof(float);
    
    srand(42);
    initMatrix(h_A, N);
    initMatrix(h_B, N);
    
    printf("CPU");
    fflush(stdout);
    double startCPU = getTime();
    matrixMultiplyCPU(h_A, h_B, h_C_cpu, N);
    double endCPU = getTime();
    double timeCPU = endCPU - startCPU;
    printf("Time: %.4f sec\n", timeCPU);
    
    printf("GPU basic");
    fflush(stdout);
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    matrixMultiplyBase<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float timeGpuBasic = 0;
    cudaEventElapsedTime(&timeGpuBasic, start, stop);
    timeGpuBasic /= 1000.0f;
    
    cudaMemcpy(h_C_gpu_basic, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("Time: %.4f sec, ", timeGpuBasic);
    
    if (verifyResult(h_C_cpu, h_C_gpu_basic, N)) {
        printf("result correct\n");
    } else {
        printf("result incorrect!\n");
    }
    
    printf("GPU with tiling + shared memory");
    fflush(stdout);
    
    cudaEventRecord(start);
    matrixMultiplyTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float timeGpuTiled = 0;
    cudaEventElapsedTime(&timeGpuTiled, start, stop);
    timeGpuTiled /= 1000.0f;
    
    cudaMemcpy(h_C_gpu_tiled, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("Time: %.4f sec, ", timeGpuTiled);
    
    if (verifyResult(h_C_cpu, h_C_gpu_tiled, N)) {
        printf("result correct\n");
    } else {
        printf("result incorrect!\n");
    }
    
    printf("Speedup (basic vs CPU): %.2fx\n", timeCPU / timeGpuBasic);
    printf("Speedup (tiled vs CPU): %.2fx\n", timeCPU / timeGpuTiled);
    printf("Speedup (tiled vs basic): %.2fx\n", timeGpuBasic / timeGpuTiled);
    
    fflush(stdout);
}

int main() {
    fflush(stdout);
    
    int maxN = 4096;
    size_t maxBytes = maxN * maxN * sizeof(float);
    float* h_A = (float*)malloc(maxBytes);
    float* h_B = (float*)malloc(maxBytes);
    float* h_C_cpu = (float*)malloc(maxBytes);
    float* h_C_gpu_basic = (float*)malloc(maxBytes);
    float* h_C_gpu_tiled = (float*)malloc(maxBytes);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, maxBytes);
    cudaMalloc(&d_B, maxBytes);
    cudaMalloc(&d_C, maxBytes);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    runBenchmark(32, h_A, h_B, h_C_cpu, h_C_gpu_basic, h_C_gpu_tiled,
                 d_A, d_B, d_C, start, stop);
    
    runBenchmark(128, h_A, h_B, h_C_cpu, h_C_gpu_basic, h_C_gpu_tiled,
                 d_A, d_B, d_C, start, stop);
    
    runBenchmark(512, h_A, h_B, h_C_cpu, h_C_gpu_basic, h_C_gpu_tiled,
                 d_A, d_B, d_C, start, stop);
    
    runBenchmark(1024, h_A, h_B, h_C_cpu, h_C_gpu_basic, h_C_gpu_tiled,
                 d_A, d_B, d_C, start, stop);
    
    runBenchmark(2048, h_A, h_B, h_C_cpu, h_C_gpu_basic, h_C_gpu_tiled,
                 d_A, d_B, d_C, start, stop);
    
    runBenchmark(4096, h_A, h_B, h_C_cpu, h_C_gpu_basic, h_C_gpu_tiled,
                 d_A, d_B, d_C, start, stop);
    
    // Освобождаем память
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu_basic);
    free(h_C_gpu_tiled);

    fflush(stdout);
    return 0;
}