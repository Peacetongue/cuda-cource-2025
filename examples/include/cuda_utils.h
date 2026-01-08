#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <cudnn.h>

// Макрос для проверки CUDA-статуса
#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Макрос проверки cuBLAS-статуса
#define CUBLAS_CHECK(x) do { \
    cublasStatus_t status = x; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Макрос проверки cuSPARSE-статуса
#define CUSPARSE_CHECK(x) do { \
    cusparseStatus_t status = x; \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        std::cerr << "cuSPARSE Error: " << status \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Макрос проверки cuSOLVER-статуса
#define CUSOLVER_CHECK(x) do { \
    cusolverStatus_t status = x; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        std::cerr << "cuSOLVER Error: " << status \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Макрос проверки cuDNN-статуса
#define CUDNN_CHECK(x) do { \
    cudnnStatus_t status = x; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Макрос для вывода информации
#define INFO(msg) std::cout << "[INFO] " << msg << std::endl

// Макрос для вывода результата
#define RESULT(msg) std::cout << "[RESULT] " << msg << std::endl

// Макрос для вывода заголовка
#define HEADER(msg) std::cout << "\n=== " << msg << " ===" << std::endl

#endif // CUDA_UTILS_H
