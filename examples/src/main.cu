#include "../include/cuda_utils.h"
#include <iostream>

// Объявления функций из других модулей
void run_gemm_example();
void run_spmv_example();
void run_solver_example();
void run_thrust_example();
void run_cudnn_mlp_example();
void run_tensorrt_example();

void print_gpu_info() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    HEADER("GPU Information");
    INFO("Device name: " << prop.name);
    INFO("Compute capability: " << prop.major << "." << prop.minor);
    INFO("Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB");
    INFO("Multiprocessors: " << prop.multiProcessorCount);
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [option]\n";
    std::cout << "Options:\n";
    std::cout << "  gemm      - Run cuBLAS GEMM example\n";
    std::cout << "  spmv      - Run cuSPARSE SpMV + cuBLAS pipeline example\n";
    std::cout << "  solver    - Run cuSOLVER sparse linear system solver example\n";
    std::cout << "  thrust    - Run Thrust parallel algorithms example\n";
    std::cout << "  cudnn     - Run cuDNN MLP (neural network) example\n";
    std::cout << "  tensorrt  - Run TensorRT inference example (requires cudnn model)\n";
    std::cout << "  all       - Run all examples (default)\n";
}

int main(int argc, char** argv) {
    HEADER("cuBLAS, cuSPARSE, cuSOLVER, Thrust, cuDNN & TensorRT Examples");
    
    // Выводим информацию о GPU
    print_gpu_info();
    
    // Определяем, какой пример запускать
    std::string mode = "all";
    if (argc > 1) {
        mode = argv[1];
    }
    
    try {
        if (mode == "gemm") {
            run_gemm_example();
        } else if (mode == "spmv") {
            run_spmv_example();
        } else if (mode == "solver") {
            run_solver_example();
        } else if (mode == "thrust") {
            run_thrust_example();
        } else if (mode == "cudnn") {
            run_cudnn_mlp_example();
        } else if (mode == "tensorrt") {
            run_tensorrt_example();
        } else if (mode == "all") {
            run_gemm_example();
            std::cout << std::endl;
            run_spmv_example();
            std::cout << std::endl;
            run_solver_example();
            std::cout << std::endl;
            run_thrust_example();
            std::cout << std::endl;
            run_cudnn_mlp_example();
            std::cout << std::endl;
            run_tensorrt_example();
        } else {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        
        HEADER("All examples completed successfully");
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
