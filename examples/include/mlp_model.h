#ifndef MLP_MODEL_H
#define MLP_MODEL_H

#include "cuda_utils.h"
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm>

// Структура для хранения параметров слоя
struct LayerParams {
    float *d_weights;  // веса на GPU
    float *d_bias;     // смещения на GPU
    float *d_dweights; // градиенты весов
    float *d_dbias;    // градиенты смещений
    int input_size;
    int output_size;
};

// Простая активация ReLU на GPU
__global__ void relu_forward(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Градиент ReLU
__global__ void relu_backward(float *grad, const float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = (data[idx] > 0.0f) ? grad[idx] : 0.0f;
    }
}

// Вычисление MSE loss и градиента
__global__ void mse_loss_backward(float *grad, const float *pred, const float *target, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 2.0f * (pred[idx] - target[idx]) / size;
    }
}

// SGD update
__global__ void sgd_update(float *params, const float *grads, int size, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= lr * grads[idx];
    }
}

class SimpleMLPWithCuDNN {
private:
    cudnnHandle_t cudnn_handle;
    cublasHandle_t cublas_handle;
    
    // Параметры сети
    LayerParams layer1;  // Input -> Hidden
    LayerParams layer2;  // Hidden -> Output
    
    // Размеры
    int batch_size;
    int input_size;
    int hidden_size;
    int output_size;
    
    // Буферы для forward/backward pass
    float *d_input;
    float *d_hidden;
    float *d_hidden_activated;
    float *d_output;
    float *d_target;
    
    // Градиенты
    float *d_grad_output;
    float *d_grad_hidden;
    
    // Дескрипторы для cuDNN
    cudnnTensorDescriptor_t input_desc, hidden_desc, output_desc;
    cudnnActivationDescriptor_t relu_desc;
    
public:
    SimpleMLPWithCuDNN(int batch_sz, int in_sz, int hid_sz, int out_sz)
        : batch_size(batch_sz), input_size(in_sz), hidden_size(hid_sz), output_size(out_sz) {
        
        // Создание handles
        CUDNN_CHECK(cudnnCreate(&cudnn_handle));
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        
        // Инициализация параметров слоев
        init_layer(layer1, input_size, hidden_size);
        init_layer(layer2, hidden_size, output_size);
        
        // Выделение памяти для буферов
        CUDA_CHECK(cudaMalloc(&d_input, batch_size * input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hidden, batch_size * hidden_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hidden_activated, batch_size * hidden_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, batch_size * output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_target, batch_size * output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_output, batch_size * output_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_hidden, batch_size * hidden_size * sizeof(float)));
        
        // Создание дескрипторов тензоров
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&hidden_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                batch_size, input_size, 1, 1));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(hidden_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                batch_size, hidden_size, 1, 1));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                batch_size, output_size, 1, 1));
        
        // Создание дескриптора активации ReLU
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&relu_desc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(relu_desc, CUDNN_ACTIVATION_RELU,
                                                  CUDNN_NOT_PROPAGATE_NAN, 0.0));
        
        INFO("MLP initialized: " << input_size << " -> " << hidden_size << " -> " << output_size);
    }
    
    ~SimpleMLPWithCuDNN() {
        // Очистка памяти
        cudaFree(d_input);
        cudaFree(d_hidden);
        cudaFree(d_hidden_activated);
        cudaFree(d_output);
        cudaFree(d_target);
        cudaFree(d_grad_output);
        cudaFree(d_grad_hidden);
        
        free_layer(layer1);
        free_layer(layer2);
        
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(hidden_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyActivationDescriptor(relu_desc);
        
        cudnnDestroy(cudnn_handle);
        cublasDestroy(cublas_handle);
    }
    
    void init_layer(LayerParams &layer, int in_sz, int out_sz) {
        layer.input_size = in_sz;
        layer.output_size = out_sz;
        
        // Выделение памяти
        CUDA_CHECK(cudaMalloc(&layer.d_weights, in_sz * out_sz * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layer.d_bias, out_sz * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layer.d_dweights, in_sz * out_sz * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&layer.d_dbias, out_sz * sizeof(float)));
        
        // Инициализация Xavier
        std::vector<float> h_weights(in_sz * out_sz);
        std::vector<float> h_bias(out_sz, 0.0f);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        float scale = sqrtf(2.0f / (in_sz + out_sz));
        std::normal_distribution<float> dist(0.0f, scale);
        
        for (auto &w : h_weights) w = dist(gen);
        
        CUDA_CHECK(cudaMemcpy(layer.d_weights, h_weights.data(), 
                              in_sz * out_sz * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(layer.d_bias, h_bias.data(), 
                              out_sz * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    void free_layer(LayerParams &layer) {
        cudaFree(layer.d_weights);
        cudaFree(layer.d_bias);
        cudaFree(layer.d_dweights);
        cudaFree(layer.d_dbias);
    }
    
    // Forward pass через слой: output = input * weights^T + bias
    void forward_linear(const float *input, float *output, const LayerParams &layer) {
        float alpha = 1.0f, beta = 0.0f;
        
        // Умножение матриц: output = weights * input (column-major)
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 layer.output_size, batch_size, layer.input_size,
                                 &alpha,
                                 layer.d_weights, layer.output_size,
                                 input, layer.input_size,
                                 &beta,
                                 output, layer.output_size));
        
        // Добавление bias к каждому элементу батча
        alpha = 1.0f; beta = 1.0f;
        for (int b = 0; b < batch_size; ++b) {
            CUBLAS_CHECK(cublasSaxpy(cublas_handle, layer.output_size,
                                     &alpha, layer.d_bias, 1,
                                     output + b * layer.output_size, 1));
        }
    }
    
    // Forward pass через всю сеть
    float forward(const std::vector<float> &h_input, const std::vector<float> &h_target) {
        // Копируем данные на GPU
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), 
                              batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_target, h_target.data(), 
                              batch_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Layer 1: Input -> Hidden
        forward_linear(d_input, d_hidden, layer1);
        
        // Activation: ReLU
        int hidden_total = batch_size * hidden_size;
        int threads = 256;
        int blocks = (hidden_total + threads - 1) / threads;
        
        // Копируем для backward pass
        CUDA_CHECK(cudaMemcpy(d_hidden_activated, d_hidden, 
                              hidden_total * sizeof(float), cudaMemcpyDeviceToDevice));
        relu_forward<<<blocks, threads>>>(d_hidden_activated, hidden_total);
        CUDA_CHECK(cudaGetLastError());
        
        // Layer 2: Hidden -> Output
        forward_linear(d_hidden_activated, d_output, layer2);
        
        // Вычисление MSE loss на CPU для мониторинга
        std::vector<float> h_output(batch_size * output_size);
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, 
                              batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        float loss = 0.0f;
        for (int i = 0; i < batch_size * output_size; ++i) {
            float diff = h_output[i] - h_target[i];
            loss += diff * diff;
        }
        return loss / (batch_size * output_size);
    }
    
    // Backward pass и обновление весов
    void backward_and_update(float learning_rate) {
        // Градиент loss по output
        int output_total = batch_size * output_size;
        int threads = 256;
        int blocks = (output_total + threads - 1) / threads;
        mse_loss_backward<<<blocks, threads>>>(d_grad_output, d_output, d_target, output_total);
        CUDA_CHECK(cudaGetLastError());
        
        // Backward через layer2
        backward_linear(d_hidden_activated, d_grad_output, d_grad_hidden, layer2, learning_rate);
        
        // Backward через ReLU
        int hidden_total = batch_size * hidden_size;
        blocks = (hidden_total + threads - 1) / threads;
        relu_backward<<<blocks, threads>>>(d_grad_hidden, d_hidden, hidden_total);
        CUDA_CHECK(cudaGetLastError());
        
        // Backward через layer1
        float *dummy_grad = nullptr;
        backward_linear(d_input, d_grad_hidden, dummy_grad, layer1, learning_rate);
    }
    
    // Backward pass через линейный слой
    void backward_linear(const float *input, const float *grad_output, float *grad_input,
                         LayerParams &layer, float lr) {
        float alpha = 1.0f, beta = 0.0f;
        
        // Градиент по весам: dW = grad_output^T * input
        CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 layer.output_size, layer.input_size, batch_size,
                                 &alpha,
                                 grad_output, layer.output_size,
                                 input, layer.input_size,
                                 &beta,
                                 layer.d_dweights, layer.output_size));
        
        // Градиент по bias: сумма по батчу
        CUDA_CHECK(cudaMemset(layer.d_dbias, 0, layer.output_size * sizeof(float)));
        alpha = 1.0f;
        for (int b = 0; b < batch_size; ++b) {
            CUBLAS_CHECK(cublasSaxpy(cublas_handle, layer.output_size,
                                     &alpha, grad_output + b * layer.output_size, 1,
                                     layer.d_dbias, 1));
        }
        
        // Градиент по входу (если нужен)
        if (grad_input != nullptr) {
            CUBLAS_CHECK(cublasSgemm(cublas_handle,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     layer.input_size, batch_size, layer.output_size,
                                     &alpha,
                                     layer.d_weights, layer.output_size,
                                     grad_output, layer.output_size,
                                     &beta,
                                     grad_input, layer.input_size));
        }
        
        // Обновление параметров (SGD)
        int weight_count = layer.input_size * layer.output_size;
        int threads = 256;
        int blocks_w = (weight_count + threads - 1) / threads;
        int blocks_b = (layer.output_size + threads - 1) / threads;
        
        sgd_update<<<blocks_w, threads>>>(layer.d_weights, layer.d_dweights, weight_count, lr);
        sgd_update<<<blocks_b, threads>>>(layer.d_bias, layer.d_dbias, layer.output_size, lr);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Предсказание для одного входа
    float predict(float x) {
        std::vector<float> input(batch_size, x);
        std::vector<float> dummy_target(batch_size, 0.0f);
        forward(input, dummy_target);
        
        float result;
        CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
        return result;
    }
    
    // Экспорт в упрощенный ONNX-подобный формат
    void export_to_onnx(const std::string &filename) {
        INFO("Exporting model to " << filename);
        
        // Копируем веса на host
        std::vector<float> h_w1(layer1.input_size * layer1.output_size);
        std::vector<float> h_b1(layer1.output_size);
        std::vector<float> h_w2(layer2.input_size * layer2.output_size);
        std::vector<float> h_b2(layer2.output_size);
        
        CUDA_CHECK(cudaMemcpy(h_w1.data(), layer1.d_weights, 
                              h_w1.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b1.data(), layer1.d_bias, 
                              h_b1.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_w2.data(), layer2.d_weights, 
                              h_w2.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b2.data(), layer2.d_bias, 
                              h_b2.size() * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Сохранение в простой текстовый формат (упрощенный ONNX)
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
        
        file << std::fixed << std::setprecision(8);
        
        // Метаданные
        file << "# Simple MLP ONNX-like export\n";
        file << "model_type: sequential\n";
        file << "input_size: " << input_size << "\n";
        file << "hidden_size: " << hidden_size << "\n";
        file << "output_size: " << output_size << "\n\n";
        
        // Layer 1
        file << "layer1_type: linear\n";
        file << "layer1_weights: " << layer1.input_size << "x" << layer1.output_size << "\n";
        for (const auto &w : h_w1) file << w << " ";
        file << "\nlayer1_bias: " << layer1.output_size << "\n";
        for (const auto &b : h_b1) file << b << " ";
        file << "\n\n";
        
        // Activation
        file << "activation: relu\n\n";
        
        // Layer 2
        file << "layer2_type: linear\n";
        file << "layer2_weights: " << layer2.input_size << "x" << layer2.output_size << "\n";
        for (const auto &w : h_w2) file << w << " ";
        file << "\nlayer2_bias: " << layer2.output_size << "\n";
        for (const auto &b : h_b2) file << b << " ";
        file << "\n";
        
        file.close();
        INFO("Model exported successfully");
    }
};

#endif // MLP_MODEL_H
