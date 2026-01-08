#include "../include/cuda_utils.h"
#include "../include/mlp_model.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include <string>
#include <vector>

/**
 * Пример использования cuDNN для обучения простого персептрона (MLP)
 * Задача: аппроксимация функции sin(x)
 * Архитектура: Input(1) -> Hidden(16) -> Output(1)
 *
 * Этапы:
 * 1. Генерация обучающих данных
 * 2. Создание и инициализация нейросети
 * 3. Обучение с использованием cuDNN
 * 4. Экспорт модели в ONNX формат
 */

void run_cudnn_mlp_example() {
    HEADER("cuDNN MLP Example: Learning sin(x)");
    
    // Гиперпараметры
    const int batch_size = 32;
    const int input_size = 1;
    const int hidden_size = 16;
    const int output_size = 1;
    const int epochs = 1000;
    const float learning_rate = 0.01f;
    const int num_samples = 100;
    
    INFO("Training parameters:");
    INFO("  Batch size: " << batch_size);
    INFO("  Hidden units: " << hidden_size);
    INFO("  Learning rate: " << learning_rate);
    INFO("  Epochs: " << epochs);
    
    // Создание модели
    SimpleMLPWithCuDNN model(batch_size, input_size, hidden_size, output_size);
    
    // Генерация обучающих данных: sin(x) для x в [-Pi, Pi]
    std::vector<float> train_x(num_samples);
    std::vector<float> train_y(num_samples);
    
    for (int i = 0; i < num_samples; ++i) {
        train_x[i] = -M_PI + (2.0f * M_PI * i) / (num_samples - 1);
        train_y[i] = sinf(train_x[i]);
    }
    
    INFO("Generated " << num_samples << " training samples");
    
    // Обучение
    std::random_device rd;
    std::mt19937 gen(rd());
    
    HEADER("Training Progress");
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Перемешивание данных
        std::vector<int> indices(num_samples);
        for (int i = 0; i < num_samples; ++i) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), gen);
        
        float total_loss = 0.0f;
        int num_batches = 0;
        
        // Мини-батчи
        for (int start = 0; start < num_samples; start += batch_size) {
            // Пропускаем неполный батч
            if (start + batch_size > num_samples) break;

            std::vector<float> batch_x(batch_size);
            std::vector<float> batch_y(batch_size);

            for (int i = 0; i < batch_size; ++i) {
                int idx = indices[start + i];
                batch_x[i] = train_x[idx];
                batch_y[i] = train_y[idx];
            }
            
            // Forward pass
            float loss = model.forward(batch_x, batch_y);
            total_loss += loss;
            num_batches++;
            
            // Backward pass
            model.backward_and_update(learning_rate);
        }
        
        // Вывод прогресса
        if ((epoch + 1) % 100 == 0 || epoch == 0) {
            if (num_batches > 0) {
                float avg_loss = total_loss / num_batches;
                INFO("Epoch " << std::setw(4) << (epoch + 1) << "/" << epochs 
                     << " | Loss: " << std::fixed << std::setprecision(6) << avg_loss);
            } else {
                INFO("Epoch " << std::setw(4) << (epoch + 1) << "/" << epochs
                     << " | No complete batches processed");
            }
        }
    }
    
    HEADER("Training Completed");
    
    // Тестирование модели
    HEADER("Testing Model");
    std::cout << "\nPredictions vs Ground Truth:\n";
    std::cout << std::setw(10) << "x" << std::setw(15) << "sin(x)" 
              << std::setw(15) << "predicted" << std::setw(15) << "error\n";
    std::cout << std::string(55, '-') << "\n";
    
    for (int i = 0; i < 10; ++i) {
        float x = -M_PI + (2.0f * M_PI * i) / 9.0f;
        float true_y = sinf(x);
        float pred_y = model.predict(x);
        float error = fabsf(true_y - pred_y);
        
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(10) << x
                  << std::setw(15) << true_y
                  << std::setw(15) << pred_y
                  << std::setw(15) << error << "\n";
    }
    
    // Экспорт в ONNX
    HEADER("Exporting Model");
    model.export_to_onnx("mlp_sin_model.onnx");
    
    RESULT("Model trained and exported successfully!");
}
