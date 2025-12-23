#include "../include/cuda_utils.h"
#include "../include/mlp_model.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <iomanip>
#include <memory>
#include <sstream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>

/**
 * Пример использования TensorRT для инференса модели
 * Задача: обучение персептрона для sin(x) с cuDNN, экспорт в ONNX и запуск инференса с TensorRT
 * 
 * Этапы:
 * 1. Обучение модели с использованием cuDNN (переиспользуем код из cudnn_mlp_example)
 * 2. Экспорт обученной модели в ONNX формат
 * 3. Построение TensorRT engine из ONNX
 * 4. Запуск инференса на 10 примерах
 * 5. Сравнение результатов TensorRT с эталонными значениями sin(x)
 */

// Logger для TensorRT
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Фильтруем слишком подробные сообщения
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// Структура для хранения параметров модели из файла
struct MLPModel {
    int input_size;
    int hidden_size;
    int output_size;
    std::vector<float> w1;  // layer1 weights
    std::vector<float> b1;  // layer1 bias
    std::vector<float> w2;  // layer2 weights
    std::vector<float> b2;  // layer2 bias
};

// Парсинг упрощенного ONNX файла
bool parse_model_file(const std::string& filename, MLPModel& model) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open model file: " << filename << std::endl;
        return false;
    }

    // Инициализация значений по умолчанию
    model.input_size = 0;
    model.hidden_size = 0;
    model.output_size = 0;
    model.w1.clear();
    model.b1.clear();
    model.w2.clear();
    model.b2.clear();

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("input_size:") != std::string::npos) {
            if (sscanf(line.c_str(), "input_size: %d", &model.input_size) != 1) {
                std::cerr << "Failed to parse input_size" << std::endl;
            }
        } else if (line.find("hidden_size:") != std::string::npos) {
            if (sscanf(line.c_str(), "hidden_size: %d", &model.hidden_size) != 1) {
                std::cerr << "Failed to parse hidden_size" << std::endl;
            }
        } else if (line.find("output_size:") != std::string::npos) {
            if (sscanf(line.c_str(), "output_size: %d", &model.output_size) != 1) {
                std::cerr << "Failed to parse output_size" << std::endl;
            }
        } else if (line.find("layer1_weights:") != std::string::npos) {
            if (!std::getline(file, line)) {
                std::cerr << "Failed to read layer1_weights data" << std::endl;
                file.close();
                return false;
            }
            std::istringstream iss(line);
            float value;
            while (iss >> value) {
                model.w1.push_back(value);
            }
        } else if (line.find("layer1_bias:") != std::string::npos) {
            if (!std::getline(file, line)) {
                std::cerr << "Failed to read layer1_bias data" << std::endl;
                file.close();
                return false;
            }
            std::istringstream iss(line);
            float value;
            while (iss >> value) {
                model.b1.push_back(value);
            }
        } else if (line.find("layer2_weights:") != std::string::npos) {
            if (!std::getline(file, line)) {
                std::cerr << "Failed to read layer2_weights data" << std::endl;
                file.close();
                return false;
            }
            std::istringstream iss(line);
            float value;
            while (iss >> value) {
                model.w2.push_back(value);
            }
        } else if (line.find("layer2_bias:") != std::string::npos) {
            if (!std::getline(file, line)) {
                std::cerr << "Failed to read layer2_bias data" << std::endl;
                file.close();
                return false;
            }
            std::istringstream iss(line);
            float value;
            while (iss >> value) {
                model.b2.push_back(value);
            }
        }
    }

    file.close();

    // Проверка, что все необходимые данные были прочитаны
    if (model.input_size <= 0 || model.hidden_size <= 0 || model.output_size <= 0) {
        std::cerr << "Invalid or missing model dimensions in file" << std::endl;
        return false;
    }

    if (model.w1.empty() || model.b1.empty() || model.w2.empty() || model.b2.empty()) {
        std::cerr << "Missing weight or bias data in file" << std::endl;
        return false;
    }

    return true;
}

// Преобразование весов из column-major (cuBLAS) в row-major (TensorRT)
std::vector<float> convertColumnMajorToRowMajor(const std::vector<float>& data,
                                                int rows, int cols) {
    std::vector<float> row_major(data.size());
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            row_major[r * cols + c] = data[c * rows + r];
        }
    }
    return row_major;
}

// Deleter для TensorRT объектов
template <typename T>
struct TRTDestroy {
    void operator()(T* obj) const {
        if (obj) {
            obj->destroy();
        }
    }
};

// Класс для работы с TensorRT
class TensorRTInference {
private:
    std::unique_ptr<nvinfer1::IRuntime, TRTDestroy<nvinfer1::IRuntime>> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroy<nvinfer1::ICudaEngine>> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroy<nvinfer1::IExecutionContext>> context;
    
    void* d_input;
    void* d_output;
    int input_size;
    int output_size;
    
    cudaStream_t stream;
    
public:
    TensorRTInference() : d_input(nullptr), d_output(nullptr), input_size(0), output_size(0) {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    cudaStream_t getStream() const {
        return stream;
    }

    ~TensorRTInference() {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (stream) cudaStreamDestroy(stream);
    }
    
    // Построение TensorRT engine из модели
    bool build_engine(const MLPModel& model) {
        INFO("Building TensorRT engine...");

        // Проверка корректности параметров модели
        if (model.input_size <= 0 || model.hidden_size <= 0 || model.output_size <= 0) {
            std::cerr << "Invalid model dimensions" << std::endl;
            return false;
        }

        if (model.w1.empty() || model.b1.empty() || model.w2.empty() || model.b2.empty()) {
            std::cerr << "Model weights are empty" << std::endl;
            return false;
        }

        // Проверка размеров весов
        size_t expected_w1_size = static_cast<size_t>(model.input_size) * model.hidden_size;
        size_t expected_w2_size = static_cast<size_t>(model.hidden_size) * model.output_size;

        if (model.w1.size() != expected_w1_size) {
            std::cerr << "Layer 1 weights size mismatch: expected " << expected_w1_size
                      << ", got " << model.w1.size() << std::endl;
            return false;
        }

        if (model.b1.size() != static_cast<size_t>(model.hidden_size)) {
            std::cerr << "Layer 1 bias size mismatch: expected " << model.hidden_size
                      << ", got " << model.b1.size() << std::endl;
            return false;
        }

        if (model.w2.size() != expected_w2_size) {
            std::cerr << "Layer 2 weights size mismatch: expected " << expected_w2_size
                      << ", got " << model.w2.size() << std::endl;
            return false;
        }

        if (model.b2.size() != static_cast<size_t>(model.output_size)) {
            std::cerr << "Layer 2 bias size mismatch: expected " << model.output_size
                      << ", got " << model.b2.size() << std::endl;
            return false;
        }

        input_size = model.input_size;
        output_size = model.output_size;
        
        // Создание builder
        auto builder = std::unique_ptr<nvinfer1::IBuilder, TRTDestroy<nvinfer1::IBuilder>>(
            nvinfer1::createInferBuilder(gLogger));
        if (!builder) {
            std::cerr << "Failed to create builder" << std::endl;
            return false;
        }
        
        // Создание network
        const auto explicitBatch = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition, TRTDestroy<nvinfer1::INetworkDefinition>>(
            builder->createNetworkV2(explicitBatch));
        if (!network) {
            std::cerr << "Failed to create network" << std::endl;
            return false;
        }
        
        // Добавление входного слоя
        // Используем Dims для MLP: {batch_size, input_features}
        nvinfer1::Dims inputDims;
        inputDims.nbDims = 2;
        inputDims.d[0] = 1;  // batch_size
        inputDims.d[1] = model.input_size;  // input_features
        auto input = network->addInput("input", nvinfer1::DataType::kFLOAT, inputDims);
        if (!input) {
            std::cerr << "Failed to add input layer" << std::endl;
            return false;
        }
        
        // Layer 1: Fully Connected
        std::vector<float> w1_row = convertColumnMajorToRowMajor(
            model.w1, model.hidden_size, model.input_size);
        nvinfer1::Weights w1{nvinfer1::DataType::kFLOAT, w1_row.data(), 
                             static_cast<int64_t>(w1_row.size())};
        nvinfer1::Weights b1{nvinfer1::DataType::kFLOAT, model.b1.data(), 
                             static_cast<int64_t>(model.b1.size())};
        
        auto fc1 = network->addFullyConnected(*input, model.hidden_size, w1, b1);
        if (!fc1) {
            std::cerr << "Failed to add FC1 layer" << std::endl;
            return false;
        }
        
        // Activation: ReLU
        auto relu = network->addActivation(*fc1->getOutput(0), 
                                          nvinfer1::ActivationType::kRELU);
        if (!relu) {
            std::cerr << "Failed to add ReLU layer" << std::endl;
            return false;
        }
        
        // Layer 2: Fully Connected
        std::vector<float> w2_row = convertColumnMajorToRowMajor(
            model.w2, model.output_size, model.hidden_size);
        nvinfer1::Weights w2{nvinfer1::DataType::kFLOAT, w2_row.data(), 
                             static_cast<int64_t>(w2_row.size())};
        nvinfer1::Weights b2{nvinfer1::DataType::kFLOAT, model.b2.data(), 
                             static_cast<int64_t>(model.b2.size())};
        
        auto fc2 = network->addFullyConnected(*relu->getOutput(0), model.output_size, w2, b2);
        if (!fc2) {
            std::cerr << "Failed to add FC2 layer" << std::endl;
            return false;
        }
        
        // Маркируем выход
        network->markOutput(*fc2->getOutput(0));
        fc2->getOutput(0)->setName("output");
        
        // Создание конфигурации
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig, TRTDestroy<nvinfer1::IBuilderConfig>>(
            builder->createBuilderConfig());
        if (!config) {
            std::cerr << "Failed to create builder config" << std::endl;
            return false;
        }
        
        // Устанавливаем максимальный размер workspace (256 MB)
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 256 * (1 << 20));
        
        // Включаем FP16 если доступно
        if (builder->platformHasFastFp16()) {
            INFO("Enabling FP16 mode");
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        
        // Строим engine
        INFO("Serializing engine (this may take a while)...");
        auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory, TRTDestroy<nvinfer1::IHostMemory>>(
            builder->buildSerializedNetwork(*network, *config));
        if (!serializedEngine) {
            std::cerr << "Failed to build engine" << std::endl;
            return false;
        }
        
        // Создание runtime и десериализация engine
        runtime.reset(nvinfer1::createInferRuntime(gLogger));
        if (!runtime) {
            std::cerr << "Failed to create runtime" << std::endl;
            return false;
        }
        
        engine.reset(runtime->deserializeCudaEngine(serializedEngine->data(), 
                                                    serializedEngine->size()));
        if (!engine) {
            std::cerr << "Failed to deserialize engine" << std::endl;
            return false;
        }
        
        // Создание execution context
        context.reset(engine->createExecutionContext());
        if (!context) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
        
        // Выделение памяти для входа и выхода
        CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));
        
        INFO("TensorRT engine built successfully");
        return true;
    }
    
    // Инференс для одного примера (batch_size = 1)
    float infer(float input_value) {
        // Проверка: этот метод работает только для моделей с input_size = 1
        if (input_size != 1) {
            std::cerr << "infer() requires input_size = 1, got " << input_size << std::endl;
            return 0.0f;
        }
        
        // Копируем вход на GPU (используем синхронное копирование для stack variable)
        CUDA_CHECK(cudaMemcpy(d_input, &input_value, sizeof(float),
                              cudaMemcpyHostToDevice));

        // Получаем индексы для input и output
        // Можно использовать getIOTensorName() + setTensorAddress() с enqueueV3()
        int32_t inputIndex = engine->getBindingIndex("input");
        int32_t outputIndex = engine->getBindingIndex("output");

        if (inputIndex == -1 || outputIndex == -1) {
            std::cerr << "Failed to get binding indices" << std::endl;
            return 0.0f;
        }

        // Создаем массив указателей на buffers в правильном порядке
        // Массив должен содержать указатели в порядке индексов биндингов
        void* bindings[2];
        bindings[inputIndex] = d_input;
        bindings[outputIndex] = d_output;

        // Выполняем инференс
        bool status = context->enqueueV2(bindings, stream, nullptr);
        if (!status) {
            std::cerr << "Failed to enqueue inference" << std::endl;
            return 0.0f;
        }

        // Копируем результат обратно (синхронное копирование)
        float result;
        CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        return result;
    }
    
    // Пакетный инференс
    void infer_batch(const std::vector<float>& inputs, std::vector<float>& outputs) {
        if (inputs.empty()) {
            outputs.clear();
            return;
        }

        if (inputs.size() % static_cast<size_t>(input_size) != 0) {
            std::cerr << "Input buffer size (" << inputs.size()
                      << ") is not divisible by input_size (" << input_size << ")" << std::endl;
            outputs.clear();
            return;
        }

        size_t sample_count = inputs.size() / input_size;
        outputs.resize(sample_count);

        // Получаем индексы биндингов один раз для всех элементов
        int32_t inputIndex = engine->getBindingIndex("input");
        int32_t outputIndex = engine->getBindingIndex("output");

        if (inputIndex == -1 || outputIndex == -1) {
            std::cerr << "Failed to get binding indices in batch inference" << std::endl;
            outputs.clear();
            return;
        }

        void* bindings[2];
        bindings[inputIndex] = d_input;
        bindings[outputIndex] = d_output;

        // Обрабатываем каждый элемент последовательно
        for (size_t i = 0; i < sample_count; ++i) {
            const float* sample_ptr = inputs.data() + i * input_size;
            CUDA_CHECK(cudaMemcpy(d_input, sample_ptr, input_size * sizeof(float),
                                  cudaMemcpyHostToDevice));

            bool status = context->enqueueV2(bindings, stream, nullptr);
            if (!status) {
                std::cerr << "Failed to enqueue inference for element " << i << std::endl;
                outputs[i] = 0.0f;
                continue;
            }

            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaMemcpy(&outputs[i], d_output, sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
    }
};

void run_tensorrt_example() {
    HEADER("TensorRT Inference Example: sin(x) Perceptron");
    
    const std::string model_filename = "mlp_sin_model.onnx";
    
    // ==== ШАГ 1: Обучение модели с cuDNN (реюзаем код) ====
    HEADER("Step 1: Training Model with cuDNN");
    
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
    
    // Генерация обучающих данных
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
    
    INFO("Training in progress...");
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<int> indices(num_samples);
        for (int i = 0; i < num_samples; ++i) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), gen);
        
        float total_loss = 0.0f;
        int num_batches = 0;
        
        for (int start = 0; start < num_samples; start += batch_size) {
            int actual_batch = std::min(batch_size, num_samples - start);
            if (actual_batch < batch_size) break;
            
            std::vector<float> batch_x(batch_size);
            std::vector<float> batch_y(batch_size);
            
            for (int i = 0; i < actual_batch; ++i) {
                int idx = indices[start + i];
                batch_x[i] = train_x[idx];
                batch_y[i] = train_y[idx];
            }
            
            float loss = model.forward(batch_x, batch_y);
            total_loss += loss;
            num_batches++;
            
            model.backward_and_update(learning_rate);
        }
        
        if ((epoch + 1) % 200 == 0) {
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
    
    RESULT("Training completed!");
    
    // ==== ШАГ 2: Экспорт в ONNX ====
    HEADER("Step 2: Exporting Model to ONNX");
    model.export_to_onnx(model_filename);
    
    // Парсим ONNX для получения параметров модели
    INFO("Loading model from " << model_filename);
    MLPModel mlp_params;
    if (!parse_model_file(model_filename, mlp_params)) {
        std::cerr << "Failed to parse model file" << std::endl;
        return;
    }
    
    INFO("Model loaded: " << mlp_params.input_size << " -> " 
         << mlp_params.hidden_size << " -> " << mlp_params.output_size);
    
    // ==== ШАГ 3: Построение TensorRT engine ====
    HEADER("Step 3: Building TensorRT Engine");
    TensorRTInference trt_inference;
    if (!trt_inference.build_engine(mlp_params)) {
        std::cerr << "Failed to build TensorRT engine" << std::endl;
        return;
    }
    
    // ==== ШАГ 4: Запуск инференса на 10 примерах ====
    HEADER("Step 4: Running TensorRT Inference on 10 Test Examples");
    
    std::vector<float> test_inputs;
    std::vector<float> ground_truth;
    std::vector<float> cudnn_predictions;
    std::vector<float> trt_input_buffer;
    
    // Подготовка тестовых данных
    INFO("Preparing test data...");
    for (int i = 0; i < 10; ++i) {
        float x = -M_PI + (2.0f * M_PI * i) / 9.0f;
        test_inputs.push_back(x);
        ground_truth.push_back(sinf(x));
        cudnn_predictions.push_back(model.predict(x));  // Эталон от cuDNN
        trt_input_buffer.push_back(x);
    }
    
    // Запуск TensorRT инференса
    INFO("Running TensorRT inference...");
    std::vector<float> trt_predictions;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, trt_inference.getStream()));
    trt_inference.infer_batch(trt_input_buffer, trt_predictions);
    CUDA_CHECK(cudaEventRecord(stop, trt_inference.getStream()));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float inference_time;
    CUDA_CHECK(cudaEventElapsedTime(&inference_time, start, stop));

    // Проверка, что инференс прошел успешно
    size_t expected_predictions = trt_input_buffer.size() / input_size;
    if (trt_predictions.size() != expected_predictions) {
        std::cerr << "TensorRT inference failed: expected " << expected_predictions
                  << " predictions, got " << trt_predictions.size() << std::endl;
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return;
    }
    
    // ==== ШАГ 5: Сравнение результатов ====
    HEADER("Step 5: Comparing Results");
    
    std::cout << "\n" << std::setw(10) << "x" 
              << std::setw(15) << "sin(x)" 
              << std::setw(15) << "cuDNN Pred"
              << std::setw(18) << "TensorRT Pred"
              << std::setw(15) << "TRT Error\n";
    std::cout << std::string(73, '-') << "\n";
    
    float total_trt_error = 0.0f;
    float max_trt_error = 0.0f;
    float total_cudnn_trt_diff = 0.0f;
    
    for (size_t i = 0; i < test_inputs.size(); ++i) {
        float trt_error = fabsf(ground_truth[i] - trt_predictions[i]);
        float cudnn_trt_diff = fabsf(cudnn_predictions[i] - trt_predictions[i]);
        
        total_trt_error += trt_error;
        max_trt_error = std::max(max_trt_error, trt_error);
        total_cudnn_trt_diff += cudnn_trt_diff;
        
        std::cout << std::fixed << std::setprecision(6)
                  << std::setw(10) << test_inputs[i]
                  << std::setw(15) << ground_truth[i]
                  << std::setw(15) << cudnn_predictions[i]
                  << std::setw(18) << trt_predictions[i]
                  << std::setw(15) << trt_error << "\n";
    }
    
    // Статистика
    HEADER("Performance Statistics");

    // Проверка на деление на ноль (хотя test_inputs всегда имеет 10 элементов)
    if (test_inputs.empty()) {
        std::cerr << "No test inputs to compute statistics" << std::endl;
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return;
    }

    float avg_trt_error = total_trt_error / test_inputs.size();
    float avg_cudnn_trt_diff = total_cudnn_trt_diff / test_inputs.size();
    
    INFO("TensorRT vs Ground Truth:");
    INFO("  Average absolute error: " << std::fixed << std::setprecision(8) << avg_trt_error);
    INFO("  Maximum absolute error: " << std::fixed << std::setprecision(8) << max_trt_error);
    INFO("");
    INFO("TensorRT vs cuDNN:");
    INFO("  Average difference: " << std::fixed << std::setprecision(8) << avg_cudnn_trt_diff);
    INFO("  (Should be near zero - validates correct implementation)");
    INFO("");
    INFO("Inference Performance:");
    INFO("  Total time (10 samples): " << std::fixed << std::setprecision(3) 
         << inference_time << " ms");
    INFO("  Average per sample: " << std::fixed << std::setprecision(3) 
         << inference_time / test_inputs.size() << " ms");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    RESULT("TensorRT inference completed successfully!");
    INFO("Workflow: cuDNN Training -> ONNX Export -> TensorRT Engine -> Inference");
}
