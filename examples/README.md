# cuBLAS, cuSPARSE, cuSOLVER, Thrust, cuDNN & TensorRT Examples

Демонстрационный проект с примерами использования библиотек **cuBLAS**, **cuSPARSE**, **cuSOLVER**, **Thrust**, **cuDNN** и **TensorRT**.

## Описание

Проект содержит шесть основных примеров:

1. **GEMM (cuBLAS)** - умножение плотных матриц `C = alpha*A*B + beta*C`
2. **Гибридный конвейер (cuSPARSE + cuBLAS)** - демонстрация совместной работы обеих библиотек в едином вычислительном pipeline:
   - SpMV с разреженной матрицей (cuSPARSE)
   - GEMV с плотной матрицей (cuBLAS)
   - Нормализация вектора (cuBLAS)
   - Скалярное произведение (cuBLAS)
3. **Решение разреженных систем (cuSOLVER + cuSPARSE)** - решение системы линейных уравнений `A*x = b` с разреженной матрицей:
   - Использование формата CSR для хранения разреженной матрицы
   - QR факторизация для решения системы
   - Верификация решения через вычисление невязки
4. **Thrust Parallel Algorithms** - высокоуровневые параллельные алгоритмы на GPU:
   - Работа с device/host векторами
   - Transform, Reduce, Sort операции
   - Пользовательские функторы
   - Скалярное произведение и нормализация
   - Сравнение производительности CPU vs GPU
5. **cuDNN MLP (Neural Network)** - обучение простой нейронной сети (персептрон) для аппроксимации функции sin(x):
   - Архитектура: Input(1) -> Hidden(16, ReLU) -> Output(1)
   - Обучение с использованием cuDNN и cuBLAS для матричных операций
   - Forward и backward propagation
   - SGD оптимизатор
   - Экспорт обученной модели в ONNX-подобный формат
6. **TensorRT Inference** - полный цикл: обучение с cuDNN -> экспорт в ONNX -> инференс с TensorRT:
   - Переиспользование кода обучения из cuDNN модуля
   - Построение TensorRT engine из обученной модели
   - Оптимизированный инференс на GPU
   - Сравнение результатов TensorRT с cuDNN и эталонными значениями
   - Замер производительности инференса

Все модули используют общий заголовочный файл с макросами для проверки ошибок.

## Компиляция

```bash
make
```

Или с указанием архитектуры GPU:

```bash
nvcc -O2 -std=c++14 -arch=sm_80 -I./include \
     -o cublas_cusparse_demo \
     src/main.cu src/gemm_example.cu src/spmv_example.cu src/solver_example.cu src/thrust_example.cu src/cudnn_mlp_example.cu src/tensorrt_example.cu \
     -lcublas -lcusparse -lcusolver -lcudnn -lnvinfer -lnvonnxparser
```

## Запуск примеров

Запустить все примеры:
```bash
make run
# или
./cublas_cusparse_demo all
```

Запустить отдельные примеры:
```bash
make run-gemm     # только GEMM
make run-spmv     # только SpMV pipeline
make run-solver   # только sparse solver
make run-thrust   # только Thrust
make run-cudnn    # только cuDNN MLP
make run-tensorrt # только TensorRT inference

# или напрямую:
./cublas_cusparse_demo gemm
./cublas_cusparse_demo spmv
./cublas_cusparse_demo solver
./cublas_cusparse_demo thrust
./cublas_cusparse_demo cudnn
./cublas_cusparse_demo tensorrt
```

### Модуль solver_example.cu

Демонстрирует решение разреженных систем линейных уравнений:
- Создание cuSOLVER handle для работы с разреженными матрицами
- Решение системы `A*x = b` методом QR факторизации
- Формат CSR для хранения разреженной матрицы (совместимость с cuSPARSE)
- Проверка на сингулярность матрицы
- Верификация решения через вычисление невязки `||A*x - b||`
- Применение: метод конечных элементов, анализ электрических цепей, оптимизация

**Основные функции:**
- `cusolverSpScsrlsvqr` - решение системы методом QR
- `cusparseCreateMatDescr` - создание дескриптора разреженной матрицы
- Верификация результата на CPU для проверки корректности

**Ключевые концепции:**
- Разреженные системы линейных уравнений
- QR факторизация vs Cholesky факторизация
- Численная стабильность решений

### Модуль thrust_example.cu

Демонстрирует использование Thrust для высокоуровневых параллельных вычислений:
- **Векторы:** `thrust::host_vector` и `thrust::device_vector` с автоматическим управлением памятью
- **Transform:** поэлементные преобразования с пользовательскими функторами (квадрат, модуль)
- **Reduce:** свёртка для вычисления суммы и максимума
- **Sort:** быстрая сортировка на GPU
- **Inner Product:** скалярное произведение векторов
- **Transform Reduce:** комбинированные операции (L2 норма)
- **Бинарные операции:** SAXPY (`z = a*x + y`)
- **Сравнение производительности:** CPU vs GPU для больших векторов

**Основные алгоритмы:**
- `thrust::transform` - применение функции к каждому элементу
- `thrust::reduce` - агрегация (сумма, max, min)
- `thrust::sort` - сортировка
- `thrust::inner_product` - dot product
- `thrust::transform_reduce` - комбинированная операция

**Ключевые концепции:**
- STL-подобный интерфейс для GPU
- Автоматическое управление памятью
- Функторы (встроенные и пользовательские)
- Высокая производительность без написания CUDA ядер
- Применение: data processing, scientific computing, ML preprocessing

### Модуль cudnn_mlp_example.cu

Демонстрирует использование cuDNN для обучения простой нейронной сети (персептрон):
- **Архитектура:** Input(1) -> Dense(16, ReLU) -> Dense(1)
- **Задача:** аппроксимация функции sin(x) на интервале [-Pi, Pi]
- **Обучение:**
  - Генерация 100 обучающих примеров
  - Mini-batch SGD (batch_size=32)
  - 1000 эпох с learning rate = 0.01
  - MSE loss функция
- **Компоненты:**
  - cuDNN для управления дескрипторами и активаций (ReLU)
  - cuBLAS для матричных операций (forward/backward pass)
  - Custom CUDA kernels для loss и градиентов
  - Xavier инициализация весов
- **Экспорт модели:**
  - Сохранение весов и bias в текстовый ONNX-подобный формат
  - Включает архитектуру, размеры слоёв и все параметры
  - Файл: `mlp_sin_model.onnx`

**Основные функции:**
- `cudnnCreate/cudnnDestroy` - управление cuDNN handle
- `cudnnSetTensor4dDescriptor` - описание тензоров
- `cudnnCreateActivationDescriptor` - настройка активационных функций
- `cublasSgemm` - матричные умножения для forward/backward
- Custom kernels: `relu_forward`, `relu_backward`, `mse_loss_backward`, `sgd_update`

**Ключевые концепции:**
- Forward и backward propagation
- Automatic differentiation через chain rule
- Batch processing для эффективного обучения
- SGD оптимизация
- Transfer learning готовность (экспорт модели)
- Применение: function approximation, time series prediction, regression tasks

### Модуль tensorrt_example.cu

Демонстрирует полный цикл: обучение -> экспорт -> оптимизированный инференс:
- **Этап 1: Обучение модели**
  - Переиспользование класса `SimpleMLPWithCuDNN` из заголовочного файла
  - Обучение персептрона на sin(x) (та же архитектура 1->16->1)
  - Автоматический экспорт в ONNX после обучения
- **Этап 2: Построение TensorRT Engine**
  - Парсинг ONNX файла с весами обученной модели
  - Программное построение сети через TensorRT API
  - Layer-by-layer: Input -> FC1 -> ReLU -> FC2 -> Output
  - Оптимизация engine (включая FP16 если доступно)
- **Этап 3: Инференс и валидация**
  - Запуск инференса на 10 тестовых примерах
  - Сравнение результатов TensorRT vs cuDNN vs sin(x)
  - Замер производительности (время инференса)
  - Валидация корректности (разница TensorRT-cuDNN ≈ 0)

**Основные функции:**
- `nvinfer1::createInferBuilder` - создание builder для TensorRT
- `IBuilder::createNetworkV2` - создание сети с явным batch
- `INetworkDefinition::addInput` - добавление входного слоя
- `INetworkDefinition::addFullyConnected` - добавление FC слоя
- `INetworkDefinition::addActivation` - добавление активации (ReLU)
- `IBuilder::buildSerializedNetwork` - построение и сериализация engine
- `IRuntime::deserializeCudaEngine` - десериализация engine
- `ICudaEngine::createExecutionContext` - создание контекста для инференса
- `IExecutionContext::enqueueV3` - асинхронный инференс

**Ключевые концепции:**
- Model deployment pipeline (training -> export -> optimization -> inference)
- TensorRT оптимизации: layer fusion, precision calibration, kernel auto-tuning
- Engine serialization для переиспользования
- Асинхронный инференс с CUDA streams
- Валидация точности после оптимизации
- Применение: production ML inference, real-time systems, embedded devices

**Особенности:**
- Полная интеграция с cuDNN модулем через shared header
- Максимальная переиспользуемость кода (класс MLP в `mlp_model.h`)
- Демонстрация всех этапов ML pipeline в одном примере
- Проверка численной эквивалентности TensorRT и cuDNN
- Измерение ускорения от TensorRT оптимизаций

## Особенности реализации

### Модуль gemm_example.cu

Демонстрирует:
- Создание cuBLAS handle
- Выделение памяти и копирование данных
- Вызов `cublasSgemm` для умножения матриц
- Column-major формат хранения матриц
- Правильное освобождение ресурсов

### Модуль spmv_example.cu

Демонстрирует совместную работу cuSPARSE и cuBLAS в едином вычислительном конвейере:
- **Разреженная матрица A** (3x3) в формате CSR (cuSPARSE)
- **Плотная матрица B** (3x3) в column-major формате (cuBLAS)
- **Вычислительный pipeline:**
  1. SpMV: y = Axx (разреженное умножение, 3x3 matrix x 3x1 vector)
  2. GEMV: z = Bxy (плотное умножение, 3x3 matrix x 3x1 vector)
  3. Вычисление нормы: ||z||_2 (L2-норма вектора)
  4. Нормализация: z = z / ||z||_2 (единичный вектор)
  5. Скалярное произведение: dot(x, z) (similarity метрика)
- Демонстрирует типичный паттерн в ML и численных методах
- Все операции выполняются на GPU без промежуточных копирований
- Применение: Graph Neural Networks, Sparse Linear Systems, Recommender Systems

## Полезные ссылки

- [NVIDIA cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [NVIDIA cuSPARSE Documentation](https://docs.nvidia.com/cuda/cusparse/)
- [NVIDIA cuSOLVER Documentation](https://docs.nvidia.com/cuda/cusolver/)
- [NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [NVIDIA Thrust Documentation](https://docs.nvidia.com/cuda/thrust/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Sparse Matrix Formats](https://en.wikipedia.org/wiki/Sparse_matrix)
- [ONNX Format Specification](https://onnx.ai/)
