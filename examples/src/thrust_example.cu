#include "../include/cuda_utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/inner_product.h>
#include <vector>
#include <cmath>
#include <iomanip>

/**
 * Пример использования библиотеки Thrust для параллельных вычислений на GPU
 * 
 * ЧТО ТАКОЕ THRUST?
 * -----------------
 * Thrust - это высокоуровневая C++ библиотека параллельных алгоритмов для CUDA.
 * Она предоставляет STL-подобный интерфейс для работы с GPU, что делает код
 * проще и понятнее по сравнению с написанием низкоуровневых CUDA ядер.
 * 
 * ОСНОВНЫЕ ПРЕИМУЩЕСТВА:
 * ----------------------
 * 1. Высокая производительность: оптимизированные реализации алгоритмов
 * 2. Простота использования: STL-подобный интерфейс
 * 3. Меньше кода: не нужно писать CUDA ядра для типовых операций
 * 4. Переносимость: код может работать и на CPU, и на GPU
 * 5. Безопасность: автоматическое управление памятью
 * 
 * ОСНОВНЫЕ КОНЦЕПЦИИ:
 * -------------------
 * 1. Векторы:
 *    - thrust::host_vector<T>   - вектор в памяти CPU
 *    - thrust::device_vector<T> - вектор в памяти GPU
 *    - Автоматическое копирование между host и device
 * 
 * 2. Алгоритмы:
 *    - transform: поэлементное преобразование
 *    - reduce: свёртка (сложение, умножение, min/max)
 *    - sort: сортировка
 *    - scan: префиксная сумма
 *    - inner_product: скалярное произведение
 * 
 * 3. Функторы:
 *    - Встроенные: thrust::plus, thrust::minus, thrust::multiplies, etc.
 *    - Пользовательские: собственные операции с __host__ __device__
 * 
 * КОГДА ИСПОЛЬЗОВАТЬ THRUST?
 * ---------------------------
 * ok: Нужны типовые операции: сортировка, свёртка, преобразования
 * ok: Прототипирование: быстро проверить идею
 * ok: Простая логика: не требуется сложная синхронизация
 * ok: Читаемость важнее контроля: STL-подобный код понятен всем
 * 
 * bad: Нужен полный контроль над памятью и потоками
 * bad: Очень специфичные операции, не покрываемые библиотекой
 * bad: Критична максимальная производительность (можно оптимизировать ядра)
 * 
 * ЧТО МЫ ДЕЛАЕМ В ЭТОМ ПРИМЕРЕ?
 * ------------------------------
 * Демонстрируем основные операции Thrust:
 * 1. Работа с векторами (host/device)
 * 2. Transform: вычисление квадратов элементов
 * 3. Reduce: нахождение суммы и максимума
 * 4. Sort: сортировка данных
 * 5. Inner product: скалярное произведение векторов
 * 6. Custom functors: пользовательские операции
 * 
 * ПРИМЕНЕНИЯ THRUST:
 * ------------------
 * - Обработка данных: фильтрация, трансформация, агрегация
 * - Научные вычисления: линейная алгебра, статистика
 * - Компьютерная графика: обработка точек, mesh processing
 * - Machine Learning: операции с тензорами, preprocessing
 * - Анализ данных: сортировка, поиск, группировка
 * 
 * ==========================================================================
 */

// Пользовательский функтор для возведения в квадрат
struct square_functor {
    __host__ __device__
    float operator()(float x) const {
        return x * x;
    }
};

// Пользовательский функтор для вычисления модуля
struct abs_functor {
    __host__ __device__
    float operator()(float x) const {
        return fabsf(x);
    }
};

// Пользовательский функтор для вычисления z = a*x + y (AXPY operation)
struct saxpy_functor {
    float a;
    
    saxpy_functor(float _a) : a(_a) {}
    
    __host__ __device__
    float operator()(float x, float y) const {
        return a * x + y;
    }
};

void run_thrust_example() {
    HEADER("Thrust Parallel Algorithms Example");
    
    INFO("1. Creating and initializing vectors");
    
    const int N = 10;
    
    // Создаём вектор на CPU
    thrust::host_vector<float> h_vec(N);
    
    // Заполняем данными: [1.0, 2.0, 3.0, ..., 10.0]
    for (int i = 0; i < N; i++) {
        h_vec[i] = static_cast<float>(i + 1);
    }
    
    std::cout << "  Initial vector: [ ";
    for (int i = 0; i < N; i++) {
        std::cout << h_vec[i] << " ";
    }
    std::cout << "]" << std::endl;
    
    // Автоматическое копирование на GPU
    thrust::device_vector<float> d_vec = h_vec;
    INFO("  Copied to device (GPU)");
    
    std::cout << std::endl;
    
    // ========== 2. Transform: Возведение в квадрат ==========
    INFO("2. Transform: Computing squares using custom functor");
    
    thrust::device_vector<float> d_squares(N);
    thrust::transform(d_vec.begin(), d_vec.end(), 
                     d_squares.begin(), 
                     square_functor());
    
    // Копируем результат обратно на CPU для вывода
    thrust::host_vector<float> h_squares = d_squares;
    
    std::cout << "  Squares: [ ";
    for (int i = 0; i < N; i++) {
        std::cout << h_squares[i] << " ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << std::endl;
    
    // ========== 3. Reduce: Сумма и максимум ==========
    INFO("3. Reduce: Computing sum and maximum");
    
    float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 
                               0.0f, 
                               thrust::plus<float>());
    
    float max_val = thrust::reduce(d_vec.begin(), d_vec.end(), 
                                   d_vec[0], 
                                   thrust::maximum<float>());
    
    RESULT("Sum of elements: " << sum);
    RESULT("Maximum element: " << max_val);
    
    std::cout << std::endl;
    
    // ========== 4. Inner Product: Скалярное произведение ==========
    INFO("4. Inner Product: Computing dot product");
    
    thrust::device_vector<float> d_vec2(N);
    thrust::sequence(d_vec2.begin(), d_vec2.end(), 1.0f);  // [1, 2, 3, ..., N]
    
    float dot_product = thrust::inner_product(d_vec.begin(), d_vec.end(),
                                             d_vec2.begin(),
                                             0.0f);
    
    RESULT("Dot product (vec1 · vec2): " << dot_product);
    
    std::cout << std::endl;
    
    // ========== 5. Transform with Binary Operation: SAXPY ==========
    INFO("5. Binary Transform: Computing z = 2.5*x + y (SAXPY)");
    
    thrust::device_vector<float> d_result(N);
    thrust::transform(d_vec.begin(), d_vec.end(),
                     d_vec2.begin(),
                     d_result.begin(),
                     saxpy_functor(2.5f));
    
    thrust::host_vector<float> h_result = d_result;
    
    std::cout << "  Result (2.5*x + y): [ ";
    for (int i = 0; i < N; i++) {
        std::cout << h_result[i] << " ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << std::endl;
    
    // ========== 6. Sort: Сортировка ==========
    INFO("6. Sort: Sorting vector in descending order");
    
    // Создаём неупорядоченный вектор
    thrust::device_vector<float> d_unsorted(N);
    h_vec[0] = 5.0f; h_vec[1] = 2.0f; h_vec[2] = 9.0f; 
    h_vec[3] = 1.0f; h_vec[4] = 7.0f; h_vec[5] = 3.0f;
    h_vec[6] = 8.0f; h_vec[7] = 4.0f; h_vec[8] = 6.0f; 
    h_vec[9] = 10.0f;
    
    d_unsorted = h_vec;
    
    thrust::host_vector<float> h_unsorted = d_unsorted;
    std::cout << "  Before sort: [ ";
    for (int i = 0; i < N; i++) {
        std::cout << h_unsorted[i] << " ";
    }
    std::cout << "]" << std::endl;
    
    // Сортировка по убыванию
    thrust::sort(d_unsorted.begin(), d_unsorted.end(), thrust::greater<float>());
    
    thrust::host_vector<float> h_sorted = d_unsorted;
    std::cout << "  After sort:  [ ";
    for (int i = 0; i < N; i++) {
        std::cout << h_sorted[i] << " ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << std::endl;
    
    // ========== 7. Transform with Lambda-like Functor: Нормализация ==========
    INFO("7. Advanced: Computing L2 norm and normalization");
    
    // Вычисляем сумму квадратов (для L2 нормы)
    float sum_of_squares = thrust::transform_reduce(
        d_vec.begin(), d_vec.end(),
        square_functor(),
        0.0f,
        thrust::plus<float>()
    );
    
    float l2_norm = sqrtf(sum_of_squares);
    RESULT("L2 norm: " << l2_norm);
    
    // Нормализуем вектор (делим каждый элемент на норму)
    thrust::device_vector<float> d_normalized(N);
    thrust::transform(d_vec.begin(), d_vec.end(),
                     d_normalized.begin(),
                     [l2_norm] __host__ __device__ (float x) {
                         return x / l2_norm;
                     });
    
    thrust::host_vector<float> h_normalized = d_normalized;
    std::cout << "  Normalized vector: [ ";
    for (int i = 0; i < N; i++) {
        std::cout << std::fixed << std::setprecision(3) << h_normalized[i] << " ";
    }
    std::cout << "]" << std::endl;
    
    // Проверяем, что норма нормализованного вектора = 1
    float normalized_norm = sqrtf(thrust::transform_reduce(
        d_normalized.begin(), d_normalized.end(),
        square_functor(),
        0.0f,
        thrust::plus<float>()
    ));
    
    RESULT("Norm of normalized vector: " << normalized_norm << " (should be ~1.0)");
    
    std::cout << std::endl;
    
    INFO("8. Performance comparison: Thrust vs CPU");
    
    const int LARGE_N = 1000000;
    
    // CPU версия
    std::vector<float> cpu_data(LARGE_N);
    for (int i = 0; i < LARGE_N; i++) {
        cpu_data[i] = static_cast<float>(i + 1);
    }
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Измеряем CPU
    CUDA_CHECK(cudaEventRecord(start));
    
    float cpu_sum = 0.0f;
    for (int i = 0; i < LARGE_N; i++) {
        cpu_sum += cpu_data[i] * cpu_data[i];
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float cpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&cpu_time, start, stop));
    
    // Измеряем Thrust (GPU)
    thrust::device_vector<float> d_large(cpu_data.begin(), cpu_data.end());
    
    CUDA_CHECK(cudaEventRecord(start));
    
    float gpu_sum = thrust::transform_reduce(
        d_large.begin(), d_large.end(),
        square_functor(),
        0.0f,
        thrust::plus<float>()
    );
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    
    std::cout << "  Vector size: " << LARGE_N << " elements" << std::endl;
    RESULT("CPU time: " << cpu_time << " ms (sum: " << cpu_sum << ")");
    RESULT("GPU time (Thrust): " << gpu_time << " ms (sum: " << gpu_sum << ")");
    RESULT("Speedup: " << (cpu_time / gpu_time) << "x");
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    std::cout << std::endl;
    INFO("Thrust example completed successfully");
}
