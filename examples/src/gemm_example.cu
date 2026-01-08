#include "../include/cuda_utils.h"
#include <vector>
#include <iomanip>

/**
 * ЧТО ТАКОЕ GEMM?
 * ===============
 * GEMM = General Matrix Multiply - это одна из самых важных операций в
 * линейной алгебре и научных вычислениях.
 *
 * Формула: C = alpha * A * B + beta * C
 *
 * Где:
 *   A - матрица размером MxK
 *   B - матрица размером KxN
 *   C - матрица размером MxN (результат)
 *   alpha, beta - скалярные множители
 *
 * Типичные варианты использования:
 *   - alpha=1.0, beta=0.0: простое умножение C = A*B
 *   - alpha=1.0, beta=1.0: накопление C = A*B + C
 *   - alpha!=1, beta!=1: масштабированное накопление C = α*A*B + β*C
 *
 * ЗАЧЕМ НУЖЕН cuBLAS?
 * ===================
 * cuBLAS - это GPU-ускоренная библиотека линейной алгебры от NVIDIA.
 * Преимущества:
 *   - Оптимизирована для GPU архитектуры (использует Tensor Cores на новых GPU)
 *   - В 10-100x быстрее CPU реализации для больших матриц
 *   - Автоматическая оптимизация под разные размеры матриц
 *   - Поддержка различных типов данных (float, double, half, int8)
 *
 * COLUMN-MAJOR ФОРМАТ
 * ===================
 * cuBLAS использует column-major формат хранения матриц (как в Fortran/BLAS).
 *
 * Пример для матрицы 3x4:
 *   Row-major (C/C++):     Column-major (cuBLAS):
 *   [ a b c d ]            Хранится: [a, e, i, b, f, j, c, g, k, d, h, l]
 *   [ e f g h ]
 *   [ i j k l ]            Колонка 0: [a, e, i]
 *                          Колонка 1: [b, f, j]
 *                          Колонка 2: [c, g, k]
 *                          Колонка 3: [d, h, l]
 *
 * ПРИМЕНЕНИЯ GEMM
 * ===============
 * 1. Глубокое обучение:
 *    - Forward/backward pass в fully-connected слоях
 *    - Convolution как im2col + GEMM
 *    - Attention механизмы в трансформерах (Q*K^T, attention*V)
 *
 * 2. Научные вычисления:
 *    - Решение систем линейных уравнений
 *    - Моделирование физических процессов
 *    - Обработка сигналов
 *
 * 3. Компьютерное зрение:
 *    - Трансформации изображений
 *    - Feature extraction
 *    - 3D графика (матрицы трансформаций)
 *
 * ============================================================================
 */
void run_gemm_example() {
    HEADER("cuBLAS GEMM Example");

    // Размеры матриц для операции C = A*B
    const int M = 3;   // строки A и C
    const int K = 4;   // столбцы A, строки B
    const int N = 2;   // столбцы B и C

    // Скалярные множители для GEMM: C = alpha*A*B + beta*C
    // alpha=1.0, beta=0.0 означает простое умножение: C = A*B
    float alpha = 1.0f;
    float beta = 0.0f;

    INFO("Matrix sizes: A[" << M << "x" << K << "], B[" << K << "x" << N << "], C[" << M << "x" << N << "]");
    INFO("Operation: C = " << alpha << "*A*B + " << beta << "*C");

    // Матрица A (3x4) в column-major формате:
    // В row-major представлении (для наглядности):
    //   [ 1   2   3  10 ]
    //   [ 4   5   6  11 ]
    //   [ 7   8   9  12 ]
    //
    // В column-major (как хранится в памяти):
    //   Колонка 0: [1, 4, 7]
    //   Колонка 1: [2, 5, 8]
    //   Колонка 2: [3, 6, 9]
    //   Колонка 3: [10, 11, 12]
    std::vector<float> h_A = {
        1, 4, 7,    // колонка 0 матрицы A
        2, 5, 8,    // колонка 1
        3, 6, 9,    // колонка 2
        10, 11, 12  // колонка 3
    };

    // Матрица B (4x2) в column-major формате:
    // В row-major представлении:
    //   [ 1  5 ]
    //   [ 2  6 ]
    //   [ 3  7 ]
    //   [ 4  8 ]
    //
    // В column-major:
    //   Колонка 0: [1, 2, 3, 4]
    //   Колонка 1: [5, 6, 7, 8]
    std::vector<float> h_B = {
        1, 2, 3, 4,  // колонка 0 матрицы B
        5, 6, 7, 8   // колонка 1
    };

    // Матрица C (3x2) - результат, инициализируется нулями
    std::vector<float> h_C(M * N, 0.0f);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));  // 3*4 = 12 floats для A
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));  // 4*2 = 8 floats для B
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));  // 3*2 = 6 floats для C

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Handle управляет контекстом библиотеки и содержит информацию о
    // конфигурации (stream, режим pointer mode и т.д.)

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    INFO("Performing GEMM: C = alpha*A*B + beta*C");

    // cublasSgemm выполняет: C = alpha * op(A) * op(B) + beta * C
    //
    // Параметры:
    //   handle        - контекст cuBLAS
    //   transa, transb - операции над A и B (N = no transpose, T = transpose)
    //   m, n, k       - размеры: A(mxk), B(kxn), C(mxn)
    //   alpha         - скаляр для произведения A*B
    //   A, lda        - матрица A и её leading dimension (расстояние между столбцами)
    //   B, ldb        - матрица B и её leading dimension
    //   beta          - скаляр для старого значения C
    //   C, ldc        - матрица C и её leading dimension
    //
    // Для column-major без транспонирования: lda=m, ldb=k, ldc=m

    CUBLAS_CHECK(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,  // A не транспонируется
            CUBLAS_OP_N,  // B не транспонируется
            M, N, K,      // размеры: A(3x4), B(4x2), C(3x2)
            &alpha,       // scalar alpha = 1.0
            d_A, M,       // матрица A, lda = M = 3
            d_B, K,       // матрица B, ldb = K = 4
            &beta,        // scalar beta = 0.0
            d_C, M        // матрица C, ldc = M = 3
        )
    );

    // cudaMemcpy автоматически синхронизирует GPU перед копированием

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Выводим матрицу C в человекочитаемом формате (по строкам)
    // В column-major формате элемент C[row][col] находится по индексу: col*M + row

    RESULT("Matrix C (result of A*B):");
    for (int row = 0; row < M; ++row) {
        std::cout << "  [ ";
        for (int col = 0; col < N; ++col) {
            // Индекс в column-major: row + col * M
            std::cout << std::fixed << std::setprecision(1) << std::setw(6) << h_C[row + col * M] << " ";
        }
        std::cout << "]" << std::endl;
    }

    // ВАЖНО: всегда освобождайте ресурсы в обратном порядке их создания

    // Уничтожаем cuBLAS контекст
    CUBLAS_CHECK(cublasDestroy(handle));

    // Освобождаем память на GPU
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    INFO("GEMM example completed successfully");

    /*
     * 1. TENSOR CORES (для Volta/Turing/Ampere/Hopper архитектур)
     *    - Используйте cublasGemmEx с CUDA_R_16F (half precision) или TF32
     *    - Tensor Cores дают до 8x ускорение для больших матриц
     *    - Требуют кратности размеров 8 (для FP16) или 16 (для TF32)
     *
     * 2. BATCHED GEMM (множественные умножения)
     *    - cublasSgemmBatched: массив независимых GEMM операций
     *    - cublasSgemmStridedBatched: регулярно расположенные матрицы
     *    - Эффективно для нейросетей (batch processing)
     *
     * 3. STREAMS (асинхронное выполнение)
     *    - cublasSetStream для привязки к CUDA stream
     *    - Позволяет перекрывать копирование и вычисления
     *    - Критично для производительности в production
     *
     * 4. НАСТРОЙКА ПРОИЗВОДИТЕЛЬНОСТИ
     *    - cublasSetMathMode(CUBLAS_TENSOR_OP_MATH) для Tensor Cores
     *    - Выбор правильного типа данных (FP32, FP16, INT8)
     *    - Профилирование с nvprof/Nsight для поиска bottlenecks
     *
     * 5. АЛЬТЕРНАТИВЫ cuBLAS
     *    - cutlass (NVIDIA template library) для custom kernels
     *    - Triton (OpenAI) для высокоуровневого программирования GPU
     *    - Eigen (CPU/GPU) для прототипирования
     *
     * 6. ТИПИЧНЫЕ ОШИБКИ
     *    - Неправильный формат данных (row-major вместо column-major)
     *    - Неверный leading dimension (lda != m для нетранспонированной)
     *    - Забыли создать/уничтожить handle
     *    - Неправильные размеры (k у A и B должны совпадать!)
     *
     * ПОЛЕЗНЫЕ ССЫЛКИ
     * ===============
     * - cuBLAS Guide: https://docs.nvidia.com/cuda/cublas/
     * - cuBLAS Examples: https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLAS
     * - Tensor Cores: https://developer.nvidia.com/tensor-cores
     * - CUTLASS: https://github.com/NVIDIA/cutlass
     */
}
