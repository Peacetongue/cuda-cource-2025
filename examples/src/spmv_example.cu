#include "../include/cuda_utils.h"
#include <vector>

/**
 * Этот пример демонстрирует работу с разреженными матрицами (sparse matrices)
 * с использованием библиотеки cuSPARSE и cuBLAS.
 * 
 * ЧТО ТАКОЕ РАЗРЕЖЕННАЯ МАТРИЦА?
 * ------------------------------
 * Разреженная матрица - это матрица, в которой большинство элементов равны нулю.
 * Например:
 *     [ 1  0  2 ]
 *     [ 0  3  0 ]
 *     [ 4  0  5 ]
 * 
 * Здесь только 5 элементов ненулевые из 9 возможных (55% нулей).
 * В реальных задачах (графы, физические симуляции, ML) бывают матрицы,
 * где 99%+ элементов - нули
 * 
 * ЗАЧЕМ НУЖЕН cuSPARSE?
 * ---------------------
 * Хранить миллионы нулей в памяти GPU - расточительно.
 * cuSPARSE позволяет хранить только ненулевые элементы и работать с ними эффективно.
 * 
 * Для матрицы 10000x10000 с 0.1% заполнения:
 *   - Плотный формат: 10000x10000x4 bytes = 400 MB
 *   - Разреженный (CSR): 10000x4 + 10000x4 + 12 bytes ≈ 80 KB
 * 
 * ФОРМАТ CSR (Compressed Sparse Row)
 * ----------------------------------
 * Это один из самых популярных форматов хранения разреженных матриц.
 * Для матрицы A размером mxn с nnz ненулевыми элементами нужно 3 массива:
 * 
 * 1. values[nnz]      - значения ненулевых элементов (слева направо, сверху вниз)
 * 2. col_indices[nnz] - индексы столбцов для каждого элемента
 * 3. row_offsets[m+1] - указатели на начало каждой строки в массивах values/col_indices
 * 
 * Пример для матрицы:
 *     [ 1  0  2 ]    строка 0
 *     [ 0  3  0 ]    строка 1
 *     [ 4  0  5 ]    строка 2
 * 
 * Преобразование в CSR:
 *   values      = [1, 2,  3,  4, 5]     // ненулевые элементы по строкам
 *   col_indices = [0, 2,  1,  0, 2]     // номера столбцов
 *   row_offsets = [0, 2,  3,  5]        // row_offsets[i] = начало строки i
 * 
 * Как это читать:
 *   - Строка 0: элементы с индексов 0 до 2 (не включая) -> values[0..1] = [1, 2]
 *   - Строка 1: элементы с индексов 2 до 3 (не включая) -> values[2] = [3]
 *   - Строка 2: элементы с индексов 3 до 5 (не включая) -> values[3..4] = [4, 5]
 * 
 * ЧТО ТАКОЕ SpMV?
 * ---------------
 * SpMV = Sparse Matrix-Vector multiplication (умножение разреженной матрицы на вектор)
 * y = A * x, где A - разреженная матрица, x и y - плотные векторы
 * 
 * Это одна из самых важных операций в научных вычислениях!
 * Используется в:
 *   - Решении систем линейных уравнений
 *   - Анализе графов (PageRank)
 *   - Методе конечных элементов
 *   - Машинном обучении (нейронные сети с разреженными связями)
 * 
 * ЧТО МЫ ДЕЛАЕМ В ЭТОМ ПРИМЕРЕ?
 * --------------------------------------------------
 * Вычислительный конвейер (pipeline), где cuSPARSE и cuBLAS работают вместе:
 * 
 * 1. SpMV (cuSPARSE): y = A * x, где A - разреженная матрица 3x3
 * 2. GEMV (cuBLAS):   z = B * y, где B - плотная матрица 3x3
 * 3. Normalization (cuBLAS):
 *    - Вычисляем norm = ||z||_2 (L2-норма вектора z)
 *    - Нормализуем: z_normalized = z / norm
 * 4. Dot Product (cuBLAS): scalar = dot(x, z_normalized)
 * 
 * ЗАЧЕМ ЭТО НУЖНО?
 * ----------------
 * Это типичный паттерн в машинном обучении и численных методах:
 *   - Разреженная матрица A моделирует связи/граф (например, adjacency matrix)
 *   - Плотная матрица B - это трансформация/проекция (например, weight matrix)
 *   - Нормализация нужна для стабильности (например, в нейросетях)
 * 
 * Пример использования: Graph Neural Networks (GNN)
 *   1. A - adjacency matrix графа (разреженная)
 *   2. x - признаки узлов
 *   3. y = A*x - агрегация признаков от соседей
 *   4. z = B*y - трансформация признаков
 *   5. normalize(z) - нормализация для следующего слоя
 * 
 * ============================================================================
 */

void run_spmv_example() {
    HEADER("cuSPARSE + cuBLAS: Hybrid Sparse-Dense Pipeline Example");
    
    INFO("COMPUTATIONAL PIPELINE:");
    INFO("  Step 1: y = A*x     (cuSPARSE SpMV, A is sparse)");
    INFO("  Step 2: z = B*y     (cuBLAS GEMV, B is dense)");
    INFO("  Step 3: norm = ||z||_2  (cuBLAS norm)");
    INFO("  Step 4: z = z/norm  (cuBLAS scal)");
    INFO("  Step 5: scalar = dot(x, z) (cuBLAS dot)");
    std::cout << std::endl;
    
    // Размеры для конвейера:
    // A: 3x3 sparse matrix
    // x: 3x1 vector
    // y: 3x1 vector (result of A*x)
    // B: 3x3 dense matrix
    // z: 3x1 vector (result of B*y)

    const int sparse_rows = 3;
    const int sparse_cols = 3;
    const int nnz = 5;

    const int dense_rows = 3;
    const int dense_cols = 3;

    INFO("Matrix A (sparse): " << sparse_rows << "x" << sparse_cols << ", nnz = " << nnz);
    INFO("Matrix B (dense):  " << dense_rows << "x" << dense_cols);

    // Проверка совместимости размеров для конвейера:
    // - Для SpMV (y = A*x): sparse_cols должен соответствовать размеру x
    // - Для GEMV (z = B*y): dense_cols должен равняться sparse_rows (размер y)
    // - Для dot(x, z): sparse_cols должен равняться dense_rows (размер z)
    if (dense_cols != sparse_rows) {
        std::cerr << "Error: incompatible dimensions for GEMV: "
                  << "B has " << dense_cols << " columns, but y has size " << sparse_rows << std::endl;
        return;
    }
    if (sparse_cols != dense_rows) {
        std::cerr << "Warning: vectors x and z have different sizes ("
                  << sparse_cols << " vs " << dense_rows << "), dot product may fail" << std::endl;
    }
    
    // Разреженная матрица A в формате CSR:
    //     [ 1  0  2 ]
    //     [ 0  3  0 ]
    //     [ 4  0  5 ]
    std::vector<float> h_csr_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<int> h_csr_col_indices = {0, 2, 1, 0, 2};
    std::vector<int> h_csr_row_offsets = {0, 2, 3, 5};
    
    // Плотная матрица B в column-major формате (для cuBLAS):
    //     [ 1  2  1 ]
    //     [ 0  1  2 ]
    //     [ 1  0  1 ]
    std::vector<float> h_B = {
        1, 0, 1,   // колонка 0
        2, 1, 0,   // колонка 1
        1, 2, 1    // колонка 2
    };
    
    // Входной вектор x (например, начальные признаки узлов графа)
    std::vector<float> h_x = {1.0f, 2.0f, 3.0f};
    
    // Промежуточные и выходные векторы
    std::vector<float> h_y(sparse_rows, 0.0f);  // y = A*x
    std::vector<float> h_z(dense_rows, 0.0f);   // z = B*y
    
    INFO("Input vector x: [" << h_x[0] << ", " << h_x[1] << ", " << h_x[2] << "]");
    
    // Скаляры для операций
    float alpha_sparse = 1.0f;
    float beta_sparse = 0.0f;
    
    // Память для разреженной матрицы A (CSR формат)
    float *d_csr_values;
    int *d_csr_col_indices, *d_csr_row_offsets;
    CUDA_CHECK(cudaMalloc(&d_csr_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_csr_col_indices, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_row_offsets, (sparse_rows + 1) * sizeof(int)));
    
    // Память для плотной матрицы B
    float *d_B;
    CUDA_CHECK(cudaMalloc(&d_B, dense_rows * dense_cols * sizeof(float)));
    
    // Память для векторов
    float *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, sparse_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, sparse_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z, dense_rows * sizeof(float)));
    
    // Копируем данные на GPU
    CUDA_CHECK(cudaMemcpy(d_csr_values, h_csr_values.data(), 
                          nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_col_indices, h_csr_col_indices.data(), 
                          nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_row_offsets, h_csr_row_offsets.data(), 
                          (sparse_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), 
                          dense_rows * dense_cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), 
                          sparse_cols * sizeof(float), cudaMemcpyHostToDevice));
    
    // Создаём контексты для обеих библиотек
    cusparseHandle_t cusparse_handle;
    cublasHandle_t cublas_handle;
    
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    INFO("Both cuSPARSE and cuBLAS handles created");
    
    // Создаём дескриптор разреженной матрицы A
    cusparseSpMatDescr_t mat_A;
    CUSPARSE_CHECK(
        cusparseCreateCsr(
            &mat_A,
            sparse_rows, sparse_cols, nnz,
            d_csr_row_offsets,
            d_csr_col_indices,
            d_csr_values,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F
        )
    );
    
    // Дескрипторы для плотных векторов x и y
    cusparseDnVecDescr_t vec_x, vec_y;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x, sparse_cols, d_x, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y, sparse_rows, d_y, CUDA_R_32F));
    
    // Вычисляем размер буфера для SpMV
    size_t buffer_size = 0;
    CUSPARSE_CHECK(
        cusparseSpMV_bufferSize(
            cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_sparse,
            mat_A,
            vec_x,
            &beta_sparse,
            vec_y,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            &buffer_size
        )
    );
    
    void* d_buffer = nullptr;
    if (buffer_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));
    }
    
    HEADER("PIPELINE EXECUTION");
    INFO("Step 1: Computing y = A*x using cuSPARSE SpMV...");
    
    // Выполняем SpMV с разреженной матрицей
    CUSPARSE_CHECK(
        cusparseSpMV(
            cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_sparse,
            mat_A,
            vec_x,
            &beta_sparse,
            vec_y,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            d_buffer
        )
    );
    
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, 
                          sparse_rows * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "  ok y = [";
    for (int i = 0; i < sparse_rows; ++i) {
        std::cout << h_y[i];
        if (i < sparse_rows - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    INFO("Step 2: Computing z = B*y using cuBLAS GEMV...");

    // ВАЖНО: Теперь y становится входом для cuBLAS операции!

    // Скаляры для операции GEMV
    float alpha_dense = 1.0f;
    float beta_dense = 0.0f;

    // cublasSgemv выполняет: z = alpha*B*y + beta*z
    // B - это dense_rows x dense_cols матрица в column-major формате
    CUBLAS_CHECK(
        cublasSgemv(
            cublas_handle,
            CUBLAS_OP_N,              // B не транспонируется
            dense_rows,               // количество строк B
            dense_cols,               // количество столбцов B
            &alpha_dense,             // scalar alpha
            d_B,                      // матрица B на GPU
            dense_rows,               // leading dimension (lda)
            d_y,                      // вектор y на GPU (ВЫХОД cuSPARSE!)
            1,                        // stride для y
            &beta_dense,              // scalar beta
            d_z,                      // результат z на GPU
            1                         // stride для z
        )
    );
    
    CUDA_CHECK(cudaMemcpy(h_z.data(), d_z, 
                          dense_rows * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "  ok z = [";
    for (int i = 0; i < dense_rows; ++i) {
        std::cout << h_z[i];
        if (i < dense_rows - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    INFO("Step 3: Computing norm ||z||_2 using cuBLAS...");
    
    float norm_z = 0.0f;
    
    // cublasSnrm2 вычисляет L2-норму: sqrt(z[0]^2 + z[1]^2 + ... + z[n-1]^2)
    CUBLAS_CHECK(
        cublasSnrm2(
            cublas_handle,
            dense_rows,               // размер вектора
            d_z,                      // вектор z на GPU
            1,                        // stride
            &norm_z                   // результат нормы (на CPU)
        )
    );
    
    std::cout << "  ok ||z||_2 = " << norm_z << std::endl;
    
    INFO("Step 4: Normalizing z = z/||z||_2 using cuBLAS...");

    // Проверка деления на ноль
    if (norm_z < 1e-10f) {
        std::cerr << "Error: norm is too close to zero (" << norm_z << "), cannot normalize" << std::endl;
        // Очистка ресурсов перед выходом
        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_x));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_y));
        CUSPARSE_CHECK(cusparseDestroySpMat(mat_A));
        CUSPARSE_CHECK(cusparseDestroy(cusparse_handle));
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
        if (d_buffer) CUDA_CHECK(cudaFree(d_buffer));
        CUDA_CHECK(cudaFree(d_csr_values));
        CUDA_CHECK(cudaFree(d_csr_col_indices));
        CUDA_CHECK(cudaFree(d_csr_row_offsets));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_z));
        return;
    }

    // cublasSscal выполняет: x = alpha * x
    // Мы хотим z = z / norm, что эквивалентно z = (1/norm) * z
    float inv_norm = 1.0f / norm_z;
    
    CUBLAS_CHECK(
        cublasSscal(
            cublas_handle,
            dense_rows,               // размер вектора
            &inv_norm,                // scalar (1/norm)
            d_z,                      // вектор z на GPU (модифицируется in-place!)
            1                         // stride
        )
    );
    
    // Копируем нормализованный результат
    CUDA_CHECK(cudaMemcpy(h_z.data(), d_z, 
                          dense_rows * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "  ok z_normalized = [";
    for (int i = 0; i < dense_rows; ++i) {
        std::cout << h_z[i];
        if (i < dense_rows - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    INFO("Step 5: Computing dot(x, z) using cuBLAS...");

    float dot_result = 0.0f;

    // Проверка совместимости размеров для dot product
    // d_x имеет размер sparse_cols, d_z имеет размер dense_rows
    if (sparse_cols != dense_rows) {
        std::cerr << "Error: incompatible vector sizes for dot product: "
                  << "x has size " << sparse_cols << ", z has size " << dense_rows << std::endl;
        // Очистка ресурсов перед выходом
        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_x));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_y));
        CUSPARSE_CHECK(cusparseDestroySpMat(mat_A));
        CUSPARSE_CHECK(cusparseDestroy(cusparse_handle));
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
        if (d_buffer) CUDA_CHECK(cudaFree(d_buffer));
        CUDA_CHECK(cudaFree(d_csr_values));
        CUDA_CHECK(cudaFree(d_csr_col_indices));
        CUDA_CHECK(cudaFree(d_csr_row_offsets));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_z));
        return;
    }

    // Вычисляем скалярное произведение dot(x, z_normalized)
    // Это типичная операция в ML: similarity между входом и трансформированным выходом
    CUBLAS_CHECK(
        cublasSdot(
            cublas_handle,
            sparse_cols,              // размер векторов (проверено выше что sparse_cols == dense_rows)
            d_x,                      // входной вектор x (размер: sparse_cols)
            1,
            d_z,                      // нормализованный вектор z (размер: dense_rows)
            1,
            &dot_result
        )
    );

    std::cout << "  ok dot(x, z_normalized) = " << dot_result << std::endl;
    
    HEADER("PIPELINE RESULTS SUMMARY");
    RESULT("Original input x:         [" << h_x[0] << ", " << h_x[1] << ", " << h_x[2] << "]");
    RESULT("After sparse matmul y:    [" << h_y[0] << ", " << h_y[1] << ", " << h_y[2] << "]");
    RESULT("After dense matmul z:     [" << h_z[0] << ", " << h_z[1] << ", " << h_z[2] << "]");
    RESULT("Dot product dot(x, z):    " << dot_result);
    std::cout << std::endl;
    RESULT("ok Successfully demonstrated cuSPARSE + cuBLAS hybrid pipeline!");
    
    // Уничтожаем дескрипторы cuSPARSE
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_x));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_y));
    CUSPARSE_CHECK(cusparseDestroySpMat(mat_A));
    CUSPARSE_CHECK(cusparseDestroy(cusparse_handle));
    
    // Уничтожаем handle cuBLAS
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    
    // Освобождаем память на GPU
    if (d_buffer) CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaFree(d_csr_values));
    CUDA_CHECK(cudaFree(d_csr_col_indices));
    CUDA_CHECK(cudaFree(d_csr_row_offsets));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    
    INFO("All resources cleaned up successfully");
    
    /*
     * ЧТО МЫ ПРОДЕМОНСТРИРОВАЛИ?
     * 
     * 1. ГИБРИДНЫЙ КОНВЕЙЕР (Hybrid Pipeline)
     *    - cuSPARSE для работы с разреженными данными (SpMV)
     *    - cuBLAS для работы с плотными данными (GEMV, norm, scal, dot)
     *    - Плавный переход данных между операциями (всё на GPU!)
     * 
     * 2. ТИПИЧНЫЕ ПАТТЕРНЫ ИСПОЛЬЗОВАНИЯ
     *    - SpMV -> GEMV: разреженная агрегация + плотная трансформация
     *    - Normalization: важна для стабильности в ML/численных методах
     *    - Dot product: метрика similarity/distance
     * 
     * 3. РЕАЛЬНЫЕ ПРИМЕНЕНИЯ
     *    a) Graph Neural Networks (GNN):
     *       - A: adjacency matrix графа (sparse)
     *       - x: node features
     *       - B: learnable weight matrix (dense)
     *       - Pipeline: агрегация от соседей -> трансформация -> нормализация
     * 
     *    b) Sparse Linear Systems:
     *       - Iterative solvers (Conjugate Gradient, GMRES)
     *       - SpMV для системной матрицы
     *       - BLAS операции для векторов (dot, axpy, norm)
     * 
     *    c) Recommender Systems:
     *       - A: user-item interaction matrix (sparse)
     *       - B: item embedding matrix (dense)
     *       - Pipeline: sparse lookup -> dense projection -> similarity
     * 
     * 4. ОПТИМИЗАЦИИ
     *    - Используйте unified memory для прототипирования
     *    - Batched операции для множественных SpMV
     *    - cuSPARSELt для structured sparsity (2:4 sparsity на Ampere+)
     *    - Tensor Cores через cublasGemmEx для плотных матриц
     * 
     * 5. АЛЬТЕРНАТИВНЫЕ ФОРМАТЫ
     *    - CSR: лучший для SpMV (row-wise access pattern)
     *    - CSC: лучший для sparse matrix x dense matrix
     *    - COO: проще для построения, медленнее для вычислений
     *    - Hybrid (ELL+COO): балансирует производительность и гибкость
     * 
     * 6. DEBUGGING TIPS
     *    - Проверяйте результаты на CPU для малых размеров
     *    - Используйте CUDA_CHECK/CUBLAS_CHECK/CUSPARSE_CHECK всегда!
     *    - cuda-memcheck для поиска memory errors
     *    - nvprof/Nsight для профилирования
     * 
     * ПОЛЕЗНЫЕ ССЫЛКИ
     * ===============
     * - cuSPARSE Guide: https://docs.nvidia.com/cuda/cusparse/
     * - cuBLAS Guide: https://docs.nvidia.com/cuda/cublas/
     * - NVIDIA GitHub Examples: https://github.com/NVIDIA/CUDALibrarySamples
     * - GNN с cuSPARSE: https://developer.nvidia.com/blog/accelerating-graph-neural-networks-with-nvidia-libraries/
     * - Sparse DL: https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/
     */
}
