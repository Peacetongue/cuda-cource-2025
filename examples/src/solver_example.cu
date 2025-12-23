#include "../include/cuda_utils.h"
#include <cusolverSp.h>
#include <vector>
#include <cmath>
#include <iomanip>

static void cleanup_solver_resources(
    float* d_csr_values,
    int* d_csr_col_indices,
    int* d_csr_row_offsets,
    float* d_b,
    float* d_x,
    cusparseMatDescr_t mat_desc,
    cusolverSpHandle_t solver_handle
) {
    if (mat_desc) {
        CUSPARSE_CHECK(cusparseDestroyMatDescr(mat_desc));
    }
    if (solver_handle) {
        CUSOLVER_CHECK(cusolverSpDestroy(solver_handle));
    }
    if (d_csr_values) {
        CUDA_CHECK(cudaFree(d_csr_values));
    }
    if (d_csr_col_indices) {
        CUDA_CHECK(cudaFree(d_csr_col_indices));
    }
    if (d_csr_row_offsets) {
        CUDA_CHECK(cudaFree(d_csr_row_offsets));
    }
    if (d_b) {
        CUDA_CHECK(cudaFree(d_b));
    }
    if (d_x) {
        CUDA_CHECK(cudaFree(d_x));
    }
}

/**
 * Тут используем cuSOLVER с разреженными матрицами.
 *
 * ЧТО ТАКОЕ cuSOLVER?
 * -------------------
 * cuSOLVER - это библиотека NVIDIA для решения систем линейных уравнений
 * и задач линейной алгебры на GPU. Она включает три основных API:
 *   - cuSOLVER Dense: для плотных матриц (LU, QR, SVD, eigenvalues)
 *   - cuSOLVER Sparse: для разреженных матриц (QR, Cholesky)
 *   - cuSOLVER Refactorization: для повторных факторизаций
 *
 * ЧТО ТАКОЕ СИСТЕМА ЛИНЕЙНЫХ УРАВНЕНИЙ?
 * -------------------------------------
 * Это система вида: A*x = b
 * где:
 *   - A - известная матрица коэффициентов (n x n)
 *   - b - известный вектор правых частей (n x 1)
 *   - x - неизвестный вектор решений (n x 1)
 *
 * Пример для 3x3:
 *     A = [ 4  1  0 ]     x = [ x0 ]     b = [ 9 ]
 *         [ 1  3  1 ]         [ x1 ]         [ 7 ]
 *         [ 0  1  2 ]         [ x2 ]         [ 5 ]
 *
 * Решение: x = [2, 1, 2]
 *
 * ЗАЧЕМ ИСПОЛЬЗОВАТЬ cuSOLVER С РАЗРЕЖЕННЫМИ МАТРИЦАМИ?
 * ------------------------------------------------------
 * В реальных задачах (физические симуляции, метод конечных элементов,
 * анализ графов) системы могут иметь миллионы уравнений, но матрица A
 * разреженная (большинство элементов = 0).
 *
 * Преимущества:
 *   1. Экономия памяти: храним только ненулевые элементы
 *   2. Ускорение вычислений: операции только с ненулевыми элементами
 *   3. Численная стабильность: специализированные алгоритмы
 *
 * МЕТОДЫ РЕШЕНИЯ В cuSOLVER Sparse
 * ---------------------------------
 * 1. QR факторизация:
 *    - A = Q*R, где Q - ортогональная, R - верхнетреугольная
 *    - Универсальный метод, работает для любых матриц
 *    - Используется в этом примере
 *
 * 2. Cholesky факторизация:
 *    - A = L*L^T, где L - нижнетреугольная
 *    - Только для симметричных положительно определённых матриц
 *    - Быстрее QR, но более ограниченная применимость
 *
 * ЧТО МЫ ДЕЛАЕМ В ЭТОМ ПРИМЕРЕ?
 * ------------------------------
 * Решаем систему A*x = b с разреженной матрицей A методом QR:
 *
 * 1. Создаём разреженную матрицу A в формате CSR (из cuSPARSE примера)
 * 2. Задаём вектор правых частей b
 * 3. Используем cusolverSpScsrlsvqr для решения системы
 * 4. Проверяем решение: вычисляем A*x и сравниваем с b
 *
 * ФОРМАТ CSR (напоминание)
 * ------------------------
 * Для разреженной матрицы A размером n×n с nnz ненулевыми элементами:
 *   - values[nnz]: значения ненулевых элементов
 *   - col_indices[nnz]: индексы столбцов
 *   - row_offsets[n+1]: указатели на начало каждой строки
 *
 * ПРИМЕНЕНИЯ
 * ----------
 * - Метод конечных элементов (Finite Element Method)
 * - Анализ электрических цепей
 * - Структурная механика
 * - Computational Fluid Dynamics (CFD)
 * - Оптимизация (Interior Point Methods)
 * - Graph partitioning
 *
 * ПРОИЗВОДИТЕЛЬНОСТЬ И РЕКОМЕНДАЦИИ
 * ----------------------------------
 * 1. ВЫБОР МЕТОДА
 *
 *    QR факторизация (cusolverSpScsrlsvqr):
 *      ok: Универсальный метод для любых матриц
 *      ok: Численно стабильный
 *      ok: Работает с прямоугольными матрицами
 *      bad: Медленнее Cholesky для SPD матриц
 *
 *    Cholesky факторизация (cusolverSpScsrlsvchol):
 *      ok: Быстрее QR (~2x для разреженных SPD)
 *      ok: Меньше требования к памяти
 *      bad: Только для симметричных положительно определённых матриц
 *
 * 2. ПЕРЕУПОРЯДОЧИВАНИЕ (Reordering)
 *
 *    Уменьшает fill-in (новые ненулевые элементы) при факторизации:
 *      - 0: No reordering (простой случай, может быть неэффективно)
 *      - 1: SYMRCM (Symmetric Reverse Cuthill-McKee) - хорош для bandwidth reduction
 *      - 2: SYMAMD (Symmetric Approximate Minimum Degree) - минимизирует fill-in
 *      - 3: CSRMETISND (METIS nested dissection) - лучший для больших систем
 *
 *    Рекомендация: используйте reorder=2 или 3 для реальных задач
 *
 * 3. PRECISION CONSIDERATIONS
 *
 *    - float (используется здесь): достаточно для многих задач, быстрее
 *    - double: используйте для плохо обусловленных матриц
 *    - Проверяйте condition number матрицы A
 *
 * 4. АЛЬТЕРНАТИВНЫЕ ПОДХОДЫ
 *
 *    Для очень больших разреженных систем:
 *      - Итерационные методы (CG, BiCGSTAB, GMRES) из cuSPARSE
 *      - Preconditioned iterative solvers
 *      - cuSOLVER можно использовать как preconditioner
 *
 * 5. КОГДА ИСПОЛЬЗОВАТЬ cuSOLVER Sparse?
 *
 *    Хорошо подходит для:
 *      ok: Средние системы (10^3 - 10^6 неизвестных)
 *      ok: Умеренная разреженность (0.1% - 10% ненулевых)
 *      ok: Множественные правые части (batched solve)
 *      ok: Интеграция с cuSPARSE/cuBLAS pipeline
 *
 *    Не рекомендуется для:
 *      bad: Очень большие системы (>10^7) - используйте итерационные методы
 *      bad: Очень плотные матрицы (>50% ненулевых) - используйте cuBLAS
 *      bad: Ill-conditioned matrices - может потребоваться preconditioning
 *
 * 6. DEBUGGING & VALIDATION
 *
 *    - Всегда проверяйте singularity flag
 *    - Вычисляйте residual norm для верификации
 *    - Сравнивайте с CPU решением на малых размерах
 *    - Используйте double precision для плохо обусловленных матриц
 *
 * ПОЛЕЗНЫЕ ССЫЛКИ
 * ===============
 * - cuSOLVER Guide: https://docs.nvidia.com/cuda/cusolver/
 * - Sparse Direct Solvers: https://en.wikipedia.org/wiki/Sparse_matrix#Direct_methods
 * - FEM with CUDA: https://developer.nvidia.com/blog/finite-element-method-cuda/
 * - Graph Laplacian: https://en.wikipedia.org/wiki/Laplacian_matrix
 *
 * ==========================================================================
 */

void run_solver_example() {
    HEADER("cuSOLVER + cuSPARSE: Sparse Linear System Solver Example");

    INFO("PROBLEM: Solve A*x = b, where A is sparse");
    INFO("METHOD: QR factorization with cusolverSpScsrlsvqr");
    std::cout << std::endl;

    // Размер системы
    const int n = 3;      // размер матрицы A (n x n)
    const int nnz = 7;    // количество ненулевых элементов

    INFO("System size: " << n << "x" << n);
    INFO("Non-zero elements: " << nnz);

    // Валидация входных параметров
    if (n <= 0) {
        std::cerr << "Error: invalid matrix size n=" << n << std::endl;
        return;
    }
    if (nnz < 0) {
        std::cerr << "Error: invalid number of non-zeros nnz=" << nnz << std::endl;
        return;
    }

    // Разреженная симметричная положительно определённая матрица A в формате CSR:
    // A = [ 4  1  0 ]
    //     [ 1  3  1 ]
    //     [ 0  1  2 ]
    //
    // Это типичная матрица из задач МКЭ (трёхдиагональная структура)
    std::vector<float> h_csr_values = {4.0f, 1.0f, 1.0f, 3.0f, 1.0f, 1.0f, 2.0f};
    std::vector<int> h_csr_col_indices = {0, 1, 0, 1, 2, 1, 2};
    std::vector<int> h_csr_row_offsets = {0, 2, 5, 7};

    // Вектор правых частей b
    // Для проверки выберем b так, чтобы решение было x = [2, 1, 2]
    std::vector<float> h_b = {9.0f, 7.0f, 5.0f};

    // Вектор для решения (будет заполнен cuSOLVER)
    std::vector<float> h_x(n, 0.0f);

    // Валидация формата CSR
    if (h_csr_row_offsets.size() != static_cast<size_t>(n + 1)) {
        std::cerr << "Error: row_offsets size must be n+1=" << (n+1) 
                  << ", got " << h_csr_row_offsets.size() << std::endl;
        return;
    }
    if (h_csr_row_offsets[n] != nnz) {
        std::cerr << "Error: row_offsets[n] must equal nnz=" << nnz
                  << ", got " << h_csr_row_offsets[n] << std::endl;
        return;
    }
    if (h_csr_values.size() != static_cast<size_t>(nnz) || 
        h_csr_col_indices.size() != static_cast<size_t>(nnz)) {
        std::cerr << "Error: values and col_indices size must equal nnz=" << nnz << std::endl;
        return;
    }

    INFO("Matrix A (CSR format):");
    std::cout << "  [ 4  1  0 ]" << std::endl;
    std::cout << "  [ 1  3  1 ]" << std::endl;
    std::cout << "  [ 0  1  2 ]" << std::endl;

    INFO("Right-hand side b: [" << h_b[0] << ", " << h_b[1] << ", " << h_b[2] << "]");
    INFO("Expected solution x: [2, 1, 2]");
    std::cout << std::endl;

    // Инициализация указателей на NULL для безопасной очистки
    float *d_csr_values = nullptr;
    int *d_csr_col_indices = nullptr;
    int *d_csr_row_offsets = nullptr;
    float *d_b = nullptr;
    float *d_x = nullptr;
    cusolverSpHandle_t solver_handle = nullptr;
    cusparseMatDescr_t mat_desc = nullptr;

    // Выделение памяти на GPU
    CUDA_CHECK(cudaMalloc(&d_csr_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_csr_col_indices, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csr_row_offsets, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));

    // Копирование данных на GPU
    CUDA_CHECK(cudaMemcpy(d_csr_values, h_csr_values.data(),
                          nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_col_indices, h_csr_col_indices.data(),
                          nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_row_offsets, h_csr_row_offsets.data(),
                          (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(),
                          n * sizeof(float), cudaMemcpyHostToDevice));

    // Создание cuSOLVER handle
    CUSOLVER_CHECK(cusolverSpCreate(&solver_handle));

    INFO("cuSOLVER handle created");

    // Создание дескриптора матрицы для cuSPARSE
    CUSPARSE_CHECK(cusparseCreateMatDescr(&mat_desc));

    // Устанавливаем свойства матрицы
    CUSPARSE_CHECK(cusparseSetMatType(mat_desc, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(mat_desc, CUSPARSE_INDEX_BASE_ZERO));

    INFO("Matrix descriptor created");

    // cuSOLVER проверяет сингулярность матрицы во время решения
    // и возвращает индекс проблемной строки (если есть)
    int singularity = 0;

    HEADER("SOLVING SYSTEM");
    INFO("Using QR factorization (cusolverSpScsrlsvqr)...");

    // tolerance - точность решения (для iterative refinement)
    const float tol = 1.e-6f;

    // reorder - переупорядочивание для уменьшения fill-in:
    //   0 - no reordering
    //   1 - symrcm (Symmetric Reverse Cuthill-McKee)
    //   2 - symamd (Symmetric Approximate Minimum Degree) - рекомендуется
    //   3 - csrmetisnd (METIS nested dissection) - для больших систем
    const int reorder = 0;  // для простоты не используем переупорядочивание

    // Решение системы A*x = b
    CUSOLVER_CHECK(
        cusolverSpScsrlsvqr(
            solver_handle,
            n,                      // размер системы
            nnz,                    // количество ненулевых элементов
            mat_desc,               // дескриптор матрицы
            d_csr_values,           // значения CSR
            d_csr_row_offsets,      // row offsets CSR
            d_csr_col_indices,      // column indices CSR
            d_b,                    // вектор правых частей
            tol,                    // tolerance
            reorder,                // reordering algorithm
            d_x,                    // решение (выход)
            &singularity            // флаг сингулярности
        )
    );

    // Проверка сингулярности матрицы
    if (singularity >= 0) {
        std::cerr << "ERROR: Matrix is singular at row " << singularity << std::endl;
        std::cerr << "Cannot compute solution for singular matrix" << std::endl;
        cleanup_solver_resources(
            d_csr_values, d_csr_col_indices, d_csr_row_offsets,
            d_b, d_x, mat_desc, solver_handle
        );
        return;
    }
    
    INFO("Matrix is non-singular, solution found");

    // Копирование решения обратно на CPU
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x,
                          n * sizeof(float), cudaMemcpyDeviceToHost));

    // Вывод решения
    HEADER("SOLUTION");
    std::cout << "Computed solution x: [";
    for (int i = 0; i < n; ++i) {
        std::cout << std::fixed << std::setprecision(4) << h_x[i];
        if (i < n - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    INFO("Verifying solution (computing residual ||A*x - b||)...");

    // Вычисляем A*x на CPU для проверки
    std::vector<float> Ax(n, 0.0f);

    for (int row = 0; row < n; ++row) {
        float sum = 0.0f;
        int row_start = h_csr_row_offsets[row];
        int row_end = h_csr_row_offsets[row + 1];

        for (int j = row_start; j < row_end; ++j) {
            int col = h_csr_col_indices[j];
            float val = h_csr_values[j];
            sum += val * h_x[col];
        }

        Ax[row] = sum;
    }

    // Вычисляем норму невязки ||A*x - b||
    float residual_norm = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = Ax[i] - h_b[i];
        residual_norm += diff * diff;
    }
    residual_norm = std::sqrt(residual_norm);

    // Вывод верификации
    HEADER("VERIFICATION");
    std::cout << "A*x = [";
    for (int i = 0; i < n; ++i) {
        std::cout << std::fixed << std::setprecision(4) << Ax[i];
        if (i < n - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "b   = [";
    for (int i = 0; i < n; ++i) {
        std::cout << std::fixed << std::setprecision(4) << h_b[i];
        if (i < n - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    RESULT("Residual ||A*x - b|| = " << residual_norm);

    // Оценка точности решения
    if (residual_norm < 1e-5f) {
        RESULT("ok: Solution is ACCURATE (residual < 1e-5)");
    } else if (residual_norm < 1e-3f) {
        RESULT("ok: Solution is acceptable (residual < 1e-3)");
    } else {
        RESULT("bad: Solution may be inaccurate (residual >= 1e-3)");
    }

    // Освобождение ресурсов через централизованную функцию cleanup
    cleanup_solver_resources(
        d_csr_values, d_csr_col_indices, d_csr_row_offsets,
        d_b, d_x, mat_desc, solver_handle
    );

    INFO("All resources cleaned up successfully");
}
