## Lab 4: Radix Sort - Поразрядная сортировка

### Описание задачи
Реализовать алгоритм поразрядной сортировки (Radix Sort) на CUDA.

## Компиляция

```bash
make
```

## Запуск

```bash
./radix_sort
```

## Результаты

```
Radix Sort Benchmark
Array size: 1000000

init array with random data
Running CPU qsort
CPU time: 75.55 ms

Running GPU Radix Sort
GPU time: 8.37 ms
Speedup: 9.03x

Running Thrust sort
Thrust sort time: 1.07 ms

Verifying
Radix Sort: OK
Match with Thrust: OK
```

После исправлиний (добавил warp синхронизацию, убрал лишние операции, заменил алгоритм подсчёта локальных смещений на параллезированный префикс скан, сделал prefix scan по блокам в одном kernel вместо параллельных блоков с циклом по предыдущим элементам) получился результат:
 ```
Radix Sort Benchmark
Array size: 1000000

init array with random data
Running CPU qsort
CPU time: 74.80 ms

Running GPU Radix Sort
GPU time: 19.59 ms
Speedup: 3.82x

Running Thrust sort
Thrust sort time: 1.11 ms

Verifying
Radix Sort: OK
Match with Thrust: OK
```
Время увеличилось из-за параллезированного префикс скан, где запускается 256 отдельных сканирований (для каждой возможной цифры) и каждое такое сканирование требует синхронизации всех потоков...
Поэтому было принято волевое решение вернуть алгоритм с квадратичной сложностью тк он подходит лучше, тк запускает всего 2 синхронизации
и того:
```
Radix Sort Benchmark
Array size: 1000000

init array with random data

Running GPU Radix Sort        
GPU time: 5.30 ms
Speedup: 14.19x

Running Thrust sort
Thrust sort time: 0.99 ms

Verifying
Radix Sort: OK
Match with Thrust: OK
```