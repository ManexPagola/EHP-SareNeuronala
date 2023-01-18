#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "device_launch_parameters.h"
#include "matrix_cuda.cuh"
#include "matrix.h"
#include "nn_aux.h"
#include "globals.h"

#define THR_PER_BLOCK 1024

#ifdef TIMING
#include <time.h>
#include "utils.h"
#endif

__global__ void cuda_vec_add(double *A, double *B, double *C, int N)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

__global__ void cuda_vec_sub(double *A, double *B, double *C, int N)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] - B[i];
}

__global__ void cuda_vec_mulcnt(double *A, int N, double cnt)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        A[i] *= cnt;
}

__global__ void cuda_vec_zero(double *A, int N)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        A[i] = 0.0;
}

__global__ void cuda_vec_muldot(double *A, double *B, double *C, int N)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] * B[i];
}

__global__ void cuda_mat_transpose(double *A, int N)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int errenkada, zutabea;
    double lag;

    for (int i = 0; i < N; i++)
    {

        errenkada = i / N;
        zutabea = i % N;

        if (zutabea > errenkada)
        {
            lag = A[zutabea * N + errenkada];
            A[zutabea * N + errenkada] = A[errenkada * N + zutabea];
            A[errenkada * N + zutabea] = lag;
        }
    }
}

// Suposatuz tamaina berekoak A eta B
__global__ void cuda_prod_matrixes(double *A, double *B, double *C, int N)
{

    double m = 0.0;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < N; i++)
    {
        m += A[y * N + i] * B[i * N + x];
    }
    C[y * N + x] = m;
}

//??
__global__ void cuda_prod_add_matrixes(double *A, double *B, double *C, double *D, int N)
{

    double m = 0.0;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < N; i++)
    {
        m += A[y * N + i] * B[i * N + x];
    }
    C[y * N + x] = m + D[y*N + x];
    
}

__global__ void cuda_vec_func(double *n, double *m, int N, double (*func)(double))
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        n[i] = func(m[i]);
}

double **alloc_matrix_2v(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void))
{

    double **m;
    int i, j;

    if ((m = (double **)malloc(n_layers * sizeof(double *))) == NULL)
    {
        return (NULL);
    }

    for (i = 0; i < n_layers; i++)
        if ((m[i] = (double *)malloc(size[i] * size_prev[i] * sizeof(double))) == NULL)
        {
            matrix_free_2D(m, n_layers);
            return (NULL);
        }

    for (i = 0; i < n_layers; i++)
    {
        for (j = 0; j < size[i] * size_prev[i]; j++)
        {
            m[i][j] = init_weight_ptr();
        }
    }

    return (m);
}

double **alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void))
{

    double **m;
    int i, j;

    if ((m = (double **)malloc(n_layers * sizeof(double *))) == NULL)
    {
        return (NULL);
    }

    for (i = 0; i < n_layers; i++)
        if ((m[i] = (double *)malloc(size[i] * sizeof(double))) == NULL)
        {
            matrix_free_2D(m, n_layers);
            return (NULL);
        }

    for (i = 0; i < n_layers; i++)
    {
        for (j = 0; j < size[i]; j++)
        {
            m[i][j] = init_weight_ptr();
        }
    }

    return (m);
}

double *alloc_array(int length)
{

    double *v;
    int i;

    if ((v = (double *)malloc(length * sizeof(double))) == NULL)
    {
        return (NULL);
    }

    for (i = 0; i < length; i++)
    {
        v[i] = 0.0;
    }

    return (v);
}

double *alloc_matrix(int rows, int cols)
{

    double *m;
    int i;

    if ((m = (double *)malloc(rows * cols * sizeof(double))) == NULL)
    {
        return (NULL);
    }

    for (i = 0; i < rows * cols; i++)
    {
        m[i] = 0.0;
    }

    return (m);
}

void matrix_free_2D(double **m, int n_layers)
{

    int i;

    for (i = 0; i < n_layers; ++i)
    {
        if (m[i] != NULL)
        {
            free(m[i]);
        }
    }
    free(m);
}

void matrix_free(double *m)
{

    if (m != NULL)
        free(m);
}

double *m_elem(double *m, int length, int x, int y)
{

    return (double *)&m[length * x + y];
}

/*void matrix_sum(double *c, double *a, double *b, int rows, int cols){

    int  col, row;
    double sum;

    for (row = 0; row < rows; row++) {
        for(col = 0; col < cols; col++) {
            sum = *m_elem(a, cols, row, col) + *m_elem(b, cols, row, col);
            //printf("- %f %f %f \n ", *m_elem(a, cols, row, col), *m_elem(b, cols, row, col),sum);
            *m_elem(c, cols, row, col) = sum;
        }
    }
}*/

void add_vectors_GPU(double *A, double *B, double *C, size_t N)
{

    cudaEvent_t start, stop;
    double *d_A, *d_B, *d_C;
    double milliseconds = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuErrchk(cudaMalloc(&d_A, N * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_B, N * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_C, N * sizeof(double)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B, N * sizeof(double), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil((double)N / thr_per_blk);

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_vec_add<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C, N);
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array d_C to host array C
    cudaMemcpy(C, d_C, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matrix_sum(double *c, double *a, double *b, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        add_vectors_GPU(a + cols * i, b + cols * i, c + cols * i, cols);
    }
}

void sub_vectors_GPU(double *A, double *B, double *C, size_t N)
{

    cudaEvent_t start, stop;
    double *d_A, *d_B, *d_C;
    double milliseconds = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuErrchk(cudaMalloc(&d_A, N * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_B, N * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_C, N * sizeof(double)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B, N * sizeof(double), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil((double)N / thr_per_blk);

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_vec_sub<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C, N);
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array d_C to host array C
    cudaMemcpy(C, d_C, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matrix_sub(double *c, double *a, double *b, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        sub_vectors_GPU(a + cols * i, b + cols * i, c + cols * i, cols);
    }
}

/*void matrix_sub(double *c, double *a, double *b, int rows, int cols){

    int col, row;
    double sum;

    for (row = 0; row < rows; row++) {
        for(col = 0; col < cols; col++) {
            sum = *m_elem(a, cols, row, col) - *m_elem(b, cols, row, col);
            *m_elem(c, cols, row, col) = sum;
        }
    }
}*/

/*void matrix_mul_cnt(double *m, int rows, int cols, double cnt){

    int col, row;

    for (row = 0; row < rows; row++) {
        for(col = 0; col < cols; col++) {
            *m_elem(m, cols, row, col) *= cnt;
        }
    }
}*/

void mulcnt_vectors_GPU(double *A, size_t N, double cnt)
{

    cudaEvent_t start, stop;
    double *d_A;
    double milliseconds = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuErrchk(cudaMalloc(&d_A, N * sizeof(double)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil((double)N / thr_per_blk);

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_vec_mulcnt<<<blk_in_grid, thr_per_blk>>>(d_A, N, cnt);
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array d_C to host array C
    cudaMemcpy(A, d_A, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
}

void matrix_mul_cnt(double *m, int rows, int cols, double cnt)
{
    for (int i = 0; i < rows; i++)
    {
        mulcnt_vectors_GPU(m + cols * i, cols, cnt);
    }
}

/*void matrix_zero(double *m, int rows, int cols){

    int col, row;

    for (row = 0; row < rows; row++) {
        for(col = 0; col < cols; col++) {
            *m_elem(m, cols, row, col) = 0.0;
        }
    }
}*/

void zero_vectors_GPU(double *A, size_t N)
{

    cudaEvent_t start, stop;
    double *d_A;
    double milliseconds = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuErrchk(cudaMalloc(&d_A, N * sizeof(double)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil((double)N / thr_per_blk);

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_vec_zero<<<blk_in_grid, thr_per_blk>>>(d_A, N);
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array d_C to host array C
    cudaMemcpy(A, d_A, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
}

void matrix_zero(double *m, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        zero_vectors_GPU(m + cols * i, cols);
    }
}

/*void matrix_mul_dot(double *c, double *a, double *b, int rows, int cols){

    int col, row;
    double prod;

    for (row = 0; row < rows; row++) {
        for(col = 0; col < cols; col++) {
            prod = *m_elem(a, cols, row, col) * *m_elem(b, cols, row, col);
            //printf("- %f %f %f \n ", *m_elem(a, rows, row, col), *m_elem(b, rows, row, col),sum);
            *m_elem(c, cols, row, col) = prod;
        }
    }
}*/

void muldot_vectors_GPU(double *A, double *B, double *C, size_t N)
{

    cudaEvent_t start, stop;
    double *d_A, *d_B, *d_C;
    double milliseconds = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuErrchk(cudaMalloc(&d_A, N * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_B, N * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_C, N * sizeof(double)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B, N * sizeof(double), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil((double)N / thr_per_blk);

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_vec_muldot<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C, N);
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array d_C to host array C
    cudaMemcpy(C, d_C, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void matrix_mul_dot(double *c, double *a, double *b, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        muldot_vectors_GPU(a + i * cols, b + i * cols, c + i * cols, cols);
    }
}

/*double *matrix_transpose(double *m, int rows, int cols){

    double *m_t;
    int i, j;

    if ((m_t = (double*)malloc(rows * cols * sizeof(double))) == NULL) {
        return(NULL);
    }

    for (i = 0; i < rows; i++){
        for (j = 0; j < cols; j++){
            *m_elem(m_t, rows, j, i) = *m_elem(m, cols, i, j);
        }
    }

    return(m_t);
}*/

void matrix_transpose(double *A, int rows, int cols)
{

    cudaEvent_t start, stop;
    double *d_A;
    float milliseconds = 0;
    int N = rows * cols;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuErrchk(cudaMalloc(&d_A, N * sizeof(double)));

    dim3 hb = dim3(8, 8); // hari kopurua blokeko
    dim3 b = dim3(2, 2);  // bloke kopurua

    // thr_per_blk = THR_PER_BLOCK;
    // blk_in_grid = ceil( (float)N / thr_per_blk );

    gpuErrchk(cudaEventRecord(start));
    cuda_mat_transpose<<<b, hb>>>(d_A, N);
    gpuErrchk(cudaEventRecord(stop));

    cudaMemcpy(A, d_A, sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
}

/*void matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols){

    assert(a_cols == b_rows);

    int i, col, row;
    double sum;

#ifdef TIMING
    int res_time;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
    res_time = clock_gettime(clk_id, &t1);
#endif

    for (row = 0; row < a_rows; row++) {
        for(col = 0; col < b_cols; col++) {
            sum = 0.0;
            for (i = 0; i < a_cols; i++) {
                sum += *m_elem(a, a_cols, row, i) * *m_elem(b, b_cols, i, col);
                //printf("%lf %lf\n", *m_elem(a, a_cols, row, i), *m_elem(b, b_cols, i, col));
            }
            *m_elem(c, b_cols, row, col) = sum;
        }
    }

#ifdef TIMING
    res_time = clock_gettime(clk_id, &t2);
    printf("Matrix mul execution time: %ld us \n", diff_time(t2, t1));
#endif

}*/

void matrix_mul(double *A, double *B, double *C, int a_rows, int a_cols, int b_rows, int b_cols)
{

    if (a_cols == b_rows)
    {
        int N = a_rows*b_cols;
        cudaEvent_t start, stop;
        double *d_A, *d_B, *d_C;
        float milliseconds = 0;
        int thr_per_blk, blk_in_grid;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        gpuErrchk(cudaMalloc(&d_A, N * sizeof(double)));
        gpuErrchk(cudaMalloc(&d_B, N * sizeof(double)));
        gpuErrchk(cudaMalloc(&d_C, N * sizeof(double)));

        thr_per_blk = THR_PER_BLOCK;
        blk_in_grid = ceil((double)N / thr_per_blk);

        gpuErrchk(cudaEventRecord(start));
        cuda_prod_matrixes<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C, N);
        gpuErrchk(cudaEventRecord(stop));

        cudaMemcpy(C, d_C, sizeof(double), cudaMemcpyDeviceToHost);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    else
    {
        printf("Matrizeak ezin dira biderkatu\n");
    }
}

/*void matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d)
{

    int i, col, row;
    double sum;

    for (row = 0; row < a_rows; row++)
    {
        for (col = 0; col < b_cols; col++)
        {
            sum = 0.0;
            for (i = 0; i < a_cols; i++)
            {
                sum += *m_elem(a, a_cols, row, i) * *m_elem(b, b_cols, i, col);
                // printf("%lf %lf\n", *m_elem(a, a_cols, row, i), *m_elem(b, b_cols, i, col));
            }
            *m_elem(c, b_cols, row, col) = sum + *m_elem(d, b_cols, row, col);
        }
    }
}*/

void matrix_mul_add(double *A, double *B, double *C, int a_rows, int a_cols, int b_rows, int b_cols, double *D)
{

    if (a_cols == b_rows)
    {
        int N = a_rows*b_cols;
        cudaEvent_t start, stop;
        double *d_A, *d_B, *d_C, *d_D;
        float milliseconds = 0;
        int thr_per_blk, blk_in_grid;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        gpuErrchk(cudaMalloc(&d_A, N * sizeof(double)));
        gpuErrchk(cudaMalloc(&d_B, N * sizeof(double)));
        gpuErrchk(cudaMalloc(&d_C, N * sizeof(double)));
        gpuErrchk(cudaMalloc(&d_D, N * sizeof(double)));

        thr_per_blk = THR_PER_BLOCK;
        blk_in_grid = ceil((double)N / thr_per_blk);

        gpuErrchk(cudaEventRecord(start));
        cuda_prod_add_matrixes<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C, d_D, N);
        gpuErrchk(cudaEventRecord(stop));

        cudaMemcpy(C, d_C, sizeof(double), cudaMemcpyDeviceToHost);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_D);
    }
    else
    {
        printf("Matrizeak ezin dira biderkatu\n");
    }
}

/*void matrix_func(double *n, double *m, int rows, int cols, double (*func)(double))
{

    int col, row;

    for (row = 0; row < rows; row++)
    {
        for (col = 0; col < cols; col++)
        {
            *m_elem(n, cols, row, col) = func(*m_elem(m, cols, row, col));
        }
    }
}*/

void func_vectors_GPU(double *n, double *m, int N, double (*func)(double))
{

    cudaEvent_t start, stop;
    double *d_M, *d_N;
    double milliseconds = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuErrchk(cudaMalloc(&d_M, N * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_N, N * sizeof(double)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_M, m, N * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_N, n, N * sizeof(double), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil((double)N / thr_per_blk);

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_vec_func<<<blk_in_grid, thr_per_blk>>>(d_N, d_M, N, func);
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array d_C to host array C
    cudaMemcpy(n, d_N, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_M);
    cudaFree(d_N);
}

void matrix_func(double *n, double *m, int rows, int cols, double (*func)(double))
{
    for (int i = 0; i < rows; i++)
    {
        func_vectors_GPU(n + i * cols, m + i * cols, cols, func);
    }
}



void print_matrix(double *m, int m_rows, int m_cols)
{

    int col, row;
    printf("%d %d\n", m_rows, m_cols);
    for (row = 0; row < m_rows; row++)
    {
        for (col = 0; col < m_cols; col++)
        {
            printf("(%d %d) %.*lf ", row, col, 10, *m_elem(m, m_cols, row, col));
        }
        printf("\n");
    }
    printf("\n");
}
