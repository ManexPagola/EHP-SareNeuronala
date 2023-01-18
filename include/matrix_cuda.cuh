#include <stdbool.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

/* Macro for checking cuda errors following a cuda launch or api call
 Taken from: https://gist.github.com/jefflarkin/5390993 */
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

#define gpuErrchk(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)



void matrix_sum(double *c, double *a, double *b, int rows, int cols);
void matrix_sub(double *c, double *a, double *b, int rows, int cols);
void matrix_mul_cnt(double *m, int rows, int cols, double cnt);
void matrix_zero(double *m, int rows, int cols);
void matrix_mul_dot(double *c, double *a, double *b, int rows, int cols);
void matrix_transpose(double* A, int rows, int cols);
void matrix_mul(double *A, double *B, double *C, int a_rows, int a_cols, int b_rows, int b_cols);
void matrix_mul_add(double *A, double *B, double *C, int a_rows, int a_cols, int b_rows, int b_cols, double *D);
void matrix_func(double *n, double *m, int rows, int cols, double (*func)(double));
void add_vectors_GPU(float *A, float *B, float *C, size_t N);
void sub_vectors_GPU(float *A, float *B, float *C, size_t N);
void mulcnt_vectors_GPU(float* A, size_t N, double cnt);
void zero_vectors_GPU(float* A, size_t N);
void muldot_vectors_GPU(double* A, double* B, double* C, size_t N);
void func_vectors_GPU(double *m, double *n, int N, double (*func)(double));
void cuda_mat_transpose(double* A, int N);
void cuda_prod_matrixes(double *A, double *B, double *C, int N);
void cuda_prod_add_matrixes(double *A, double *B, double *C, double *D, int N);
void cuda_vec_func(double *m, double *n, int N, double (*func)(double));
//double matrizea_biderkatu_GPU(float* A, float* B, float* C, size_t N);
//double atomicAdd_vectors_GPU(float* A, float* B, float* C, size_t N);