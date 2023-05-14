#ifndef DELTAAUC_KERNELS_CUH
#define DELTAAUC_KERNELS_CUH

__device__ float deltaauc_kernel(int32_t* y_true, int32_t* y_pred_ranks,
                                 size_t n_ones, size_t n_zeroes,
                                 size_t i, size_t j);

__device__ float deltaauc_exact_kernel(int32_t* y_true, float* y_pred,
                                       int32_t* counters_p, int32_t* counters_n,
                                       int32_t* y_pred_left, int32_t* y_pred_right,
                                       size_t n_ones, size_t n_zeroes, size_t i, size_t j);

__global__ void deltaauc_kernel_wrapper(int32_t* y_true, int32_t* y_pred_ranks,
                                        size_t n_ones, size_t n_zeroes, 
                                        size_t i, size_t j, float* _deltaauc);


__global__ void deltaauc_exact_kernel_wrapper(int32_t* y_true, float* y_pred,
                                              int32_t* counters_p, int32_t* counters_n,
                                              int32_t* y_pred_left, int32_t* y_pred_right,
                                              size_t n_ones, size_t n_zeroes, size_t i, size_t j, float* _deltaauc);

#endif