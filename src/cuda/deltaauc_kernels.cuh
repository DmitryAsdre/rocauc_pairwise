#ifndef DELTAAUC_KERNELS_CUH
#define DELTAAUC_KERNELS_CUH

__device__ float deltaauc_kernel(int32_t* y_true, int32_t* y_pred_ranks,
                                 uint n_ones, uint n_zeroes,
                                 uint i, uint j);

__device__ float deltaauc_exact_kernel(int32_t* y_true, float* y_pred,
                                       int32_t* counters_p, int32_t* counters_n,
                                       int32_t* y_pred_left, int32_t* y_pred_right,
                                       uint n_ones, uint n_zeroes, uint i, uint j);

__global__ void deltaauc_kernel_wrapper(int32_t* y_true, int32_t* y_pred_ranks,
                                        uint n_ones, uint n_zeroes, 
                                        uint i, uint j, float* _deltaauc);


__global__ void deltaauc_exact_kernel_wrapper(int32_t* y_true, float* y_pred,
                                              int32_t* counters_p, int32_t* counters_n,
                                              int32_t* y_pred_left, int32_t* y_pred_right,
                                              uint n_ones, uint n_zeroes, uint i, uint j, float* _deltaauc);

#endif