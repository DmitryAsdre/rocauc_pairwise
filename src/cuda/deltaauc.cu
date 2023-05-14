#include <cmath>
#include <cstddef>
#include <assert.h>

#include "deltaauc_kernels.cuh"

float deltaauc(int32_t* y_true, int32_t* y_pred_ranks,
               size_t n_ones, size_t n_zeroes, 
               size_t i, size_t j, size_t N){
    float _deltaauc = 0.f;
    float* _deltaauc_device;

    size_t size = N*sizeof(int32_t);
    
    int32_t* y_true_device;
    int32_t* y_pred_ranks_device;

    cudaError_t err;

    err = cudaMalloc((void**)&y_true_device, size);
    assert(err == 0);
    err = cudaMemcpy((void*)y_true_device, (void*)y_true, size, cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&y_pred_ranks_device, size);
    assert(err == 0);
    err = cudaMemcpy((void*)y_pred_ranks_device, (void*)y_pred_ranks, size, cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&_deltaauc_device, sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)_deltaauc_device, (void*)&_deltaauc, sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);

    deltaauc_kernel_wrapper<<<2, 2>>>(y_true_device, y_pred_ranks_device, n_ones, n_zeroes, i, j, _deltaauc_device);

    err = cudaGetLastError();
    assert(err == 0);

    err = cudaMemcpy((void*)&_deltaauc, (void*)_deltaauc_device, sizeof(float), cudaMemcpyDeviceToHost);
    assert(err == 0);

    cudaFree(y_true_device);
    cudaFree(y_pred_ranks_device);
    cudaFree(_deltaauc_device);

    return _deltaauc;
}

float deltaauc_exact(int32_t* y_true, float* y_pred,
                     int32_t* counters_p, int32_t* counters_n,
                     int32_t* y_pred_left, int32_t* y_pred_right,
                     size_t n_ones, size_t n_zeroes, size_t i, size_t j, size_t N){
    float _deltaauc = 0.f;
    float* _deltaauc_device;

    int32_t* y_true_device;
    float* y_pred_device;
    int32_t* counters_p_device;
    int32_t* counters_n_device;
    int32_t* y_pred_left_device;
    int32_t* y_pred_right_device;

    cudaError_t err;

    err = cudaMalloc((void**)&y_true_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)y_true_device, (void*)y_true, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&y_pred_device, N*sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)y_pred_device, (void*)y_pred, N*sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&counters_p_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)counters_p_device, (void*)counters_p, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&counters_n_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)counters_n_device, (void*)counters_n, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&y_pred_left_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)y_pred_left_device, (void*)y_pred_left, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&y_pred_right_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)y_pred_right_device, (void*)y_pred_right, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&_deltaauc_device, sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)_deltaauc_device, (void*)&_deltaauc, sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);

    deltaauc_exact_kernel_wrapper<<<2, 2>>>(y_true_device, y_pred_device, counters_p_device, counters_n_device,
                                            y_pred_left_device, y_pred_right_device, n_ones, n_zeroes, i, j, _deltaauc_device);

    err = cudaGetLastError();
    assert(err == 0);
    
    err = cudaMemcpy((void*)&_deltaauc, (void*)_deltaauc_device, sizeof(float), cudaMemcpyDeviceToHost);
    assert(err == 0);

    cudaFree(y_true_device);
    cudaFree(y_pred_device);
    cudaFree(counters_p_device);
    cudaFree(counters_n_device);
    cudaFree(y_pred_left_device);
    cudaFree(y_pred_right_device);
    cudaFree(_deltaauc_device);

    return _deltaauc;
}
