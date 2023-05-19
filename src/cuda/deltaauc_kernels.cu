#include <cmath>
#include <cstddef>
#include "constants.cuh"
#include "deltaauc_kernels.cuh"

__device__ float deltaauc_kernel(int32_t* y_true, int32_t* y_pred_ranks,
                                 uint n_ones, uint n_zeroes,
                                 uint i, uint j){
    float deltaauc_ = 0.f;
    float ranki = y_pred_ranks[i];
    float rankj = y_pred_ranks[j];

    deltaauc_ = (1.f*(y_true[i] - y_true[j]) * (rankj - ranki) / (n_ones * n_zeroes));
    return deltaauc_;
}

__global__ void deltaauc_kernel_wrapper(int32_t* y_true, int32_t* y_pred_ranks,
                                        uint n_ones, uint n_zeroes, 
                                        uint i, uint j, float* _deltaauc){
    int gid = threadIdx.x + blockDim.x*blockIdx.x;

    if(gid == 0){
        *_deltaauc = deltaauc_kernel(y_true, y_pred_ranks, n_ones, n_zeroes, i, j);
    }
}

__device__ float deltaauc_exact_kernel(int32_t* y_true, float* y_pred,
                                       int32_t* counters_p, int32_t* counters_n,
                                       int32_t* y_pred_left, int32_t* y_pred_right,
                                       uint n_ones, uint n_zeroes, uint i, uint j){
    float ypredi = y_pred[i];
    float ypredj = y_pred[j];
    
    if(ypredi < ypredj){
        uint tmpi = i;
        i = j;
        j = tmpi;
    }

    ypredi = y_pred[i];
    ypredj = y_pred[j];

    float deltaji = y_true[j] - y_true[i];

    float deltai =  0.5f*counters_p[i]*counters_n[i] - 0.5f*(counters_p[i] + deltaji) * (counters_n[i] - deltaji);
    float deltaj =  0.5f*counters_p[j]*counters_n[j] - 0.5f*(counters_p[j] - deltaji) * (counters_n[j] + deltaji);

    float delta_eq = 0.f;
    float multiplicate = 1.f;
    
    if(fabsf(deltaji + 1.f) < EPS_CU)
        delta_eq = counters_p[i] + counters_n[j] - 2.f;
    else
        delta_eq = -(counters_p[i] + counters_n[j]);
    
    if(fabsf(deltaji) < EPS_CU || fabsf(ypredi - ypredj) < EPS_CU)
        multiplicate *= 0;
    
    return multiplicate * (delta_eq + deltai + deltaj - deltaji * fabsf(y_pred_right[i] - y_pred_left[j])) / (n_ones * n_zeroes);
}

__global__ void deltaauc_exact_kernel_wrapper(int32_t* y_true, float* y_pred,
                                              int32_t* counters_p, int32_t* counters_n,
                                              int32_t* y_pred_left, int32_t* y_pred_right,
                                              uint n_ones, uint n_zeroes, uint i, uint j, float* _deltaauc)
{
    int gid = threadIdx.x + blockDim.x*blockIdx.x;

    if(gid == 0){
        *_deltaauc = deltaauc_exact_kernel(y_true, y_pred, counters_p, counters_n, y_pred_left, y_pred_right, n_ones, n_zeroes, i, j);
    }
}