#include <cmath>
#include <cstddef>
#include "constants.cuh"
#include "deltaauc_kernels.cu"
#include "sigmoid_pairwise_kernels.cuh"

__global__ void sigmoid_pairwise_loss_auc_kernel(int32_t* y_true, float* exp_pred,
                                                 int32_t* y_pred_ranks, float* sigmoid_loss,
                                                 uint n_ones, uint n_zeroes, uint N){
    int gid = threadIdx.x + blockDim.x*blockIdx.x;

    __shared__ float _loss;

    if(threadIdx.x == 0){
        _loss = 0;
    }
    __syncthreads();

    while(gid < N){
        uint _i = (gid%2 == 1) ? (N - uint(gid/2) - 1) : uint(gid/2);
        for(uint j = 0; j <= _i; j++){
            if(y_true[_i] == y_true[j])
                continue;

            float _deltaauc = deltaauc_kernel(y_true, y_pred_ranks, n_ones, n_zeroes, _i, j);
            float P_hat = 0.5f*(y_true[_i] - y_true[j]) + 0.5f;
            float P = 1.f / (1.f + (exp_pred[j] / exp_pred[_i]));
            float delta_loss = fabsf(_deltaauc)*(P_hat*logf(P + EPS_CU) + (1.f - P_hat)*logf(1.f - P - EPS_CU));
            atomicAdd(&_loss, delta_loss);
        }
        gid += blockDim.x*gridDim.x;
    }
    __syncthreads();

    if(threadIdx.x == 0){
        atomicAdd(sigmoid_loss, _loss);
    }
}

__global__ void sigmoid_pairwise_grad_hess_auc_kernel(int32_t* y_true, float* exp_pred,
                                                      int32_t* y_pred_ranks,
                                                      float* grad, float* hess, 
                                                      uint n_ones, uint n_zeroes, uint N){
    int gid = threadIdx.x + blockDim.x*blockIdx.x;

    float exp_tmp_diff = 0.;
    float cur_d_dx_i = 0.;
    float cur_d_dx_j = 0.;
    float cur_d2_dx2_i = 0.;
    float cur_d2_dx2_j = 0.;

    float P_hat = 0.;

    while(gid < N){
        uint _i = (gid%2 == 1) ? (N - uint(gid/2) - 1) : uint(gid/2);
        for(uint j = 0; j < _i; j++){
            if(y_true[_i] == y_true[j])
                continue;
            
            P_hat = 0.5f*(y_true[_i] - y_true[j]) + 0.5f;
            exp_tmp_diff = exp_pred[_i] / exp_pred[j];

            float _deltaauc = deltaauc_kernel(y_true, y_pred_ranks, n_ones, n_zeroes, _i, j);

            cur_d_dx_i = fabsf(_deltaauc)*((P_hat - 1.f)*exp_tmp_diff + P_hat) / (exp_tmp_diff + 1.f);
            cur_d_dx_j = -cur_d_dx_i;
            cur_d2_dx2_i = fabsf(_deltaauc)*(-exp_pred[_i]/(exp_pred[_i] + exp_pred[j]))*(exp_pred[j]/(exp_pred[_i] + exp_pred[j]));
            cur_d2_dx2_j = cur_d2_dx2_i;


        
            atomicAdd(&(grad[_i]), cur_d_dx_i);
            atomicAdd(&(grad[j]), cur_d_dx_j);
            atomicAdd(&(hess[_i]), cur_d2_dx2_i);
            atomicAdd(&(hess[j]), cur_d2_dx2_j);
        }
        gid += blockDim.x*gridDim.x;
    }
}

__global__ void sigmoid_pairwise_loss_auc_exact_kernel(int32_t* y_true, float* exp_pred,
                                                       int32_t* counters_p, int32_t* counters_n,
                                                       int32_t* y_pred_left, int32_t* y_pred_right, 
                                                       float* sigmoid_loss,
                                                       uint n_ones, uint n_zeroes, uint N){
    int gid = threadIdx.x + blockDim.x*blockIdx.x;

    __shared__ float _loss;

    if(threadIdx.x == 0){
        _loss = 0;
    }
    __syncthreads();

    while(gid < N){
        uint _i = (gid%2 == 1) ? (N - uint(gid/2) - 1) : uint(gid/2);
        for(uint j = 0; j <= _i; j++){
            if(y_true[_i] == y_true[j])
                continue;

            float P_hat = 0.5f*(y_true[_i] - y_true[j]) + 0.5f;
            float P = 1.f / (1.f + (exp_pred[j] / exp_pred[_i]));

            float _deltaauc = deltaauc_exact_kernel(y_true, exp_pred, counters_p, counters_n, y_pred_left, y_pred_right, n_ones, n_zeroes, _i, j);

            float delta_loss = fabsf(_deltaauc)*(P_hat*logf(P + EPS_CU) + (1.f - P_hat)*logf(1.f - P - EPS_CU));
            atomicAdd(&_loss, delta_loss);
        }
        gid += blockDim.x*gridDim.x;
    }
    __syncthreads();

    if(threadIdx.x == 0){
        atomicAdd(sigmoid_loss, _loss);
    }
}

__global__ void sigmoid_pairwise_grad_hess_auc_exact_kernel(int32_t* y_true, float* exp_pred,
                                                            int32_t* counters_p, int32_t* counters_n,
                                                            int32_t* y_pred_left, int32_t* y_pred_right,
                                                            float* grad, float* hess,
                                                            uint n_ones, uint n_zeroes, uint N){
    int gid = threadIdx.x + blockDim.x*blockIdx.x;

    float exp_tmp_diff = 0.;
    float cur_d_dx_i = 0.;
    float cur_d_dx_j = 0.;
    float cur_d2_dx2_i = 0.;
    float cur_d2_dx2_j = 0.;

    float P_hat = 0.;

    while(gid < N){
        uint _i = (gid%2 == 1) ? (N - uint(gid/2) - 1) : uint(gid/2);
        for(uint j = 0; j < _i; j++){
            if(y_true[_i] == y_true[j])
                continue;

            P_hat = 0.5f*(y_true[_i] - y_true[j]) + 0.5f;
            exp_tmp_diff = exp_pred[_i] / exp_pred[j];

            float _deltaauc = deltaauc_exact_kernel(y_true, exp_pred, counters_p, counters_n, y_pred_left, y_pred_right, n_ones, n_zeroes, _i, j);    

            cur_d_dx_i = fabsf(_deltaauc)*((P_hat - 1.f)*exp_tmp_diff + P_hat) / (exp_tmp_diff + 1.f);
            cur_d_dx_j = -cur_d_dx_i;
            cur_d2_dx2_i = fabsf(_deltaauc)*(-exp_pred[_i]/(exp_pred[_i] + exp_pred[j]))*(exp_pred[j]/(exp_pred[_i] + exp_pred[j]));
            cur_d2_dx2_j = cur_d2_dx2_i;
        
            atomicAdd(&(grad[_i]), cur_d_dx_i);
            atomicAdd(&(grad[j]), cur_d_dx_j);
            atomicAdd(&(hess[_i]), cur_d2_dx2_i);
            atomicAdd(&(hess[j]), cur_d2_dx2_j);
        }
        gid += blockDim.x*gridDim.x;
    }
}

__global__ void sigmoid_pairwise_loss_auc_exact_sm_kernel(int32_t* y_true, float* exp_pred,
                                                          int32_t* counters_p, int32_t* counters_n,
                                                          int32_t* y_pred_left, int32_t* y_pred_right, 
                                                          float* sigmoid_loss, float eps,
                                                          uint n_ones, uint n_zeroes, uint N){
    int gid = threadIdx.x + blockDim.x*blockIdx.x;

    __shared__ float _loss;

    if(threadIdx.x == 0){
        _loss = 0;
    }
    __syncthreads();

    while(gid < N){
        uint _i = (gid%2 == 1) ? (N - uint(gid/2) - 1) : uint(gid/2);
        for(uint j = 0; j <= _i; j++){
            if(y_true[_i] == y_true[j])
                continue;

            float P_hat = 0.5f*(y_true[_i] - y_true[j]) + 0.5f;
            float P = 1.f / (1.f + (exp_pred[j] / exp_pred[_i]));
            
            float _deltaauc = 0.f;

            if(y_true[_i] != y_true[j])
                _deltaauc = deltaauc_exact_kernel(y_true, exp_pred, counters_p, counters_n, y_pred_left, y_pred_right, n_ones, n_zeroes, _i, j);

            float delta_loss = (fabsf(_deltaauc) + eps)*(P_hat*logf(P + EPS_CU) + (1.f - P_hat)*logf(1.f - P - EPS_CU));
            atomicAdd(&_loss, delta_loss);
        }
        gid += blockDim.x*gridDim.x;
    }
    __syncthreads();

    if(threadIdx.x == 0){
        atomicAdd(sigmoid_loss, _loss);
    }
}

__global__ void sigmoid_pairwise_grad_hess_auc_exact_sm_kernel(int32_t* y_true, float* exp_pred,
                                                               int32_t* counters_p, int32_t* counters_n,
                                                               int32_t* y_pred_left, int32_t* y_pred_right,
                                                               float* grad, float* hess, float eps,
                                                               uint n_ones, uint n_zeroes, uint N){
    int gid = threadIdx.x + blockDim.x*blockIdx.x;

    float exp_tmp_diff = 0.;
    float cur_d_dx_i = 0.;
    float cur_d_dx_j = 0.;
    float cur_d2_dx2_i = 0.;
    float cur_d2_dx2_j = 0.;

    float P_hat = 0.;

    while(gid < N){
        uint _i = (gid%2 == 1) ? (N - uint(gid/2) - 1) : uint(gid/2);
        for(uint j = 0; j < _i; j++){
            P_hat = 0.5f*(y_true[_i] - y_true[j]) + 0.5f;
            exp_tmp_diff = exp_pred[_i] / exp_pred[j];

            float _deltaauc = 0.f;
            if(y_true[_i] != y_true[j])
                _deltaauc = deltaauc_exact_kernel(y_true, exp_pred, counters_p, counters_n, y_pred_left, y_pred_right, n_ones, n_zeroes, _i, j);    

            cur_d_dx_i = (fabsf(_deltaauc) + eps)*((P_hat - 1.f)*exp_tmp_diff + P_hat) / (exp_tmp_diff + 1.f);
            cur_d_dx_j = -cur_d_dx_i;
            cur_d2_dx2_i = (fabsf(_deltaauc) + eps)*(-exp_pred[_i]/(exp_pred[_i] + exp_pred[j]))*(exp_pred[j]/(exp_pred[_i] + exp_pred[j]));
            cur_d2_dx2_j = cur_d2_dx2_i;
        
            atomicAdd(&(grad[_i]), cur_d_dx_i);
            atomicAdd(&(grad[j]), cur_d_dx_j);
            atomicAdd(&(hess[_i]), cur_d2_dx2_i);
            atomicAdd(&(hess[j]), cur_d2_dx2_j);
        }
        gid += blockDim.x*gridDim.x;
    }
}