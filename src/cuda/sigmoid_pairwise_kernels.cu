#include <cmath>
#include <cstddef>
#include "constants.cuh"

__global__ void sigmoid_pairwise_loss_kernel(int32_t* y_true, float* exp_pred, float* sigmoid_loss, uint N){
    int gid = threadIdx.x + blockDim.x*blockIdx.x;

    __shared__ float _loss;

    if(threadIdx.x == 0){
        _loss = 0;
    }
    __syncthreads();

    while(gid < N){
        uint _i = (gid%2 == 1) ? (N - uint(gid/2) - 1) : uint(gid/2);
        for(uint j = 0; j <= _i; j++){
            float P_hat = 0.5f*(y_true[_i] - y_true[j]) + 0.5f;
            float P = 1.f / (1.f + (exp_pred[j] / exp_pred[_i]));
            float delta_loss = P_hat*logf(P + EPS_CU) + (1.f - P_hat)*logf(1.f - P - EPS_CU);
            atomicAdd(&_loss, delta_loss);
        }
        gid += blockDim.x*gridDim.x;
    }
    __syncthreads();

    if(threadIdx.x == 0){
        atomicAdd(sigmoid_loss, _loss);
    }
}

__global__ void sigmoid_pairwise_grad_hess_kernel(int32_t* y_true, float* exp_pred, float* grad, float* hess, uint N){
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

            cur_d_dx_i = ((P_hat - 1.f)*exp_tmp_diff + P_hat) / (exp_tmp_diff + 1.f);
            cur_d_dx_j = -cur_d_dx_i;
            cur_d2_dx2_i = (-exp_pred[_i]/(exp_pred[_i] + exp_pred[j]))*(exp_pred[j]/(exp_pred[_i] + exp_pred[j]));
            cur_d2_dx2_j = cur_d2_dx2_i;
        
            atomicAdd(&(grad[_i]), cur_d_dx_i);
            atomicAdd(&(grad[j]), cur_d_dx_j);
            atomicAdd(&(hess[_i]), cur_d2_dx2_i);
            atomicAdd(&(hess[j]), cur_d2_dx2_j);
        }
        gid += blockDim.x*gridDim.x;
    }
}