#include <math.h>

__global__ void sigmoid_pairwise_loss_gpu(int size, float * loss, 
                                          const int * y_true, 
                                          const float * exp_pred)
{
    float eps = 1e-20;
    float P_hat = 0.;
    float P = 0.;
    int i = 0;

    float cur_loss = 0.;

    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int bdimx = blockDim.x;

    i = (bidx*bdimx) + tidx;

    if(i < size){
        for(int j = i; j >= 0; j--){
            P_hat = 0.5 *(y_true[i] - y_true[j]) + 0.5;
            P = 1.0 / (1.0 + (exp_pred[j] / exp_pred[i]));
            cur_loss += P_hat*log(P + eps) + (1.0 - P_hat)*log(1.0 - P + eps);
        }
        atomicAdd(loss, cur_loss);
    }
}

__global__ void sigmoid_pairwise_grad_hess_gpu(int size, float * grad,
                                               float * hess,
                                               const int * y_true,
                                               const float * exp_pred)
{
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int bdimx = blockDim.x;

    int i = (bidx*bdimx) + tidx;

    float exp_tmp_diff = 0.;
    float cur_d_dx_i = 0.;
    float cur_d_dx_j = 0.;
    float cur_d2_dx2_i = 0.;
    float cur_d2_dx2_j = 0.;

    float P_hat = 0.;

    if(i < size)
    {
        for(int j = 0; j < i + 1; j++){
            P_hat = 0.5 *(y_true[i] - y_true[j]) + 0.5;
            exp_tmp_diff = exp_pred[i] / exp_pred[j];

            cur_d_dx_i = ((P_hat - 1.) * exp_tmp_diff + P_hat) / (exp_tmp_diff + 1.);
            cur_d_dx_j = -cur_d_dx_i;
            cur_d2_dx2_i = ((-exp_pred[i]*exp_pred[j]) / ((exp_pred[i] + exp_pred[j])*(exp_pred[i] + exp_pred[j])));
            cur_d2_dx2_j = cur_d2_dx2_i;
        
            atomicAdd(&(grad[i]), cur_d_dx_i);
            atomicAdd(&(grad[j]), cur_d_dx_j);
            atomicAdd(&(hess[i]), cur_d2_dx2_i);
            atomicAdd(&(hess[j]), cur_d2_dx2_j);
        }
    }
}