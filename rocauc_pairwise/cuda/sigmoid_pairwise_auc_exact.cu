#include <math.h>
#include <stdio.h>

__device__ float deltaauc_exact(const int * y_true,
                                const float * y_pred,
                                const int * counters_p,
                                const int * counters_n,
                                const int * y_pred_left,
                                const int * y_pred_right,
                                int n_ones,
                                int n_zeros,
                                int i, int j)
{
    float ypredi = y_pred[i];
    float ypredj = y_pred[j];
    
    if(ypredi < ypredj){
        int tmp = i;
        i = j;
        j = tmp;
    }

    ypredi = y_pred[i];
    ypredj = y_pred[j];

    int li = y_true[i];
    int lj = y_true[j];


    float deltaji = lj - li;
    
   
    float deltai = 0.5 * counters_p[i]*counters_n[i] - 0.5*(1.0*counters_p[i] + deltaji) * (1.0 * counters_n[i] - deltaji);
    float deltaj = 0.5 * counters_p[j]*counters_n[j] - 0.5*(counters_p[j] - deltaji) * (1.0 * counters_n[j] + deltaji);
    
    float delta_eq = 0.;
    float multiplicate = 1.;

    if(deltaji == -1)
        delta_eq = counters_p[i] + counters_n[j] - 2.;
    else
        delta_eq = -(counters_p[i] + counters_n[j]);
    if(deltaji == 0)
        multiplicate *= 0;
    if(ypredi == ypredj)
       multiplicate *= 0;
    return multiplicate * (float(delta_eq) + float(deltai) + float(deltaj) - float(deltaji) * float(abs(y_pred_right[i] - y_pred_left[j]))) / float(n_ones * n_zeros);
    
}

__global__ void deltaauc_exact_wrapper(float * deltaauc_,
                                       const int * y_true,
                                       const float * y_pred,
                                       const int * counters_n,
                                       const int * counters_p,
                                       const int * y_pred_left,
                                       const int * y_pred_right,
                                       int n_ones,
                                       int n_zeros,
                                       int i, int j)
{
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int bdimx = blockDim.x;

    int s = (bidx*bdimx) + tidx;

    if(s == 0){
        float deltaauc_tmp = deltaauc_exact(y_true, y_pred, counters_n, counters_p, y_pred_left, y_pred_right, n_ones, n_zeros, i, j);
        atomicAdd(deltaauc_, deltaauc_tmp);
    }
}

__global__ void sigmoid_pairwise_loss_auc_exact_gpu(int size,
                                                    float * loss,
                                                    const int * y_true,
                                                    const float * exp_pred,
                                                    const int * counters_n,
                                                    const int * counters_p,
                                                    const int * y_pred_left,
                                                    const int * y_pred_right,
                                                    int n_ones, 
                                                    int n_zeros)
{
    float eps = 1e-20;
    float deltaauc_ij = 0;
    float cur_loss = 0.;

    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int bdimx = blockDim.x;

    int i = (bidx*bdimx) + tidx;
    float P = 0.;
    float P_hat = 0.;

    if (i < size){
        for(int j = 0; j < i + 1; j++){
            P_hat = 0.5 *(y_true[i] - y_true[j]) + 0.5;
            P = 1.0 / (1.0 + (exp_pred[j] / exp_pred[i]));
            deltaauc_ij = deltaauc_exact(y_true, exp_pred, counters_n, counters_p, y_pred_left, y_pred_right, n_ones, n_zeros, i, j);
            deltaauc_ij = abs(deltaauc_ij);
            cur_loss += deltaauc_ij*(P_hat*log(P + eps) + (1.0 - P_hat)*log(1.0 - P + eps));
        }
        atomicAdd(loss, cur_loss);
    }
}

__global__ void sigmoid_pairwise_grad_hess_auc_exact_gpu(int size,
                                                         float * grad,
                                                         float * hess,
                                                         const int * y_true,
                                                         const float * exp_pred,
                                                         const int * counters_n,
                                                         const int * counters_p,
                                                         const int * y_pred_left, 
                                                         const int * y_pred_right,
                                                         int n_ones,
                                                         int n_zeros)
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

    float deltaauc_ij = 0.;

    if(i < size)
    {
        for(int j = 0; j < i + 1; j++){
            deltaauc_ij = deltaauc_exact(y_true, exp_pred, counters_n, counters_p, y_pred_left, y_pred_right, n_ones, n_zeros, i, j);
            deltaauc_ij = abs(deltaauc_ij);

            P_hat = 0.5 *(y_true[i] - y_true[j]) + 0.5;
            exp_tmp_diff = exp_pred[i] / exp_pred[j];

            cur_d_dx_i = deltaauc_ij*((P_hat - 1.) * exp_tmp_diff + P_hat) / (exp_tmp_diff + 1.);
            cur_d_dx_j = -cur_d_dx_i;
            cur_d2_dx2_i = deltaauc_ij*(-exp_pred[i]*exp_pred[j]) / ((exp_pred[i] + exp_pred[j])*(exp_pred[i] + exp_pred[j]));
            cur_d2_dx2_j = cur_d2_dx2_i;
        
            atomicAdd(&(grad[i]), cur_d_dx_i);
            atomicAdd(&(grad[j]), cur_d_dx_j);
            atomicAdd(&(hess[i]), cur_d2_dx2_i);
            atomicAdd(&(hess[j]), cur_d2_dx2_j);
        }
    }
}