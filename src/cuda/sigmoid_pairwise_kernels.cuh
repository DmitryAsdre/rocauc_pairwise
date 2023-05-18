#ifndef SIGMOID_PAIRWISE_KERNELS_CUH
#define SIGMOID_PAIRWISE_KERNELS_CUH

__global__ void sigmoid_pairwise_loss_kernel(int32_t* y_true, float* exp_pred, float* sigmoid_loss, uint N);
__global__ void sigmoid_pairwise_grad_hess_kernel(int32_t* y_true, float* exp_pred, float* grad, float* hess, uint N);

#endif