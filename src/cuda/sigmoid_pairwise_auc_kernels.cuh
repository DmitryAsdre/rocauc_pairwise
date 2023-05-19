#ifndef SIGMOID_PAIRWISE_AUC_KERNELS
#define SIGMOID_PAIRWISE_AUC_KERNELS
#include <cstddef>
#include "sigmoid_pairwise_kernels.cuh"
#include "deltaauc_kernels.cuh"
#include "../cpu/utils.hpp"

__global__ void sigmoid_pairwise_loss_auc_kernel(int32_t* y_true, float* exp_pred,
                                            int32_t* y_pred_ranks, float* sigmoid_loss,
                                            uint n_ones, uint n_zeroes, uint N);

__global__ void sigmoid_pairwise_grad_hess_auc_kernel(int32_t* y_true, float* exp_pred,
                                                      int32_t* y_pred_ranks,
                                                      float* grad, float* hess, 
                                                      uint n_ones, uint n_zeroes, uint N);

__global__ void sigmoid_pairwise_loss_auc_exact_kernel(int32_t* y_true, float* exp_pred,
                                                       int32_t* counters_p, int32_t* counters_n,
                                                       int32_t* y_pred_left, int32_t* y_pred_right, 
                                                       float* sigmoid_loss,
                                                       uint n_ones, uint n_zeroes, uint N);

__global__ void sigmoid_pairwise_grad_hess_auc_exact_kernel(int32_t* y_true, float* exp_pred,
                                                            int32_t* counters_p, int32_t* counters_n,
                                                            int32_t* y_pred_left, int32_t* y_pred_right,
                                                            float* grad, float* hess,
                                                            uint n_ones, uint n_zeroes, uint N);

__global__ void sigmoid_pairwise_loss_auc_exact_sm_kernel(int32_t* y_true, float* exp_pred,
                                                          int32_t* counters_p, int32_t* counters_n,
                                                          int32_t* y_pred_left, int32_t* y_pred_right, 
                                                          float* sigmoid_loss, float eps,
                                                          uint n_ones, uint n_zeroes, uint N);

__global__ void sigmoid_pairwise_grad_hess_auc_exact_sm_kernel(int32_t* y_true, float* exp_pred,
                                                               int32_t* counters_p, int32_t* counters_n,
                                                               int32_t* y_pred_left, int32_t* y_pred_right,
                                                               float* grad, float* hess, float eps,
                                                               uint n_ones, uint n_zeroes, uint N);
#endif