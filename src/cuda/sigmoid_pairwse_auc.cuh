#ifndef SIGMOID_PAIRWISE_AUC_CUH
#define SIGMOID_PAIRWISE_AUC_CUH
#include <utility>
#include <cstddef>

float sigmoid_pairwise_loss_auc(int32_t* y_true, float* exp_pred,
                                long* y_pred_argsorted,
                                size_t n_ones, size_t n_zeroes, size_t N);

std::pair<float*, float*> sigmoid_pairwise_grad_hess_auc(int32_t* y_true, float* exp_pred,
                                                         long* y_pred_argsorted, 
                                                         size_t n_ones, size_t n_zeroes, size_t N);

float sigmoid_pairwise_loss_auc_exact(int32_t* y_true, float* exp_pred, 
                                      long* y_pred_argsorted, 
                                      size_t n_ones, size_t n_zeroes, size_t N);

std::pair<float*, float*> sigmoid_pairwise_grad_hess_auc_exact(int32_t* y_true, float* exp_pred, 
                                                               long* y_pred_argsorted, 
                                                               size_t n_ones, size_t n_zeroes, size_t N);

#endif
