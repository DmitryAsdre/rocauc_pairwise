#ifndef SIGMOID_PAIRWISE_CUH
#define SIGMOID_PAIRWISE_CUH
#include <utility>
#include <cstddef>

float sigmoid_pairwise_loss(int32_t* y_true, float* exp_pred, size_t N);

std::pair<float*, float*> sigmoid_pairwise_grad_hess(int32_t* y_true, float* exp_pred, size_t N);

#endif