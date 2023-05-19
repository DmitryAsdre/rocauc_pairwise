#ifndef SIGMOID_PAIRWISE_HPP
#define SIGMOID_PAIRWISE_HPP

#include <cstddef>
#include <cmath>
#include <utility>
#include <cstring>
#include <tuple>

template<class T_true, class T_pred>
double sigmoid_pairwise_loss(T_true* y_true, T_pred* exp_pred, size_t N);

template<class T_true, class T_pred>
std::pair<double*, double*> sigmoid_pairwise_diff_hess(T_true* y_true, T_pred* exp_pred, size_t N);

#endif