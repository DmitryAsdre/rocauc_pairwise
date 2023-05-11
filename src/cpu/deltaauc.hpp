#ifndef DELTAAUC_HPP
#define DELTAAUC_HPP

#include <cstddef>
#include <cmath>
#include <utility>
#include <cstring>
#include <tuple>

template<class T_true, class T_ranks>
double deltaauc(T_true* y_true, T_ranks* y_pred_ranks,
                size_t n_ones, size_t n_zeroes,
                size_t i, size_t j);

template<class T_true, class T_pred>
double deltaauc_exact(T_true y_true, T_pred y_pred,
                      int32_t* counters_p, int32_t* counters_n,
                      int32_t* y_pred_left, int32_t* y_pred_right,
                      size_t n_ones, size_t n_zeroes, size_t i, size_t j);

#endif