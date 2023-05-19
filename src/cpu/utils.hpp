#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstddef>
#include <cmath>
#include <utility>
#include <cstring>
#include <tuple>

template <class T_true, class T_pred, class T_argsorted>
std::pair<int32_t*, int32_t*> get_non_unique_labels_count(T_true* y_true, T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N);

template<class T_pred, class T_argsorted> 
std::pair<int32_t*, int32_t*> get_non_unique_borders(T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N);

template<class T_true, class T_pred, class T_argsorted>
std::tuple<int32_t*, int32_t*, int32_t*, int32_t*> get_labelscount_borders(T_true* y_true, T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N);

template<class T_out, class T_true, class T_pred, class T_argsorted>
T_out* get_inverse_argsort(T_true* y_true, T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N);

template<class T_true, class T_pred, class T_argsorted>
long* get_inverse_argsort_wrapper(T_true* y_true, T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N);

#endif