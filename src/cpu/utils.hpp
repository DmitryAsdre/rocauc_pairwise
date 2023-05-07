#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstddef>
#include <cmath>
#include <utility>
#include <cstring>
#include <tuple>

template <typename T>
std::pair<int*, int*> get_non_unique_labels_count(int * y_true, T * y_pred, long * y_pred_argsorted, size_t N);

template<typename T> 
std::pair<int*, int*> get_non_unique_borders(T * y_pred, long * y_pred_argsorted, size_t N);

template<typename T>
std::tuple<int*, int*, int*, int*> get_labelscount_borders(int * y_true, T * y_pred, long * y_pred_argsorted, size_t N);

template<typename T>
long * get_inverse_argsort(int * y_true, T * y_pred, long * y_pred_argsorted, size_t N);

#endif