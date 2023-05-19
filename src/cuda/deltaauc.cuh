#ifndef DELTAAUC_CUH
#define DELTAAUC_CUH
#include <cstddef>

float deltaauc(int32_t* y_true, int32_t* y_pred_ranks,
               size_t n_ones, size_t n_zeroes, 
               size_t i, size_t j, size_t N);

float deltaauc_exact(int32_t* y_true, float* y_pred,
                     int32_t* counters_p, int32_t* counters_n,
                     int32_t* y_pred_left, int32_t* y_pred_right,
                     size_t n_ones, size_t n_zeroes, size_t i, size_t j, size_t N);

#endif