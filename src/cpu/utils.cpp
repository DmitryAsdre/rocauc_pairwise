#include <cstddef>
#include <cmath>
#include <utility>
#include <cstring>
#include <tuple>
#include "utils.hpp"

template <class T_true, class T_pred, class T_argsorted>
std::pair<int32_t*, int32_t*> get_non_unique_labels_count(T_true* y_true, T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N){
    int32_t counter_p(0), counter_n(0);
    int32_t l_pointer(0), r_pointer(0);

    int32_t * counters_p = new int32_t[N];
    int32_t * counters_n = new int32_t[N];

    memset((void*) counters_p, 0, N*sizeof(int32_t));
    memset((void*) counters_n, 0, N*sizeof(int32_t));

    for(r_pointer = 0; r_pointer < N; r_pointer++){
        if(y_true[y_pred_argsorted[r_pointer]] == 1)
            counter_p += 1;
        else
            counter_n += 1;

        if((r_pointer + 1 != N) &&(y_pred[y_pred_argsorted[r_pointer]] == y_pred[y_pred_argsorted[r_pointer + 1]]))
            continue;
        else{
            while(l_pointer <= r_pointer){
                counters_p[y_pred_argsorted[l_pointer]] = counter_p;
                counters_n[y_pred_argsorted[l_pointer]] = counter_n;
                l_pointer += 1;
            }
            counter_n = 0;
            counter_p = 0;
        }
    }
    
    return std::make_pair<int32_t*, int32_t*>(&*counters_p, &*counters_n);
}

template<class T_pred, class T_argsorted> 
std::pair<int32_t*, int32_t*> get_non_unique_borders(T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N){
    int32_t left_p(0), right_p(0), j(0), s(0);

    int32_t * y_pred_left = new int32_t[N];
    int32_t * y_pred_right = new int32_t[N];

    memset((void*)y_pred_left, 0, N*sizeof(int32_t));
    memset((void*)y_pred_right, 0, N*sizeof(int32_t));

    for(j = 0; j < N; j++){
        s = N - j - 1;
        if((j + 1 != N) && (y_pred[y_pred_argsorted[s]] == y_pred[y_pred_argsorted[s - 1]]))
            y_pred_left[y_pred_argsorted[s]] = left_p;
        else{
            y_pred_left[y_pred_argsorted[s]] = left_p;
            left_p = j + 1;
        }
    }

    right_p = N - 1;
    for(j = N - 1; j >= 0; j--){
        s = N - j - 1;
        if((j - 1 != -1) && (y_pred[y_pred_argsorted[s]] == y_pred[y_pred_argsorted[s + 1]])){
            y_pred_right[y_pred_argsorted[s]] = right_p;
            continue;
        }
        else{
            y_pred_right[y_pred_argsorted[s]] = right_p;
            right_p = j - 1;
        }
    }
    
    return std::make_pair<int32_t*, int32_t*>(&*y_pred_left, &*y_pred_right);
}

template<class T_true, class T_pred, class T_argsorted>
std::tuple<int32_t*, int32_t*, int32_t*, int32_t*> get_labelscount_borders(T_true* y_true, T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N){
    std::pair<int32_t*, int32_t*> counters = get_non_unique_labels_count<T_true, T_pred, T_argsorted>(y_true, y_pred, y_pred_argsorted, N);
    std::pair<int32_t*, int32_t*> y_pred_sides = get_non_unique_borders<T_pred, T_argsorted>(y_pred, y_pred_argsorted, N);

    return std::make_tuple<int32_t*, int32_t*, int32_t*, int32_t*>(&*(counters.first), &*(counters.second), &*(y_pred_sides.first), &*(y_pred_sides.second));
}

template<class T_out, class T_true, class T_pred, class T_argsorted>
T_out * get_inverse_argsort(T_true* y_true, T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N){
    T_out * y_pred_ranks = new T_out[N];
    memset((void*)y_pred_ranks, 0, N*sizeof(T_out));

    for(size_t k = 0; k < N; k++){
        y_pred_ranks[y_pred_argsorted[N - k - 1]] = k;
    }
    return y_pred_ranks;
}

template<class T_true, class T_pred, class T_argsorted>
long* get_inverse_argsort_wrapper(T_true* y_true, T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N){
    return get_inverse_argsort<long, T_true, T_pred, T_argsorted>(y_true, y_pred, y_pred_argsorted, N);
}