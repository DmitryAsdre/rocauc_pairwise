#include <cstddef>
#include <cmath>
#include <utility>
#include <cstring>
#include <tuple>
#include "utils.hpp"

template <typename T>
std::pair<int*, int*> get_non_unique_labels_count(int * y_true, T * y_pred, long * y_pred_argsorted, size_t N){
    int counter_p(0), counter_n(0);
    int l_pointer(0), r_pointer(0);

    int * counters_p = new int[N];
    int * counters_n = new int[N];

    memset((void*) counters_p, 0, N*sizeof(int));
    memset((void*) counters_n, 0, N*sizeof(int));

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
    
    return std::make_pair<int*, int*>(&*counters_p, &*counters_n);
}

template<typename T> 
std::pair<int*, int*> get_non_unique_borders(T * y_pred, long * y_pred_argsorted, size_t N){
    int left_p(0), right_p(0), j(0), s(0);

    int * y_pred_left = new int[N];
    int * y_pred_right = new int[N];

    memset((void*)y_pred_left, 0, N*sizeof(int));
    memset((void*)y_pred_right, 0, N*sizeof(int));

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
    
    std::make_pair<int*, int*>(&*y_pred_left, &*y_pred_right);
}

template<typename T>
std::tuple<int*, int*, int*, int*> get_labelscount_borders(int * y_true, T * y_pred, long * y_pred_argsorted, size_t N){
    std::pair<int*, int*> counters = get_non_unique_labels_count<T>(y_true, y_pred, y_pred_argsorted, N);
    std::pair<int*, int*> y_pred_sides = get_non_unique_borders<T>(y_pred, y_pred_argsorted, N);

    return std::make_tuple<int*, int*, int*, int*>(&*(counters.first), &*(counters.second), &*(y_pred_sides.first), &*(y_pred_sides.second));
}

template<typename T>
long * get_inverse_argsort(int * y_true, T * y_pred, long * y_pred_argsorted, size_t N){
    long * y_pred_ranks = new long[N];
    memset((void*)y_pred_ranks, 0, N*sizeof(long));

    for(size_t k = 0; k < N; k++){
        y_pred_ranks[y_pred_argsorted[N - k - 1]] = k;
    }
    return y_pred_ranks;
}
