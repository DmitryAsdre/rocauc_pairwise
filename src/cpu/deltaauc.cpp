#include <cstddef>
#include <cmath>
#include <utility>
#include <cstring>
#include <tuple>
#include "deltaauc.hpp"
#include "constants.hpp"

template<class T_true, class T_ranks>
double deltaauc(T_true* y_true, T_ranks* y_pred_ranks,
                size_t n_ones, size_t n_zeroes,
                size_t i, size_t j){
    double deltaauc_ = 0.d; 
    double ranki = y_pred_ranks[i];
    double rankj = y_pred_ranks[j];

    deltaauc_ = (1.d*(y_true[i] - y_true[j]) * (rankj - ranki) / (n_ones * n_zeroes));
    return deltaauc_;
}

template<class T_true, class T_pred>
double deltaauc_exact(T_true* y_true, T_pred* y_pred,
                      int32_t* counters_p, int32_t* counters_n,
                      int32_t* y_pred_left, int32_t* y_pred_right,
                      size_t n_ones, size_t n_zeroes, size_t i, size_t j){
    double ypredi = y_pred[i];
    double ypredj = y_pred[j];

    if(ypredi < ypredj){
        size_t tmpi = i;
        i = j;
        j = tmpi;
    }

    ypredi = y_pred[i];
    ypredj = y_pred[j];

    double deltaji = y_true[j] - y_true[i];

    double deltai =  0.5d*counters_p[i]*counters_n[i] - 0.5d*(counters_p[i] + deltaji) * (counters_n[i] - deltaji);
    double deltaj =  0.5d*counters_p[j]*counters_n[j] - 0.5d*(counters_p[j] - deltaji) * (counters_n[j] + deltaji);

    double delta_eq = 0.d;
    double multiplicate = 1.d;

    if(fabs(deltaji + 1.d) < EPS)
        delta_eq = counters_p[i] + counters_n[j] - 2.d;
    else
        delta_eq = -(counters_p[i] + counters_n[j]);
    
    if(fabs(deltaji) < EPS || fabs(ypredi - ypredj) < EPS)
        multiplicate *= 0;

    return multiplicate * (delta_eq + deltai + deltaj - deltaji * fabs(y_pred_right[i] - y_pred_left[j])) / (n_ones * n_zeroes);


}