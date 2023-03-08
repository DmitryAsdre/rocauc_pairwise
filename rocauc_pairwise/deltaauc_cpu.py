import numpy as np

from numba import jit

@jit(nopython=True)
def deltaauc_exact(y_true,
                   y_pred,
                   counters_p,
                   counters_n,
                   y_pred_left,
                   y_pred_right,
                   n_ones,
                   n_zeroes,
                   i, j):
    ypredi = y_pred[i]
    ypredj = y_pred[j]
    
    if ypredi < ypredj:
        i, j = j, i

    ypredi = y_pred[i]
    ypredj = y_pred[j]
    
    li = y_true[i]
    lj = y_true[j]


    deltaji = lj - li
    
    deltai = 0.5 * counters_p[i]*counters_n[i] - 0.5*(counters_p[i] + deltaji) * (counters_n[i] - deltaji)
    deltaj = 0.5 * counters_p[j]*counters_n[j] - 0.5*(counters_p[j] - deltaji) * (counters_n[j] + deltaji)
    
    if deltaji == -1:
        delta_eq = counters_p[i] + counters_n[j] - 2
    else:
        delta_eq = -(counters_p[i] + counters_n[j])
    
    multiplicate = 1
    if deltaji == 0:
        multiplicate *= 0
    if ypredi == ypredj:
        multiplicate *= 0
    
    return multiplicate * (delta_eq + deltai + deltaj - deltaji * abs((y_pred_right[i] - y_pred_left[j]))) / (n_ones * n_zeroes)
           
@jit(nopython=True)
def deltaauc(y_true,
             y_pred_ranks,
             n_ones,
             n_zeroes,
             i, j):
    i_, j_ = y_pred_ranks[i], y_pred_ranks[j]
    deltaauc_ =  (y_true[i] - y_true[j]) * (i_ - j_) / (n_ones * n_zeroes)
    return deltaauc_
