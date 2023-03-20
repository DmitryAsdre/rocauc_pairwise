import json
import numpy as np
import sys
from copy import deepcopy


sys.path.append('../')

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
from sklearn.metrics import roc_auc_score


from rocauc_pairwise.utils import get_inverse_argsort
from rocauc_pairwise.utils import get_labelscount_borders
from rocauc_pairwise.sigmoid_pairwise_auc_gpu import get_gpu_kernel


def test_sigmoid_pairwise_loss_auc_gpu_grad_on_permutation_all_unique_10_percent():
    """Half zeros ten percent permutations for non exact auc computation"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int32)
    y_pred = np.array([np.log(i) for i in range(1, 9)], dtype=np.float32)
    exp_pred = np.exp(y_pred)

    with open('grads_on_permutations_all_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
        
    sigmoid_pairwise_loss_auc_gpu = get_gpu_kernel(kernel_name='sigmoid_pairwise_loss_auc')
    
    size = y_true.shape[0]
    n_ones = np.sum(y_true)
    n_zeros = size - n_ones
    
    bdim = (32, 1, 1)
    dx, mx = divmod(size, bdim[0])
    gdim = ( (dx + (mx>0)), 1)
    
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        loss_auc_true = perm_grad[-1]
        exp_pred_permuted = exp_pred[permutation].copy()
        inversed_argsort = get_inverse_argsort(y_true, exp_pred_permuted).astype(np.int32)
        loss_auc = np.zeros(1).astype(np.float32)
        sigmoid_pairwise_loss_auc_gpu(np.int32(size), 
                                      drv.Out(loss_auc),
                                      drv.In(y_true),
                                      drv.In(exp_pred_permuted), 
                                      drv.In(inversed_argsort),
                                      np.int32(n_ones),
                                      np.int32(n_zeros),
                                      block=bdim, grid=gdim)
        
        
        if np.abs(loss_auc_true - loss_auc[0]) > 1e-4 or np.isnan(loss_auc).any():
            raise Exception(f"true loss : {loss_auc_true}, loss_auc : {loss_auc}, {permutation}")
    
def test_sigmoid_pairwise_diff_hess_auc_half_zeros_10_percent_permutations_all_unique_8():
    """Half zeros ten percent permutations for non exact auc computation gradient and hessian check"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int32)
    y_pred = np.array([np.log(i) for i in range(1, 9)], dtype=np.float32)
    exp_pred = np.exp(y_pred)

    with open('grads_on_permutations_all_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
        
    sigmoid_pairwise_grad_hess_auc_gpu = get_gpu_kernel('sigmoid_pairwise_grad_hess_auc')
    
    size = y_true.shape[0]
    n_ones = np.sum(y_true)
    n_zeros = size - n_ones
    
    bdim = (32, 1, 1)
    dx, mx = divmod(size, bdim[0])
    gdim = ( (dx + (mx>0)), 1)
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        grad_true, hess_true = np.array(perm_grad[1]), np.array(perm_grad[2])
        
        exp_pred_permuted = exp_pred[permutation].copy()
        inversed_argsort = get_inverse_argsort(y_true, exp_pred_permuted).astype(np.int32)
        
        grad = np.zeros(grad_true.shape[0]).astype(np.float32)
        hess = np.zeros(grad_true.shape[0]).astype(np.float32)
        
        sigmoid_pairwise_grad_hess_auc_gpu(np.int32(size),
                                        drv.Out(grad),
                                        drv.Out(hess),
                                        drv.In(y_true),
                                        drv.In(exp_pred_permuted),
                                        drv.In(inversed_argsort),
                                        np.int32(n_ones),
                                        np.int32(n_zeros),
                                        block=bdim, grid=gdim)
        
        
        if np.sum(np.abs(grad_true - grad)) > 1e-4 or np.isnan(grad).any():
            raise Exception(f"true grad : {grad_true}, grad : {grad}, permutation : {permutation}")
        if np.sum(np.abs(hess_true - hess)) > 1e-4 or np.isnan(hess).any():
            raise Exception(f"true hess : {hess_true}, grad : {hess}, permutation : {permutation}")
        
def test_sigmoid_pairwise_loss_auc_exact_gpu_grad_on_permutation_all_unique_10_percent():
    """Half zeros ten percent permutations for exact auc computation"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int32)
    y_pred = np.array([np.log(i) for i in range(1, 9)], dtype=np.float32)
    exp_pred = np.exp(y_pred)

    with open('grads_on_permutations_all_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
        
    sigmoid_pairwise_loss_auc_exact_gpu = get_gpu_kernel(kernel_name='sigmoid_pairwise_loss_auc_exact')
    
    size = y_true.shape[0]
    n_ones = np.sum(y_true)
    n_zeros = size - n_ones
    
    bdim = (32, 1, 1)
    dx, mx = divmod(size, bdim[0])
    gdim = ( (dx + (mx>0)), 1)
    
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        loss_auc_true = perm_grad[-1]
        exp_pred_permuted = exp_pred[permutation].copy()
        exp_pred_argsorted = np.argsort(exp_pred_permuted)
        loss_auc = np.zeros(1).astype(np.float32)
        counters_n, counters_p, y_pred_left, y_pred_right = get_labelscount_borders(y_true, 
                                                                                    exp_pred_permuted, 
                                                                                    exp_pred_argsorted)
        
        
        sigmoid_pairwise_loss_auc_exact_gpu(np.int32(size),
                                            drv.Out(loss_auc),
                                            drv.In(y_true),
                                            drv.In(exp_pred_permuted),
                                            drv.In(counters_n),
                                            drv.In(counters_p),
                                            drv.In(y_pred_left),
                                            drv.In(y_pred_right),
                                            np.int32(n_ones),
                                            np.int32(n_zeros),
                                            block=bdim, grid=gdim)

        if np.abs(loss_auc_true - loss_auc[0]) > 1e-4 or np.isnan(loss_auc).any():
            raise Exception(f"true loss : {loss_auc_true}, loss_auc : {loss_auc}, {permutation}")
        
        
def test_sigmoid_pairwise_diff_hess_auc_exact_half_zeros_10_percent_permutations_all_unique_8():
    """Half zeros ten percent permutations for exact auc computation gradient and hessian check"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int32)
    y_pred = np.array([np.log(i) for i in range(1, 9)], dtype=np.float32)
    exp_pred = np.exp(y_pred)

    with open('grads_on_permutations_all_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
        
    sigmoid_pairwise_grad_hess_auc_gpu = get_gpu_kernel('sigmoid_pairwise_grad_hess_auc_exact')
    
    size = y_true.shape[0]
    n_ones = np.sum(y_true)
    n_zeros = size - n_ones
    
    bdim = (32, 1, 1)
    dx, mx = divmod(size, bdim[0])
    gdim = ( (dx + (mx>0)), 1)
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        grad_true, hess_true = np.array(perm_grad[1]), np.array(perm_grad[2])
    
        exp_pred_permuted = exp_pred[permutation].copy().astype(np.float32)
        exp_pred_argsorted = np.argsort(exp_pred_permuted)
        
        counters_n, counters_p, y_pred_left, y_pred_right = get_labelscount_borders(y_true, 
                                                                                    y_pred[permutation], 
                                                                                    exp_pred_argsorted)
        
        counters_n = counters_n.astype(np.int32)
        counters_p = counters_p.astype(np.int32)
        y_pred_left = y_pred_left.astype(np.int32)
        y_pred_right = y_pred_right.astype(np.int32)
        
        grad = np.zeros(grad_true.shape[0]).astype(np.float32)
        hess = np.zeros(grad_true.shape[0]).astype(np.float32)
    
        
        
        sigmoid_pairwise_grad_hess_auc_gpu(np.int32(size),
                                        drv.Out(grad),
                                        drv.Out(hess),
                                        drv.In(y_true),
                                        drv.In(exp_pred_permuted),
                                        drv.In(counters_n),
                                        drv.In(counters_p),
                                        drv.In(y_pred_left),
                                        drv.In(y_pred_right),
                                        np.int32(n_ones),
                                        np.int32(n_zeros),
                                        block=bdim, grid=gdim)
        
        
        if np.sum(np.abs(grad_true - grad)) > 1e-4 or np.isnan(grad).any():
            raise Exception(f"true grad : {grad_true[:3]}, grad : {grad[:3]}, permutation : {permutation}")
        if np.sum(np.abs(hess_true - hess)) > 1e-4 or np.isnan(hess).any():
            raise Exception(f"true hess : {hess_true}, grad : {hess}, permutation : {permutation}")
        
def test_sigmoid_pairwise_loss_auc_exact_gpu_grad_on_permutation_non_unique_10_percent():
    """Half zeros ten percent permutations for exact auc computation"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int32)
    y_pred = np.array([np.log(i) for i in range(1, 9)], dtype=np.float32)
    exp_pred = np.exp(y_pred)

    with open('grads_on_permutations_all_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
        
    sigmoid_pairwise_loss_auc_exact_gpu = get_gpu_kernel(kernel_name='sigmoid_pairwise_loss_auc_exact')

    size = y_true.shape[0]
    n_ones = np.sum(y_true)
    n_zeros = size - n_ones

    bdim = (32, 1, 1)
    dx, mx = divmod(size, bdim[0])
    gdim = ((dx + (mx>0)), 1)


    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        loss_auc_true = perm_grad[-1]
        exp_pred_permuted = exp_pred[permutation].copy()
        exp_pred_argsorted = np.argsort(exp_pred_permuted)
        loss_auc = np.zeros(1).astype(np.float32)
        counters_n, counters_p, y_pred_left, y_pred_right = get_labelscount_borders(y_true, 
                                                                                    exp_pred_permuted, 
                                                                                    exp_pred_argsorted)
        
        sigmoid_pairwise_loss_auc_exact_gpu(np.int32(size),
                                            drv.Out(loss_auc),
                                            drv.In(y_true),
                                            drv.In(exp_pred_permuted),
                                            drv.In(counters_n),
                                            drv.In(counters_p),
                                            drv.In(y_pred_left),
                                            drv.In(y_pred_right),
                                            np.int32(n_ones),
                                            np.int32(n_zeros),
                                            block=bdim, grid=gdim)


        if np.abs(loss_auc_true - loss_auc[0]) > 1e-4 or np.isnan(loss_auc).any():
            raise Exception(f"true loss : {loss_auc_true}, loss_auc : {loss_auc}, {permutation}")
        
        
def test_sigmoid_pairwise_diff_hess_auc_exact_half_zeros_10_percent_permutations_non_unique_8():
    """Half zeros ten percent permutations for exact auc computation gradient and hessian check"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int32)
    y_pred = np.array([np.log(1 + i%3) for i in range(1, 9)], dtype=np.float32)
    exp_pred = np.exp(y_pred)

    with open('grads_on_permutations_non_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
        
    sigmoid_pairwise_grad_hess_auc_gpu = get_gpu_kernel('sigmoid_pairwise_grad_hess_auc_exact')
    
    size = y_true.shape[0]
    n_ones = np.sum(y_true)
    n_zeros = size - n_ones
    
    bdim = (32, 1, 1)
    dx, mx = divmod(size, bdim[0])
    gdim = ( (dx + (mx>0)), 1)
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        grad_true, hess_true = np.array(perm_grad[1]), np.array(perm_grad[2])
    
        exp_pred_permuted = exp_pred[permutation].copy()
        exp_pred_argsorted = np.argsort(exp_pred_permuted)
        
        counters_n, counters_p, y_pred_left, y_pred_right = get_labelscount_borders(y_true, 
                                                                                    y_pred[permutation], 
                                                                                    exp_pred_argsorted)
        
        grad = np.zeros(grad_true.shape[0]).astype(np.float32)
        hess = np.zeros(grad_true.shape[0]).astype(np.float32)
        
        sigmoid_pairwise_grad_hess_auc_gpu(np.int32(size),
                                        drv.Out(grad),
                                        drv.Out(hess),
                                        drv.In(y_true),
                                        drv.In(exp_pred_permuted),
                                        drv.In(counters_n),
                                        drv.In(counters_p),
                                        drv.In(y_pred_left),
                                        drv.In(y_pred_right),
                                        np.int32(n_ones),
                                        np.int32(n_zeros),
                                        block=bdim, grid=gdim)
        if np.sum(np.abs(grad_true - grad)) > 1e-4 or np.isnan(grad).any():
           raise Exception(f"true grad : {grad_true[:3]}, grad : {grad[:3]}, permutation : {permutation}")
        if np.sum(np.abs(hess_true - hess)) > 1e-4 or np.isnan(hess).any():
            raise Exception(f"true hess : {hess_true[:3]}, grad : {hess[:3]}, permutation : {permutation}")
        
        
        
###############################################################################################

def delta_auc_score(y_true, y_pred, i, j):
    auc_1 = roc_auc_score(y_true, y_pred)
    y_pred_ = deepcopy(y_pred)
    y_pred_[i], y_pred_[j] = y_pred_[j], y_pred_[i]
    auc_2 = roc_auc_score(y_true, y_pred_)
    return auc_1 - auc_2


def compute_deltaauc_exact_true_auc(deltaauc_gpu_kernel, y_true, y_pred, i, j):
    auc_true = delta_auc_score(y_true, y_pred, i, j)
    
    y_pred_argsorted = np.argsort(y_pred)
    n_ones = np.sum(y_true)
    n_zeroes = len(y_true) - np.sum(y_true)
    
    counters_p, counters_n, y_pred_left, y_pred_right = get_labelscount_borders(y_true, y_pred, y_pred_argsorted)

    auc_deltaauc_exact = np.zeros(1).astype(np.float32)
    
    size = y_true.shape[0]
    
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.float32)
    
    bdim = (32, 1, 1)
    dx, mx = divmod(size, bdim[0])
    gdim = ( (dx + (mx>0)), 1)
    
    
    deltaauc_gpu_kernel(drv.Out(auc_deltaauc_exact),
                        drv.In(y_true), 
                        drv.In(y_pred),
                        drv.In(counters_n), 
                        drv.In(counters_p),
                        drv.In(y_pred_left), 
                        drv.In(y_pred_right),
                        np.int32(n_ones), 
                        np.int32(n_zeroes),
                        np.int32(i), 
                        np.int32(j),
                        block=bdim, grid=gdim)
    
    return auc_true, auc_deltaauc_exact[0]

def test_deltaauc_exact_1():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    deltaauc_exact_gpu = get_gpu_kernel('deltaauc_exact_wrapper')
    
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[0]):
            auc_true, auc_deltaauc_exact_gpu = compute_deltaauc_exact_true_auc(deltaauc_exact_gpu, y_true, y_pred, i, j)
            assert np.abs(auc_deltaauc_exact_gpu- auc_true) < 1e-5