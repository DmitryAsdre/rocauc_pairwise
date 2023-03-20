import json
from functools import partial
import numpy as np

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

from rocauc_pairwise.utils import get_inverse_argsort
from rocauc_pairwise.utils import get_labelscount_borders
from rocauc_pairwise.sigmoid_pairwise_auc_gpu import get_gpu_kernel


def sigmoid_pairwise_loss_gpu_wrapper(y_true, 
                                      y_pred,
                                      gpu_kernel,
                                      n_blocks = 32):
    size = y_true.shape[0]
    
    bdim = (n_blocks, 1, 1)
    dx, mx = divmod(size, bdim[0])
    gdim = ((dx + (mx>0)), 1)
        
    
    loss_auc = np.zeros(1).astype(np.float32)
    y_true = y_true.astype(np.int32)
    exp_pred = np.exp(y_pred).astype(np.float32)
    
    gpu_kernel(np.int32(size), 
               drv.Out(loss_auc),
               drv.In(y_true),
               drv.In(exp_pred),
               block=bdim, grid=gdim)
    
    return loss_auc[0]

def sigmoid_pairwise_grad_hess_gpu_wrapper(y_true,
                                           y_pred,
                                           gpu_kernel,
                                           n_blocks=32):
    
    size = y_true.shape[0]
    
    bdim = (n_blocks, 1, 1)
    dx, mx = divmod(size, bdim[0])
    gdim = ((dx + (mx>0)), 1)
    
    grad = np.zeros(y_true.shape[0], dtype=np.float32)
    hess = np.zeros(y_true.shape[0], dtype=np.float32)
    
    exp_pred = np.exp(y_pred).astype(np.float32)
    
    gpu_kernel(np.int32(size), 
               drv.Out(grad),
               drv.Out(hess),
               drv.In(y_true),
               drv.In(exp_pred),
               block=bdim, grid=gdim)

def sigmoid_pairwise_loss_auc_gpu_wrapper(y_true,
                                          y_pred,
                                          gpu_kernel,
                                          n_blocks=32):
    
    size = y_true.shape[0]
    n_ones = np.sum(y_true)
    n_zeros = size - n_ones
    
    bdim = (n_blocks, 1, 1)
    dx, mx = divmod(size, bdim[0])
    gdim = ((dx + (mx>0)), 1)
    
    loss_auc = np.zeros(1).astype(np.float32)
    y_true = y_true.astype(np.int32)
    exp_pred = np.exp(y_pred).astype(np.float32)
    
    inversed_argsort = get_inverse_argsort(y_true, exp_pred).astype(np.int32)
    
    gpu_kernel(np.int32(size),
               drv.Out(loss_auc),
               drv.In(y_true),
               drv.In(exp_pred),
               drv.In(inversed_argsort),
               np.int32(n_ones),
               np.int32(n_zeros),
               block=bdim, grid=gdim)
    
    return loss_auc[0]

def sigmoid_pairwise_grad_hess_auc_gpu_wrapper(y_true,
                                               y_pred,
                                               gpu_kernel,
                                               n_blocks=32):
    
    size = y_true.shape[0]
    n_ones = np.sum(y_true)
    n_zeros = size - n_ones
    
    bdim = (n_blocks, 1, 1)
    dx, mx = divmod(size, bdim[0])
    gdim = ((dx + (mx>0)), 1)
    
    y_true = y_true.astype(np.int32)
    exp_pred = np.exp(y_pred).astype(np.float32)
    
    grad = np.zeros(y_true.shape[0], dtype=np.float32)
    hess = np.zeros(y_true.shape[0], dtype=np.float32)
    
    inversed_argsort = get_inverse_argsort(y_true, exp_pred).astype(np.int32)
    
    gpu_kernel(np.int32(size),
              drv.Out(grad),
              drv.Out(hess),
              drv.In(y_true),
              drv.In(exp_pred),
              drv.In(inversed_argsort),
              np.int32(n_ones),
              np.int32(n_zeros),
              block=bdim, grid=gdim)
    
    return grad, hess


def sigmoid_pairwise_loss_auc_exact_gpu_wrapper(y_true,
                                                y_pred,
                                                gpu_kernel,
                                                n_blocks=32):
    
    size = y_true.shape[0]
    n_ones = np.sum(y_true)
    n_zeros = size - n_ones
    
    bdim = (n_blocks, 1, 1)
    dx, mx = divmod(size, bdim[0])
    gdim = ((dx + (mx>0)), 1)
    
    loss_auc = np.zeros(1).astype(np.float32)
    y_true = y_true.astype(np.int32)
    exp_pred = np.exp(y_pred).astype(np.float32)
    exp_pred_argsorted = np.argsort(exp_pred)
    
    counters_n, counters_p, y_pred_left, y_pred_right = get_labelscount_borders(y_true, 
                                                                                exp_pred, 
                                                                                exp_pred_argsorted)
    
    gpu_kernel(np.int32(size),
               drv.Out(loss_auc),
               drv.In(y_true),
               drv.In(exp_pred),
               drv.In(counters_p),
               drv.In(counters_n),
               drv.In(y_pred_left),
               drv.In(y_pred_right),
               np.int32(n_ones),
               np.int32(n_zeros),
               block=bdim, grid=gdim)
    
    return loss_auc[0]

def sigmoid_pairwise_grad_hess_auc_exact_gpu_wrapper(y_true,
                                                     y_pred,
                                                     gpu_kernel,
                                                     n_blocks=32):
    
    size = y_true.shape[0]
    n_ones = np.sum(y_true)
    n_zeros = size - n_ones
    
    bdim = (n_blocks, 1, 1)
    dx, mx = divmod(size, bdim[0])
    gdim = ((dx + (mx>0)), 1)
    
    y_true = y_true.astype(np.int32)
    exp_pred = np.exp(y_pred).astype(np.float32)
    exp_pred_argsorted = np.argsort(exp_pred)
    
    grad = np.zeros(y_true.shape[0], dtype=np.float32)
    hess = np.zeros(y_true.shape[0], dtype=np.float32)
    
    counters_n, counters_p, y_pred_left, y_pred_right = get_labelscount_borders(y_true, 
                                                                                exp_pred, 
                                                                                exp_pred_argsorted)
    
    gpu_kernel(np.int32(size),
               drv.Out(grad),
               drv.Out(hess),
               drv.In(y_true),
               drv.In(exp_pred),
               drv.In(counters_p),
               drv.In(counters_n),
               drv.In(y_pred_left),
               drv.In(y_pred_right),
               np.int32(n_ones),
               np.int32(n_zeros),
               block=bdim, grid=gdim)
    
    return grad, hess

def get_sigmoid_pairwise_wrapper_gpu(kernel_name,
                                     n_blocks=32):
    gpu_kernel = get_gpu_kernel(kernel_name)
    
    gpu_wrapper = None
    
    if kernel_name == 'sigmoid_pairwise_loss':
        gpu_wrapper = sigmoid_pairwise_loss_gpu_wrapper
    elif kernel_name == 'sigmoid_pairwise_grad_hess':
        gpu_wrapper = sigmoid_pairwise_grad_hess_gpu_wrapper
    elif kernel_name == 'sigmoid_pairwise_loss_auc':
        gpu_wrapper = sigmoid_pairwise_loss_auc_gpu_wrapper
    elif kernel_name == 'sigmoid_pairwise_grad_hess_auc':
        gpu_wrapper = sigmoid_pairwise_grad_hess_auc_gpu_wrapper
    elif kernel_name == 'sigmoid_pairwise_loss_auc_exact':
        gpu_wrapper = sigmoid_pairwise_loss_auc_exact_gpu_wrapper
    elif kernel_name == 'sigmoid_pairwise_grad_hess_auc_exact':
        gpu_wrapper = sigmoid_pairwise_grad_hess_auc_exact_gpu_wrapper
    else:
        raise Exception(f'Unknown kernel name : kernel_name - {kernel_name}')
    
    return partial(gpu_wrapper, gpu_kernel=gpu_kernel, n_blocks=n_blocks)