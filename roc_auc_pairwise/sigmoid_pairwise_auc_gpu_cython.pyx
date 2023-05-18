cimport numpy as np
np.import_array()
import numpy as np
cimport cython
from libc.stdint cimport int32_t
from libcpp.utility cimport pair


cdef extern from '../src/cpu/utils.cpp':
    pair[int*, int*] get_non_unique_labels_count[T_true, T_pred, T_argsorted](T_true* y_true, T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N)
    pair[int*, int*] get_non_unique_borders[T_pred, T_argsorted](T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N)
    T_out* get_inverse_argsort[T_out, T_true, T_pred, T_argsorted](T_true* y_true, T_pred* y_pred, T_argsorted* y_pred_argsorted, size_t N)

cdef extern from '../src/cuda/sigmoid_pairwise_auc.cuh':
    float sigmoid_pairwise_loss_auc(int32_t* y_true, float* exp_pred,
                                    long* y_pred_argsorted, 
                                    size_t n_ones, size_t n_zeroes, size_t N)
    pair[float*, float*] sigmoid_pairwise_grad_hess_auc(int32_t* y_true, float* exp_pred,
                                                        long* y_pred_argsorted,
                                                        size_t n_ones, size_t n_zeroes, size_t N)
    float sigmoid_pairwise_loss_auc_exact(int32_t* y_true, float* exp_pred,
                                          long* y_pred_argsorted, float eps,
                                          size_t n_ones, size_t n_zeroes, size_t N)
    pair[float*, float*] sigmoid_pairwise_grad_hess_auc_exact(int32_t* y_true, float* exp_pred, 
                                                              long* y_pred_argsorted, float eps,
                                                              size_t n_ones, size_t n_zeroes, size_t N)

@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_pairwise_loss_auc_gpu_py(np.ndarray[np.int32_t, ndim=1, mode='c'] y_true,
                                     np.ndarray[np.float32_t, ndim=1, mode='c'] y_pred):
    cdef:
        size_t n_ones = np.sum(y_true)
        size_t N = y_true.shape[0]
        size_t n_zeroes = N - n_ones

        np.ndarray[np.float32_t, ndim=1, mode='c'] exp_pred = np.exp(y_pred)
        np.ndarray[np.int64_t, ndim=1, mode='c'] y_pred_argsorted = np.argsort(y_pred).copy()

        float loss = 0

    loss = sigmoid_pairwise_loss_auc(&y_true[0], &exp_pred[0], <long*>&y_pred_argsorted[0], n_ones, n_zeroes, N)

    return loss


@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_pairwise_diff_hess_auc_gpu_py(np.ndarray[np.int32_t, ndim=1, mode='c'] y_true,
                                          np.ndarray[np.float32_t, ndim=1, mode='c'] y_pred):
    cdef:
        size_t n_ones = np.sum(y_true)
        size_t N = y_true.shape[0]
        size_t n_zeroes = N - n_ones
    
        np.ndarray[np.float32_t, ndim=1, mode='c'] exp_pred = np.exp(y_pred)
        np.ndarray[np.int64_t, ndim=1, mode='c'] y_pred_argsorted = np.argsort(y_pred).copy()

        pair[float*, float*] grad_hess
    
    grad_hess = sigmoid_pairwise_grad_hess_auc(&y_true[0], &exp_pred[0],
                                               <long*>&y_pred_argsorted[0], n_ones, n_zeroes, N)

    cdef:
        np.npy_intp dims = y_true.shape[0]
        np.ndarray[np.float32_t, ndim=1, mode='c'] grad = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT32, <void*>grad_hess.first)
        np.ndarray[np.float32_t, ndim=1, mode='c'] hess = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT32, <void*>grad_hess.second)

    return grad, hess


@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_pairwise_loss_auc_exact_gpu_py(np.ndarray[np.int32_t, ndim=1, mode='c'] y_true,
                                           np.ndarray[np.float32_t, ndim=1, mode='c'] y_pred,
                                           float eps = 0.):
    cdef:
        size_t n_ones = np.sum(y_true)
        size_t N = y_true.shape[0]
        size_t n_zeroes = N - n_ones

        np.ndarray[np.float32_t, ndim=1, mode='c'] exp_pred = np.exp(y_pred)
        np.ndarray[np.int64_t, ndim=1, mode='c'] y_pred_argsorted = np.argsort(y_pred).copy()

        float loss = 0
    
    loss = sigmoid_pairwise_loss_auc_exact(&y_true[0], &exp_pred[0], <long*>&y_pred_argsorted[0], eps, n_ones, n_zeroes, N)

    return loss

@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_pairwise_diff_hess_auc_exact_gpu_py(np.ndarray[np.int32_t, ndim=1, mode='c'] y_true,
                                                np.ndarray[np.float32_t, ndim=1, mode='c'] y_pred,
                                                float eps = 0.):
    cdef:
        size_t n_ones = np.sum(y_true)
        size_t N = y_true.shape[0]
        size_t n_zeroes = N - n_ones

        np.ndarray[np.float32_t, ndim=1, mode='c'] exp_pred = np.exp(y_pred)
        np.ndarray[np.int64_t, ndim=1, mode='c'] y_pred_argsorted = np.argsort(y_pred).copy()

        pair[float*, float*] grad_hess
    
    grad_hess = sigmoid_pairwise_grad_hess_auc_exact(&y_true[0], &exp_pred[0], <long*>&y_pred_argsorted[0], eps, n_ones, n_zeroes, N)

    cdef:
        np.npy_intp dims = y_true.shape[0]
        np.ndarray[np.float32_t, ndim=1, mode='c'] grad = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT32, <void*>grad_hess.first)
        np.ndarray[np.float32_t, ndim=1, mode='c'] hess = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT32, <void*>grad_hess.second)

    return grad, hess