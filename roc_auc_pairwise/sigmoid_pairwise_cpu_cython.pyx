cimport numpy as np
np.import_array()
import numpy as np
cimport cython
from libcpp.utility cimport pair
from libc.stdint cimport int32_t

ctypedef fused int_t:
    np.int64_t
    np.int32_t

ctypedef fused _int_t:
    np.int64_t
    np.int32_t

ctypedef fused float_t:
    np.float32_t
    np.float64_t

cdef extern from '../src/cpu/sigmoid_pairwise.cpp':
    double sigmoid_pairwise_loss[T_true, T_pred](T_true* y_true, T_pred* exp_pred, size_t N)
    pair[double*, double*] sigmoid_pairwise_diff_hess[T_true, T_pred](T_true* y_true, T_pred* exp_pred, size_t N)


@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_pairwise_loss_py(np.ndarray[int_t, ndim=1, mode='c'] y_true,
                             np.ndarray[float_t, ndim=1, mode='c'] y_pred):
    cdef:
        double sigmoid_loss = 0
        np.ndarray[float_t, ndim=1, mode='c'] exp_pred = np.exp(y_pred)

    sigmoid_loss = sigmoid_pairwise_loss(&y_true[0], &exp_pred[0], y_true.shape[0])

    return sigmoid_loss

@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_pairwise_diff_hess_py(np.ndarray[int_t, ndim=1, mode='c'] y_true,
                                  np.ndarray[float_t, ndim=1, mode='c'] y_pred):
    cdef:
        pair[double*, double*] grad_hess
        np.ndarray[float_t, ndim=1, mode='c'] exp_pred = np.exp(y_pred)

    grad_hess = sigmoid_pairwise_diff_hess(&y_true[0], &exp_pred[0], y_true.shape[0])

    cdef:
        np.npy_intp dims = y_true.shape[0]
        np.ndarray[np.float64_t, ndim=1, mode='c'] grad = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT64, <void*>grad_hess.first)
        np.ndarray[np.float64_t, ndim=1, mode='c'] hess = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT64, <void*>grad_hess.second)

    return grad, hess
