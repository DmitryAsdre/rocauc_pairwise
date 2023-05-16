cimport numpy as np
np.import_array()
import numpy as np
cimport cython
from libc.stdint cimport int32_t
from libcpp.utility cimport pair


cdef extern from '../src/cuda/sigmoid_pairwise.cuh':
    float sigmoid_pairwise_loss(int32_t* y_true, float* exp_pred, size_t N)
    pair[float*, float*] sigmoid_pairwise_grad_hess(int32_t* y_true, float* exp_pred, size_t N)


@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_pairwise_loss_gpu_py(np.ndarray[np.int32_t, ndim=1, mode='c'] y_true,
                             np.ndarray[np.float32_t, ndim=1, mode='c'] y_pred):
    cdef:
        np.ndarray[np.float32_t, ndim=1, mode='c'] exp_pred = np.exp(y_pred)
        float sigmoid_loss = 0
    
    sigmoid_loss = sigmoid_pairwise_loss(&y_true[0], &exp_pred[0], y_true.shape[0])

    return sigmoid_loss

@cython.boundscheck(False)
@cython.wraparound(False)
def sigmoid_pairwise_diff_hess_gpu_py(np.ndarray[np.int32_t, ndim=1, mode='c'] y_true,
                                  np.ndarray[np.float32_t, ndim=1, mode='c'] y_pred):
    cdef:
        np.ndarray[np.float32_t, ndim=1, mode='c'] exp_pred = np.exp(y_pred)
        pair[float*, float*] grad_hess
    
    grad_hess = sigmoid_pairwise_grad_hess(&y_true[0], &exp_pred[0], y_true.shape[0])

    cdef:
        np.npy_intp dims = y_true.shape[0]
        np.ndarray[np.float32_t, ndim=1, mode='c'] grad = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT32, <void*>grad_hess.first)
        np.ndarray[np.float32_t, ndim=1, mode='c'] hess = np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT32, <void*>grad_hess.second)

    return grad, hess