import os

import numpy as np

import pycuda.driver as drv
import pycuda.autoinit
from  pycuda.compiler import SourceModule

from .utils import get_inverse_argsort 
from .utils import get_labelscount_borders

def get_sigmoid_pairwise_loss_gpu_kernel():
    pkg_path = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(pkg_path, 'cuda', 'sigmoid_pairwise.cu')
    
    with open(cu_path, 'r') as r:
        src_cu = r.read()
    
    mod_cu = SourceModule(src_cu)
    sigmoid_pairwise_loss_gpu = mod_cu.get_function("sigmoid_pairwise_loss_gpu")
    return sigmoid_pairwise_loss_gpu

def get_sigmoid_pairwise_grad_hess_gpu_kernel():
    pkg_path = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(pkg_path, 'cuda', 'sigmoid_pairwise.cu')
    
    with open(cu_path, 'r') as r:
        src_cu = r.read()
    
    mod_cu = SourceModule(src_cu)
    sigmoid_pairwise_grad_hess_gpu = mod_cu.get_function("sigmoid_pairwise_grad_hess_gpu")
    return sigmoid_pairwise_grad_hess_gpu

def get_sigmoid_pairwise_loss_auc_gpu_kernel():
    pkg_path = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(pkg_path, 'cuda', 'sigmoid_pairwise_auc.cu')
    
    with open(cu_path, 'r') as r:
        src_cu = r.read()
    
    mod_cu = SourceModule(src_cu)
    sigmoid_pairwise_loss_auc_gpu = mod_cu.get_function("sigmoid_pairwise_loss_auc_gpu")
    return sigmoid_pairwise_loss_auc_gpu

def get_sigmoid_pairwise_grad_hess_auc_gpu_kernel():
    pkg_path = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(pkg_path, 'cuda', 'sigmoid_pairwise_auc.cu')
    
    with open(cu_path, 'r') as r:
        src_cu = r.read()
    
    mod_cu = SourceModule(src_cu)
    sigmoid_pairwise_grad_hess_auc_gpu = mod_cu.get_function("sigmoid_pairwise_grad_hess_auc_gpu")
    return sigmoid_pairwise_grad_hess_auc_gpu

def get_sigmoid_pairwise_loss_auc_exact_gpu_kernel():
    pkg_path = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(pkg_path, 'cuda', 'sigmoid_pairwise_auc_exact.cu')
    
    with open(cu_path, 'r') as r:
        src_cu = r.read()
    
    mod_cu = SourceModule(src_cu)
    sigmoid_pairwise_loss_auc_exact_gpu = mod_cu.get_function("sigmoid_pairwise_loss_auc_exact_gpu")
    return sigmoid_pairwise_loss_auc_exact_gpu

def get_sigmoid_pairwise_grad_hess_auc_exact_gpu_kernel():
    pkg_path = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(pkg_path, 'cuda', 'sigmoid_pairwise_auc_exact.cu')
    
    with open(cu_path, 'r') as r:
        src_cu = r.read()
    
    mod_cu = SourceModule(src_cu)
    sigmoid_pairwise_grad_hess_auc_exact_gpu = mod_cu.get_function("sigmoid_pairwise_grad_hess_auc_exact_gpu")
    return sigmoid_pairwise_grad_hess_auc_exact_gpu

def get_sigmoid_pairwise_deltaauc_exact_wrapper_gpu_kernel():
    pkg_path = os.path.dirname(os.path.abspath(__file__))
    cu_path = os.path.join(pkg_path, 'cuda', 'sigmoid_pairwise_auc_exact.cu')
    
    with open(cu_path, 'r') as r:
        src_cu = r.read()
    
    mod_cu = SourceModule(src_cu)
    deltaauc_exact_wrapper_gpu = mod_cu.get_function("deltaauc_exact_wrapper")
    return deltaauc_exact_wrapper_gpu
 
def get_gpu_kernel(kernel_name):
    if kernel_name == 'sigmoid_pairwise_loss':
        return get_sigmoid_pairwise_loss_gpu_kernel()
    elif kernel_name == 'sigmoid_pairwise_grad_hess':
        return get_sigmoid_pairwise_grad_hess_gpu_kernel()
    elif kernel_name == 'sigmoid_pairwise_loss_auc':
        return get_sigmoid_pairwise_loss_auc_gpu_kernel()
    elif kernel_name == 'sigmoid_pairwise_grad_hess_auc':
        return get_sigmoid_pairwise_grad_hess_auc_gpu_kernel()
    elif kernel_name == 'sigmoid_pairwise_loss_auc_exact':
        return get_sigmoid_pairwise_loss_auc_exact_gpu_kernel()
    elif kernel_name == 'sigmoid_pairwise_grad_hess_auc_exact':
        return get_sigmoid_pairwise_grad_hess_auc_exact_gpu_kernel()
    elif kernel_name == 'deltaauc_exact_wrapper':
        return get_sigmoid_pairwise_deltaauc_exact_wrapper_gpu_kernel()
    else:
        raise Exception(f'Unknown kernel name : kernel_name - {kernel_name}')
