import json
import numpy as np

from rocauc_pairwise.sigmoid_pariwise_auc_gpu_wrappers import get_sigmoid_pairwise_wrapper_gpu


def test_sigmoid_pairwise_auc_half_zeros_10_percent_permutations_all_unique_8_wrapper():
    """Half zeros ten percent permutations for non exact auc computation, wrapper"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int32)
    y_pred = np.array([np.log(i) for i in range(1, 9)], dtype=np.float32)
    
    sigmoid_pairwise_auc_loss = get_sigmoid_pairwise_wrapper_gpu('sigmoid_pairwise_loss_auc')

    with open('grads_on_permutations_all_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        loss_auc_true = perm_grad[-1]
        loss_auc = sigmoid_pairwise_auc_loss(y_true, y_pred[permutation])
        if np.abs(loss_auc_true - loss_auc) > 1e-4 or np.isnan(loss_auc):
            raise Exception(f"true loss : {loss_auc_true}, loss_auc : {loss_auc}, {permutation}")
        
        
def test_sigmoid_pairwise_auc_exact_half_zeros_10_percent_permutations_non_unique_8_wrapper():
    """Half zeros ten percent permutations non unique for exact auc computation, wrapper"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int32)
    y_pred = np.array([np.log(i) for i in range(1, 9)], dtype=np.float32)

    with open('grads_on_permutations_all_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
        
    sigmoid_pairwise_loss_auc_exact_gpu = get_sigmoid_pairwise_wrapper_gpu('sigmoid_pairwise_loss_auc_exact')
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        loss_auc_true = perm_grad[-1]
        loss_auc = sigmoid_pairwise_loss_auc_exact_gpu(y_true, y_pred[permutation])
        if np.abs(loss_auc_true - loss_auc) > 1e-4 or np.isnan(loss_auc):
            raise Exception(f"true loss : {loss_auc_true}, loss_auc : {loss_auc}, {permutation}")

def test_sigmoid_pairwise_auc_exact_half_zeros_10_percent_permutations_all_unique_8_wrapper():
    """Half zeros ten percent permutations non unique for exact auc computation all unique, wrapper"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int32)
    y_pred = np.array([np.log(1 + i%3) for i in range(1, 9)], dtype=np.float32)

    with open('grads_on_permutations_non_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
        
    sigmoid_pairwise_loss_auc_exact_gpu = get_sigmoid_pairwise_wrapper_gpu('sigmoid_pairwise_loss_auc_exact')
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        loss_auc_true = perm_grad[-1]
        loss_auc = sigmoid_pairwise_loss_auc_exact_gpu(y_true, y_pred[permutation])
        if np.abs(loss_auc_true - loss_auc) > 1e-4 or np.isnan(loss_auc):
            raise Exception(f"true loss : {loss_auc_true}, loss_auc : {loss_auc}, {permutation}")
        
        
def test_sigmoid_pairwise_diff_hess_auc_half_zeros_10_percent_permutations_all_unique_8_wrapper():
    """Half zeros ten percent permutations for non exact auc computation gradient and hessian check"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int32)
    y_pred = np.array([np.log(i) for i in range(1, 9)], dtype=np.float32)

    with open('grads_on_permutations_all_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
        
    sigmoid_pairwise_grad_hess_auc = get_sigmoid_pairwise_wrapper_gpu('sigmoid_pairwise_grad_hess_auc')
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        grad_true, hess_true = np.array(perm_grad[1]), np.array(perm_grad[2])
        grad, hess = sigmoid_pairwise_grad_hess_auc(y_true, y_pred[permutation])
        if np.sum(np.abs(grad_true - grad)) > 1e-4 or np.isnan(grad).any():
            raise Exception(f"true grad : {grad_true}, grad : {grad}, permutation : {permutation}")
        if np.sum(np.abs(hess_true - hess)) > 1e-4 or np.isnan(hess).any():
            raise Exception(f"true hess : {hess_true}, grad : {hess}, permutation : {permutation}")
        

def sigmoid_pairwise_auc_diff_hess_exact_half_zeros_10_percent_permutations_non_unique_8_wrapper():
    """Half zeros ten percent permutations non unique for exact auc computation gradient and hessian check"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int32)
    y_pred = np.array([np.log(1 + i%3) for i in range(1, 9)], dtype=np.float32)

    with open('grads_on_permutations_non_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
        
    sigmoid_pairwise_grad_hess_auc_exact = get_sigmoid_pairwise_wrapper_gpu('sigmoid_pairwise_grad_hess_auc_exact')
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        grad_true, hess_true = np.array(perm_grad[1]), np.array(perm_grad[2])
        grad, hess = sigmoid_pairwise_grad_hess_auc_exact(y_true, y_pred[permutation])
        if np.sum(np.abs(grad_true - grad)) > 1e-4 or np.isnan(grad).any():
            raise Exception(f"true grad : {grad_true}, grad : {grad}, permutation : {permutation}")
        if np.sum(np.abs(hess_true - hess)) > 1e-4 or np.isnan(hess).any():
            raise Exception(f"true hess : {hess_true}, grad : {hess}, permutation : {permutation}")
        

def sigmoid_pairwise_auc_diff_hess_exact_half_zeros_10_percent_permutations_all_unique_8_wrapper():
    """Half zeros ten percent permutations all unique for exact auc computation gradient and hessian check"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int32)
    y_pred = np.array([np.log(i) for i in range(1, 9)], dtype=np.float32)

    with open('grads_on_permutations_all_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
    
    sigmoid_pairwise_grad_hess_auc_exact = get_sigmoid_pairwise_wrapper_gpu('sigmoid_pairwise_grad_hess_auc_exact')
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        grad_true, hess_true = np.array(perm_grad[1]), np.array(perm_grad[2])
        grad, hess = sigmoid_pairwise_grad_hess_auc_exact(y_true, y_pred[permutation])
        grad, hess = hess, grad
        if np.sum(np.abs(grad_true - grad)) > 1e-4 or np.isnan(grad).any():
            raise Exception(f"true grad : {grad_true}, grad : {grad}, permutation : {permutation}")
        if np.sum(np.abs(hess_true - hess)) > 1e-4 or np.isnan(hess).any():
            raise Exception(f"true hess : {hess_true}, grad : {hess}, permutation : {permutation}")