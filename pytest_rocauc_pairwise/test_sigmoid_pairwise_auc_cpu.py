import numpy as np
import json

from rocauc_pairwise.sigmoid_pairwise_auc_cpu import sigmoid_pairwise_loss_auc_cpu
from rocauc_pairwise.sigmoid_pairwise_auc_cpu import sigmoid_pairwise_loss_auc_exact_cpu
from rocauc_pairwise.sigmoid_pairwise_auc_cpu import sigmoid_pairwise_diff_hess_auc_cpu
from rocauc_pairwise.sigmoid_pairwise_auc_cpu import sigmoid_pairwise_diff_hess_auc_exact_cpu

from itertools import permutations


def test_sigmoid_pairwise_auc_all_zeros_but_one():
    """Test all ones but one in one thread" and multithreading"""
    y_true = np.array([1, 0, 0, 0], dtype=np.int64)
    y_pred = np.array([np.log(3), np.log(1), np.log(1), np.log(1)], dtype=np.float64)
    exp_pred = np.exp(y_pred)

    loss_auc_true = -0.5753641449035618
    
    losses_auc = []
    for i in range(1, 11):
        loss_auc = sigmoid_pairwise_loss_auc_cpu(y_true, exp_pred, i)
        losses_auc.append(loss_auc)
    for i in range(10):
        assert np.abs(losses_auc[i] - loss_auc_true) < 1e-5

def test_sigmoid_pairwise_auc_all_zeros_but_one_32():
    """Test float32 and int32 support"""
    y_true = np.array([1, 0, 0, 0], dtype=np.int32)
    y_pred = np.array([np.log(3), np.log(1), np.log(1), np.log(1)], dtype=np.float32)
    exp_pred = np.exp(y_pred)

    loss_auc_true = -0.5753641449035618

    loss_auc = sigmoid_pairwise_loss_auc_cpu(y_true, exp_pred, 1)

    assert np.abs(loss_auc - loss_auc_true) < 1e-5

def test_sigmoid_pairwise_auc_half_zeros_10_percent_permutations_all_unique_8():
    """Half zeros ten percent permutations for non exact auc computation"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int64)
    y_pred = np.array([np.log(i) for i in range(1, 9)], dtype=np.float64)
    exp_pred = np.exp(y_pred)

    with open('grads_on_permutations_all_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        loss_auc_true = perm_grad[-1]
        loss_auc = sigmoid_pairwise_loss_auc_cpu(y_true, exp_pred[permutation], 1)
        if np.abs(loss_auc_true - loss_auc) > 1e-4:
            raise Exception(f"true loss : {loss_auc_true}, loss_auc : {loss_auc}, {permutation}")
        
def test_sigmoid_pairwise_auc_exact_half_zeros_10_percent_permutations_non_unique_8():
    """Half zeros ten percent permutations non unique for exact auc computation"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int64)
    y_pred = np.array([np.log(i) for i in range(1, 9)], dtype=np.float64)
    exp_pred = np.exp(y_pred)

    with open('grads_on_permutations_all_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        loss_auc_true = perm_grad[-1]
        loss_auc = sigmoid_pairwise_loss_auc_exact_cpu(y_true, exp_pred[permutation], 1)
        if np.abs(loss_auc_true - loss_auc) > 1e-4:
            raise Exception(f"true loss : {loss_auc_true}, loss_auc : {loss_auc}, {permutation}")

def test_sigmoid_pairwise_auc_exact_half_zeros_10_percent_permutations_all_unique_8():
    """Half zeros ten percent permutations non unique for exact auc computation all unique"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int64)
    y_pred = np.array([np.log(1 + i%3) for i in range(1, 9)], dtype=np.float64)
    exp_pred = np.exp(y_pred)

    with open('grads_on_permutations_non_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        loss_auc_true = perm_grad[-1]
        loss_auc = sigmoid_pairwise_loss_auc_exact_cpu(y_true, exp_pred[permutation], 1)
        if np.abs(loss_auc_true - loss_auc) > 1e-4:
            raise Exception(f"true loss : {loss_auc_true}, loss_auc : {loss_auc}, {permutation}")
        
def test_sigmoid_pairwise_auc_exact_all_zeros_but_one_32():
    """Test float32 and int32 support"""
    y_true = np.array([1, 0, 0, 0], dtype=np.int32)
    y_pred = np.array([np.log(3), np.log(1), np.log(1), np.log(1)], dtype=np.float32)
    exp_pred = np.exp(y_pred)

    loss_auc_true = -0.5753641449035618

    loss_auc = sigmoid_pairwise_loss_auc_exact_cpu(y_true, exp_pred, 1)

    assert np.abs(loss_auc - loss_auc_true) < 1e-5


def test_sigmoid_pairwise_diff_hess_auc_half_zeros_10_percent_permutations_all_unique_8():
    """Half zeros ten percent permutations for non exact auc computation gradient and hessian check"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int64)
    y_pred = np.array([np.log(i) for i in range(1, 9)], dtype=np.float64)
    exp_pred = np.exp(y_pred)

    with open('grads_on_permutations_all_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        grad_true, hess_true = np.array(perm_grad[1]), np.array(perm_grad[2])
        grad, hess = sigmoid_pairwise_diff_hess_auc_cpu(y_true, exp_pred[permutation], 1)
        if np.sum(np.abs(grad_true - grad)) > 1e-4:
            raise Exception(f"true grad : {grad_true}, grad : {grad}, permutation : {permutation}")
        if np.sum(np.abs(hess_true - hess)) > 1e-4:
            raise Exception(f"true hess : {hess_true}, grad : {hess}, permutation : {permutation}")
        

def test_sigmoid_pairwise_auc_diff_hess_exact_half_zeros_10_percent_permutations_non_unique_8():
    """Half zeros ten percent permutations non unique for exact auc computation gradient and hessian check"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int64)
    y_pred = np.array([np.log(1 + i%3) for i in range(1, 9)], dtype=np.float64)
    exp_pred = np.exp(y_pred)

    with open('grads_on_permutations_non_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        grad_true, hess_true = np.array(perm_grad[1]), np.array(perm_grad[2])
        grad, hess = sigmoid_pairwise_diff_hess_auc_exact_cpu(y_true, exp_pred[permutation], 1)
        if np.sum(np.abs(grad_true - grad)) > 1e-4:
            raise Exception(f"true grad : {grad_true}, grad : {grad}, permutation : {permutation}")
        if np.sum(np.abs(hess_true - hess)) > 1e-4:
            raise Exception(f"true hess : {hess_true}, grad : {hess}, permutation : {permutation}")
        

def test_sigmoid_pairwise_auc_diff_hess_exact_half_zeros_10_percent_permutations_all_unique_8():
    """Half zeros ten percent permutations all unique for exact auc computation gradient and hessian check"""
    y_true = np.array([i % 2 for i in range(1, 9)], dtype=np.int64)
    y_pred = np.array([np.log(i) for i in range(1, 9)], dtype=np.float64)
    exp_pred = np.exp(y_pred)

    with open('grads_on_permutations_all_unique_8_10_percent_half_zeros.json', 'r') as r:
        grads_on_permutations = json.load(r)
    
    for perm_grad in grads_on_permutations:
        permutation = perm_grad[0]
        grad_true, hess_true = np.array(perm_grad[1]), np.array(perm_grad[2])
        grad, hess = sigmoid_pairwise_diff_hess_auc_exact_cpu(y_true, exp_pred[permutation], 1)
        if np.sum(np.abs(grad_true - grad)) > 1e-4:
            raise Exception(f"true grad : {grad_true}, grad : {grad}, permutation : {permutation}")
        if np.sum(np.abs(hess_true - hess)) > 1e-4:
            raise Exception(f"true hess : {hess_true}, grad : {hess}, permutation : {permutation}")





