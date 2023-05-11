import numpy as np

from roc_auc_pairwise.sigmoid_pairwise_cpu import sigmoid_pairwise_loss_py
from roc_auc_pairwise.sigmoid_pairwise_cpu import sigmoid_pairwise_diff_hess_py

def sigmoid(deltax_ij):
    return 1. / (1. + np.exp(-deltax_ij))

def log_loss(p_true, p_pred, reduction='sum'):
    eps = 1e-20
    tmp = p_true*np.log(p_pred + eps) + (1 - p_true)*np.log(1. - p_pred - eps)
    if reduction == 'sum':
        return np.sum(tmp)
    elif reduction == 'mean':
        return np.mean(tmp)
    
def test_sigmoid_pairwise_loss_4x4_binary():
    """Simple binary classification
       pairwise sigmoid loss"""
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([3., 4., -1., 0.])

    P_hat_ij  = np.array([0.5, 0.5, 0.5, 0.5, 1., 0., 1., 0.5, 0.5, 1.0])
    deltax_ij = np.array([0., 0., 0., 0., -1., 5., -1, 4., 4., 3.])

    P_ij = sigmoid(deltax_ij=deltax_ij)
    log_loss_true = log_loss(P_hat_ij, P_ij, reduction='sum')

    log_loss_ = sigmoid_pairwise_loss_py(y_true, y_pred)
    assert np.abs(log_loss_true - log_loss_) < 1e-5
    
def test_sigmoid_pairwise_loss_4x4_ranking():
    """Sigmoid pairwise loss for ranks"""
    y_true = np.array([1, 0, 2, 0])
    y_pred = np.array([3., 4., -1., 0.])

    P_hat_ij = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 1.0, 0.0, 0.5, 1.0])
    deltax_ij = np.array([0., 0., 0., 0., -1., 5., -1, 4., 4., 3.])

    P_ij = sigmoid(deltax_ij=deltax_ij)
    log_loss_true = log_loss(P_hat_ij, P_ij, reduction='sum')

    log_loss_ = sigmoid_pairwise_loss_py(y_true, y_pred)

    assert np.abs(log_loss_true - log_loss_) < 1e-5
    
def test_sigmoid_pairwise_all_zeros():
    """All zeros sigmoid pairwise loss"""

    y_true = np.array([0, 0, 0, 0, 0, 0])
    y_pred = np.array([0., 0., 0., 0., 0., 0.])

    log_loss_true = -21. * np.log(2)


    log_loss_ = sigmoid_pairwise_loss_py(y_true, y_pred)

    assert np.abs(log_loss_true - log_loss_) < 1e-5


def test_sigmoid_pairwise_all_zeros_big():
    """All zeros sigmoid pairwise loss"""

    y_true = np.array([0 for i in range(1000)])
    y_pred = np.array([10. for i in range(1000)])

    log_loss_true = -1001. * 500. * np.log(2)


    log_loss_ = sigmoid_pairwise_loss_py(y_true, y_pred)

    assert np.abs(log_loss_true - log_loss_) < 1e-5

def test_sigmoid_pairwise_all_ones_big():
    """All ones sigmoid pairwise loss"""

    y_true = np.array([1 for i in range(1000)])
    y_pred = np.array([-1. for i in range(1000)])

    log_loss_true = -1001. * 500. * np.log(2)


    log_loss_ = sigmoid_pairwise_loss_py(y_true, y_pred)

    assert np.abs(log_loss_true - log_loss_) < 1e-5
    
def test_sigmoid_pairwise_diff_hess_all_threads():
    """Test hess and grad function in all threads"""

    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([np.log(1), np.log(2), np.log(3), np.log(4)])

    grad_true = [1.7166666666666668, -0.9, 0.7214285714285713, -1.538095238095238]
    hess_true = [-1.0697222222222222, -1.1844444444444444, -1.1723979591836735, -1.1271201814058955]

    grad_true = np.array(grad_true)
    hess_true = np.array(hess_true)

    grad, hess = sigmoid_pairwise_diff_hess_py(y_true, y_pred)

    assert np.sum(np.abs(grad_true - grad)) < 1e-5
    assert np.sum(np.abs(hess_true - hess)) < 1e-5
    
    
def test_sigmoid_pairwise_diff_hess_all_zeros_except_one_all_threads():
    """Test hess and grad with all zeros in y_true except one"""

    y_true = np.array([1, 0, 0,  0])
    y_pred = np.array([np.log(3), np.log(1), np.log(1), np.log(1)])

    grad_true = [0.75, -0.25, -0.25, -0.25]
    hess_true = [-1.0625, -1.1875, -1.1875, -1.1875]

    grad_true = np.array(grad_true)
    hess_true = np.array(hess_true)

    grad, hess = sigmoid_pairwise_diff_hess_py(y_true, y_pred)

    assert np.sum(np.abs(grad_true - grad)) < 1e-5
    assert np.sum(np.abs(hess_true - hess)) < 1e-5
    
def test_sigmoid_pairwise_diff_hess_all_zeroes_big_chunk_one_thread():
    """Test all zeroes with all zeroes in y_pred in one thread"""

    y_true = np.array([0 for i in range(1000)])
    y_pred = np.array([0. for i in range(1000)])

    grad_true = [0. for i in range(1000)]
    hess_true = [-250.25 for i in range(1000)]

    grad, hess = sigmoid_pairwise_diff_hess_py(y_true, y_pred)

    assert np.sum(np.abs(grad_true - grad)) < 1e-5
    assert np.sum(np.abs(hess_true - hess)) < 1e-5