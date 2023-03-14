import numpy as np

from rocauc_pairwise.sigmoid_pairwise_cpu import sigmoid_pairwise_loss
from rocauc_pairwise.sigmoid_pairwise_cpu import sigmoid_pairwise_diff_hess

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

    log_loss_ = sigmoid_pairwise_loss(y_true, np.exp(y_pred), 1)
    assert np.abs(log_loss_true - log_loss_) < 1e-5

def test_sigmoid_pairwise_loss_4x4_ranking():
    """Sigmoid pairwise loss for ranks"""
    y_true = np.array([1, 0, 2, 0])
    y_pred = np.array([3., 4., -1., 0.])

    P_hat_ij = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 1.0, 0.0, 0.5, 1.0])
    deltax_ij = np.array([0., 0., 0., 0., -1., 5., -1, 4., 4., 3.])

    P_ij = sigmoid(deltax_ij=deltax_ij)
    log_loss_true = log_loss(P_hat_ij, P_ij, reduction='sum')

    log_loss_ = sigmoid_pairwise_loss(y_true, np.exp(y_pred), 1)

    assert np.abs(log_loss_true - log_loss_) < 1e-5


def test_sigmoid_pairwise_all_zeros():
    """All zeros sigmoid pairwise loss"""

    y_true = np.array([0, 0, 0, 0, 0, 0])
    y_pred = np.array([0., 0., 0., 0., 0., 0.])

    log_loss_true = -21. * np.log(2)


    log_loss_ = sigmoid_pairwise_loss(y_true, np.exp(y_pred), 1)

    assert np.abs(log_loss_true - log_loss_) < 1e-5


def test_sigmoid_pairwise_all_zeros_big():
    """All zeros sigmoid pairwise loss"""

    y_true = np.array([0 for i in range(1000)])
    y_pred = np.array([10. for i in range(1000)])

    log_loss_true = -1001. * 500. * np.log(2)


    log_loss_ = sigmoid_pairwise_loss(y_true, np.exp(y_pred), 6)

    assert np.abs(log_loss_true - log_loss_) < 1e-5

def test_sigmoid_pairwise_all_ones_big():
    """All ones sigmoid pairwise loss"""

    y_true = np.array([1 for i in range(1000)])
    y_pred = np.array([-1. for i in range(1000)])

    log_loss_true = -1001. * 500. * np.log(2)


    log_loss_ = sigmoid_pairwise_loss(y_true, np.exp(y_pred), 6)

    assert np.abs(log_loss_true - log_loss_) < 1e-5