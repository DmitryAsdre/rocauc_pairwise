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


def test_sigmoid_pairwise_loss_4x4():
    y_true = np.array([1, 0, 2, 3])
    y_pred = np.array([3., 4., -1., 0.])
    P_hat_ij = []
    deltax_ij = []

    for i in range(4):
        for j in range(i + 1):
            if y_true[i] == y_true[j]:
                P_hat_ij.append(0.5)
            else:
                P_hat_ij.append(float(y_true[i] > y_true[j]))
            
            deltax_ij.append(y_pred[i] - y_pred[j])
    deltax_ij = np.array(deltax_ij)
    P_hat_ij = np.array(P_hat_ij)

    #raise Exception(P_hat_ij, deltax_ij)

    #P_hat_ij  = np.array([0.5, 0.5, 0.5, 0.5, 0., 1., 0., 0.5, 0.5, 0.])
    #deltax_ij = np.array([0., 0., 0., 0., 1., -5., 1., -4., -4., -3.])

    P_ij = sigmoid(deltax_ij=deltax_ij)

    log_loss_true = log_loss(P_hat_ij, P_ij, reduction='sum')

    log_loss_ = sigmoid_pairwise_loss(y_true, np.exp(y_pred), 1)

    #raise Exception(log_loss_true, log_loss_)

    #assert np.abs(log_loss_true - log_loss_) < 1e-5