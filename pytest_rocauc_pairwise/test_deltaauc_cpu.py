from copy import deepcopy
import sys
sys.path.append('../')

import numpy as np
from sklearn.metrics import roc_auc_score

from rocauc_pairwise.deltaauc_cpu import deltaauc
from rocauc_pairwise.deltaauc_cpu import deltaauc_exact

from rocauc_pairwise.utils import get_inverse_argsort
from rocauc_pairwise.utils import get_labelscount_borders

def delta_auc_score(y_true, y_pred, i, j):
    auc_1 = roc_auc_score(y_true, y_pred)
    y_pred_ = deepcopy(y_pred)
    y_pred_[i], y_pred_[j] = y_pred_[j], y_pred_[i]
    auc_2 = roc_auc_score(y_true, y_pred_)
    return auc_1 - auc_2

def compute_deltaauc_true_auc(y_true, y_pred, i, j):
    auc_true = delta_auc_score(y_true, y_pred, i, j)
    
    y_pred_ranks = get_inverse_argsort(y_true, y_pred)
    n_ones = np.sum(y_true)
    n_zeroes = len(y_true) - np.sum(y_true)    
    
    auc_deltaauc = deltaauc(y_true, y_pred_ranks, n_ones, n_zeroes, i, j)
    
    return auc_true, auc_deltaauc

def compute_deltaauc_exact_true_auc(y_true, y_pred, i, j):
    auc_true = delta_auc_score(y_true, y_pred, i, j)
    
    y_pred_argsorted = np.argsort(y_pred)
    n_ones = np.sum(y_true)
    n_zeroes = len(y_true) - np.sum(y_true)
    
    counters_p, counters_n, y_pred_left, y_pred_right = get_labelscount_borders(y_true, y_pred, y_pred_argsorted)
    
    auc_deltaauc_exact = deltaauc_exact(y_true, y_pred,
                                        counters_n, counters_p,
                                        y_pred_left, y_pred_right,
                                        n_ones, n_zeroes, i, j)
    
    return auc_true, auc_deltaauc_exact
    

def test_deltaauc_1():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    auc_true, auc_deltaauc = compute_deltaauc_true_auc(y_true, y_pred, 4, 9)
    
    assert np.abs(auc_deltaauc - auc_true) < 1e-5
    
def test_deltaauc_2():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [.5, .4, .3, .2, .1, .05, .02, .01]
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    auc_true, auc_deltaauc = compute_deltaauc_true_auc(y_true, y_pred, 0, 6)
    
    assert np.abs(auc_deltaauc - auc_true) < 1e-5
    
    
def test_deltaauc_3():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [.5, .4, .3, .2, .1, .05, .02, .01]
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    auc_true, auc_deltaauc = compute_deltaauc_true_auc(y_true, y_pred, 0, 1)
    
    assert np.abs(auc_deltaauc - auc_true) < 1e-5
    
def test_deltaauc_4():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [.5, .4, .3, .2, .1, .05, .02, .01]
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    auc_true, auc_deltaauc = compute_deltaauc_true_auc(y_true, y_pred, 0, 7)
    
    assert np.abs(auc_deltaauc - auc_true) < 1e-5

def test_deltaauc_5():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [.5, .4, .3, .2, .1, .05, .02, .01]
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    auc_true, auc_deltaauc = compute_deltaauc_true_auc(y_true, y_pred, 1, 0)
    
    assert np.abs(auc_deltaauc - auc_true) < 1e-5        

def test_deltaauc_exact_1():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    auc_true, auc_deltaauc_exact = compute_deltaauc_exact_true_auc(y_true, y_pred, 0, 8)
    
    
    assert np.abs(auc_deltaauc_exact - auc_true) < 1e-5
    
def test_deltaauc_exact_2():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    auc_true, auc_deltaauc_exact = compute_deltaauc_exact_true_auc(y_true, y_pred, 8, 0)
    
    
    assert np.abs(auc_deltaauc_exact - auc_true) < 1e-5
    
def test_deltaauc_exact_3():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    auc_true, auc_deltaauc_exact = compute_deltaauc_exact_true_auc(y_true, y_pred, 0, 2)
    
    
    assert np.abs(auc_deltaauc_exact - auc_true) < 1e-5
    
def test_deltaauc_exact_4():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    auc_true, auc_deltaauc_exact = compute_deltaauc_exact_true_auc(y_true, y_pred, 0, 0)
    
    
    assert np.abs(auc_deltaauc_exact - auc_true) < 1e-5
    
def test_deltaauc_exact_5():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    auc_true, auc_deltaauc_exact = compute_deltaauc_exact_true_auc(y_true, y_pred, 0, 2)
    
    
    assert np.abs(auc_deltaauc_exact - auc_true) < 1e-5
    
def test_deltaauc_exact_6():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    auc_true, auc_deltaauc_exact = compute_deltaauc_exact_true_auc(y_true, y_pred, 3, 4)
    
    
    assert np.abs(auc_deltaauc_exact - auc_true) < 1e-5