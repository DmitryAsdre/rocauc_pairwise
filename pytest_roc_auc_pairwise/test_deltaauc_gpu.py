from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from roc_auc_pairwise.deltaauc_gpu import deltaauc_py
from roc_auc_pairwise.deltaauc_gpu import deltaauc_exact_py

from roc_auc_pairwise.utils import get_inverse_argsort_py
from roc_auc_pairwise.utils import get_non_unique_borders_py, get_non_unique_labels_count_py

def get_labelscount_borders(y_true, y_pred, y_pred_argsorted):
    
    y_pred_left, y_pred_right = get_non_unique_borders_py(y_pred, y_pred_argsorted)
    counters_p, counters_n = get_non_unique_labels_count_py(y_true, y_pred, y_pred_argsorted)
    
    return counters_p, counters_n, y_pred_left, y_pred_right

def delta_auc_score(y_true, y_pred, i, j):
    auc_1 = roc_auc_score(y_true, y_pred)
    
    y_pred_ = deepcopy(y_pred)
    y_pred_[i], y_pred_[j] = y_pred_[j], y_pred_[i]
    
    auc_2 = roc_auc_score(y_true, y_pred_)
    
    return auc_1 - auc_2

def compute_deltaauc_true_auc(y_true, y_pred, i, j):
    auc_true = delta_auc_score(y_true, y_pred, i, j)
    
    y_pred_argsorted = np.argsort(y_pred)
    
    y_pred_ranks = get_inverse_argsort_py(y_true, y_pred, y_pred_argsorted)
    n_ones = np.sum(y_true)
    n_zeroes = len(y_true) - np.sum(y_true)
    
    auc_deltaauc = deltaauc_py(y_true, y_pred_ranks.astype(np.int32), n_ones, n_zeroes, i, j)
    
    return auc_true, auc_deltaauc

def compute_deltaauc_exact_true_auc(y_true, y_pred, i, j):
    auc_true = delta_auc_score(y_true, y_pred, i, j)
    
    y_pred_argsorted = np.argsort(y_pred)
    n_ones = np.sum(y_true)
    n_zeroes = len(y_true) - n_ones

    counters_p, counters_n, y_pred_left, y_pred_right = get_labelscount_borders(y_true, y_pred, y_pred_argsorted)
    
    auc_deltaauc_exact = deltaauc_exact_py(y_true, y_pred, 
                                           counters_n, counters_p,
                                           y_pred_left, y_pred_right,
                                           n_ones, n_zeroes, i, j)
    
    return auc_true, auc_deltaauc_exact


def test_deltaauc_1():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    y_pred = np.array(y_pred).astype(np.float32)
    y_true = np.array(y_true).astype(np.int32)
    
    auc_true, auc_deltaauc = compute_deltaauc_true_auc(y_true, y_pred, 4, 9)
    
    assert np.abs(auc_deltaauc - auc_true) < 1e-5
    
def test_deltaauc_2():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [.5, .4, .3, .2, .1, .05, .02, .01]
    y_pred = np.array(y_pred).astype(np.float32)
    y_true = np.array(y_true).astype(np.int32)
    
    auc_true, auc_deltaauc = compute_deltaauc_true_auc(y_true, y_pred, 0, 6)
    
    assert np.abs(auc_deltaauc - auc_true) < 1e-5
    
    
def test_deltaauc_3():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [.5, .4, .3, .2, .1, .05, .02, .01]
    y_pred = np.array(y_pred).astype(np.float32)
    y_true = np.array(y_true).astype(np.int32)
    
    auc_true, auc_deltaauc = compute_deltaauc_true_auc(y_true, y_pred, 0, 1)
    
    assert np.abs(auc_deltaauc - auc_true) < 1e-5
    
def test_deltaauc_4():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [.5, .4, .3, .2, .1, .05, .02, .01]
    y_pred = np.array(y_pred).astype(np.float32)
    y_true = np.array(y_true).astype(np.int32)
    
    auc_true, auc_deltaauc = compute_deltaauc_true_auc(y_true, y_pred, 0, 7)
    
    assert np.abs(auc_deltaauc - auc_true) < 1e-5

def test_deltaauc_unique():
    y_true = np.array([i%2 for i in range(1, 9)], dtype=np.int32)
    y_pred = np.array([np.log(i) for i in range(1, 9)], dtype=np.float32)
    permutation = [0, 1, 2, 3, 5, 6, 7, 4]
    y_pred = y_pred[permutation]
    for i in range(8):
        for j in range(8):
            auc_true, auc_deltaauc = compute_deltaauc_true_auc(y_true, y_pred, i, j)
    
            assert np.abs(auc_deltaauc - auc_true) < 1e-5

def test_deltaauc_5():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [.5, .4, .3, .2, .1, .05, .02, .01]
    y_pred = np.array(y_pred).astype(np.float32)
    y_true = np.array(y_true).astype(np.int32)
    
    auc_true, auc_deltaauc = compute_deltaauc_true_auc(y_true, y_pred, 1, 0)
    
    assert np.abs(auc_deltaauc - auc_true) < 1e-5        

def test_deltaauc_exact_1():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    y_pred = np.array(y_pred).astype(np.float32)
    y_true = np.array(y_true).astype(np.int32)
    
    auc_true, auc_deltaauc_exact = compute_deltaauc_exact_true_auc(y_true, y_pred, 0, 8)
    
    
    assert np.abs(auc_deltaauc_exact - auc_true) < 1e-5
    
def test_deltaauc_exact_2():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    y_pred = np.array(y_pred).astype(np.float32)
    y_true = np.array(y_true).astype(np.int32)
    
    auc_true, auc_deltaauc_exact = compute_deltaauc_exact_true_auc(y_true, y_pred, 8, 0)
    
    
    assert np.abs(auc_deltaauc_exact - auc_true) < 1e-5
    
def test_deltaauc_exact_3():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    y_pred = np.array(y_pred).astype(np.float32)
    y_true = np.array(y_true).astype(np.int32)
    
    auc_true, auc_deltaauc_exact = compute_deltaauc_exact_true_auc(y_true, y_pred, 0, 2)
    
    
    assert np.abs(auc_deltaauc_exact - auc_true) < 1e-5
    
def test_deltaauc_exact_4():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    y_pred = np.array(y_pred).astype(np.float32)
    y_true = np.array(y_true).astype(np.int32)
    
    auc_true, auc_deltaauc_exact = compute_deltaauc_exact_true_auc(y_true, y_pred, 0, 0)
   
    
    assert np.abs(auc_deltaauc_exact - auc_true) < 1e-5
    
def test_deltaauc_exact_5():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    y_pred = np.array(y_pred).astype(np.float32)
    y_true = np.array(y_true).astype(np.int32)
    
    auc_true, auc_deltaauc_exact = compute_deltaauc_exact_true_auc(y_true, y_pred, 0, 2)
    
    
    assert np.abs(auc_deltaauc_exact - auc_true) < 1e-5
    
def test_deltaauc_exact_6():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    y_pred = np.array(y_pred).astype(np.float32)
    y_true = np.array(y_true).astype(np.int32)
    
    auc_true, auc_deltaauc_exact = compute_deltaauc_exact_true_auc(y_true, y_pred, 3, 4)
    
    
    assert np.abs(auc_deltaauc_exact - auc_true) < 1e-5
    
def test_deltaauc_equal_numbers():
    y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
    y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01]
    y_pred = np.array(y_pred).astype(np.float32)
    y_true = np.array(y_true).astype(np.int32)
    
    auc_true, auc_deltaauc = compute_deltaauc_true_auc(y_true, y_pred, 4, 4)
    
    assert np.abs(auc_deltaauc - auc_true) < 1e-5

 
#########################################################################################

#################################################################################################


def test_deltaauc_unique_8_parquet():
    df = pd.read_parquet('./tests_unique_8.parquet')
    
    for s in range(5):
        y_true, y_pred = df.iloc[s].y_true, df.iloc[s].y_pred
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        y_pred_argsorted = np.argsort(y_pred)
        y_pred_ranks = get_inverse_argsort_py(y_true, y_pred, y_pred_argsorted).astype(np.int32)
        n_ones = np.sum(y_true)
        n_zeroes = y_true.shape[0] - n_ones
        for i in range(8):
            for j in range(8):
                auc_true = delta_auc_score(y_true, y_pred, i, j)
                auc_deltaauc = deltaauc_py(y_true, y_pred_ranks, n_ones, n_zeroes, i, j)

                assert(np.abs(auc_true - auc_deltaauc) < 1e-5)
                
def test_deltaauc_unique_10_parquet():
    df = pd.read_parquet('./tests_unique_10.parquet')
    
    for s in range(5):
        y_true, y_pred = df.iloc[s].y_true, df.iloc[s].y_pred
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        y_pred_argsorted = np.argsort(y_pred)
        y_pred_ranks = get_inverse_argsort_py(y_true, y_pred, y_pred_argsorted).astype(np.int32)
        n_ones = np.sum(y_true)
        n_zeroes = y_true.shape[0] - n_ones
        for i in range(10):
            for j in range(10):
                auc_true = delta_auc_score(y_true, y_pred, i, j)
                auc_deltaauc = deltaauc_py(y_true, y_pred_ranks, n_ones, n_zeroes, i, j)

                assert(np.abs(auc_true - auc_deltaauc) < 1e-5)
                
                
def test_deltaauc_unique_100_parquet():
    df = pd.read_parquet('./tests_unique_100.parquet')
    
    for s in range(2):
        y_true, y_pred = df.iloc[s].y_true, df.iloc[s].y_pred
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        y_pred_argsorted = np.argsort(y_pred)
        y_pred_ranks = get_inverse_argsort_py(y_true, y_pred, y_pred_argsorted).astype(np.int32)
        n_ones = np.sum(y_true)
        n_zeroes = y_true.shape[0] - n_ones
        for i in range(10):
            for j in range(100):
                auc_true = delta_auc_score(y_true, y_pred, i, j)
                auc_deltaauc = deltaauc_py(y_true, y_pred_ranks, n_ones, n_zeroes, i, j)

                assert(np.abs(auc_true - auc_deltaauc) < 1e-5)
                
#######################################################################################################

def test_deltaauc_exact_unique_8_parquet():
    df = pd.read_parquet('./tests_unique_8.parquet')
    
    for s in range(5):
        y_true, y_pred = df.iloc[s].y_true, df.iloc[s].y_pred
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        y_pred_argsorted = np.argsort(y_pred)
        counters_p, counters_n, y_pred_left, y_pred_right = get_labelscount_borders(y_true, y_pred, y_pred_argsorted)
        n_ones = np.sum(y_true)
        n_zeroes = y_true.shape[0] - n_ones
        for i in range(8):
            for j in range(4):
                auc_true = delta_auc_score(y_true, y_pred, i, j)
                
                auc_deltaauc_exact = deltaauc_exact_py(y_true, y_pred, 
                                           counters_n, counters_p,
                                           y_pred_left, y_pred_right,
                                           n_ones, n_zeroes, i, j)

                assert(np.abs(auc_true - auc_deltaauc_exact) < 1e-5)
                
                
def test_deltaauc_exact_unique_10_parquet():
    df = pd.read_parquet('./tests_unique_10.parquet')
    
    for s in range(5):
        y_true, y_pred = df.iloc[s].y_true, df.iloc[s].y_pred
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        y_pred_argsorted = np.argsort(y_pred)
        counters_p, counters_n, y_pred_left, y_pred_right = get_labelscount_borders(y_true, y_pred, y_pred_argsorted)
        n_ones = np.sum(y_true)
        n_zeroes = y_true.shape[0] - n_ones
        for i in range(10):
            for j in range(5):
                auc_true = delta_auc_score(y_true, y_pred, i, j)
                
                auc_deltaauc_exact = deltaauc_exact_py(y_true, y_pred, 
                                           counters_n, counters_p,
                                           y_pred_left, y_pred_right,
                                           n_ones, n_zeroes, i, j)

                assert(np.abs(auc_true - auc_deltaauc_exact) < 1e-5)


def test_deltaauc_exact_non_unique_8_parquet():
    df = pd.read_parquet('./tests_non_unique_8.parquet')
    
    for s in range(5):
        y_true, y_pred = df.iloc[s].y_true, df.iloc[s].y_pred
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        y_pred_argsorted = np.argsort(y_pred)
        counters_p, counters_n, y_pred_left, y_pred_right = get_labelscount_borders(y_true, y_pred, y_pred_argsorted)
        n_ones = np.sum(y_true)
        n_zeroes = y_true.shape[0] - n_ones
        for i in range(8):
            for j in range(4):
                auc_true = delta_auc_score(y_true, y_pred, i, j)
                
                auc_deltaauc_exact = deltaauc_exact_py(y_true, y_pred, 
                                           counters_n, counters_p,
                                           y_pred_left, y_pred_right,
                                           n_ones, n_zeroes, i, j)

                assert(np.abs(auc_true - auc_deltaauc_exact) < 1e-5)
                
                
def test_deltaauc_exact_non_unique_10_parquet():
    df = pd.read_parquet('./tests_non_unique_10.parquet')
    
    for s in range(5):
        y_true, y_pred = df.iloc[s].y_true, df.iloc[s].y_pred
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        y_pred_argsorted = np.argsort(y_pred)
        counters_p, counters_n, y_pred_left, y_pred_right = get_labelscount_borders(y_true, y_pred, y_pred_argsorted)
        n_ones = np.sum(y_true)
        n_zeroes = y_true.shape[0] - n_ones
        for i in range(10):
            for j in range(5):
                auc_true = delta_auc_score(y_true, y_pred, i, j)
                
                auc_deltaauc_exact = deltaauc_exact_py(y_true, y_pred, 
                                           counters_n, counters_p,
                                           y_pred_left, y_pred_right,
                                           n_ones, n_zeroes, i, j)

                assert(np.abs(auc_true - auc_deltaauc_exact) < 1e-5)
                
                
def test_deltaauc_exact_non_unique_100_parquet():
    df = pd.read_parquet('./tests_non_unique_100.parquet')
    
    for s in range(1):
        y_true, y_pred = df.iloc[s].y_true, df.iloc[s].y_pred
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        y_pred_argsorted = np.argsort(y_pred)
        counters_p, counters_n, y_pred_left, y_pred_right = get_labelscount_borders(y_true, y_pred, y_pred_argsorted)
        n_ones = np.sum(y_true)
        n_zeroes = y_true.shape[0] - n_ones
        for i in range(10):
            for j in range(100):
                auc_true = delta_auc_score(y_true, y_pred, i, j)
                
                auc_deltaauc_exact = deltaauc_exact_py(y_true, y_pred, 
                                           counters_n, counters_p,
                                           y_pred_left, y_pred_right,
                                           n_ones, n_zeroes, i, j)

                assert(np.abs(auc_true - auc_deltaauc_exact) < 1e-5)