import sys
sys.path.append('../')

import os
import numpy as np

os.environ["C_INCLUDE_PATH"] = np.get_include()

import pyximport
pyximport.install()

import numpy as np

from rocauc_pairwise.utils import get_non_unique_labels_count
from rocauc_pairwise.utils import get_non_unique_borders
from rocauc_pairwise.utils import get_labelscount_borders
from rocauc_pairwise.utils import get_inverse_argsort

y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0.5, 0.5, 0.5, 0.1, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02]
y_pred = np.array(y_pred)
y_true = np.array(y_true)

def test_get_inverse_argsort():
    inversed_argsort = get_inverse_argsort(y_true, y_pred)
    sorted_array = []
    for i in inversed_argsort:
        sorted_array.append(y_pred[i])
    sorted_array_numpy = np.sort(y_pred)[::-1]
    
    assert (np.array(sorted_array) == sorted_array_numpy).all()
    

def test_get_non_unique_borders():
    y_pred_argsorted = np.argsort(y_pred)#[::-1].copy()
    y_pred_left, y_pred_right = get_non_unique_borders(y_pred, y_pred_argsorted)
    
    y_pred_left_true = [0, 0, 0, 3, 4, 5, 5, 5, 5, 5]
    y_pred_right_true = [2, 2, 2, 3, 4, 9, 9, 9, 9, 9]
    
    assert (np.array(y_pred_left_true) == y_pred_left).all()
    assert (np.array(y_pred_right_true) == y_pred_right).all()
    
def test_get_non_unique_labels_count():
    y_pred_argsorted = np.argsort(y_pred)
    counters_p, counters_n = get_non_unique_labels_count(y_true, y_pred, y_pred_argsorted)
    
    counters_p_true = [1, 1, 1, 1, 0, 3, 3, 3, 3, 3]
    counters_n_true = [2, 2, 2, 0, 1, 2, 2, 2, 2, 2]
    
    assert (np.array(counters_p_true) == counters_p).all()
    assert (np.array(counters_n_true) == counters_n).all()
    
def test_get_labelscount_borders():
    y_pred_argsorted = np.argsort(y_pred)#[::-1].copy()
    
    counters_p, counters_n, y_pred_left, y_pred_right = get_labelscount_borders(y_true, y_pred, y_pred_argsorted)
    
    y_pred_left_true = [0, 0, 0, 3, 4, 5, 5, 5, 5, 5]
    y_pred_right_true = [2, 2, 2, 3, 4, 9, 9, 9, 9, 9]
    counters_p_true = [1, 1, 1, 1, 0, 3, 3, 3, 3, 3]
    counters_n_true = [2, 2, 2, 0, 1, 2, 2, 2, 2, 2]
    
    assert (np.array(y_pred_left_true) == y_pred_left).all()
    assert (np.array(y_pred_right_true) == y_pred_right).all()
    assert (np.array(counters_p_true) == counters_p).all()
    assert (np.array(counters_n_true) == counters_n).all()
    
    