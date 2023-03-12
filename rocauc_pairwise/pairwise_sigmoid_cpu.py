import numpy as np

from numba import jit


def sigmoid_pairwise(exp_pred, y_true, y_pre)