import numpy as np
import pandas as pd

from tqdm import tqdm

from roc_auc_pairwise.sigmoid_pairwise_gpu import sigmoid_pairwise_loss_gpu_py as sigmoid_pairwise_loss_py
from roc_auc_pairwise.sigmoid_pairwise_gpu import sigmoid_pairwise_diff_hess_gpu_py as sigmoid_pairwise_diff_hess_py

EPS_LOSS = 5e-4
EPS_GRAD_HESS = 5e-4

def sigmoid(deltax_ij):
    return 1. / (1. + np.exp(-deltax_ij))

def log_loss(p_true, p_pred, reduction='sum'):
    eps = 1e-90
    tmp = p_true*np.log(p_pred + eps) + (1 - p_true)*np.log(1. - p_pred - eps)
    if reduction == 'sum':
        return np.sum(tmp)
    elif reduction == 'mean':
        return np.mean(tmp)
    
def test_sigmoid_loss_unique_8_parquet():
    """Test sigmoid loss for unique 8 len y_true"""
    df = pd.read_parquet('./tests_unique_8.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        sigm_loss = df.iloc[i].loss_sigm
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        
        sigm_loss_ = sigmoid_pairwise_loss_py(y_true, y_pred)
        
        assert(np.abs(sigm_loss_ - sigm_loss) < EPS_LOSS)

def test_sigmoid_loss_non_unique_8_parquet():
    """Test sigmoid loss for non unique 8 len y_true"""
    df = pd.read_parquet('./tests_non_unique_8.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        sigm_loss = df.iloc[i].loss_sigm
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        
        sigm_loss_ = sigmoid_pairwise_loss_py(y_true, y_pred)
        
        assert(np.abs(sigm_loss_ - sigm_loss) < EPS_LOSS)

      
def test_sigmoid_loss_unique_10_parquet():
    """Test sigmoid loss for unique 10 len y_true"""
    df = pd.read_parquet('./tests_unique_10.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        sigm_loss = df.iloc[i].loss_sigm
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        
        sigm_loss_ = sigmoid_pairwise_loss_py(y_true, y_pred)
        
        assert(np.abs(sigm_loss_ - sigm_loss) < EPS_LOSS)
        
def test_sigmoid_loss_non_unique_10_parquet():
    """Test sigmoid loss for non unique 10 len y_true"""
    df = pd.read_parquet('./tests_non_unique_10.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        sigm_loss = df.iloc[i].loss_sigm
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        
        sigm_loss_ = sigmoid_pairwise_loss_py(y_true, y_pred)
        
        assert(np.abs(sigm_loss_ - sigm_loss) < EPS_LOSS)
        
def test_sigmoid_loss_non_unique_100_parquet():
    """Test sigmoid loss for non unique 100 len y_true"""
    df = pd.read_parquet('./tests_non_unique_100.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        sigm_loss = df.iloc[i].loss_sigm / len(y_true)
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        
        sigm_loss_ = sigmoid_pairwise_loss_py(y_true, y_pred) / y_true.shape[0]
        
        assert(np.abs(sigm_loss_ - sigm_loss) < EPS_LOSS)
        
def test_sigmoid_loss_unique_100_parquet():
    """Test sigmoid loss for unique 100 len y_true"""
    df = pd.read_parquet('./tests_unique_100.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        sigm_loss = df.iloc[i].loss_sigm / len(y_true)
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        
        sigm_loss_ = sigmoid_pairwise_loss_py(y_true, y_pred) / y_true.shape[0]
        
        assert(np.abs(sigm_loss_ - sigm_loss) < EPS_LOSS)


def test_sigmoid_diff_hess_unique_8_parquet():
    """Test sigmoid diff hess for unique 8 len y_true"""
    df = pd.read_parquet('./tests_unique_8.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        grad_true = df.iloc[i].grad_sigm
        hess_true = df.iloc[i].hess_sigm
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        
        grad, hess = sigmoid_pairwise_diff_hess_py(y_true, y_pred)
        
        assert(np.mean(np.abs(grad - grad_true)) < EPS_GRAD_HESS)
        assert(np.mean(np.abs(hess - hess_true)) < EPS_GRAD_HESS)
        
        
def test_sigmoid_diff_hess_non_unique_8_parquet():
    """Test sigmoid diff hess for non unique 8 len y_true"""
    df = pd.read_parquet('./tests_non_unique_8.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        grad_true = df.iloc[i].grad_sigm
        hess_true = df.iloc[i].hess_sigm
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        
        grad, hess = sigmoid_pairwise_diff_hess_py(y_true, y_pred)
        
        assert(np.mean(np.abs(grad - grad_true)) < EPS_GRAD_HESS)
        assert(np.mean(np.abs(hess - hess_true)) < EPS_GRAD_HESS)
        
def test_sigmoid_diff_hess_unique_10_parquet():
    """Test sigmoid diff hess for unique 10 len y_true"""
    df = pd.read_parquet('./tests_non_unique_10.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        grad_true = df.iloc[i].grad_sigm
        hess_true = df.iloc[i].hess_sigm
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        
        grad, hess = sigmoid_pairwise_diff_hess_py(y_true, y_pred)
        
        assert(np.mean(np.abs(grad - grad_true)) < EPS_GRAD_HESS)
        assert(np.mean(np.abs(hess - hess_true)) < EPS_GRAD_HESS)
        
def test_sigmoid_diff_hess_non_unique_10_parquet():
    """Test sigmoid diff hess for non unique 10 len y_true"""
    df = pd.read_parquet('./tests_non_unique_10.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        grad_true = df.iloc[i].grad_sigm
        hess_true = df.iloc[i].hess_sigm
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        
        grad, hess = sigmoid_pairwise_diff_hess_py(y_true, y_pred)
        
        assert(np.mean(np.abs(grad - grad_true)) < EPS_GRAD_HESS)
        assert(np.mean(np.abs(hess - hess_true)) < EPS_GRAD_HESS)
        
def test_sigmoid_diff_hess_unique_100_parquet():
    """Test sigmoid diff hess for unique 100 len y_true"""
    df = pd.read_parquet('./tests_unique_100.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        grad_true = df.iloc[i].grad_sigm
        hess_true = df.iloc[i].hess_sigm
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        
        grad, hess = sigmoid_pairwise_diff_hess_py(y_true, y_pred)
        
        assert(np.mean(np.abs(grad - grad_true)) < EPS_GRAD_HESS)
        assert(np.mean(np.abs(hess - hess_true)) < EPS_GRAD_HESS)

def test_sigmoid_diff_hess_non_unique_100_parquet():
    """Test sigmoid diff hess for non unique 100 len y_true"""
    df = pd.read_parquet('./tests_non_unique_100.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        grad_true = df.iloc[i].grad_sigm
        hess_true = df.iloc[i].hess_sigm
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float32)
        
        grad, hess = sigmoid_pairwise_diff_hess_py(y_true, y_pred)
        
        assert(np.mean(np.abs(grad - grad_true)) < EPS_GRAD_HESS)
        assert(np.mean(np.abs(hess - hess_true)) < EPS_GRAD_HESS)