import pandas as pd
import numpy as np

from tqdm import tqdm


from roc_auc_pairwise.sigmoid_pairwise_auc_cpu import sigmoid_pairwise_loss_auc_gpu_py
from roc_auc_pairwise.sigmoid_pairwise_auc_cpu import sigmoid_pairwise_loss_auc_exact_gpu_py
from roc_auc_pairwise.sigmoid_pairwise_auc_cpu import sigmoid_pairwise_diff_hess_auc_gpu_py
from roc_auc_pairwise.sigmoid_pairwise_auc_cpu import sigmoid_pairwise_diff_hess_auc_exact_gpu_py

def test_sigmoid_pairwise_loss_auc_cpu_py_unique_8():
    df = pd.read_parquet('./tests_unique_8.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        sigm_loss = df.iloc[i].loss_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        sigm_loss_ = sigmoid_pairwise_loss_auc_gpu_py(y_true, y_pred)
        
        assert(np.abs(sigm_loss_ - sigm_loss) < 1e-5)
    
def test_sigmoid_pairwise_loss_auc_cpu_py_unique_10():
    df = pd.read_parquet('./tests_unique_10.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        sigm_loss = df.iloc[i].loss_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        sigm_loss_ = sigmoid_pairwise_loss_auc_gpu_py(y_true, y_pred)
        
        assert(np.abs(sigm_loss_ - sigm_loss) < 1e-5)
        
def test_sigmoid_pairwise_loss_auc_cpu_py_unique_100():
    df = pd.read_parquet('./tests_unique_100.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        sigm_loss = df.iloc[i].loss_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        sigm_loss_ = sigmoid_pairwise_loss_auc_gpu_py(y_true, y_pred)
        
        assert(np.abs(sigm_loss_ - sigm_loss) < 1e-5)
        

def test_sigmoid_pairwise_loss_auc_exact_cpu_py_unique_8():
    df = pd.read_parquet('./tests_unique_8.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        sigm_loss = df.iloc[i].loss_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        sigm_loss_ = sigmoid_pairwise_loss_auc_exact_gpu_py(y_true, y_pred)
        
        assert(np.abs(sigm_loss_ - sigm_loss) < 1e-5)
    
def test_sigmoid_pairwise_loss_auc_exact_cpu_py_unique_10():
    df = pd.read_parquet('./tests_unique_10.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        sigm_loss = df.iloc[i].loss_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        sigm_loss_ = sigmoid_pairwise_loss_auc_exact_gpu_py(y_true, y_pred)
        
        assert(np.abs(sigm_loss_ - sigm_loss) < 1e-5)
        
def test_sigmoid_pairwise_loss_auc_exact_cpu_py_unique_100():
    df = pd.read_parquet('./tests_unique_100.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        sigm_loss = df.iloc[i].loss_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        sigm_loss_ = sigmoid_pairwise_loss_auc_exact_gpu_py(y_true, y_pred)
        
        assert(np.abs(sigm_loss_ - sigm_loss) < 1e-5)
        
def test_sigmoid_pairwise_loss_auc_exact_cpu_py_non_unique_8():
    df = pd.read_parquet('./tests_non_unique_8.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        sigm_loss = df.iloc[i].loss_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        sigm_loss_ = sigmoid_pairwise_loss_auc_exact_gpu_py(y_true, y_pred)
        
        assert(np.abs(sigm_loss_ - sigm_loss) < 1e-5)
    
def test_sigmoid_pairwise_loss_auc_exact_cpu_py_non_unique_10():
    df = pd.read_parquet('./tests_non_unique_10.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        sigm_loss = df.iloc[i].loss_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        sigm_loss_ = sigmoid_pairwise_loss_auc_exact_gpu_py(y_true, y_pred)
        
        assert(np.abs(sigm_loss_ - sigm_loss) < 1e-5)
        
def test_sigmoid_pairwise_loss_auc_exact_cpu_py_non_unique_100():
    df = pd.read_parquet('./tests_non_unique_100.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        sigm_loss = df.iloc[i].loss_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        sigm_loss_ = sigmoid_pairwise_loss_auc_exact_gpu_py(y_true, y_pred)
        
        assert(np.abs(sigm_loss_ - sigm_loss) < 1e-5)
        
def test_sigmoid_pairwise_diff_hess_auc_cpu_unique_8_parquet():
    df = pd.read_parquet('./tests_unique_8.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        grad_true = df.iloc[i].grad_auc
        hess_true = df.iloc[i].hess_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        grad, hess = sigmoid_pairwise_diff_hess_auc_gpu_py(y_true, y_pred)
        
        assert(np.mean(np.abs(grad - grad_true)) < 1e-5)
        assert(np.mean(np.abs(hess - hess_true)) < 1e-5)

def test_sigmoid_pairwise_diff_hess_auc_cpu_unique_10_parquet():
    df = pd.read_parquet('./tests_unique_10.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        grad_true = df.iloc[i].grad_auc
        hess_true = df.iloc[i].hess_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        grad, hess = sigmoid_pairwise_diff_hess_auc_gpu_py(y_true, y_pred)
        
        assert(np.mean(np.abs(grad - grad_true)) < 1e-5)
        assert(np.mean(np.abs(hess - hess_true)) < 1e-5)
        
def test_sigmoid_pairwise_diff_hess_auc_cpu_unique_100_parquet():
    df = pd.read_parquet('./tests_unique_100.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        grad_true = df.iloc[i].grad_auc
        hess_true = df.iloc[i].hess_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        grad, hess = sigmoid_pairwise_diff_hess_auc_gpu_py(y_true, y_pred)
        
        assert(np.mean(np.abs(grad - grad_true)) < 1e-5)
        assert(np.mean(np.abs(hess - hess_true)) < 1e-5)
        

def test_sigmoid_pairwise_diff_hess_auc_exact_cpu_unique_8_parquet():
    df = pd.read_parquet('./tests_unique_8.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        grad_true = df.iloc[i].grad_auc
        hess_true = df.iloc[i].hess_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        grad, hess = sigmoid_pairwise_diff_hess_auc_exact_gpu_py(y_true, y_pred)
        
        assert(np.mean(np.abs(grad - grad_true)) < 1e-5)
        assert(np.mean(np.abs(hess - hess_true)) < 1e-5)
        
def test_sigmoid_pairwise_diff_hess_auc_exact_cpu_unique_10_parquet():
    df = pd.read_parquet('./tests_unique_10.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        grad_true = df.iloc[i].grad_auc
        hess_true = df.iloc[i].hess_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        grad, hess = sigmoid_pairwise_diff_hess_auc_exact_gpu_py(y_true, y_pred)
        
        assert(np.mean(np.abs(grad - grad_true)) < 1e-5)
        assert(np.mean(np.abs(hess - hess_true)) < 1e-5)
        
def test_sigmoid_pairwise_diff_hess_auc_exact_cpu_unique_100_parquet():
    df = pd.read_parquet('./tests_unique_100.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        grad_true = df.iloc[i].grad_auc
        hess_true = df.iloc[i].hess_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        grad, hess = sigmoid_pairwise_diff_hess_auc_exact_gpu_py(y_true, y_pred)
        
        assert(np.mean(np.abs(grad - grad_true)) < 1e-5)
        assert(np.mean(np.abs(hess - hess_true)) < 1e-5)
        

def test_sigmoid_pairwise_diff_hess_auc_exact_cpu_non_unique_8_parquet():
    df = pd.read_parquet('./tests_non_unique_8.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        grad_true = df.iloc[i].grad_auc
        hess_true = df.iloc[i].hess_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        grad, hess = sigmoid_pairwise_diff_hess_auc_exact_gpu_py(y_true, y_pred)
        
        assert(np.mean(np.abs(grad - grad_true)) < 1e-5)
        assert(np.mean(np.abs(hess - hess_true)) < 1e-5)
        
def test_sigmoid_pairwise_diff_hess_auc_exact_cpu_non_unique_10_parquet():
    df = pd.read_parquet('./tests_non_unique_10.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        grad_true = df.iloc[i].grad_auc
        hess_true = df.iloc[i].hess_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        grad, hess = sigmoid_pairwise_diff_hess_auc_exact_gpu_py(y_true, y_pred)
        
        assert(np.mean(np.abs(grad - grad_true)) < 1e-5)
        assert(np.mean(np.abs(hess - hess_true)) < 1e-5)
        
def test_sigmoid_pairwise_diff_hess_auc_exact_cpu_non_unique_100_parquet():
    df = pd.read_parquet('./tests_non_unique_100.parquet')
    
    for i in tqdm(range(df.shape[0])):
        y_true = df.iloc[i].y_true
        y_pred = df.iloc[i].y_pred
        grad_true = df.iloc[i].grad_auc
        hess_true = df.iloc[i].hess_auc
    
        y_true, y_pred = np.array(y_true).astype(np.int32), np.array(y_pred).astype(np.float64)
        
        grad, hess = sigmoid_pairwise_diff_hess_auc_exact_gpu_py(y_true, y_pred)
        
        assert(np.mean(np.abs(grad - grad_true)) < 1e-5)
        assert(np.mean(np.abs(hess - hess_true)) < 1e-5)