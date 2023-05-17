import pandas as pd
import numpy as np
import time
import sys

import tqdm

from roc_auc_pairwise.sigmoid_pairwise_auc_gpu import sigmoid_pairwise_loss_auc_gpu_py
from roc_auc_pairwise.sigmoid_pairwise_auc_gpu import sigmoid_pairwise_loss_auc_exact_gpu_py
from roc_auc_pairwise.sigmoid_pairwise_auc_gpu import sigmoid_pairwise_diff_hess_auc_gpu_py
from roc_auc_pairwise.sigmoid_pairwise_auc_gpu import sigmoid_pairwise_diff_hess_auc_exact_gpu_py

from roc_auc_pairwise.sigmoid_pairwise_auc_cpu import sigmoid_pairwise_loss_auc_cpu_py
from roc_auc_pairwise.sigmoid_pairwise_auc_cpu import sigmoid_pairwise_loss_auc_exact_cpu_py
from roc_auc_pairwise.sigmoid_pairwise_auc_cpu import sigmoid_pairwise_diff_hess_auc_cpu_py
from roc_auc_pairwise.sigmoid_pairwise_auc_cpu import sigmoid_pairwise_diff_hess_auc_exact_cpu_py

from roc_auc_pairwise.sigmoid_pairwise_gpu import sigmoid_pairwise_loss_gpu_py 
from roc_auc_pairwise.sigmoid_pairwise_gpu import sigmoid_pairwise_diff_hess_gpu_py

from roc_auc_pairwise.sigmoid_pairwise_cpu import sigmoid_pairwise_loss_py
from roc_auc_pairwise.sigmoid_pairwise_cpu import sigmoid_pairwise_diff_hess_py

class params:
    N_min = 100
    N_max_gpu = 300_000
    N_max_cpu = 100_000
    N_points = 15
    N_rounds = 3
    random_state=42
    
    PATH_TO_SAVE = './report'


type_ = None
if len(sys.argv) != 1:
    type_ = sys.argv[1]

df = list()
    
if type_ == 'gpu' or type_ is None:
    for N in tqdm.tqdm(np.linspace(params.N_min, params.N_max_gpu, params.N_points)):
        y_true = np.random.randint(0, 2, int(N)).astype(np.int32)
        y_pred = np.random.randn(int(N)).astype(np.float32)

        times_ = []
        for i in range(params.N_rounds):
            begin = time.time()
            loss = sigmoid_pairwise_loss_gpu_py(y_true.copy(), y_pred.copy())
            end = time.time()
            times_.append(end - begin)
        
        df.append({"loss" : "sigmoid_pairwise_loss", "N" : N, "device" : "cuda", "mean[ms]" : np.mean(times_), "var[ms]" : np.var(times_)})
        
        times_ = []
        for i in range(params.N_rounds):
            begin = time.time()
            grad, hess = sigmoid_pairwise_diff_hess_gpu_py(y_true.copy(), y_pred.copy())
            end = time.time()
            times_.append(end - begin)
        
        df.append({"loss" : "sigmoid_pairwise_diff_hess", "N" : N, "device" : "cuda", "mean[ms]" : np.mean(times_), "var[ms]" : np.var(times_)})
        
        times_ = []
        for i in range(params.N_rounds):
            begin = time.time()
            loss = sigmoid_pairwise_loss_auc_gpu_py(y_true.copy(), y_pred.copy())
            end = time.time()
            times_.append(end - begin)
        
        df.append({"loss" : "sigmoid_pairwise_loss_auc", "N" : N, "device" : "cuda", "mean[ms]" : np.mean(times_), "var[ms]" : np.var(times_)})
        
        times_ = []
        for i in range(params.N_rounds):
            begin = time.time()
            loss = sigmoid_pairwise_loss_auc_exact_gpu_py(y_true.copy(), y_pred.copy())
            end = time.time()
            times_.append(end - begin)
        
        df.append({"loss" : "sigmoid_pairwise_loss_auc_exact", "N" : N, "device" : "cuda", "mean[ms]" : np.mean(times_), "var[ms]" : np.var(times_)})
        
        times_ = []
        for i in range(params.N_rounds):
            begin = time.time()
            grad, hess = sigmoid_pairwise_diff_hess_auc_gpu_py(y_true.copy(), y_pred.copy())
            end = time.time()
            times_.append(end - begin)
            
        df.append({"loss" : "sigmoid_pairwise_diff_hess_auc", "N" : N, "device" : "cuda", "mean[ms]" : np.mean(times_), "var[ms]" : np.var(times_)})
        
        times_ = []
        for i in range(params.N_rounds):
            begin = time.time()
            grad, hess = sigmoid_pairwise_diff_hess_auc_exact_gpu_py(y_true.copy(), y_pred.copy())
            end = time.time()
            times_.append(end - begin)
        
        df.append({"loss" : "sigmoid_pairwise_diff_hess_auc_exact", "N" : N, "device" : "cuda", "mean[ms]" : np.mean(times_), "var[ms]" : np.var(times_)})
if type_ == 'cpu' or type_ is None:
    gpu_linspace = np.linspace(params.N_min, params.N_max_gpu, params.N_points)
    cpu_linspace = gpu_linspace[gpu_linspace < params.N_max_cpu]
    for N in tqdm.tqdm(cpu_linspace):
        y_true = np.random.randint(0, 2, int(N)).astype(np.int32)
        y_pred = np.random.randn(int(N)).astype(np.float32)

        times_ = []
        for i in range(params.N_rounds):
            begin = time.time()
            loss = sigmoid_pairwise_loss_py(y_true.copy(), y_pred.copy())
            end = time.time()
            times_.append(end - begin)
        
        df.append({"loss" : "sigmoid_pairwise_loss", "N" : N, "device" : "cpu", "mean[ms]" : np.mean(times_), "var[ms]" : np.var(times_)})
        
        times_ = []
        for i in range(params.N_rounds):
            begin = time.time()
            grad, hess = sigmoid_pairwise_diff_hess_py(y_true.copy(), y_pred.copy())
            end = time.time()
            times_.append(end - begin)
        
        df.append({"loss" : "sigmoid_pairwise_diff_hess", "N" : N, "device" : "cpu", "mean[ms]" : np.mean(times_), "var[ms]" : np.var(times_)})
        
        times_ = []
        for i in range(params.N_rounds):
            begin = time.time()
            loss = sigmoid_pairwise_loss_auc_cpu_py(y_true.copy(), y_pred.copy())
            end = time.time()
            times_.append(end - begin)
        
        df.append({"loss" : "sigmoid_pairwise_loss_auc", "N" : N, "device" : "cpu", "mean[ms]" : np.mean(times_), "var[ms]" : np.var(times_)})
        
        times_ = []
        for i in range(params.N_rounds):
            begin = time.time()
            loss = sigmoid_pairwise_loss_auc_exact_cpu_py(y_true.copy(), y_pred.copy())
            end = time.time()
            times_.append(end - begin)
        
        df.append({"loss" : "sigmoid_pairwise_loss_auc_exact", "N" : N, "device" : "cpu", "mean[ms]" : np.mean(times_), "var[ms]" : np.var(times_)})
        
        times_ = []
        for i in range(params.N_rounds):
            begin = time.time()
            grad, hess = sigmoid_pairwise_diff_hess_auc_cpu_py(y_true.copy(), y_pred.copy())
            end = time.time()
            times_.append(end - begin)
            
        df.append({"loss" : "sigmoid_pairwise_diff_hess_auc", "N" : N, "device" : "cpu", "mean[ms]" : np.mean(times_), "var[ms]" : np.var(times_)})
        
        times_ = []
        for i in range(params.N_rounds):
            begin = time.time()
            grad, hess = sigmoid_pairwise_diff_hess_auc_exact_cpu_py(y_true.copy(), y_pred.copy())
            end = time.time()
            times_.append(end - begin)
        
        df.append({"loss" : "sigmoid_pairwise_diff_hess_auc_exact", "N" : N, "device" : "cpu", "mean[ms]" : np.mean(times_), "var[ms]" : np.var(times_)})

df = pd.DataFrame(df)
if type_ is not None:        
    df.to_csv(params.PATH_TO_SAVE + f'_{type_}_' + '.csv')
else:
    df.to_csv(params.PATH_TO_SAVE + '.csv')
