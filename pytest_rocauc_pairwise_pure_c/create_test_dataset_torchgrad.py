from copy import deepcopy
from functools import partial
import itertools

import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
from torch.autograd.functional import hessian
from torch.autograd import grad
from torch.autograd.functional import hessian


def delta_auc_score(y_true, y_pred, i, j):
    auc_1 = roc_auc_score(y_true, y_pred)
    y_pred_ = deepcopy(y_pred)
    y_pred_[i], y_pred_[j] = y_pred_[j], y_pred_[i]
    auc_2 = roc_auc_score(y_true, y_pred_)
    return auc_1 - auc_2

def get_roc_auc(y_true, y_pred):
    roc_aucs = np.zeros((y_true.shape[0], y_true.shape[0]))
    for i in range(y_true.shape[0]):
        for j in range(i):
            roc_aucs[i, j] = delta_auc_score(y_true, y_pred, i, j)
            roc_aucs[j, i] = roc_aucs[i, j]
    return roc_aucs

def sigmoid_pairwise_torch(y_pred, y_true):
    P_hat = 0.5*(y_true.reshape(-1, 1) - y_true) + 0.5
    P = 1. / (1. + torch.exp(-(y_pred.reshape(-1, 1) - y_pred)))
    doubled_diag = torch.ones(P_hat.shape) + torch.eye(P_hat.shape[0])
    return torch.sum(doubled_diag*P_hat*torch.log(P + 1e-60))

def sigmoid_pairwise_roc_auc_torch(y_pred, y_true):
    delta_aucs = get_roc_auc(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
    delta_aucs = np.abs(delta_aucs)
    delta_aucs = torch.tensor(delta_aucs, requires_grad=False)
    P_hat = 0.5*(y_true.reshape(-1, 1) - y_true) + 0.5
    P = 1. / (1. + torch.exp(-(y_pred.reshape(-1, 1) - y_pred)))
    return torch.sum(delta_aucs*P_hat*torch.log(P + 1e-60))

def generate_grad_hess_torch(y_pred, y_true):
    y_true = torch.tensor(y_true.astype(np.int32))
    y_pred_sigm = torch.tensor(y_pred.astype(np.float32), requires_grad=True)
    y_pred_auc = torch.tensor(y_pred.astype(np.float32), requires_grad=True)
    
    loss_sigm = sigmoid_pairwise_torch(y_pred_sigm, y_true)
    loss_auc = sigmoid_pairwise_roc_auc_torch(y_pred_auc, y_true)
    
    d_loss_dx_sigm = grad(outputs=loss_sigm, inputs=y_pred_sigm)
    d_loss_dx_auc = grad(outputs=loss_auc, inputs=y_pred_auc)
    
    d2_loss_dx2_sigm = hessian(partial(sigmoid_pairwise_torch, y_true=y_true), y_pred_auc)
    d2_loss_dx2_auc = hessian(partial(sigmoid_pairwise_torch, y_true=y_true), y_pred_sigm)
    
    return loss_sigm.detach().numpy(),\
           loss_auc.detach().numpy(),\
           d_loss_dx_sigm[0].detach().numpy(),\
           d_loss_dx_auc[0].detach().numpy(),\
           d2_loss_dx2_sigm.detach().diagonal().numpy(),\
           d2_loss_dx2_auc.detach().diagonal().numpy()
           
           
def create_tests_on_permutations(y_pred, y_true, random_t, n_max=None):
    counter = 0
    grad_hess_dicts = []
    for permutation in tqdm.tqdm(itertools.permutations([i for i in range(len(y_true))])):
        if np.random.randint(0, random_t) != 0:
            continue
        
        counter += 1
        
        permutation = np.array(permutation)
        
        res = generate_grad_hess_torch(y_pred, y_true[permutation])
        
        res_dict = {'y_pred' : y_pred,
                    'y_true' : y_true[permutation],
                    'loss_sigm' : res[0].item(),
                    'loss_auc' : res[1].item(),
                    'grad_sigm' : res[2],
                    'grad_auc' : res[3],
                    'hess_sigm' : res[4],
                    'hess_auc' : res[5]}
        
        grad_hess_dicts.append(res_dict)
        if n_max is not None and counter > n_max:
            break
        
    return pd.DataFrame(grad_hess_dicts)

def create_tests_on_random_permutations(y_pred, y_true, n_max):
    
    random_permutations = []
    
    for i in range(n_max * 3):
        random_permutations.append(tuple(np.random.permutation(len(y_true))))
    
    random_permutations = set(random_permutations)
    grad_hess_dicts = []
    
    for counter, permutation in tqdm.tqdm(enumerate(random_permutations), total=n_max):
        permutation = np.array(permutation)
        
        res = generate_grad_hess_torch(y_pred, y_true[permutation])
        
        res_dict = {'y_pred' : y_pred,
                    'y_true' : y_true[permutation],
                    'loss_sigm' : res[0].item(),
                    'loss_auc' : res[1].item(),
                    'grad_sigm' : res[2],
                    'grad_auc' : res[3],
                    'hess_sigm' : res[4],
                    'hess_auc' : res[5]}
        
        grad_hess_dicts.append(res_dict)
        
        if counter > n_max:
            break
    
    return pd.DataFrame(grad_hess_dicts)


#8 samples, unique predictions
y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1]).astype(np.int32)
y_pred = np.array([np.log(1.0 + i) for i in range(8)]).astype(np.float32)

df_8_unique = create_tests_on_random_permutations(y_pred, y_true, 250)
df_8_unique.to_parquet('./tests_unique_8.parquet')

#10 samples, unique predictions
y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).astype(np.int32)
y_pred = np.array([np.log(1.0 + i) for i in range(10)]).astype(np.float32)

df_8_unique = create_tests_on_random_permutations(y_pred, y_true, 250)
df_8_unique.to_parquet('./tests_unique_10.parquet')

#8 samples, non unique predictions
y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1]).astype(np.int32)
y_pred = np.array([np.log(1.0 + i % 3) for i in range(8)]).astype(np.float32)

df_8_unique = create_tests_on_random_permutations(y_pred, y_true, 250)
df_8_unique.to_parquet('./tests_non_unique_8.parquet')

#10 samples, non unique predictions
y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).astype(np.int32)
y_pred = np.array([np.log(1.0 + i % 3) for i in range(10)]).astype(np.float32)

df_8_unique = create_tests_on_random_permutations(y_pred, y_true, 250)
df_8_unique.to_parquet('./tests_non_unique_10.parquet')

#100 samples, unique predictions, random_permutations
y_true = np.array([int(i > 50) for i in range(100)])
y_pred = np.array([np.log(1. + i) for i in range(100)])

df_1000_unique = create_tests_on_random_permutations(y_pred, y_true, 100)
df_1000_unique.to_parquet('./tests_unique_100.parquet')

#1000 samples, non unique predictions, random_permutations
y_true = np.array([int(i > 50) for i in range(100)])
y_pred = np.array([np.log(1. + i % 10) for i in range(100)])

df_1000_unique = create_tests_on_random_permutations(y_pred, y_true, 100)
df_1000_unique.to_parquet('./tests_non_unique_100.parquet')
