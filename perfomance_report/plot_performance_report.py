import pandas as pd
import matplotlib.pyplot as plt

class param:
    figsize = (15, 15)
    PATH_TO_SAVE = './performance_report.jpg'
    REPORT_PATH = './report.csv'

df = pd.read_csv(param.REPORT_PATH, index_col=0).set_index('loss')

losses = [['sigmoid_pairwise_loss', 'sigmoid_pairwise_diff_hess'],
          ['sigmoid_pairwise_loss_auc','sigmoid_pairwise_diff_hess_auc'],
          ['sigmoid_pairwise_loss_auc_exact', 'sigmoid_pairwise_diff_hess_auc_exact']]

losses_full_names = {'sigmoid_pairwise_loss' : 'Sigmoid Pairwise Loss', 
                   'sigmoid_pairwise_diff_hess' : "Sigmoid Pairwise Grad&Hess",
                   'sigmoid_pairwise_loss_auc' : "Sigmoid Pairwise Loss approx. AUC",
                   'sigmoid_pairwise_diff_hess_auc' : "Sigmoid Pairwise Grad&Hess approx. AUC",
                   'sigmoid_pairwise_loss_auc_exact' : "Sigmoid Pairwise Loss exact AUC", 
                   'sigmoid_pairwise_diff_hess_auc_exact' : "Sigmoid Pairwise Grad&Hess exact AUC"}

fig, ax = plt.subplots(3, 2, figsize=param.figsize)
for i, _loss_name in enumerate(losses):
    for j, loss_name in enumerate(_loss_name):
        loss_full_name = losses_full_names[loss_name]
        
        df_loss = df.loc[loss_name]
        df_loss_cpu = df_loss[df_loss.device == 'cpu']
        df_loss_gpu = df_loss[df_loss.device == 'cuda']

        ax[i, j].set_title(loss_full_name)
        ax[i, j].set_xlabel('N')
        ax[i, j].set_ylabel('[ms]')
        
        ax[i, j].plot(df_loss_cpu.N, df_loss_cpu["mean[ms]"], 'bd')
        ax[i, j].plot(df_loss_cpu.N, df_loss_cpu["mean[ms]"], 'b', label='i5 10600KF')

        ax[i, j].plot(df_loss_gpu.N, df_loss_gpu["mean[ms]"], 'gd')
        ax[i, j].plot(df_loss_gpu.N, df_loss_gpu["mean[ms]"], 'g', label = 'RTX 3060')
        ax[i, j].legend()
plt.savefig(param.PATH_TO_SAVE)
        
    