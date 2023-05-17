import numpy as np
import time
from roc_auc_pairwise.sigmoid_pairwise_auc_gpu import sigmoid_pairwise_loss_auc_gpu_py

y_true = np.random.randint(0, 2, 100000).astype(np.int32)
y_pred = np.random.randn(100000).astype(np.float32)


begin = time.time()

sigmoid_pairwise_loss_auc_gpu_py(y_true, y_pred)

end = time.time()

print(end - begin)