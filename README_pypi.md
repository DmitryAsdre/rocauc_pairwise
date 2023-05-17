# RocAuc Pairwise Loss/Objective
 This is gpu implementation of rocauc pairwise objectives for gradient boosting:
$$L = \sum_{i, j} (\hat P_{ij}\log{P_{ij}} + (1 - \hat P_{ij})\log{(1 - P_{ij})}) |\Delta_{AUC_{ij}}|$$
This package could be used to solve classification problems with relative small numbers of objects, where you need to improve rocauc score. \
Also there is cpu multithread implementation of this losses.
## Losses that are implemented in this package
1. **Sigmoid pairwise loss.** (GPU or CPU implementations)
$$L = \sum_{i, j}\hat P_{ij}\log{P_{ij}} + (1 - \hat P_{ij})\log{(1 - P_{ij})}$$
2. **RocAuc Pairwise Loss** with approximate auc computation. (GPU or CPU implementations)
$$L = \sum_{i, j} (\hat P_{ij}\log{P_{ij}} + (1 - \hat P_{ij})\log{(1 - P_{ij})})|\Delta_{AUC^{approx}_{ij}}|$$
3. **RocAuc Pairwise Loss Exact** (GPU or CPU implementations) with exact auc computation. This could be more compute intensive, but this loss might be helpfull for first boosting rounds (if you are using gradient boosting)
$$L = \sum_{i, j} (\hat P_{ij}\log{P_{ij}} + (1 - \hat P_{ij})\log{(1 - P_{ij})})|\Delta_{AUC^{exact}_{ij}}|$$
4. **RocAuc Pairwise Loss Exact Smoothened** (GPU or CPU implementations). This loss allows you to incorporate information about equal instances. Because $\Delta_{AUC_{ij}} = 0$ if $y_i = y_j$. So we just add small $\epsilon > 0$ in equation.
$$L = \sum_{i, j} (\hat P_{ij}\log{P_{ij}} + (1 - \hat P_{ij})\log{(1 - P_{ij})})(\epsilon + |\Delta_{AUC^{exact}_{ij}}|)$$

### For more information 
- You can see [GitHub repository](https://github.com/DmitryAsdre/rocauc_pairwise).
- Or you can use example notebook on [Google Colab](https://colab.research.google.com/drive/1w7BN0XGjB5vgFp2pbiCaejabc91xWmI0?usp=sharing)
- Or you can use example notebook on [Kaggle](https://www.kaggle.com/code/michailindmitry/gradient-boosting-roc-auc-pairwise-example-ipynb)

## References
[1] Sean J. Welleck, Efficient AUC Optimization for Information Ranking Applications, IBM USA (2016) <br />
[2] Burges, C.J.: From ranknet to lambdarank to lambdamart: An overview. Learning (2010) <br />
[3] Calders, T., Jaroszewicz, S.: Efficient auc optimization for classification. Knowledge
Discovery in Databases. (2007)