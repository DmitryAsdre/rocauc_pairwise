#ifndef SIGMOID_PAIRWISE_AUC_HPP
#define SIGMOID_PAIRWISE_AUC_HPP

template<class T_true, class T_pred, class T_argsorted>
double sigmoid_pairwise_loss_auc_cpu(T_true* y_true, T_pred* exp_pred, 
                                     T_argsorted* y_pred_argsorted, size_t n_ones, 
                                     size_t n_zeroes, size_t N);


template<class T_true, class T_pred, class T_argsorted>
double sigmoid_pairwise_loss_auc_exact_cpu(T_true* y_true, T_pred* exp_pred,
                                           T_argsorted* y_pred_argsorted, double eps,
                                           size_t n_ones, size_t n_zeroes, size_t N);


template<class T_true, class T_pred, class T_argsorted>
std::pair<double*, double*> sigmoid_pairwise_diff_hess_auc_cpu(T_true* y_true, T_pred* exp_pred,
                                                               T_argsorted* y_pred_argsorted, 
                                                               size_t n_ones, size_t n_zeroes, size_t N);


template<class T_true, class T_pred, class T_argsorted>
std::pair<double*, double*> sigmoid_pairwise_diff_hess_auc_exact_cpu(T_true* y_true, T_pred* exp_pred,
                                                                     T_argsorted* y_pred_argsorted, double eps,
                                                                     size_t n_ones, size_t n_zeroes, size_t N);

#endif