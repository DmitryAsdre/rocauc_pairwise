#include <cstddef>
#include <cmath>
#include <utility>
#include <cstring>
#include <tuple>
#include "sigmoid_pairwise.hpp"
#include "utils.hpp"
#include "deltaauc.hpp"
#include "constants.hpp"


template<class T_true, class T_pred, class T_argsorted>
double sigmoid_pairwise_loss_auc_cpu(T_true* y_true, T_pred* exp_pred, 
                                     T_argsorted* y_pred_argsorted, size_t n_ones, 
                                     size_t n_zeroes, size_t N){
    double loss = 0.d;
    long* inverse_argsort = get_inverse_argsort<long, T_true, T_pred, T_argsorted>(y_true, exp_pred, y_pred_argsorted, N);

    #pragma omp parallel for reduction(+ : loss)
        for(size_t i = 0; i < N; i++)
        {
            size_t _i = (i%2 == 1) ? (N - size_t(i/2) - 1) : size_t(i/2);
            for(size_t j = 0; j <= _i; j++){
                double P_hat = (y_true[_i] == y_true[j]) ? 0.5d : double(y_true[_i] > y_true[j]);
                double P = 1.d / (1.d + (exp_pred[j] / exp_pred[_i]));
                double deltaauc_ij = deltaauc<T_true, long>(y_true, inverse_argsort, n_ones, n_zeroes, _i, j);

                loss += fabs(deltaauc_ij)*(P_hat*log(P + EPS) + (1.d - P_hat)*log(1.d - P - EPS));
            }
        }

    delete[] inverse_argsort;

    return loss;
}


template<class T_true, class T_pred, class T_argsorted>
double sigmoid_pairwise_loss_auc_exact_cpu(T_true* y_true, T_pred* exp_pred,
                                           T_argsorted* y_pred_argsorted, double eps,
                                           size_t n_ones, size_t n_zeroes, size_t N){
    double loss = 0.d;
    std::tuple<int32_t*, int32_t*, int32_t*, int32_t*> labelscount_borders;

    labelscount_borders = get_labelscount_borders<T_true, T_pred, T_argsorted>(y_true, exp_pred, y_pred_argsorted, N);

    int32_t* counters_p = std::get<0>(labelscount_borders);
    int32_t* counters_n = std::get<1>(labelscount_borders);
    int32_t* y_pred_left = std::get<2>(labelscount_borders);
    int32_t* y_pred_right = std::get<3>(labelscount_borders);

     #pragma omp parallel for reduction(+ : loss)
        for(size_t i = 0; i < N; i++)
        {
            size_t _i = (i%2 == 1) ? (N - size_t(i/2) - 1) : size_t(i/2);
            for(size_t j = 0; j < _i; j++){
                double P_hat = (y_true[_i] == y_true[j]) ? 0.5d : double(y_true[_i] > y_true[j]);
                double P = 1.d / (1.d + (exp_pred[j] / exp_pred[_i]));
                double deltaauc_ij = deltaauc_exact<T_true, T_pred>(y_true, exp_pred, counters_p, counters_n, 
                                                                    y_pred_left, y_pred_right, n_ones, n_zeroes, _i, j);

                loss += (fabs(deltaauc_ij) + eps)*(P_hat*log(P + EPS) + (1.d - P_hat)*log(1.d - P - EPS));
            }
        }

    delete[] counters_p;
    delete[] counters_n;
    delete[] y_pred_left;
    delete[] y_pred_right;

    return loss;
}


template<class T_true, class T_pred, class T_argsorted>
std::pair<double*, double*> sigmoid_pairwise_diff_hess_auc_cpu(T_true* y_true, T_pred* exp_pred,
                                                               T_argsorted* y_pred_argsorted, 
                                                               size_t n_ones, size_t n_zeroes, size_t N){
    double* grad = new double[N];
    double* hess = new double[N];

    memset((void*)grad, 0, N*sizeof(double));
    memset((void*)hess, 0, N*sizeof(double));

    long* inverse_argsort = get_inverse_argsort<long, T_true, T_pred, T_argsorted>(y_true, exp_pred, y_pred_argsorted, N);

    #pragma omp parallel for
    for(size_t i = 0; i < N; i++){
        size_t _i = (i%2 == 1) ? (N - size_t(i/2) - 1) : size_t(i/2);
        for(size_t j = 0; j < _i; j++){
            double deltaauc_ij = deltaauc<T_true, long>(y_true, inverse_argsort, n_ones, n_zeroes, _i, j);

            double exp_tmp_diff = exp_pred[_i] / exp_pred[j];
            double P_hat = (y_true[_i] == y_true[j]) ? 0.5d : double(y_true[_i] > y_true[j]);

            double cur_d_dx_i = ((P_hat - 1.d) * exp_tmp_diff + P_hat) / (exp_tmp_diff + 1.d);
            double cur_d_dx_j = -cur_d_dx_i;

            double cur_d2_dx2_i = (-exp_pred[_i] / (exp_pred[_i] + exp_pred[j]))*(exp_pred[j] / (exp_pred[_i] + exp_pred[j]));
            double cur_d2_dx2_j = cur_d2_dx2_i;

            cur_d_dx_i *= fabs(deltaauc_ij);
            cur_d_dx_j *= fabs(deltaauc_ij);
            cur_d2_dx2_i *= fabs(deltaauc_ij);
            cur_d2_dx2_j *= fabs(deltaauc_ij);

                
            #pragma omp atomic
                grad[j] += cur_d_dx_j;
            #pragma omp atomic
                hess[j] += cur_d2_dx2_j;
            #pragma omp atomic
                grad[_i] += cur_d_dx_i;
            #pragma omp atomic
                hess[_i] += cur_d2_dx2_i;
        }
    }

    delete[] inverse_argsort;

    return std::make_pair<double*, double*>(&*grad, &*hess);
}


template<class T_true, class T_pred, class T_argsorted>
std::pair<double*, double*> sigmoid_pairwise_diff_hess_auc_exact_cpu(T_true* y_true, T_pred* exp_pred,
                                                                     T_argsorted* y_pred_argsorted, double eps,
                                                                     size_t n_ones, size_t n_zeroes, size_t N){
    double* grad = new double[N];
    double* hess = new double[N];

    memset((void*)grad, 0, N*sizeof(double));
    memset((void*)hess, 0, N*sizeof(double));

    std::tuple<int32_t*, int32_t*, int32_t*, int32_t*> labelscount_borders;

    labelscount_borders = get_labelscount_borders<T_true, T_pred, T_argsorted>(y_true, exp_pred, y_pred_argsorted, N);

    int32_t* counters_p = std::get<0>(labelscount_borders);
    int32_t* counters_n = std::get<1>(labelscount_borders);
    int32_t* y_pred_left = std::get<2>(labelscount_borders);
    int32_t* y_pred_right = std::get<3>(labelscount_borders);


    #pragma omp parallel for
    for(size_t i = 0; i < N; i++){
        size_t _i = (i%2 == 1) ? (N - size_t(i/2) - 1) : size_t(i/2);
        for(size_t j = 0; j < _i; j++){
            double deltaauc_ij = deltaauc_exact<T_true, T_pred>(y_true, exp_pred, counters_p, counters_n, 
                                                                y_pred_left, y_pred_right, n_ones, n_zeroes, _i, j);

            double exp_tmp_diff = exp_pred[_i] / exp_pred[j];
            double P_hat = (y_true[_i] == y_true[j]) ? 0.5d : double(y_true[_i] > y_true[j]);

            double cur_d_dx_i = ((P_hat - 1.d) * exp_tmp_diff + P_hat) / (exp_tmp_diff + 1.d);
            double cur_d_dx_j = -cur_d_dx_i;

            double cur_d2_dx2_i = (-exp_pred[_i] / (exp_pred[_i] + exp_pred[j]))*(exp_pred[j] / (exp_pred[_i] + exp_pred[j]));
            double cur_d2_dx2_j = cur_d2_dx2_i;


            cur_d_dx_i *= (fabs(deltaauc_ij) + eps);
            cur_d_dx_j *= (fabs(deltaauc_ij) + eps);
            cur_d2_dx2_i *= (fabs(deltaauc_ij) + eps);
            cur_d2_dx2_j *= (fabs(deltaauc_ij) + eps);
                
            #pragma omp atomic
                grad[j] += cur_d_dx_j;
            #pragma omp atomic
                hess[j] += cur_d2_dx2_j;
            #pragma omp atomic
                grad[_i] += cur_d_dx_i;
            #pragma omp atomic
                hess[_i] += cur_d2_dx2_i;
        }
    }

    delete[] counters_p;
    delete[] counters_n;
    delete[] y_pred_left;
    delete[] y_pred_right;

    return std::make_pair<double*, double*>(&*grad, &*hess);
}
