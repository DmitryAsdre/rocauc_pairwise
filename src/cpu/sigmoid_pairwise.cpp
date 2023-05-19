#include <cstddef>
#include <cmath>
#include <utility>
#include <cstring>
#include <tuple>
#include "sigmoid_pairwise.hpp"
#include "constants.hpp"


template<class T_true, class T_pred>
double sigmoid_pairwise_loss(T_true* y_true, T_pred* exp_pred, size_t N){
    double loss = 0.d;

    #pragma omp parallel for reduction(+ : loss)
        for(size_t i = 0; i < N; i++) 
        {
            size_t _i = (i%2 == 1) ? (N - size_t(i/2) - 1) : size_t(i/2);
            for(size_t j = 0; j <= _i ; j++){
                double P_hat = (y_true[_i] == y_true[j]) ? 0.5d : double(y_true[_i] > y_true[j]);
                double P = 1.d / (1.d + (exp_pred[j] / exp_pred[_i]));
                loss += P_hat*log(P + EPS) + (1.d - P_hat)*log(1.d - P - EPS);
            }
        }
    return loss;
}


template<class T_true, class T_pred>
std::pair<double*, double*> sigmoid_pairwise_diff_hess(T_true* y_true, T_pred* exp_pred, size_t N){
    double* grad = new double[N];
    double* hess = new double[N];

    memset((void*)grad, 0, N*sizeof(double));
    memset((void*)hess, 0, N*sizeof(double));

    #pragma omp parallel for
        for(size_t i = 0; i < N; i++){
            size_t _i = (i % 2 == 1) ? N - size_t(i / 2) - 1 : size_t(i / 2);
            for(size_t j = 0; j < _i; j++){
                double exp_tmp_diff = exp_pred[_i] / exp_pred[j];
                double P_hat = (y_true[_i] == y_true[j]) ? 0.5d : double(y_true[_i] > y_true[j]);

                double cur_d_dx_i = ((P_hat - 1.d) * exp_tmp_diff + P_hat) / (exp_tmp_diff + 1.d);
                double cur_d_dx_j = -cur_d_dx_i;

                double cur_d2_dx2_i = (-exp_pred[_i] / (exp_pred[_i] + exp_pred[j]))*(exp_pred[j] / (exp_pred[_i] + exp_pred[j]));
                double cur_d2_dx2_j = cur_d2_dx2_i;
                
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
    return std::make_pair<double*, double*>(&*grad, &*hess);
}