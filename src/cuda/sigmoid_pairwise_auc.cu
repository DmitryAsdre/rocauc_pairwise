#include <cmath>
#include <cstddef>
#include <utility>
#include <cstring>
#include <tuple>
#include <assert.h>
#include "constants.cuh"
#include "../cpu/utils.cpp"
#include "sigmoid_pairwise_auc_kernels.cuh"
#include "deltaauc_kernels.cuh"

float sigmoid_pairwise_loss_auc(int32_t* y_true, float* exp_pred,
                                long* y_pred_argsorted,
                                size_t n_ones, size_t n_zeroes, size_t N){
    float loss = 0;
    float* loss_device;

    int32_t* y_pred_ranks = get_inverse_argsort<int32_t, int32_t, float, long>(y_true, exp_pred, y_pred_argsorted, N);

    int32_t* y_true_device, *y_pred_ranks_device;
    float* exp_pred_device;

    cudaError_t err;

    err = cudaMalloc((void**)&y_true_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)y_true_device, (void*)y_true, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&exp_pred_device, N*sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)exp_pred_device, (void*)exp_pred, N*sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&y_pred_ranks_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)y_pred_ranks_device, (void*)y_pred_ranks, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&loss_device, sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)loss_device, (void*)&loss, sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);

    sigmoid_pairwise_loss_auc_kernel<<<N_BLOCKS_LOSS, N_THREADS_LOSS>>>(y_true_device, exp_pred_device, y_pred_ranks_device, loss_device, n_ones, n_zeroes, N);

    err = cudaGetLastError();
    assert(err == 0);

    err = cudaMemcpy((void*)&loss, (void*)loss_device, sizeof(float), cudaMemcpyDeviceToHost);
    assert(err == 0);

    delete[] y_pred_ranks;

    cudaFree(y_true_device);
    cudaFree(exp_pred_device);
    cudaFree(y_pred_ranks_device);
    cudaFree(loss_device);

    return loss;
}

std::pair<float*, float*> sigmoid_pairwise_grad_hess_auc(int32_t* y_true, float* exp_pred,
                                                         long* y_pred_argsorted, 
                                                         size_t n_ones, size_t n_zeroes, size_t N){
    
    int32_t* y_pred_ranks = get_inverse_argsort<int32_t, int32_t, float, long>(y_true, exp_pred, y_pred_argsorted, N);

    float* grad, *hess;

    grad = new float[N];
    hess = new float[N];

    memset(grad, 0, N*sizeof(float));
    memset(hess, 0, N*sizeof(float));

    float* grad_device, *hess_device;

    int32_t* y_true_device, *y_pred_ranks_device;
    float* exp_pred_device;

    cudaError_t err;

    err = cudaMalloc((void**)&grad_device, N*sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)grad_device, (void*)grad, N*sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&hess_device, N*sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)hess_device, (void*)hess, N*sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);


    err = cudaMalloc((void**)&y_true_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)y_true_device, (void*)y_true, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&exp_pred_device, N*sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)exp_pred_device, (void*)exp_pred, N*sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&y_pred_ranks_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)y_pred_ranks_device, (void*)y_pred_ranks, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    sigmoid_pairwise_grad_hess_auc_kernel<<<N_BLOCKS_GRADHESS, N_THREADS_GRADHESS>>>(y_true_device, exp_pred_device, y_pred_ranks_device, grad_device, hess_device, n_ones, n_zeroes, N);

    err = cudaGetLastError();
    assert(err == 0);

    err = cudaMemcpy((void*)grad, (void*)grad_device, N*sizeof(float), cudaMemcpyDeviceToHost);
    assert(err == 0);

    err = cudaMemcpy((void*)hess, (void*)hess_device, N*sizeof(float), cudaMemcpyDeviceToHost);
    assert(err == 0);

    delete[] y_pred_ranks;

    cudaFree(y_true_device);
    cudaFree(exp_pred_device);
    cudaFree(y_pred_ranks_device);
    cudaFree(grad_device);
    cudaFree(hess_device);

    return std::make_pair<float*, float*>(&*grad, &*hess);
}


float sigmoid_pairwise_loss_auc_exact(int32_t* y_true, float* exp_pred, 
                                      long* y_pred_argsorted, float eps,
                                      size_t n_ones, size_t n_zeroes, size_t N){
    std::tuple<int32_t*, int32_t*, int32_t*, int32_t*> labelscount;

    labelscount = get_labelscount_borders<int32_t, float, long>(y_true, exp_pred, y_pred_argsorted, N);

    int32_t* counters_p, *counters_n;
    int32_t* y_pred_left, *y_pred_right;

    counters_p = std::get<0>(labelscount);
    counters_n = std::get<1>(labelscount);
    y_pred_left = std::get<2>(labelscount);
    y_pred_right = std::get<3>(labelscount);


    int32_t* y_true_device;
    float* exp_pred_device;

    int32_t *counters_p_device, *counters_n_device;
    int32_t *y_pred_left_device, *y_pred_right_device;

    float loss = 0;
    float* loss_device;

    cudaError_t err;

    err = cudaMalloc((void**)&y_true_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)y_true_device, (void*)y_true, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&exp_pred_device, N*sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)exp_pred_device, (void*)exp_pred, N*sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&counters_p_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)counters_p_device, (void*)counters_p, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&counters_n_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)counters_n_device, (void*)counters_n, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&y_pred_left_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)y_pred_left_device, (void*)y_pred_left, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&y_pred_right_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)y_pred_right_device, (void*)y_pred_right, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&loss_device, sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)loss_device, (void*)&loss, sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);

    if(eps == 0){
        sigmoid_pairwise_loss_auc_exact_kernel<<<N_BLOCKS_LOSS, N_THREADS_LOSS>>>(y_true_device, exp_pred_device, 
                                                                                  counters_p_device, counters_n_device, 
                                                                                  y_pred_left_device, y_pred_right_device, 
                                                                                  loss_device,
                                                                                  n_ones, n_zeroes, N);
    }else{
        sigmoid_pairwise_loss_auc_exact_sm_kernel<<<N_BLOCKS_LOSS, N_THREADS_LOSS>>>(y_true_device, exp_pred_device, 
                                                                                     counters_p_device, counters_n_device, 
                                                                                     y_pred_left_device, y_pred_right_device, 
                                                                                     loss_device, eps,
                                                                                     n_ones, n_zeroes, N);
    }

    
    err = cudaGetLastError();
    assert(err == 0);

    err = cudaMemcpy((void*)&loss, (void*)loss_device, sizeof(float), cudaMemcpyDeviceToHost);
    assert(err == 0);

    delete[] counters_p;
    delete[] counters_n;
    delete[] y_pred_left;
    delete[] y_pred_right;
    
    cudaFree(loss_device);
    cudaFree(counters_n_device);
    cudaFree(counters_p_device);
    cudaFree(y_pred_left_device);
    cudaFree(y_pred_right_device);
    cudaFree(exp_pred_device);
    cudaFree(y_true_device);

    return loss;
}

std::pair<float*, float*> sigmoid_pairwise_grad_hess_auc_exact(int32_t* y_true, float* exp_pred, 
                                                              long* y_pred_argsorted, float eps,
                                                              size_t n_ones, size_t n_zeroes, size_t N){
    std::tuple<int32_t*, int32_t*, int32_t*, int32_t*> labelscount;

    labelscount = get_labelscount_borders<int32_t, float, long>(y_true, exp_pred, y_pred_argsorted, N);

    int32_t* counters_p, *counters_n;
    int32_t* y_pred_left, *y_pred_right;

    counters_p = std::get<0>(labelscount);
    counters_n = std::get<1>(labelscount);
    y_pred_left = std::get<2>(labelscount);
    y_pred_right = std::get<3>(labelscount);


    int32_t* y_true_device;
    float* exp_pred_device;

    int32_t *counters_p_device, *counters_n_device;
    int32_t *y_pred_left_device, *y_pred_right_device;

    float* grad = new float[N];
    float* hess = new float[N];

    memset(grad, 0, N*sizeof(float));
    memset(hess, 0, N*sizeof(float));

    float* grad_device, *hess_device;

    cudaError_t err;

    err = cudaMalloc((void**)&y_true_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)y_true_device, (void*)y_true, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&exp_pred_device, N*sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)exp_pred_device, (void*)exp_pred, N*sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&counters_p_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)counters_p_device, (void*)counters_p, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&counters_n_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)counters_n_device, (void*)counters_n, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&y_pred_left_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)y_pred_left_device, (void*)y_pred_left, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&y_pred_right_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)y_pred_right_device, (void*)y_pred_right, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&grad_device, N*sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)grad_device, (void*)grad, N*sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&hess_device, N*sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)hess_device, (void*)hess, N*sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);

    if(eps == 0){
        sigmoid_pairwise_grad_hess_auc_exact_kernel<<<N_BLOCKS_GRADHESS, N_THREADS_GRADHESS>>>(y_true_device, exp_pred_device,
                                                                                               counters_p_device, counters_n_device,
                                                                                               y_pred_left_device, y_pred_right_device,
                                                                                               grad_device, hess_device,
                                                                                               n_ones, n_zeroes, N);
    }else{
        sigmoid_pairwise_grad_hess_auc_exact_sm_kernel<<<N_BLOCKS_GRADHESS, N_THREADS_GRADHESS>>>(y_true_device, exp_pred_device,
                                                                                                  counters_p_device, counters_n_device,
                                                                                                  y_pred_left_device, y_pred_right_device,
                                                                                                  grad_device, hess_device, eps,
                                                                                                  n_ones, n_zeroes, N);
    }
    
    err = cudaGetLastError();
    assert(err == 0);

    err = cudaMemcpy((void*)grad, (void*)grad_device, N*sizeof(float), cudaMemcpyDeviceToHost);
    assert(err == 0);

    err = cudaMemcpy((void*)hess, (void*)hess_device, N*sizeof(float), cudaMemcpyDeviceToHost);
    assert(err == 0);

    delete[] counters_p;
    delete[] counters_n;
    delete[] y_pred_left;
    delete[] y_pred_right;
    
    cudaFree(grad_device);
    cudaFree(hess_device);
    cudaFree(counters_n_device);
    cudaFree(counters_p_device);
    cudaFree(y_pred_left_device);
    cudaFree(y_pred_right_device);
    cudaFree(exp_pred_device);
    cudaFree(y_true_device);

    return std::make_pair<float*, float*>(&*grad, &*hess);
}