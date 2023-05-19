#include <cmath>
#include <cstddef>
#include <cstring>
#include <utility>
#include <assert.h>
#include "constants.cuh"
#include "sigmoid_pairwise_kernels.cuh"

float sigmoid_pairwise_loss(int32_t* y_true, float* exp_pred, size_t N){
    float sigmoid_loss = 0.f;

    int32_t* y_true_device;
    float* exp_pred_device;
    float* sigmoid_loss_device;

    cudaError_t err;

    err = cudaMalloc((void**)&y_true_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)y_true_device, (void*)y_true, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&exp_pred_device, N*sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)exp_pred_device, (void*)exp_pred, N*sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&sigmoid_loss_device, sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)sigmoid_loss_device, (void*)&sigmoid_loss, sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);

    sigmoid_pairwise_loss_kernel<<<N_BLOCKS_LOSS, N_THREADS_LOSS>>>(y_true_device, exp_pred_device, sigmoid_loss_device, N);

    err = cudaGetLastError();
    assert(err == 0);

    err = cudaMemcpy((void*)&sigmoid_loss, (void*)sigmoid_loss_device, sizeof(float), cudaMemcpyDeviceToHost);
    assert(err == 0);

    cudaFree(y_true_device);
    cudaFree(exp_pred_device);
    cudaFree(sigmoid_loss_device);

    return sigmoid_loss;
}

std::pair<float*, float*> sigmoid_pairwise_grad_hess(int32_t* y_true, float* exp_pred, size_t N){
    float* grad, *hess;
    
    float* grad_device, *hess_device;
    int32_t* y_true_device;
    float* exp_pred_device;

    grad = new float[N];
    hess = new float[N];

    memset(grad, 0, N*sizeof(float));
    memset(hess, 0, N*sizeof(float));


    cudaError_t err;

    err = cudaMalloc((void**)&y_true_device, N*sizeof(int32_t));
    assert(err == 0);
    err = cudaMemcpy((void*)y_true_device, (void*)y_true, N*sizeof(int32_t), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&exp_pred_device, N*sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)exp_pred_device, (void*)exp_pred, N*sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);


    err = cudaMalloc((void**)&grad_device, N*sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)grad_device, (void*)grad, N*sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);

    err = cudaMalloc((void**)&hess_device, N*sizeof(float));
    assert(err == 0);
    err = cudaMemcpy((void*)hess_device, (void*)hess, N*sizeof(float), cudaMemcpyHostToDevice);
    assert(err == 0);


    sigmoid_pairwise_grad_hess_kernel<<<N_BLOCKS_GRADHESS, N_THREADS_GRADHESS>>>(y_true_device, exp_pred_device, grad_device, hess_device, N);

    err = cudaGetLastError();
    assert(err == 0);


    err = cudaMemcpy((void*)grad, (void*)grad_device, N*sizeof(float), cudaMemcpyDeviceToHost);
    assert(err == 0);

    err = cudaMemcpy((void*)hess, (void*)hess_device, N*sizeof(float), cudaMemcpyDeviceToHost);
    assert(err == 0);

    
    cudaFree(grad_device);
    cudaFree(hess_device);
    cudaFree(y_true_device);
    cudaFree(exp_pred_device);
    
    return std::make_pair<float*, float*>(&*grad, &*hess);
}


