#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

// CUDA forward declarations
void sparsMMul_forward_cuda(
    torch::Tensor indx,
    torch::Tensor indy,
    torch::Tensor mtxvals,
    torch::Tensor vecvals,
    bool ist,
    torch::Tensor re
    );

void sparsMMul_backward_cuda(
    torch::Tensor indx,
    torch::Tensor indy,
    torch::Tensor mtxvals,
    torch::Tensor vecvals,
    bool ist,
    torch::Tensor grad_re,
    torch::Tensor grad_mtxvals,
    torch::Tensor grad_vecvals
    );

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void sparsMMul_forward(
    torch::Tensor indx,
    torch::Tensor indy,
    torch::Tensor mtxvals,
    torch::Tensor vecvals,
    bool ist,
    torch::Tensor re
    ) {
    CHECK_INPUT(indx)
    CHECK_INPUT(indy)
    CHECK_INPUT(mtxvals)
    CHECK_INPUT(vecvals)
    CHECK_INPUT(re)

    sparsMMul_forward_cuda(indx, indy, mtxvals, vecvals, ist, re);
    return;
}

void sparsMMul_backward(
    torch::Tensor indx,
    torch::Tensor indy,
    torch::Tensor mtxvals,
    torch::Tensor vecvals,
    bool ist,
    torch::Tensor grad_re,
    torch::Tensor grad_mtxvals,
    torch::Tensor grad_vecvals
    ) {
    CHECK_INPUT(indx)
    CHECK_INPUT(indy)
    CHECK_INPUT(mtxvals)
    CHECK_INPUT(vecvals)
    CHECK_INPUT(grad_re)
    CHECK_INPUT(grad_mtxvals)
    CHECK_INPUT(grad_vecvals)

    sparsMMul_backward_cuda(indx, indy, mtxvals, vecvals, ist, grad_re, grad_mtxvals, grad_vecvals);
    return;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparsMMul_forward", &sparsMMul_forward, "Sparse Matrix Mult forward function");
  m.def("sparsMMul_backward", &sparsMMul_backward, "Sparse Matrix Mult backward function");
}
