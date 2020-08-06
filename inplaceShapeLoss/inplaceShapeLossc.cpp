#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

// CUDA forward declarations
void inplaceShapeLoss_forward_cuda(
    torch::Tensor logdepth,
    torch::Tensor logratioh,
    torch::Tensor logratiov,
    torch::Tensor valindic,
    torch::Tensor lossrec,
    torch::Tensor countsrec,
    torch::Tensor rndseeds,
    int srw,
    int srh
    );

void inplaceShapeLoss_backward_cuda(
    torch::Tensor logdepth,
    torch::Tensor logratioh,
    torch::Tensor logratiov,
    torch::Tensor valindic,
    torch::Tensor grad_re,
    torch::Tensor gradrech,
    torch::Tensor gradrecv,
    torch::Tensor countsrec,
    torch::Tensor rndseeds,
    int srw,
    int srh
    );

void inplaceShapeLoss_integration_cuda(
    torch::Tensor logdepth,
    torch::Tensor logratioh,
    torch::Tensor logratiov,
    torch::Tensor valindic,
    torch::Tensor lossrec,
    int srw,
    int srh
    );
// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x) CHECK_CONTIGUOUS(x)


void inplaceShapeLoss_forward(
    torch::Tensor logdepth,
    torch::Tensor logratioh,
    torch::Tensor logratiov,
    torch::Tensor valindic,
    torch::Tensor lossrec,
    torch::Tensor countsrec,
    torch::Tensor rndseeds,
    int srw,
    int srh
    ) {
    CHECK_INPUT(logdepth)
    CHECK_INPUT(logratioh)
    CHECK_INPUT(logratiov)
    CHECK_INPUT(valindic)
    CHECK_INPUT(lossrec)
    CHECK_INPUT(countsrec)
    CHECK_INPUT(rndseeds)
    inplaceShapeLoss_forward_cuda(logdepth, logratioh, logratiov, valindic, lossrec, countsrec, rndseeds, srw, srh);
    return;
}

void inplaceShapeLoss_backward(
    torch::Tensor logdepth,
    torch::Tensor logratioh,
    torch::Tensor logratiov,
    torch::Tensor valindic,
    torch::Tensor grad_re,
    torch::Tensor gradrech,
    torch::Tensor gradrecv,
    torch::Tensor countsrec,
    torch::Tensor rndseeds,
    int srw,
    int srh
    ) {
    CHECK_INPUT(logdepth)
    CHECK_INPUT(logratioh)
    CHECK_INPUT(logratiov)
    CHECK_INPUT(valindic)
    CHECK_INPUT(grad_re)
    CHECK_INPUT(gradrech)
    CHECK_INPUT(gradrecv)
    CHECK_INPUT(countsrec)
    CHECK_INPUT(rndseeds)
    inplaceShapeLoss_backward_cuda(logdepth, logratioh, logratiov, valindic, grad_re, gradrech, gradrecv, countsrec, rndseeds, srw, srh);
    return;
}

void inplaceShapeLoss_integration(
    torch::Tensor logdepth,
    torch::Tensor logratioh,
    torch::Tensor logratiov,
    torch::Tensor valindic,
    torch::Tensor lossrec,
    int srw,
    int srh
    ) {
    CHECK_INPUT(logdepth)
    CHECK_INPUT(logratioh)
    CHECK_INPUT(logratiov)
    CHECK_INPUT(valindic)
    CHECK_INPUT(lossrec)
    inplaceShapeLoss_integration_cuda(logdepth, logratioh, logratiov, valindic, lossrec, srw, srh);
    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inplaceShapeLoss_forward", &inplaceShapeLoss_forward, "Inplace Shape Loss forward function");
  m.def("inplaceShapeLoss_backward", &inplaceShapeLoss_backward, "Inplace Shape Loss backward function");
  m.def("inplaceShapeLoss_integration", &inplaceShapeLoss_integration, "Inplace Shape Loss Aggregation function");
}
