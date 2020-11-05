#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

// CUDA forward declarations
void shapeIntegration_crf_forward_cuda(
    torch::Tensor log,
    torch::Tensor semantics,
    torch::Tensor mask,
    torch::Tensor depthin,
    torch::Tensor depth_optedin,
    torch::Tensor depth_optedout,
    int height,
    int width,
    int bs,
    float lambda
    );

void shapeIntegration_crf_constrain_forward_cuda(
    torch::Tensor log,
    torch::Tensor semantics,
    torch::Tensor mask,
    torch::Tensor depthin,
    torch::Tensor constrainout,
    int height,
    int width,
    int bs
    );
// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//torch::Tensor bnmorph_find_coorespond_pts(
void shapeIntegration_crf_forward(
    torch::Tensor log,
    torch::Tensor semantics,
    torch::Tensor mask,
    torch::Tensor depthin,
    torch::Tensor depth_optedin,
    torch::Tensor depth_optedout,
    int height,
    int width,
    int bs,
    float lambda
    ) {
    CHECK_INPUT(log);
    CHECK_INPUT(semantics);
    CHECK_INPUT(mask);
    CHECK_INPUT(depthin);
    CHECK_INPUT(depth_optedin);
    CHECK_INPUT(depth_optedout);
    shapeIntegration_crf_forward_cuda(log, semantics, mask, depthin, depth_optedin, depth_optedout, height, width, bs, lambda);
    return;
}

void shapeIntegration_crf_constrain_forward(
    torch::Tensor log,
    torch::Tensor semantics,
    torch::Tensor mask,
    torch::Tensor depthin,
    torch::Tensor constrainout,
    int height,
    int width,
    int bs
    ) {
    CHECK_INPUT(log);
    CHECK_INPUT(semantics);
    CHECK_INPUT(mask);
    CHECK_INPUT(depthin);
    CHECK_INPUT(constrainout);
    shapeIntegration_crf_constrain_forward_cuda(log, semantics, mask, depthin, constrainout, height, width, bs);
    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shapeIntegration_crf_forward", &shapeIntegration_crf_forward, "crf based shape integration forward function");
  m.def("shapeIntegration_crf_constrain_forward", &shapeIntegration_crf_constrain_forward, "crf based shape integration constrain forward function");
}
