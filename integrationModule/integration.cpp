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

void shapeIntegration_crf_variance_forward_cuda(
    torch::Tensor log,
    torch::Tensor semantics,
    torch::Tensor mask,
    torch::Tensor variance,
    torch::Tensor depthin,
    torch::Tensor depth_optedin,
    torch::Tensor depth_optedout,
    torch::Tensor summedconfidence,
    int height,
    int width,
    int bs,
    float lambda,
    float clipvariance,
    float maxrange
    );


void shapeIntegration_crf_star_forward_cuda(
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
    torch::Tensor counts,
    int height,
    int width,
    int bs
    );

void shapeIntegration_crf_constrain_backward_cuda(
    torch::Tensor log,
    torch::Tensor semantics,
    torch::Tensor mask,
    torch::Tensor depthin,
    torch::Tensor counts,
    torch::Tensor constraingradin,
    torch::Tensor constraingradout,
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

void shapeIntegration_crf_variance_forward(
    torch::Tensor log,
    torch::Tensor semantics,
    torch::Tensor mask,
    torch::Tensor variance,
    torch::Tensor depthin,
    torch::Tensor depth_optedin,
    torch::Tensor depth_optedout,
    torch::Tensor summedconfidence,
    int height,
    int width,
    int bs,
    float lambda,
    float clipvariance,
    float maxrange
    ) {
    CHECK_INPUT(log);
    CHECK_INPUT(semantics);
    CHECK_INPUT(mask);
    CHECK_INPUT(variance);
    CHECK_INPUT(depthin);
    CHECK_INPUT(depth_optedin);
    CHECK_INPUT(depth_optedout);
    CHECK_INPUT(summedconfidence);
    shapeIntegration_crf_variance_forward_cuda(log, semantics, mask, variance, depthin, depth_optedin, depth_optedout, summedconfidence, height, width, bs, lambda, clipvariance, maxrange);
    return;
}

void shapeIntegration_crf_star_forward(
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
    shapeIntegration_crf_star_forward_cuda(log, semantics, mask, depthin, depth_optedin, depth_optedout, height, width, bs, lambda);
    return;
}

void shapeIntegration_crf_constrain_forward(
    torch::Tensor log,
    torch::Tensor semantics,
    torch::Tensor mask,
    torch::Tensor depthin,
    torch::Tensor constrainout,
    torch::Tensor counts,
    int height,
    int width,
    int bs
    ) {
    CHECK_INPUT(log);
    CHECK_INPUT(semantics);
    CHECK_INPUT(mask);
    CHECK_INPUT(depthin);
    CHECK_INPUT(constrainout);
    shapeIntegration_crf_constrain_forward_cuda(log, semantics, mask, depthin, constrainout, counts, height, width, bs);
    return;
}

void shapeIntegration_crf_constrain_backward(
    torch::Tensor log,
    torch::Tensor semantics,
    torch::Tensor mask,
    torch::Tensor depthin,
    torch::Tensor counts,
    torch::Tensor constraingradin,
    torch::Tensor constraingradout,
    int height,
    int width,
    int bs
    ) {
    CHECK_INPUT(log);
    CHECK_INPUT(semantics);
    CHECK_INPUT(mask);
    CHECK_INPUT(depthin);
    CHECK_INPUT(counts);
    CHECK_INPUT(constraingradin);
    CHECK_INPUT(constraingradout);
    shapeIntegration_crf_constrain_backward_cuda(log, semantics, mask, depthin, counts, constraingradin, constraingradout, height, width, bs);
    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shapeIntegration_crf_forward", &shapeIntegration_crf_forward, "crf based shape integration forward function");
  m.def("shapeIntegration_crf_variance_forward", &shapeIntegration_crf_variance_forward, "crf based shape integration with confidence forward function");
  m.def("shapeIntegration_crf_star_forward", &shapeIntegration_crf_star_forward, "crf based shape star integration forward function");
  m.def("shapeIntegration_crf_constrain_forward", &shapeIntegration_crf_constrain_forward, "crf based shape integration constrain forward function");
  m.def("shapeIntegration_crf_constrain_backward", &shapeIntegration_crf_constrain_backward, "crf based shape integration constrain backward function");
}
