#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

// CUDA forward declarations
void shapeIntegration_forward_cuda(
    torch::Tensor ang,
    torch::Tensor log,
    torch::Tensor confidence,
    torch::Tensor semantics,
    torch::Tensor mask,
    torch::Tensor depthin,
    torch::Tensor depthout,
    torch::Tensor summedConfidence,
    torch::Tensor varbar,
    int height,
    int width,
    int bs
    );

void shapeIntegration_backward_cuda(
    torch::Tensor ang,
    torch::Tensor log,
    torch::Tensor confidence,
    torch::Tensor semantics,
    torch::Tensor mask,
    torch::Tensor depthin,
    torch::Tensor depthin_opted,
    torch::Tensor summedConfidence,
    torch::Tensor gradin,
    torch::Tensor gradout_depth,
    torch::Tensor gradout_confidence,
    torch::Tensor varbar,
    int height,
    int width,
    int bs
    );

void shapeIntegration_crf_forward_cuda(
    torch::Tensor ang,
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
// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//torch::Tensor bnmorph_find_coorespond_pts(
void shapeIntegration_forward(
    torch::Tensor ang,
    torch::Tensor log,
    torch::Tensor confidence,
    torch::Tensor semantics,
    torch::Tensor mask,
    torch::Tensor depthin,
    torch::Tensor depthout,
    torch::Tensor summedConfidence,
    torch::Tensor varbar,
    int height,
    int width,
    int bs
    ) {
    CHECK_INPUT(ang)
    CHECK_INPUT(log)
    CHECK_INPUT(confidence)
    CHECK_INPUT(semantics)
    CHECK_INPUT(mask)
    CHECK_INPUT(depthin)
    CHECK_INPUT(depthout)
    CHECK_INPUT(summedConfidence)
    CHECK_INPUT(varbar)
    shapeIntegration_forward_cuda(ang, log, confidence, semantics, mask, depthin, depthout, summedConfidence, varbar, height, width, bs);
    return;
}

void shapeIntegration_backward(
    torch::Tensor ang,
    torch::Tensor log,
    torch::Tensor confidence,
    torch::Tensor semantics,
    torch::Tensor mask,
    torch::Tensor depthin,
    torch::Tensor depthin_opted,
    torch::Tensor summedConfidence,
    torch::Tensor gradin,
    torch::Tensor gradout_depth,
    torch::Tensor gradout_confidence,
    torch::Tensor varbar,
    int height,
    int width,
    int bs
    ) {
    CHECK_INPUT(ang)
    CHECK_INPUT(log)
    CHECK_INPUT(confidence)
    CHECK_INPUT(semantics)
    CHECK_INPUT(mask)
    CHECK_INPUT(depthin)
    CHECK_INPUT(depthin_opted)
    CHECK_INPUT(summedConfidence)
    CHECK_INPUT(gradin)
    CHECK_INPUT(gradout_depth)
    CHECK_INPUT(gradout_confidence)
    CHECK_INPUT(varbar)
    shapeIntegration_backward_cuda(ang, log, confidence, semantics, mask, depthin, depthin_opted, summedConfidence, gradin, gradout_depth, gradout_confidence, varbar, height, width, bs);
    return;
}


void shapeIntegration_crf_forward(
    torch::Tensor ang,
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
    CHECK_INPUT(ang)
    CHECK_INPUT(log)
    CHECK_INPUT(semantics)
    CHECK_INPUT(mask)
    CHECK_INPUT(depthin)
    CHECK_INPUT(depth_optedin)
    CHECK_INPUT(depth_optedout)
    shapeIntegration_crf_forward_cuda(ang, log, semantics, mask, depthin, depth_optedin, depth_optedout, height, width, bs, lambda);
    return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shapeIntegration_forward", &shapeIntegration_forward, "shape integration forward function");
  m.def("shapeIntegration_backward", &shapeIntegration_backward, "shape integration backward function");
  m.def("shapeIntegration_crf_forward", &shapeIntegration_crf_forward, "crf based shape integration forward function");
}
