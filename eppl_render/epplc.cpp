#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> eppl_forward_cuda(
    torch::Tensor inv_r_sigma,
    torch::Tensor projected2d,
    torch::Tensor selector,
    float kws,
    int sr
    );

std::vector<torch::Tensor> eppl_backward_cuda(
    torch::Tensor grad_rimg,
    torch::Tensor depthmap,
    torch::Tensor inv_r_sigma,
    torch::Tensor projected2d,
    torch::Tensor selector,
    torch::Tensor Pcombined,
    torch::Tensor counter,
    float kws,
    int sr
    );

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

//torch::Tensor bnmorph_find_coorespond_pts(
std::vector<torch::Tensor> eppl_forward(
    torch::Tensor inv_r_sigma,
    torch::Tensor projected2d,
    torch::Tensor selector,
    float kws,
    int sr
    ) {
    CHECK_INPUT(inv_r_sigma)
    CHECK_INPUT(projected2d)
    CHECK_INPUT(selector)
    std::vector<torch::Tensor> results_bindings = eppl_forward_cuda(inv_r_sigma, projected2d, selector, kws, sr);
    return results_bindings;
}

std::vector<torch::Tensor> eppl_backward(
    torch::Tensor grad_rimg,
    torch::Tensor depthmap,
    torch::Tensor inv_r_sigma,
    torch::Tensor projected2d,
    torch::Tensor selector,
    torch::Tensor Pcombined,
    torch::Tensor counter,
    float kws,
    int sr
    ) {
    CHECK_INPUT(grad_rimg)
    CHECK_INPUT(depthmap)
    CHECK_INPUT(inv_r_sigma)
    CHECK_INPUT(projected2d)
    CHECK_INPUT(selector)
    CHECK_INPUT(Pcombined)
    CHECK_INPUT(counter)
    std::vector<torch::Tensor> results_bindings = eppl_backward_cuda(grad_rimg, depthmap, inv_r_sigma, projected2d, selector, Pcombined, counter, kws, sr);
    return results_bindings;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("eppl_forward", &eppl_forward, "Epipolar line redering forward function");
  m.def("eppl_backward", &eppl_backward, "Epipolar line redering backward function");
}
