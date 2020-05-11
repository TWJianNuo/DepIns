#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> init_lookupTable_cuda(
    torch::Tensor lookTable,
    torch::Tensor epp,
    int w,
    int h,
    int h2,
    int searchRange,
    float max_depth,
    float verRange,
    float horRange
    );


std::vector<torch::Tensor> lidar_denoise_cuda(
    torch::Tensor nvelo_projected_img,
    torch::Tensor lookTable,
    torch::Tensor noocc_mask,
    torch::Tensor epp,
    int w,
    int h2,
    float mind1d2,
    float maxd2
    );

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> init_lookupTable(
    torch::Tensor lookTable,
    torch::Tensor epp,
    int w,
    int h,
    int h2,
    int searchRange,
    float max_depth,
    float verRange,
    float horRange
    ) {
    CHECK_INPUT(lookTable)
    CHECK_INPUT(epp)
    std::vector<torch::Tensor> results_bindings = init_lookupTable_cuda(lookTable, epp, w, h, h2, searchRange, max_depth, verRange, horRange);
    return results_bindings;
}

std::vector<torch::Tensor> lidar_denoise(
    torch::Tensor nvelo_projected_img,
    torch::Tensor lookTable,
    torch::Tensor noocc_mask,
    torch::Tensor epp,
    int w,
    int h2,
    float mind1d2,
    float maxd2
    ) {
    CHECK_INPUT(nvelo_projected_img)
    CHECK_INPUT(lookTable)
    CHECK_INPUT(noocc_mask)
    CHECK_INPUT(epp)
    std::vector<torch::Tensor> results_bindings = lidar_denoise_cuda(nvelo_projected_img, lookTable, noocc_mask, epp, w, h2, mind1d2, maxd2);
    return results_bindings;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_lookupTable", &init_lookupTable, "Initialize the look up table");
  m.def("lidar_denoise", &lidar_denoise, "Denoise Lidar");
}
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//  m.def("init_lookupTable", &init_lookupTable, "Initialize the look up table");
//}
