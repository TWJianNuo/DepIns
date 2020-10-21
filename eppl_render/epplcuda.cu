#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <cmath>

#include <math_constants.h>


namespace {

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmaxf(0.0, z) + fminf(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}


template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1 / (1 + std::exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t alpha_d_weight_function(scalar_t x, scalar_t sr) {
    // sr for smooth range
    scalar_t sigmoid_sr = 12.0 / sr;
    scalar_t bias_pos = sr / 2.0 + 1.0;
    scalar_t bias_neg = 0 - sr / 2.0;
    scalar_t y;
    if ((x < 1) && (x > 0)){
        y = 1;
    }
    else if(x <= 0){
        y = 0;
    }
    else{
        y = sigmoid(-sigmoid_sr * (x - bias_pos));

    }
    return y;
}

template <typename scalar_t>
__global__ void eppl_initCounter_cuda_kernel(
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> projected2d,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> selector,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> counter,
    const float kws,
    const int srhalf,
    const int bs,
    const int samplesz,
    const int height,
    const int width,
    const float eps
    ) {
        const int lindex = blockIdx.x * blockDim.x + threadIdx.x;
        int m;
        int n;
        int c = blockIdx.y;
        int sz = blockIdx.z;

        int ctx;
        int cty;

        float inc = 1.0;
        for(int l = lindex; l < height * width; l = l + blockDim.x * gridDim.x){
            m = (l / width);
            n = (l - m * width);
            if(selector[c][sz][0][m][n] > eps){
                ctx = round(projected2d[c][sz][0][m][n]);
                cty = round(projected2d[c][sz][1][m][n]);
                for (int i = ctx - srhalf; i < ctx + srhalf + 1; i++){
                    for(int j = cty - srhalf; j < cty + srhalf + 1; j++){
                        if ((i >= 0) && (i < width) && (j >= 0) && (j < height)){
                            atomicAdd((float*)&counter[c][sz][j][i], inc);
                        }
                    }
                }
            }
        }
    }



template <typename scalar_t>
__global__ void eppl_forward_cuda_kernel(
    const torch::PackedTensorAccessor<scalar_t,6,torch::RestrictPtrTraits,size_t> inv_r_sigma,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> projected2d,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> selector,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> counter,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> rimg,
    const float kws,
    const int srhalf,
    const int height,
    const int width,
    const float eps
    ) {
        const int lindex = blockIdx.x * blockDim.x + threadIdx.x;
        int m;
        int n;
        int c = blockIdx.y;
        int sz = blockIdx.z;

        int ctx;
        int cty;

        float cx;
        float cy;

        float expv;

        for(int l = lindex; l < height * width; l = l + blockDim.x * gridDim.x){
            m = (l / width);
            n = (l - m * width);
            if(selector[c][sz][0][m][n] > eps){
                ctx = round(projected2d[c][sz][0][m][n]);
                cty = round(projected2d[c][sz][1][m][n]);

                for (int i = ctx - srhalf; i < ctx + srhalf + 1; i++){
                    for(int j = cty - srhalf; j < cty + srhalf + 1; j++){
                        if ((i >= 0) && (i < width) && (j >= 0) && (j < height)){
                            cx = (projected2d[c][sz][0][m][n] - (float)i) / kws;
                            cy = (projected2d[c][sz][1][m][n] - (float)j) / kws;
                            expv = inv_r_sigma[c][sz][m][n][0][0] * cx * cx + inv_r_sigma[c][sz][m][n][1][0] * cx * cy + inv_r_sigma[c][sz][m][n][0][1] * cx * cy + inv_r_sigma[c][sz][m][n][1][1] * cy * cy;
                            expv = expf(-expv / 2.0) / 2.0 / M_PI / (counter[c][sz][j][i] + eps);
                            atomicAdd((float*)&rimg[c][sz][j][i], expv);
                        }
                    }
                }
            }
        }
    }


template <typename scalar_t>
__global__ void eppl_backward_cuda_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grad_rimg,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> depthmap,
    const torch::PackedTensorAccessor<scalar_t,6,torch::RestrictPtrTraits,size_t> inv_r_sigma,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> projected2d,
    const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> selector,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> Pcombined,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> counter,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> gradDepth,
    const float kws,
    const int srhalf,
    const int bs,
    const int samplesz,
    const int height,
    const int width,
    const float eps
    ) {
        const int lindex = blockIdx.x * blockDim.x + threadIdx.x;
        int m;
        int n;
        int c = blockIdx.y;
        int sz = blockIdx.z;

        int ctx;
        int cty;

        float cx;
        float cy;

        float expv;

        float m11 = Pcombined[c][sz][0][0];
        float m12 = Pcombined[c][sz][0][1];
        float m13 = Pcombined[c][sz][0][2];
        float m14 = Pcombined[c][sz][0][3];

        float m21 = Pcombined[c][sz][1][0];
        float m22 = Pcombined[c][sz][1][1];
        float m23 = Pcombined[c][sz][1][2];
        float m24 = Pcombined[c][sz][1][3];

        float m31 = Pcombined[c][sz][2][0];
        float m32 = Pcombined[c][sz][2][1];
        float m33 = Pcombined[c][sz][2][2];
        float m34 = Pcombined[c][sz][2][3];

        float gradPxDep;
        float gradPyDep;
        float x;
        float y;
        float D;

        for(int l = lindex; l < height * width; l = l + blockDim.x * gridDim.x){
            m = (l / width);
            n = (l - m * width);
            if(selector[c][sz][0][m][n] > eps){
                ctx = round(projected2d[c][sz][0][m][n]);
                cty = round(projected2d[c][sz][1][m][n]);

                D = depthmap[c][0][m][n];

                x = (float)n;
                y = (float)m;

                gradPxDep = ((m11 * x + m12 * y + m13) / (m31 * x * D + m32 * y * D + m33 * D + m34)) - ((m11 * x * D + m12 * y * D + m13 * D + m14) / (m31 * x * D + m32 * y * D + m33 * D + m34)) * ((m31 * x + m32 * y + m33) / (m31 * x * D + m32 * y * D + m33 * D + m34));
                gradPyDep = ((m21 * x + m22 * y + m23) / (m31 * x * D + m32 * y * D + m33 * D + m34)) - ((m21 * x * D + m22 * y * D + m23 * D + m24) / (m31 * x * D + m32 * y * D + m33 * D + m34)) * ((m31 * x + m32 * y + m33) / (m31 * x * D + m32 * y * D + m33 * D + m34));

                for (int i = ctx - srhalf; i < ctx + srhalf + 1; i++){
                    for(int j = cty - srhalf; j < cty + srhalf + 1; j++){
                        if ((i >= 0) && (i < width) && (j >= 0) && (j < height)){
                            cx = (projected2d[c][sz][0][m][n] - (float)i) / kws;
                            cy = (projected2d[c][sz][1][m][n] - (float)j) / kws;
                            expv = inv_r_sigma[c][sz][m][n][0][0] * cx * cx + inv_r_sigma[c][sz][m][n][1][0] * cx * cy + inv_r_sigma[c][sz][m][n][0][1] * cx * cy + inv_r_sigma[c][sz][m][n][1][1] * cy * cy;
                            expv = expf(-expv / 2.0) / 2.0 / M_PI / (counter[c][sz][j][i] + eps);

                            expv = expv / (-2.0);
                            expv = expv / kws * ((2.0 * inv_r_sigma[c][sz][m][n][0][0] * cx + inv_r_sigma[c][sz][m][n][1][0] * cy + inv_r_sigma[c][sz][m][n][0][1] * cy) * gradPxDep + (2.0 * inv_r_sigma[c][sz][m][n][1][1] * cy + inv_r_sigma[c][sz][m][n][1][0] * cx + inv_r_sigma[c][sz][m][n][0][1] * cx) * gradPyDep);
                            expv = expv * grad_rimg[c][sz][j][i];
                            // if (abs(grad_rimg[c][sz][j][i]) > 0){
                            //     printf("Detected. \n");
                            // }
                            atomicAdd((float*)&gradDepth[c][0][m][n], expv);
                        }
                    }
                }
            }
        }
    }



}




std::vector<torch::Tensor> eppl_forward_cuda(
    torch::Tensor inv_r_sigma,
    torch::Tensor projected2d,
    torch::Tensor selector,
    float kws,
    int sr
    ) {
  const int bs = projected2d.size(0);
  const int samplesz = projected2d.size(1);
  const int height = projected2d.size(3);
  const int width = projected2d.size(4);
  const int srhalf = (int) ((sr - 1) / 2);
  const float eps = 1e-6;

  torch::Tensor rimg = torch::zeros({bs, samplesz, height, width}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  torch::Tensor counter = torch::zeros({bs, samplesz, height, width}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

  const int threads = 1024;
  const int num_fold = 1; // Adjust cuda performance speed
  const dim3 blocks(num_fold, bs, samplesz);

  AT_DISPATCH_FLOATING_TYPES(projected2d.type(), "initialize counter", ([&] {
  eppl_initCounter_cuda_kernel<scalar_t><<<blocks, threads>>>(
        projected2d.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
        selector.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
        counter.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        kws,
        srhalf,
        bs,
        samplesz,
        height,
        width,
        eps
        );
  }));


  AT_DISPATCH_FLOATING_TYPES(projected2d.type(), "do forward rendering", ([&] {
  eppl_forward_cuda_kernel<scalar_t><<<blocks, (int) (threads / 4)>>>(
        inv_r_sigma.packed_accessor<scalar_t,6,torch::RestrictPtrTraits,size_t>(),
        projected2d.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
        selector.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
        counter.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        rimg.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        kws,
        srhalf,
        height,
        width,
        eps
        );
  }));

  return {rimg, counter};
}


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
    ) {
  const int bs = projected2d.size(0);
  const int samplesz = projected2d.size(1);
  const int height = projected2d.size(3);
  const int width = projected2d.size(4);
  const int srhalf = (int) ((sr - 1) / 2);
  const float eps = 1e-6;

  torch::Tensor gradDepth = torch::zeros({bs, 1, height, width}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

  const int threads = 1024;
  const int num_fold = 1; // Adjust cuda performance speed
  const dim3 blocks(num_fold, bs, samplesz);


  AT_DISPATCH_FLOATING_TYPES(depthmap.type(), "do forward and backward rendering", ([&] {
  eppl_backward_cuda_kernel<scalar_t><<<blocks, (int) (threads / 4)>>>(
        grad_rimg.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        depthmap.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        inv_r_sigma.packed_accessor<scalar_t,6,torch::RestrictPtrTraits,size_t>(),
        projected2d.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
        selector.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
        Pcombined.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        counter.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        gradDepth.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        kws,
        srhalf,
        bs,
        samplesz,
        height,
        width,
        eps
        );
  }));
  return {gradDepth};
}

