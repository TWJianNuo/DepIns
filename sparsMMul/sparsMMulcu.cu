#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <cmath>

#include <math_constants.h>

namespace {


template <typename scalar_t>
__global__ void sparsMMul_forward_cuda_kernel(
    const torch::PackedTensorAccessor<long,1,torch::RestrictPtrTraits,size_t> indx,
    const torch::PackedTensorAccessor<long,1,torch::RestrictPtrTraits,size_t> indy,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> mtxvals,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> vecvals,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> re
    ) {
       for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < indx.size(0); i = i + blockDim.x * gridDim.x){
           atomicAdd((float*)&re[indy[i]], mtxvals[i] * vecvals[indx[i]]);
       }
    }

template <typename scalar_t>
__global__ void sparsMMul_forward_cuda_kernel_transpose(
    const torch::PackedTensorAccessor<long,1,torch::RestrictPtrTraits,size_t> indx,
    const torch::PackedTensorAccessor<long,1,torch::RestrictPtrTraits,size_t> indy,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> mtxvals,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> vecvals,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> re
    ) {
       for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < indx.size(0); i = i + blockDim.x * gridDim.x){
           atomicAdd((float*)&re[indx[i]], mtxvals[i] * vecvals[indy[i]]);
       }
    }

template <typename scalar_t>
__global__ void sparsMMul_backward_cuda_kernel(
    const torch::PackedTensorAccessor<long,1,torch::RestrictPtrTraits,size_t> indx,
    const torch::PackedTensorAccessor<long,1,torch::RestrictPtrTraits,size_t> indy,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> mtxvals,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> vecvals,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> grad_re,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> grad_mtxvals,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> grad_vecvals
    ) {
       for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < indx.size(0); i = i + blockDim.x * gridDim.x){
            atomicAdd((float*)&grad_mtxvals[i], vecvals[indx[i]] * grad_re[indy[i]]);
            atomicAdd((float*)&grad_vecvals[indx[i]], mtxvals[i] * grad_re[indy[i]]);
       }
    }

template <typename scalar_t>
__global__ void sparsMMul_backward_cuda_kernel_transpose(
    const torch::PackedTensorAccessor<long,1,torch::RestrictPtrTraits,size_t> indx,
    const torch::PackedTensorAccessor<long,1,torch::RestrictPtrTraits,size_t> indy,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> mtxvals,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> vecvals,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> grad_re,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> grad_mtxvals,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> grad_vecvals
    ) {
       for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < indx.size(0); i = i + blockDim.x * gridDim.x){
            atomicAdd((float*)&grad_mtxvals[i], vecvals[indy[i]] * grad_re[indx[i]]);
            atomicAdd((float*)&grad_vecvals[indy[i]], mtxvals[i] * grad_re[indx[i]]);
       }
    }

}


void sparsMMul_forward_cuda(
    torch::Tensor indx,
    torch::Tensor indy,
    torch::Tensor mtxvals,
    torch::Tensor vecvals,
    bool ist,
    torch::Tensor re
    ) {
    const int threads = 1024;
    const int blockdimx = 128;
    const dim3 blocks(blockdimx);

    if(!ist){
        AT_DISPATCH_FLOATING_TYPES(mtxvals.type(), "do forward Sparse Matrix Multuplication", ([&] {
        sparsMMul_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            indx.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
            indy.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
            mtxvals.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            vecvals.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            re.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>()
            );
        }));
    }
    else{
        AT_DISPATCH_FLOATING_TYPES(mtxvals.type(), "do forward Sparse Matrix Multuplication", ([&] {
        sparsMMul_forward_cuda_kernel_transpose<scalar_t><<<blocks, threads>>>(
            indx.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
            indy.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
            mtxvals.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            vecvals.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            re.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>()
            );
        }));
    }
    return;
}

void sparsMMul_backward_cuda(
    torch::Tensor indx,
    torch::Tensor indy,
    torch::Tensor mtxvals,
    torch::Tensor vecvals,
    bool ist,
    torch::Tensor grad_re,
    torch::Tensor grad_mtxvals,
    torch::Tensor grad_vecvals
    ) {
    const int threads = 1024;
    const int blockdimx = 128;
    const dim3 blocks(blockdimx);


    if(!ist){
        AT_DISPATCH_FLOATING_TYPES(mtxvals.type(), "do forward Sparse Matrix Multuplication", ([&] {
        sparsMMul_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            indx.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
            indy.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
            mtxvals.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            vecvals.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            grad_re.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            grad_mtxvals.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            grad_vecvals.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>()
            );
        }));
    }
    else{
        AT_DISPATCH_FLOATING_TYPES(mtxvals.type(), "do forward Sparse Matrix Multuplication", ([&] {
        sparsMMul_backward_cuda_kernel_transpose<scalar_t><<<blocks, threads>>>(
            indx.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
            indy.packed_accessor<long,1,torch::RestrictPtrTraits,size_t>(),
            mtxvals.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            vecvals.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            grad_re.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            grad_mtxvals.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
            grad_vecvals.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>()
            );
        }));
    }
    return;
}