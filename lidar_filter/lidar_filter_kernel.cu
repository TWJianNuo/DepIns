#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>


namespace {

template <typename scalar_t>
__global__ void init_lookupTable_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> lookTable,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> epp,
    const int w,
    const int h,
    const int h2,
    const int searchRange,
    const float max_depth,
    const float verRange,
    const float horRange) {

    const int linear_index_pos = blockIdx.x * blockDim.x + threadIdx.x;

    int cw;
    int ch;
    int xx;
    int yy;
    int count;

    float projx;
    float projy;
    float ratio;

    float curLen;
    for(int i = linear_index_pos; i < h2 * w; i = i + blockDim.x * gridDim.x){
        ch = linear_index_pos / w;
        cw = linear_index_pos - ch * w;
        count = 0;

        curLen = (epp[0] - cw) * (epp[0] - cw) + (epp[1] - ch) * (epp[1] - ch);
        for (int sxx = -searchRange; sxx <= searchRange; sxx++){
            for (int syy = -searchRange; syy <= searchRange; syy++){
                xx = sxx + cw;
                yy = syy + ch;
                if((xx > 0) && (yy > 0) && (xx < w) && (yy < h2) && (count < max_depth)){
                    if (!((sxx == 0) && (syy == 0))){

                        ratio = ((epp[0] - cw) * sxx + (epp[1] - ch) * syy) / curLen;
                        projx = ratio * (epp[0] - cw);
                        projy = ratio * (epp[1] - ch);

                        if((sqrt(projx * projx + projy * projy) < verRange) && (sqrt((sxx - projx) * (sxx - projx) + (syy - projy) * (syy - projy)) < horRange)){
                            if (((projx * (epp[0] - cw) + projy * (epp[1] - ch)) > 0) && ((projx * projx + projy * projy) < curLen)){
                                lookTable[ch][cw][count + 1][0] = xx;
                                lookTable[ch][cw][count + 1][1] = yy;
                                count = count + 1;
                            }
                        }

                    }
                }
            }
        }

        lookTable[ch][cw][0][0] = count;
        }

    }


template <typename scalar_t>
__global__ void lidar_denoise_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> nvelo_projected_img,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> lookTable,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> noocc_mask,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> epp,
    const int w,
    const int h2,
    const float mind1d2,
    const float maxd2) {


    const int linear_index_pos = blockIdx.x * blockDim.x + threadIdx.x;

    int cw;
    int ch;
    int xx;
    int yy;

    float distance1;
    float distance2;
    float refx;
    float refy;
    float lrefx;
    float lrefy;

    for(int i = linear_index_pos; i < h2 * w; i = i + blockDim.x * gridDim.x){
        ch = linear_index_pos / w;
        cw = linear_index_pos - ch * w;

        // if(nvelo_projected_img[ch][cw][2] > 0){
        if(noocc_mask[ch][cw] > 0){
            refx = nvelo_projected_img[ch][cw][3];
            refy = nvelo_projected_img[ch][cw][4];
            lrefx = nvelo_projected_img[ch][cw][0];
            lrefy = nvelo_projected_img[ch][cw][1];
            for(int j = 0; j < lookTable[ch][cw][0][0]; j++){
                if (noocc_mask[ch][cw] < 0.9){
                    break;
                }
                xx = lookTable[ch][cw][j+1][0];
                yy = lookTable[ch][cw][j+1][1];

                distance2 =((nvelo_projected_img[yy][xx][3] - refx) * (epp[0] - refx) + (nvelo_projected_img[yy][xx][4] - refy) * (epp[1] - refy)) / sqrt((epp[0] - refx)*(epp[0] - refx) + (epp[1] - refy)*(epp[1] - refy));
                distance1 =((nvelo_projected_img[yy][xx][0] - lrefx) * (epp[0] - lrefx) + (nvelo_projected_img[yy][xx][1] - lrefy) * (epp[1] - lrefy)) / sqrt((epp[0] - lrefx)*(epp[0] - lrefx) + (epp[1] - lrefy)*(epp[1] - lrefy));
                if((distance1 > 0) && (distance2 < 0) && ((distance1 - distance2) > mind1d2) && (abs(distance2) < maxd2)){
                // if((distance1 > 0) && (distance2 < 0)){
                    noocc_mask[ch][cw] = 0;
                }
                //if(nvelo_projected_img[yy][xx][2] > 0.1){
                //     mul1 = (nvelo_projected_img[yy][xx][3] - refx) * (epp[0] - refx) + (nvelo_projected_img[yy][xx][4] - refy) * (epp[1] - refy);
                //     mul2 = (nvelo_projected_img[yy][xx][3] - epp[0]) * (refx - epp[0]) + (nvelo_projected_img[yy][xx][4] - epp[1]) * (refy - epp[1]);
                //     if ((mul1 < 0) || (mul2 < 0)){
                //         mul1 = (nvelo_projected_img[yy][xx][0] - nvelo_projected_img[ch][cw][0]) * (epp[0] - nvelo_projected_img[ch][cw][0]) + (nvelo_projected_img[yy][xx][1] - nvelo_projected_img[ch][cw][1]) * (epp[1] - nvelo_projected_img[ch][cw][1]);
                //         mul2 = (nvelo_projected_img[yy][xx][0] - epp[0]) * (nvelo_projected_img[ch][cw][0] - epp[0]) + (nvelo_projected_img[yy][xx][1] - epp[1]) * (nvelo_projected_img[ch][cw][1] - epp[1]);
                //         if ((mul1 > 0) && (mul2 > 0)){
                //             noocc_mask[ch][cw] = 0;
                //         }
                //     }
                //}
            }
        }
    }

    }

} // namespace




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
    ) {
    const int threads = 1024;
    const dim3 blocks(((h2 * w) + threads - 1) / threads, threads);


    AT_DISPATCH_FLOATING_TYPES(lookTable.type(), "initialize look up table", ([&] {
    init_lookupTable_cuda_kernel<scalar_t><<<blocks, threads>>>(
        lookTable.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        epp.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        w,
        h,
        h2,
        searchRange,
        max_depth,
        verRange,
        horRange
        );
    }));


    return {lookTable};
}

std::vector<torch::Tensor> lidar_denoise_cuda(
    torch::Tensor nvelo_projected_img,
    torch::Tensor lookTable,
    torch::Tensor noocc_mask,
    torch::Tensor epp,
    int w,
    int h2,
    float mind1d2,
    float maxd2
    ) {
    const int threads = 1024;
    const dim3 blocks(((h2 * w) + threads - 1) / threads, threads);

    AT_DISPATCH_FLOATING_TYPES(nvelo_projected_img.type(), "denoise lidar scan", ([&] {
    lidar_denoise_cuda_kernel<scalar_t><<<blocks, threads>>>(
        nvelo_projected_img.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        lookTable.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
        noocc_mask.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        epp.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
        w,
        h2,
        mind1d2,
        maxd2
        );
    }));

    return {noocc_mask};
}
