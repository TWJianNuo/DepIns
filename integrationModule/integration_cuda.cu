#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <math_constants.h>

namespace {

}

__global__ void shapeIntegration_crf_forward_cuda_kernel(
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> log,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> semantics,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> mask,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depthin,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depth_optedin,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depth_optedout,
    const int height,
    const int width,
    const int bs,
    const float lambda
    ) {

    int m;
    int n;

    int sm;
    int sn;

    float totcounts;
    float lateralre;

    float intlog;

    int semancat;

    for(int i = threadIdx.x; i < height * width; i = i + blockDim.x){
        m = i / width;
        n = i - m * width;

        semancat = semantics[blockIdx.x][0][m][n];
        totcounts = 0;
        lateralre = 0;

        if(mask[blockIdx.x][0][m][n] == 1){
            // Left
            sm = m;
            intlog = 0;
            for(sn = n-1; sn >= 0; sn--){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                intlog += -log[blockIdx.x][0][sm][sn];
                totcounts += 1;
                lateralre += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn];
            }

            // Right
            sm = m;
            intlog = 0;
            for(sn = n+1; sn < width; sn++){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                intlog += log[blockIdx.x][0][sm][sn-1];
                totcounts += 1;
                lateralre += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn];
            }

            // Up
            sn = n;
            intlog = 0;
            for(sm = m-1; sm >= 0; sm--){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                intlog += -log[blockIdx.x][1][sm][sn];
                totcounts += 1;
                lateralre += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn];
            }

            // Down
            sn = n;
            intlog = 0;
            for(sm = m+1; sm < height; sm++){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                intlog += log[blockIdx.x][1][sm-1][sn];
                totcounts += 1;
                lateralre += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn];
            }
        }

        if(totcounts > 0){
            depth_optedout[blockIdx.x][0][m][n] = lambda * depthin[blockIdx.x][0][m][n] + (1 - lambda) * lateralre / totcounts;
        }
        else{
            depth_optedout[blockIdx.x][0][m][n] = depthin[blockIdx.x][0][m][n];
        }
    }
    return;

    }

__global__ void shapeIntegration_crf_variance_forward_cuda_kernel(
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> log,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> semantics,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> mask,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> variance,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depth_optedin,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depth_optedout,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> summedconfidence,
    const int height,
    const int width,
    const int bs,
    const float clipvariance,
    const float maxrange
    ) {

    int m;
    int n;

    int sm;
    int sn;

    float intlog;
    float intvariance;
    float intexpvariance;

    int semancat;

    for(int i = threadIdx.x; i < height * width; i = i + blockDim.x){
        m = i / width;
        n = i - m * width;

        semancat = semantics[blockIdx.x][0][m][n];

        if(mask[blockIdx.x][0][m][n] == 1){
            intexpvariance = 0;

            // Left
            sm = m;
            intvariance = 0;
            for(sn = n-1; sn >= 0; sn--){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(n - sn > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intvariance += variance[blockIdx.x][0][sm][sn];
                intexpvariance += exp(-intvariance);
            }

            // Right
            sm = m;
            intvariance = 0;
            for(sn = n+1; sn < width; sn++){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(sn - n > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intvariance += variance[blockIdx.x][0][sm][sn];
                intexpvariance += exp(-intvariance);
            }

            // Up
            sn = n;
            intvariance = 0;
            for(sm = m-1; sm >= 0; sm--){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(m - sm > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intvariance += variance[blockIdx.x][0][sm][sn];
                intexpvariance += exp(-intvariance);
            }

            // Down
            sn = n;
            intvariance = 0;
            for(sm = m+1; sm < height; sm++){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(sm - m > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intvariance += variance[blockIdx.x][0][sm][sn];
                intexpvariance += exp(-intvariance);
            }

            summedconfidence[blockIdx.x][0][m][n] = intexpvariance;

            // Left
            sm = m;
            intlog = 0;
            intvariance = 0;
            for(sn = n-1; sn >= 0; sn--){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(n - sn > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += -log[blockIdx.x][0][sm][sn];
                intvariance += variance[blockIdx.x][0][sm][sn];
                depth_optedout[blockIdx.x][0][m][n] += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance) / intexpvariance;
            }

            // Right
            sm = m;
            intlog = 0;
            intvariance = 0;
            for(sn = n+1; sn < width; sn++){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(sn - n > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += log[blockIdx.x][0][sm][sn-1];
                intvariance += variance[blockIdx.x][0][sm][sn];
                depth_optedout[blockIdx.x][0][m][n] += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance) / intexpvariance;
            }

            // Up
            sn = n;
            intlog = 0;
            intvariance = 0;
            for(sm = m-1; sm >= 0; sm--){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(m - sm > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += -log[blockIdx.x][1][sm][sn];
                intvariance += variance[blockIdx.x][0][sm][sn];
                depth_optedout[blockIdx.x][0][m][n] += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance) / intexpvariance;
            }

            // Down
            sn = n;
            intlog = 0;
            intvariance = 0;
            for(sm = m+1; sm < height; sm++){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(sm - m > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += log[blockIdx.x][1][sm-1][sn];
                intvariance += variance[blockIdx.x][0][sm][sn];
                depth_optedout[blockIdx.x][0][m][n] += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance) / intexpvariance;
            }
        }

    }
    return;

    }

__global__ void shapeIntegration_crf_variance_backward_cuda_kernel(
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> log,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> semantics,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> mask,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> variance,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depth_optedin,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depth_optedout,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> summedconfidence,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> grad_depthin,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> grad_varianceout,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> grad_depthout,
    const int height,
    const int width,
    const int bs,
    const float clipvariance,
    const float maxrange
    ) {

    int m;
    int n;

    int sm;
    int sn;

    float intlog;
    float intvariance;
    float intexpvariance;
    float lateralre;

    float sdirintexpvariance;
    float sdirintweighteddepth;
    float cursdirintexpvariance;
    float cursdirweighteddepth;

    int semancat;

    for(int i = threadIdx.x; i < height * width; i = i + blockDim.x){
        m = i / width;
        n = i - m * width;

        semancat = semantics[blockIdx.x][0][m][n];

        if(mask[blockIdx.x][0][m][n] == 1){
            intexpvariance = summedconfidence[blockIdx.x][0][m][n];
            lateralre = depth_optedout[blockIdx.x][0][m][n];

            // Left
            sm = m;
            intlog = 0;
            intvariance = 0;
            sdirintexpvariance = 0;
            sdirintweighteddepth = 0;
            for(sn = n-1; sn >= 0; sn--){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(n - sn > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += -log[blockIdx.x][0][sm][sn];
                intvariance += variance[blockIdx.x][0][sm][sn];
                sdirintexpvariance += exp(-intvariance);
                sdirintweighteddepth += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance);
            }
            intlog = 0;
            intvariance = 0;
            cursdirintexpvariance = 0;
            cursdirweighteddepth = 0;
            for(sn = n-1; sn >= 0; sn--){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(n - sn > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += -log[blockIdx.x][0][sm][sn];
                intvariance += variance[blockIdx.x][0][sm][sn];

                atomicAdd((float*)&grad_varianceout[blockIdx.x][0][sm][sn], grad_depthin[blockIdx.x][0][m][n] * ((sdirintexpvariance - cursdirintexpvariance) / intexpvariance * lateralre - (sdirintweighteddepth - cursdirweighteddepth) / intexpvariance));
                cursdirintexpvariance += exp(-intvariance);
                cursdirweighteddepth += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance);

                atomicAdd((float*)&grad_depthout[blockIdx.x][0][sm][sn], grad_depthin[blockIdx.x][0][m][n] * exp(-intlog) * exp(-intvariance) / intexpvariance);
            }

            // Right
            sm = m;
            intlog = 0;
            intvariance = 0;
            sdirintexpvariance = 0;
            sdirintweighteddepth = 0;
            for(sn = n+1; sn < width; sn++){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(sn - n > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += log[blockIdx.x][0][sm][sn-1];
                intvariance += variance[blockIdx.x][0][sm][sn];
                sdirintexpvariance += exp(-intvariance);
                sdirintweighteddepth += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance);
            }
            intlog = 0;
            intvariance = 0;
            cursdirintexpvariance = 0;
            cursdirweighteddepth = 0;
            for(sn = n+1; sn < width; sn++){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(sn - n > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += log[blockIdx.x][0][sm][sn-1];
                intvariance += variance[blockIdx.x][0][sm][sn];

                atomicAdd((float*)&grad_varianceout[blockIdx.x][0][sm][sn], grad_depthin[blockIdx.x][0][m][n] * ((sdirintexpvariance - cursdirintexpvariance) / intexpvariance * lateralre - (sdirintweighteddepth - cursdirweighteddepth) / intexpvariance));
                cursdirintexpvariance += exp(-intvariance);
                cursdirweighteddepth += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance);

                atomicAdd((float*)&grad_depthout[blockIdx.x][0][sm][sn], grad_depthin[blockIdx.x][0][m][n] * exp(-intlog) * exp(-intvariance) / intexpvariance);
            }

            // Up
            sn = n;
            intlog = 0;
            intvariance = 0;
            sdirintexpvariance = 0;
            sdirintweighteddepth = 0;
            for(sm = m-1; sm >= 0; sm--){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(m - sm > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += -log[blockIdx.x][1][sm][sn];
                intvariance += variance[blockIdx.x][0][sm][sn];
                sdirintexpvariance += exp(-intvariance);
                sdirintweighteddepth += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance);
            }
            intlog = 0;
            intvariance = 0;
            cursdirintexpvariance = 0;
            cursdirweighteddepth = 0;
            for(sm = m-1; sm >= 0; sm--){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(m - sm > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += -log[blockIdx.x][1][sm][sn];
                intvariance += variance[blockIdx.x][0][sm][sn];

                atomicAdd((float*)&grad_varianceout[blockIdx.x][0][sm][sn], grad_depthin[blockIdx.x][0][m][n] * ((sdirintexpvariance - cursdirintexpvariance) / intexpvariance * lateralre - (sdirintweighteddepth - cursdirweighteddepth) / intexpvariance));
                cursdirintexpvariance += exp(-intvariance);
                cursdirweighteddepth += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance);

                atomicAdd((float*)&grad_depthout[blockIdx.x][0][sm][sn], grad_depthin[blockIdx.x][0][m][n] * exp(-intlog) * exp(-intvariance) / intexpvariance);
            }

            // Down
            sn = n;
            intlog = 0;
            intvariance = 0;
            sdirintexpvariance = 0;
            sdirintweighteddepth = 0;
            for(sm = m+1; sm < height; sm++){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(sm - m > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += log[blockIdx.x][1][sm-1][sn];
                intvariance += variance[blockIdx.x][0][sm][sn];
                sdirintexpvariance += exp(-intvariance);
                sdirintweighteddepth += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance);
            }
            intlog = 0;
            intvariance = 0;
            cursdirintexpvariance = 0;
            cursdirweighteddepth = 0;
            for(sm = m+1; sm < height; sm++){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                if(sm - m > maxrange){break;}
                if(intvariance + variance[blockIdx.x][0][sm][sn] > clipvariance){break;}
                intlog += log[blockIdx.x][1][sm-1][sn];
                intvariance += variance[blockIdx.x][0][sm][sn];

                atomicAdd((float*)&grad_varianceout[blockIdx.x][0][sm][sn], grad_depthin[blockIdx.x][0][m][n] * ((sdirintexpvariance - cursdirintexpvariance) / intexpvariance * lateralre - (sdirintweighteddepth - cursdirweighteddepth) / intexpvariance));
                cursdirintexpvariance += exp(-intvariance);
                cursdirweighteddepth += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn] * exp(-intvariance);

                atomicAdd((float*)&grad_depthout[blockIdx.x][0][sm][sn], grad_depthin[blockIdx.x][0][m][n] * exp(-intlog) * exp(-intvariance) / intexpvariance);
            }
        }

    }
    return;

    }

__global__ void shapeIntegration_crf_star_forward_cuda_kernel(
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> log,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> semantics,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> mask,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depthin,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depth_optedin,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depth_optedout,
    const int height,
    const int width,
    const int bs,
    const float lambda
    ) {

    int m;
    int n;

    int sm;
    int sn;

    float totcounts;
    float lateralre;

    float intlog;

    int semancat;

    for(int i = threadIdx.x; i < height * width; i = i + blockDim.x){
        m = i / width;
        n = i - m * width;

        semancat = semantics[blockIdx.x][0][m][n];
        totcounts = 0;
        lateralre = 0;

        if(mask[blockIdx.x][0][m][n] == 1){

            // Left up direction
            sn = n;
            sm = m;
            intlog = 0;
            while(true){
                sn -= 1;
                sm -= 1;
                if(sn >=0 && sm >= 0 && sn < width && sm < height){
                    if((semantics[blockIdx.x][0][sm][sn] != semancat) || (semantics[blockIdx.x][0][sm+1][sn] != semancat) || (semantics[blockIdx.x][0][sm][sn+1] != semancat)){break;}
                    if((mask[blockIdx.x][0][sm][sn] != 1) || (mask[blockIdx.x][0][sm+1][sn] != 1) || (mask[blockIdx.x][0][sm][sn+1] != 1)){break;}
                    intlog += (-log[blockIdx.x][0][sm+1][sn] -log[blockIdx.x][1][sm][sn] -log[blockIdx.x][1][sm][sn+1] -log[blockIdx.x][0][sm][sn]) / 2;
                    totcounts += 1;
                    lateralre += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn];
                }
                else{break;}
            }

            // right up direction
            sn = n;
            sm = m;
            intlog = 0;
            while(true){
                sn += 1;
                sm -= 1;
                if(sn >=0 && sm >= 0 && sn < width && sm < height){
                    if((semantics[blockIdx.x][0][sm][sn] != semancat) || (semantics[blockIdx.x][0][sm+1][sn] != semancat) || (semantics[blockIdx.x][0][sm][sn-1] != semancat)){break;}
                    if((mask[blockIdx.x][0][sm][sn] != 1) || (mask[blockIdx.x][0][sm+1][sn] != 1) || (mask[blockIdx.x][0][sm][sn-1] != 1)){break;}
                    intlog += (log[blockIdx.x][0][sm+1][sn-1] -log[blockIdx.x][1][sm][sn] -log[blockIdx.x][1][sm][sn-1] +log[blockIdx.x][0][sm][sn-1]) / 2;
                    totcounts += 1;
                    lateralre += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn];
                }
                else{break;}
            }

            // Left down direction
            sn = n;
            sm = m;
            intlog = 0;
            while(true){
                sn -= 1;
                sm += 1;
                if(sn >=0 && sm >= 0 && sn < width && sm < height){
                    if((semantics[blockIdx.x][0][sm][sn] != semancat) || (semantics[blockIdx.x][0][sm-1][sn] != semancat) || (semantics[blockIdx.x][0][sm][sn+1] != semancat)){break;}
                    if((mask[blockIdx.x][0][sm][sn] != 1) || (mask[blockIdx.x][0][sm-1][sn] != 1) || (mask[blockIdx.x][0][sm][sn+1] != 1)){break;}
                    intlog += (-log[blockIdx.x][0][sm-1][sn] +log[blockIdx.x][1][sm-1][sn] +log[blockIdx.x][1][sm-1][sn+1] -log[blockIdx.x][0][sm][sn]) / 2;
                    totcounts += 1;
                    lateralre += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn];
                }
                else{break;}
            }

            // Right down direction
            sn = n;
            sm = m;
            intlog = 0;
            while(true){
                sn += 1;
                sm += 1;
                if(sn >=0 && sm >= 0 && sn < width && sm < height){
                    if((semantics[blockIdx.x][0][sm][sn] != semancat) || (semantics[blockIdx.x][0][sm][sn-1] != semancat) || (semantics[blockIdx.x][0][sm-1][sn] != semancat)){break;}
                    if((mask[blockIdx.x][0][sm][sn] != 1) || (mask[blockIdx.x][0][sm][sn-1] != 1) || (mask[blockIdx.x][0][sm-1][sn] != 1)){break;}
                    intlog += (log[blockIdx.x][0][sm-1][sn-1] +log[blockIdx.x][1][sm-1][sn] +log[blockIdx.x][1][sm-1][sn-1] +log[blockIdx.x][0][sm][sn-1]) / 2;
                    totcounts += 1;
                    lateralre += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn];
                }
                else{break;}
            }

            // Left
            sm = m;
            intlog = 0;
            for(sn = n-1; sn >= 0; sn--){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                intlog += -log[blockIdx.x][0][sm][sn];
                totcounts += 1;
                lateralre += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn];
            }

            // Right
            sm = m;
            intlog = 0;
            for(sn = n+1; sn < width; sn++){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                intlog += log[blockIdx.x][0][sm][sn-1];
                totcounts += 1;
                lateralre += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn];
            }

            // Up
            sn = n;
            intlog = 0;
            for(sm = m-1; sm >= 0; sm--){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                intlog += -log[blockIdx.x][1][sm][sn];
                totcounts += 1;
                lateralre += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn];
            }

            // Down
            sn = n;
            intlog = 0;
            for(sm = m+1; sm < height; sm++){
                if(semantics[blockIdx.x][0][sm][sn] != semancat){break;}
                if(mask[blockIdx.x][0][sm][sn] != 1){break;}
                intlog += log[blockIdx.x][1][sm-1][sn];
                totcounts += 1;
                lateralre += exp(-intlog) * depth_optedin[blockIdx.x][0][sm][sn];
            }
        }

        if(totcounts > 0){
            depth_optedout[blockIdx.x][0][m][n] = lambda * depthin[blockIdx.x][0][m][n] + (1 - lambda) * lateralre / totcounts;
        }
        else{
            depth_optedout[blockIdx.x][0][m][n] = depthin[blockIdx.x][0][m][n];
        }
    }
    return;

    }

__global__ void shapeIntegration_crf_constrain_forward_cuda_kernel(
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> log,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> semantics,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> mask,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depthin,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> constrainout,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> counts,
    const int height,
    const int width,
    const int bs
    ) {

    int m;
    int n;

    int sm;
    int sn;

    bool breakpos;
    bool breakneg;

    int countpos;
    int countneg;

    int counth;
    int countv;

    float intpos;
    float intneg;

    int semancat;

    float depthh;
    float depthv;

    for(int i = threadIdx.x; i < height * width; i = i + blockDim.x){
        m = i / width;
        n = i - m * width;

        semancat = semantics[blockIdx.x][0][m][n];
        depthh = 0;
        depthv = 0;

        counth = 0;
        countv = 0;

        if(mask[blockIdx.x][0][m][n] == 1){
            breakpos = false;
            breakneg = false;
            countpos = 0;
            countneg = 0;
            intpos = 0;
            intneg = 0;

            for(int ii = 0; ii < width * 2; ii++){
                if(breakpos && breakneg){break;}
                if(!(ii&1)){
                    if(breakpos){continue;}
                    sn = n + ii / 2 + 1;
                    if(sn>=width){breakpos=true;continue;}
                    else if(semancat != semantics[blockIdx.x][0][m][sn]){breakpos=true;continue;}
                    else if(mask[blockIdx.x][0][m][sn] == 0){breakpos=true;continue;}
                    else{
                    intpos += log[blockIdx.x][0][m][sn-1];
                    countpos++;
                    }
                }
                else{
                    if(breakneg){continue;}
                    sn = n - ii / 2 - 1;
                    if(sn<0){breakneg=true;continue;}
                    else if(semancat != semantics[blockIdx.x][0][m][sn]){breakneg=true;continue;}
                    else if(mask[blockIdx.x][0][m][sn] == 0){breakneg=true;continue;}
                    else{
                    intneg -= log[blockIdx.x][0][m][sn];
                    countneg++;
                    }
                }
            }

            for(int ii = countpos; ii > 0; ii--){
                sn = n + ii;
                depthh += abs(exp(-intpos) * depthin[blockIdx.x][0][m][sn] - depthin[blockIdx.x][0][m][n]) / (countpos + countneg);
                intpos -= log[blockIdx.x][0][m][sn-1];
            }

            for(int ii = countneg; ii > 0; ii--){
                sn = n - ii;
                depthh += abs(exp(-intneg) * depthin[blockIdx.x][0][m][sn] - depthin[blockIdx.x][0][m][n]) / (countpos + countneg);
                intneg += log[blockIdx.x][0][m][sn];
            }

            counth = counth + countpos + countneg;
        }

        if(mask[blockIdx.x][0][m][n] == 1){
            breakpos = false;
            breakneg = false;
            countpos = 0;
            countneg = 0;
            intpos = 0;
            intneg = 0;

            for(int ii = 0; ii < height * 2; ii++){
                if(breakpos && breakneg){break;}
                if(!(ii&1)){
                    if(breakpos){continue;}
                    sm = m + ii / 2 + 1;
                    if(sm>=height){breakpos=true;continue;}
                    else if(semancat != semantics[blockIdx.x][0][sm][n]){breakpos=true;continue;}
                    else if(mask[blockIdx.x][0][sm][n] == 0){breakpos=true;continue;}
                    else{
                    intpos += log[blockIdx.x][1][sm-1][n];
                    countpos++;
                    }
                }
                else{
                    if(breakneg){continue;}
                    sm = m - ii / 2 - 1;
                    if(sm<0){breakneg=true;continue;}
                    else if(semancat != semantics[blockIdx.x][0][sm][n]){breakneg=true;continue;}
                    else if(mask[blockIdx.x][0][sm][n] == 0){breakneg=true;continue;}
                    else{
                    intneg -= log[blockIdx.x][1][sm][n];
                    countneg++;
                    }
                }
            }

            for(int ii = countpos; ii > 0; ii--){
                sm = m + ii;
                depthv += abs(exp(-intpos) * depthin[blockIdx.x][0][sm][n] - depthin[blockIdx.x][0][m][n]) / (countpos + countneg);
                intpos -= log[blockIdx.x][1][sm-1][n];
            }

            for(int ii = countneg; ii > 0; ii--){
                sm = m - ii;
                depthv += abs(exp(-intneg) * depthin[blockIdx.x][0][sm][n] - depthin[blockIdx.x][0][m][n]) / (countpos + countneg);
                intneg += log[blockIdx.x][1][sm][n];
            }

            countv = countv + countpos + countneg;
        }

        if(counth + countv > 0){
            constrainout[blockIdx.x][0][m][n] = (depthh * counth / (counth + countv) + depthv * countv / (counth + countv));
            counts[blockIdx.x][0][m][n] = counth + countv;
        }
    }
    return;

    }

__global__ void shapeIntegration_crf_constrain_backward_cuda_kernel(
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> log,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> semantics,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> mask,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depthin,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> counts,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> constraingradin,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> constraingradout,
    const int height,
    const int width,
    const int bs
    ) {

    int m;
    int n;

    int sm;
    int sn;

    bool breakpos;
    bool breakneg;

    int countpos;
    int countneg;

    float intpos;
    float intneg;

    int semancat;

    float depthh;
    float depthv;

    for(int i = threadIdx.x; i < height * width; i = i + blockDim.x){
        m = i / width;
        n = i - m * width;

        semancat = semantics[blockIdx.x][0][m][n];
        depthh = 0;
        depthv = 0;

        if(mask[blockIdx.x][0][m][n] == 1){
            breakpos = false;
            breakneg = false;
            countpos = 0;
            countneg = 0;
            intpos = 0;
            intneg = 0;

            for(int ii = 0; ii < width * 2; ii++){
                if(breakpos && breakneg){break;}
                if(!(ii&1)){
                    if(breakpos){continue;}
                    sn = n + ii / 2 + 1;
                    if(sn>=width){breakpos=true;continue;}
                    else if(semancat != semantics[blockIdx.x][0][m][sn]){breakpos=true;continue;}
                    else if(mask[blockIdx.x][0][m][sn] == 0){breakpos=true;continue;}
                    else{
                    intpos += log[blockIdx.x][0][m][sn-1];
                    countpos++;
                    }
                }
                else{
                    if(breakneg){continue;}
                    sn = n - ii / 2 - 1;
                    if(sn<0){breakneg=true;continue;}
                    else if(semancat != semantics[blockIdx.x][0][m][sn]){breakneg=true;continue;}
                    else if(mask[blockIdx.x][0][m][sn] == 0){breakneg=true;continue;}
                    else{
                    intneg -= log[blockIdx.x][0][m][sn];
                    countneg++;
                    }
                }
            }

            for(int ii = countpos; ii > 0; ii--){
                sn = n + ii;
                depthh += abs(exp(-intpos) * depthin[blockIdx.x][0][m][sn] - depthin[blockIdx.x][0][m][n]) / (countpos + countneg);
                if(exp(-intpos) * depthin[blockIdx.x][0][m][sn] >= depthin[blockIdx.x][0][m][n]){
                    atomicAdd((float*)&constraingradout[blockIdx.x][0][m][sn], constraingradin[blockIdx.x][0][m][n] * exp(-intpos) / counts[blockIdx.x][0][m][n]);
                    atomicAdd((float*)&constraingradout[blockIdx.x][0][m][n], -constraingradin[blockIdx.x][0][m][n] / counts[blockIdx.x][0][m][n]);
                }
                else{
                    atomicAdd((float*)&constraingradout[blockIdx.x][0][m][sn], -constraingradin[blockIdx.x][0][m][n] * exp(-intpos) / counts[blockIdx.x][0][m][n]);
                    atomicAdd((float*)&constraingradout[blockIdx.x][0][m][n], constraingradin[blockIdx.x][0][m][n] / counts[blockIdx.x][0][m][n]);
                }

                intpos -= log[blockIdx.x][0][m][sn-1];
            }

            for(int ii = countneg; ii > 0; ii--){
                sn = n - ii;
                depthh += abs(exp(-intneg) * depthin[blockIdx.x][0][m][sn] - depthin[blockIdx.x][0][m][n]) / (countpos + countneg);
                if(exp(-intneg) * depthin[blockIdx.x][0][m][sn] >= depthin[blockIdx.x][0][m][n]){
                    atomicAdd((float*)&constraingradout[blockIdx.x][0][m][sn], constraingradin[blockIdx.x][0][m][n] * exp(-intneg) / counts[blockIdx.x][0][m][n]);
                    atomicAdd((float*)&constraingradout[blockIdx.x][0][m][n], -constraingradin[blockIdx.x][0][m][n] / counts[blockIdx.x][0][m][n]);
                }
                else{
                    atomicAdd((float*)&constraingradout[blockIdx.x][0][m][sn], -constraingradin[blockIdx.x][0][m][n] * exp(-intneg) / counts[blockIdx.x][0][m][n]);
                    atomicAdd((float*)&constraingradout[blockIdx.x][0][m][n], constraingradin[blockIdx.x][0][m][n] / counts[blockIdx.x][0][m][n]);
                }

                intneg += log[blockIdx.x][0][m][sn];
            }
        }

        if(mask[blockIdx.x][0][m][n] == 1){
            breakpos = false;
            breakneg = false;
            countpos = 0;
            countneg = 0;
            intpos = 0;
            intneg = 0;

            for(int ii = 0; ii < height * 2; ii++){
                if(breakpos && breakneg){break;}
                if(!(ii&1)){
                    if(breakpos){continue;}
                    sm = m + ii / 2 + 1;
                    if(sm>=height){breakpos=true;continue;}
                    else if(semancat != semantics[blockIdx.x][0][sm][n]){breakpos=true;continue;}
                    else if(mask[blockIdx.x][0][sm][n] == 0){breakpos=true;continue;}
                    else{
                    intpos += log[blockIdx.x][1][sm-1][n];
                    countpos++;
                    }
                }
                else{
                    if(breakneg){continue;}
                    sm = m - ii / 2 - 1;
                    if(sm<0){breakneg=true;continue;}
                    else if(semancat != semantics[blockIdx.x][0][sm][n]){breakneg=true;continue;}
                    else if(mask[blockIdx.x][0][sm][n] == 0){breakneg=true;continue;}
                    else{
                    intneg -= log[blockIdx.x][1][sm][n];
                    countneg++;
                    }
                }
            }

            for(int ii = countpos; ii > 0; ii--){
                sm = m + ii;
                depthv += abs(exp(-intpos) * depthin[blockIdx.x][0][sm][n] - depthin[blockIdx.x][0][m][n]) / (countpos + countneg);
                if(exp(-intpos) * depthin[blockIdx.x][0][sm][n] >= depthin[blockIdx.x][0][m][n]){
                    atomicAdd((float*)&constraingradout[blockIdx.x][0][sm][n], constraingradin[blockIdx.x][0][m][n] * exp(-intpos) / counts[blockIdx.x][0][m][n]);
                    atomicAdd((float*)&constraingradout[blockIdx.x][0][m][n], -constraingradin[blockIdx.x][0][m][n] / counts[blockIdx.x][0][m][n]);
                }
                else{
                    atomicAdd((float*)&constraingradout[blockIdx.x][0][sm][n], -constraingradin[blockIdx.x][0][m][n] * exp(-intpos) / counts[blockIdx.x][0][m][n]);
                    atomicAdd((float*)&constraingradout[blockIdx.x][0][m][n], constraingradin[blockIdx.x][0][m][n] / counts[blockIdx.x][0][m][n]);
                }

                intpos -= log[blockIdx.x][1][sm-1][n];
            }

            for(int ii = countneg; ii > 0; ii--){
                sm = m - ii;
                depthv += abs(exp(-intneg) * depthin[blockIdx.x][0][sm][n] - depthin[blockIdx.x][0][m][n]) / (countpos + countneg);
                if(exp(-intneg) * depthin[blockIdx.x][0][sm][n] >= depthin[blockIdx.x][0][m][n]){
                    atomicAdd((float*)&constraingradout[blockIdx.x][0][sm][n], constraingradin[blockIdx.x][0][m][n] * exp(-intneg) / counts[blockIdx.x][0][m][n]);
                    atomicAdd((float*)&constraingradout[blockIdx.x][0][m][n], -constraingradin[blockIdx.x][0][m][n] / counts[blockIdx.x][0][m][n]);
                }
                else{
                    atomicAdd((float*)&constraingradout[blockIdx.x][0][sm][n], -constraingradin[blockIdx.x][0][m][n] * exp(-intneg) / counts[blockIdx.x][0][m][n]);
                    atomicAdd((float*)&constraingradout[blockIdx.x][0][m][n], constraingradin[blockIdx.x][0][m][n] / counts[blockIdx.x][0][m][n]);
                }
                intneg += log[blockIdx.x][1][sm][n];
            }
        }
    }
    }

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
    ){
      const int threads = 512;

      shapeIntegration_crf_forward_cuda_kernel<<<bs, threads>>>(
            log.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            semantics.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            mask.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            depthin.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            depth_optedin.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            depth_optedout.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            height,
            width,
            bs,
            lambda
            );
    return;
    }

void shapeIntegration_crf_variance_forward_cuda(
    torch::Tensor log,
    torch::Tensor semantics,
    torch::Tensor mask,
    torch::Tensor variance,
    torch::Tensor depth_optedin,
    torch::Tensor depth_optedout,
    torch::Tensor summedconfidence,
    int height,
    int width,
    int bs,
    float clipvariance,
    float maxrange
    ){
      const int threads = 512;

      shapeIntegration_crf_variance_forward_cuda_kernel<<<bs, threads>>>(
            log.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            semantics.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            mask.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            variance.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            depth_optedin.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            depth_optedout.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            summedconfidence.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            height,
            width,
            bs,
            clipvariance,
            maxrange
            );

    return;
    }

void shapeIntegration_crf_variance_backward_cuda(
    torch::Tensor log,
    torch::Tensor semantics,
    torch::Tensor mask,
    torch::Tensor variance,
    torch::Tensor depth_optedin,
    torch::Tensor depth_optedout,
    torch::Tensor summedconfidence,
    torch::Tensor grad_depthin,
    torch::Tensor grad_varianceout,
    torch::Tensor grad_depthout,
    int height,
    int width,
    int bs,
    float clipvariance,
    float maxrange
    ){
      const int threads = 512;

      shapeIntegration_crf_variance_backward_cuda_kernel<<<bs, threads>>>(
            log.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            semantics.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            mask.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            variance.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            depth_optedin.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            depth_optedout.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            summedconfidence.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            grad_depthin.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            grad_varianceout.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            grad_depthout.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            height,
            width,
            bs,
            clipvariance,
            maxrange
            );

    return;
    }

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
    ){
      const int threads = 512;

      shapeIntegration_crf_star_forward_cuda_kernel<<<bs, threads>>>(
            log.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            semantics.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            mask.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            depthin.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            depth_optedin.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            depth_optedout.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            height,
            width,
            bs,
            lambda
            );
    return;
    }

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
    ){
      const int threads = 512;

      shapeIntegration_crf_constrain_forward_cuda_kernel<<<bs, threads>>>(
            log.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            semantics.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            mask.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            depthin.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            constrainout.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            counts.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            height,
            width,
            bs
            );
    return;
    }

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
    ){
      const int threads = 512;

      shapeIntegration_crf_constrain_backward_cuda_kernel<<<bs, threads>>>(
            log.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            semantics.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            mask.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            depthin.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            counts.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            constraingradin.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            constraingradout.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            height,
            width,
            bs
            );
    return;
    }

