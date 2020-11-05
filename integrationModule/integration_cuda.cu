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
                depthh += exp(-intpos) * depth_optedin[blockIdx.x][0][m][sn] / (countpos + countneg);
                intpos -= log[blockIdx.x][0][m][sn-1];
            }

            for(int ii = countneg; ii > 0; ii--){
                sn = n - ii;
                depthh += exp(-intneg) * depth_optedin[blockIdx.x][0][m][sn] / (countpos + countneg);
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
                depthv += exp(-intpos) * depth_optedin[blockIdx.x][0][sm][n] / (countpos + countneg);
                intpos -= log[blockIdx.x][1][sm-1][n];
            }

            for(int ii = countneg; ii > 0; ii--){
                sm = m - ii;
                depthv += exp(-intneg) * depth_optedin[blockIdx.x][0][sm][n] / (countpos + countneg);
                intneg += log[blockIdx.x][1][sm][n];
            }

            countv = countv + countpos + countneg;
        }

        if(counth + countv > 0){
            depth_optedout[blockIdx.x][0][m][n] = lambda * depthin[blockIdx.x][0][m][n] + (1 - lambda) * (depthh * counth / (counth + countv) + depthv * countv / (counth + countv));
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
        }
    }
    return;

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

void shapeIntegration_crf_constrain_forward_cuda(
    torch::Tensor log,
    torch::Tensor semantics,
    torch::Tensor mask,
    torch::Tensor depthin,
    torch::Tensor constrainout,
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
            height,
            width,
            bs
            );
    return;
    }