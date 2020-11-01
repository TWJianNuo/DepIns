#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <math_constants.h>

namespace {

}

__global__ void shapeIntegration_forward_cuda(
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> ang,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> log,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> confidence,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> semantics,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> mask,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depthin,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depthout,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> summedConfidence,
    const torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> varbar,
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

    float intconfidenceh;
    float intconfidencev;

    float sums;
    float sumsquare;

    int semancat;

    float depthh;
    float depthv;

    for(int i = threadIdx.x; i < height * width; i = i + blockDim.x){
        m = i / width;
        n = i - m * width;

        semancat = semantics[blockIdx.x][0][m][n];
        intconfidenceh = confidence[blockIdx.x][0][m][n];
        intconfidencev = confidence[blockIdx.x][0][m][n];
        depthh = 0;
        depthv = 0;


        if(varbar[semancat][0] > 0){
            breakpos = false;
            breakneg = false;
            countpos = 0;
            countneg = 0;
            intpos = 0;
            intneg = 0;

            sums = ang[blockIdx.x][0][m][n];
            sumsquare = ang[blockIdx.x][0][m][n] * ang[blockIdx.x][0][m][n];

            for(int ii = 0; ii < width * 2; ii++){
                if(breakpos && breakneg){break;}
                if(!(ii&1)){
                    if(breakpos){continue;}
                    sn = n + ii / 2 + 1;
                    if(sn>=width){breakpos=true;continue;}
                    else if(semancat != semantics[blockIdx.x][0][m][sn]){breakpos=true;continue;}
                    else if(mask[blockIdx.x][0][m][sn] == 0){breakpos=true;continue;}
                    else if((sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1)) > varbar[semancat][0]){breakpos=true;continue;}
                    else{
                    intpos += log[blockIdx.x][0][m][sn-1];
                    intconfidenceh += confidence[blockIdx.x][0][m][sn];
                    sums += ang[blockIdx.x][0][m][sn];
                    sumsquare += ang[blockIdx.x][0][m][sn] * ang[blockIdx.x][0][m][sn];
                    countpos++;
                    }
                }
                else{
                    if(breakneg){continue;}
                    sn = n - ii / 2 - 1;
                    if(sn<0){breakneg=true;continue;}
                    else if(semancat != semantics[blockIdx.x][0][m][sn]){breakneg=true;continue;}
                    else if(mask[blockIdx.x][0][m][sn] == 0){breakneg=true;continue;}
                    else if((sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1)) > varbar[semancat][0]){breakneg=true;continue;}
                    else{
                    intneg -= log[blockIdx.x][0][m][sn];
                    intconfidenceh += confidence[blockIdx.x][0][m][sn];
                    sums += ang[blockIdx.x][0][m][sn];
                    sumsquare += ang[blockIdx.x][0][m][sn] * ang[blockIdx.x][0][m][sn];
                    countneg++;
                    }
                }
            }

            for(int ii = countpos; ii > 0; ii--){
                sn = n + ii;
                depthh += confidence[blockIdx.x][0][m][sn] / intconfidenceh * exp(-intpos) * depthin[blockIdx.x][0][m][sn];
                intpos -= log[blockIdx.x][0][m][sn-1];
            }

            for(int ii = countneg; ii > 0; ii--){
                sn = n - ii;
                depthh += confidence[blockIdx.x][0][m][sn] / intconfidenceh * exp(-intneg) * depthin[blockIdx.x][0][m][sn];
                intneg += log[blockIdx.x][0][m][sn];
            }
        }

        if(varbar[semancat][1] > 0){
            breakpos = false;
            breakneg = false;
            countpos = 0;
            countneg = 0;
            intpos = 0;
            intneg = 0;

            sums = ang[blockIdx.x][1][m][n];
            sumsquare = ang[blockIdx.x][1][m][n] * ang[blockIdx.x][1][m][n];

            for(int ii = 0; ii < height * 2; ii++){
                if(breakpos && breakneg){break;}
                if(!(ii&1)){
                    if(breakpos){continue;}
                    sm = m + ii / 2 + 1;
                    if(sm>=height){breakpos=true;continue;}
                    else if(semancat != semantics[blockIdx.x][0][sm][n]){breakpos=true;continue;}
                    else if(mask[blockIdx.x][0][sm][n] == 0){breakpos=true;continue;}
                    else if((sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1)) > varbar[semancat][1]){breakpos=true;continue;}
                    else{
                    intpos += log[blockIdx.x][1][sm-1][n];
                    intconfidencev += confidence[blockIdx.x][0][sm][n];
                    sums += ang[blockIdx.x][1][sm][n];
                    sumsquare += ang[blockIdx.x][1][sm][n] * ang[blockIdx.x][1][sm][n];
                    countpos++;
                    }
                }
                else{
                    if(breakneg){continue;}
                    sm = m - ii / 2 - 1;
                    if(sm<0){breakneg=true;continue;}
                    else if(semancat != semantics[blockIdx.x][0][sm][n]){breakneg=true;continue;}
                    else if(mask[blockIdx.x][0][sm][n] == 0){breakneg=true;continue;}
                    else if((sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1)) > varbar[semancat][1]){breakneg=true;continue;}
                    else{
                    intneg -= log[blockIdx.x][1][sm][n];
                    intconfidencev += confidence[blockIdx.x][0][sm][n];
                    sums += ang[blockIdx.x][1][sm][n];
                    sumsquare += ang[blockIdx.x][1][sm][n] * ang[blockIdx.x][1][sm][n];
                    countneg++;
                    }
                }
            }

            for(int ii = countpos; ii > 0; ii--){
                sm = m + ii;
                depthv += confidence[blockIdx.x][0][sm][n] / intconfidencev * exp(-intpos) * depthin[blockIdx.x][0][sm][n];
                intpos -= log[blockIdx.x][1][sm-1][n];
            }

            for(int ii = countneg; ii > 0; ii--){
                sm = m - ii;
                depthv += confidence[blockIdx.x][0][sm][n] / intconfidencev * exp(-intneg) * depthin[blockIdx.x][0][sm][n];
                intneg += log[blockIdx.x][1][sm][n];
            }
        }

        depthh += confidence[blockIdx.x][0][m][n] / intconfidenceh * depthin[blockIdx.x][0][m][n];
        depthv += confidence[blockIdx.x][0][m][n] / intconfidencev * depthin[blockIdx.x][0][m][n];
        depthout[blockIdx.x][0][m][n] = depthh * intconfidenceh / (intconfidenceh + intconfidencev) + depthv * intconfidencev / (intconfidenceh + intconfidencev);
        summedConfidence[blockIdx.x][0][m][n] = (intconfidenceh + intconfidencev);
    }
    return;

    }

__global__ void shapeIntegration_backward_cuda(
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> ang,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> log,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> confidence,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> semantics,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> mask,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depthin,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> depthin_opted,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> summedConfidence,
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> gradin,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> gradout_depth,
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> gradout_confidence,
    const torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> varbar,
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

    float sums;
    float sumsquare;

    int semancat;

    float intconfidence;
    float inputgrad;
    float universal_grad;

    for(int i = threadIdx.x; i < height * width; i = i + blockDim.x){
        m = i / width;
        n = i - m * width;

        semancat = semantics[blockIdx.x][0][m][n];

        intconfidence = summedConfidence[blockIdx.x][0][m][n];
        inputgrad = gradin[blockIdx.x][0][m][n];
        universal_grad = -depthin_opted[blockIdx.x][0][m][n] / intconfidence;

        if(varbar[semancat][0] > 0){
            breakpos = false;
            breakneg = false;
            countpos = 0;
            countneg = 0;
            intpos = 0;
            intneg = 0;

            sums = ang[blockIdx.x][0][m][n];
            sumsquare = ang[blockIdx.x][0][m][n] * ang[blockIdx.x][0][m][n];

            for(int ii = 0; ii < width * 2; ii++){
                if(breakpos && breakneg){break;}
                if(!(ii&1)){
                    if(breakpos){continue;}
                    sn = n + ii / 2 + 1;
                    if(sn>=width){breakpos=true;continue;}
                    else if(semancat != semantics[blockIdx.x][0][m][sn]){breakpos=true;continue;}
                    else if(mask[blockIdx.x][0][m][sn] == 0){breakpos=true;continue;}
                    else if((sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1)) > varbar[semancat][0]){breakpos=true;continue;}
                    else{
                    intpos += log[blockIdx.x][0][m][sn-1];
                    sums += ang[blockIdx.x][0][m][sn];
                    sumsquare += ang[blockIdx.x][0][m][sn] * ang[blockIdx.x][0][m][sn];
                    countpos++;
                    }
                }
                else{
                    if(breakneg){continue;}
                    sn = n - ii / 2 - 1;
                    if(sn<0){breakneg=true;continue;}
                    else if(semancat != semantics[blockIdx.x][0][m][sn]){breakneg=true;continue;}
                    else if(mask[blockIdx.x][0][m][sn] == 0){breakneg=true;continue;}
                    else if((sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1)) > varbar[semancat][0]){breakneg=true;continue;}
                    else{
                    intneg -= log[blockIdx.x][0][m][sn];
                    sums += ang[blockIdx.x][0][m][sn];
                    sumsquare += ang[blockIdx.x][0][m][sn] * ang[blockIdx.x][0][m][sn];
                    countneg++;
                    }
                }
            }

            for(int ii = countpos; ii > 0; ii--){
                sn = n + ii;
                atomicAdd((float*)&gradout_depth[blockIdx.x][0][m][sn], confidence[blockIdx.x][0][m][sn] / intconfidence * exp(-intpos) * inputgrad);
                atomicAdd((float*)&gradout_confidence[blockIdx.x][0][m][sn], (exp(-intpos) * depthin[blockIdx.x][0][m][sn] / intconfidence + universal_grad) * inputgrad);
                intpos -= log[blockIdx.x][0][m][sn-1];
            }

            for(int ii = countneg; ii > 0; ii--){
                sn = n - ii;
                atomicAdd((float*)&gradout_depth[blockIdx.x][0][m][sn], confidence[blockIdx.x][0][m][sn] / intconfidence * exp(-intneg) * inputgrad);
                atomicAdd((float*)&gradout_confidence[blockIdx.x][0][m][sn], (exp(-intneg) * depthin[blockIdx.x][0][m][sn] / intconfidence + universal_grad) * inputgrad);
                intneg += log[blockIdx.x][0][m][sn];
            }
        }

        if(varbar[semancat][1] > 0){
            breakpos = false;
            breakneg = false;
            countpos = 0;
            countneg = 0;
            intpos = 0;
            intneg = 0;

            sums = ang[blockIdx.x][1][m][n];
            sumsquare = ang[blockIdx.x][1][m][n] * ang[blockIdx.x][1][m][n];

            for(int ii = 0; ii < height * 2; ii++){
                if(breakpos && breakneg){break;}
                if(!(ii&1)){
                    if(breakpos){continue;}
                    sm = m + ii / 2 + 1;
                    if(sm>=height){breakpos=true;continue;}
                    else if(semancat != semantics[blockIdx.x][0][sm][n]){breakpos=true;continue;}
                    else if(mask[blockIdx.x][0][sm][n] == 0){breakpos=true;continue;}
                    else if((sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1)) > varbar[semancat][1]){breakpos=true;continue;}
                    else{
                    intpos += log[blockIdx.x][1][sm-1][n];
                    sums += ang[blockIdx.x][1][sm][n];
                    sumsquare += ang[blockIdx.x][1][sm][n] * ang[blockIdx.x][1][sm][n];
                    countpos++;
                    }
                }
                else{
                    if(breakneg){continue;}
                    sm = m - ii / 2 - 1;
                    if(sm<0){breakneg=true;continue;}
                    else if(semancat != semantics[blockIdx.x][0][sm][n]){breakneg=true;continue;}
                    else if(mask[blockIdx.x][0][sm][n] == 0){breakneg=true;continue;}
                    else if((sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1)) > varbar[semancat][1]){breakneg=true;continue;}
                    else{
                    intneg -= log[blockIdx.x][1][sm][n];
                    sums += ang[blockIdx.x][1][sm][n];
                    sumsquare += ang[blockIdx.x][1][sm][n] * ang[blockIdx.x][1][sm][n];
                    countneg++;
                    }
                }
            }

            for(int ii = countpos; ii > 0; ii--){
                sm = m + ii;
                atomicAdd((float*)&gradout_depth[blockIdx.x][0][sm][n], confidence[blockIdx.x][0][sm][n] / intconfidence * exp(-intpos) * inputgrad);
                atomicAdd((float*)&gradout_confidence[blockIdx.x][0][sm][n], (exp(-intpos) * depthin[blockIdx.x][0][sm][n] / intconfidence + universal_grad) * inputgrad);
                intpos -= log[blockIdx.x][1][sm-1][n];
            }

            for(int ii = countneg; ii > 0; ii--){
                sm = m - ii;
                atomicAdd((float*)&gradout_depth[blockIdx.x][0][sm][n], confidence[blockIdx.x][0][sm][n] / intconfidence * exp(-intneg) * inputgrad);
                atomicAdd((float*)&gradout_confidence[blockIdx.x][0][sm][n], (exp(-intneg) * depthin[blockIdx.x][0][sm][n] / intconfidence + universal_grad) * inputgrad);
                intneg += log[blockIdx.x][1][sm][n];
            }
        }
        atomicAdd((float*)&gradout_depth[blockIdx.x][0][m][n], confidence[blockIdx.x][0][m][n] / intconfidence * inputgrad * 2);
        atomicAdd((float*)&gradout_confidence[blockIdx.x][0][m][n], (depthin[blockIdx.x][0][m][n] / intconfidence + universal_grad) * inputgrad * 2);
    }
    return;

    }


__global__ void shapeIntegration_crf_forward_cuda(
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> ang,
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
    ){

      const int threads = 512;

      shapeIntegration_forward_cuda<<<bs, threads>>>(
            ang.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            log.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            confidence.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            semantics.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            mask.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            depthin.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            depthout.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            summedConfidence.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            varbar.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(),
            height,
            width,
            bs
            );
    return;
    }

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
    ){

      const int threads = 512;

      shapeIntegration_backward_cuda<<<bs, threads>>>(
            ang.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            log.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            confidence.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            semantics.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            mask.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            depthin.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            depthin_opted.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            summedConfidence.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            gradin.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            gradout_depth.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            gradout_confidence.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
            varbar.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(),
            height,
            width,
            bs
            );
    return;
    }

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
    ){
      const int threads = 512;

      shapeIntegration_crf_forward_cuda<<<bs, threads>>>(
            ang.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
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