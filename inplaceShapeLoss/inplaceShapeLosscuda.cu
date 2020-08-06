#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <cmath>

#include <math_constants.h>

namespace {

template <typename scalar_t>
__global__ void inplaceShapeLoss_forward_cuda_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> logdepth,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> logratioh,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> logratiov,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> valindic,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> lossrec,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> countsrec,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> rndseeds,
    const int srw,
    const int srh
    ) {
       int pxnum = logdepth.size(2) * logdepth.size(3);
       int m;
       int n;
       int left;
       int right;
       int up;
       int down;

       int cm;
       int cn;
       float accumlog;
       float accumcount;
       for(int bz = 0; bz < logdepth.size(0); bz++){
           for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < pxnum; i = i + blockDim.x * gridDim.x){
                m = i / logdepth.size(3);
                n = i - m * logdepth.size(3);

                if(valindic[bz][0][m][n] == 1){
                    up = m - srh;
                    down = m + srh;
                    left = n - srw;
                    right = n + srw;

                    if(up < 0){
                        up = 0;
                    }
                    if(down >= logdepth.size(2)){
                        down = logdepth.size(2)-1;
                    }
                    if(left < 0){
                        left = 0;
                    }
                    if(right >= logdepth.size(3)){
                        right = logdepth.size(3) -1;
                    }

                    accumcount = 0;
                    for(int sm = up; sm <= down; sm++){
                        for(int sn = left; sn <= right; sn++){
                            if((valindic[bz][0][sm][sn] == 1) && ((sm!=m) || (sn!=n))){
                                cm = m;
                                cn = n;
                                accumlog = logdepth[bz][0][m][n];

                                while((cm != sm) || (cn != sn)){
                                    if(rndseeds[bz][0][cm][cn] > 0.5){
                                        if(cm < sm){
                                            cm = cm + 1;
                                            accumlog = accumlog + logratiov[bz][0][cm-1][cn];
                                        }
                                        else if(cm > sm){
                                            cm = cm - 1;
                                            accumlog = accumlog - logratiov[bz][0][cm][cn];
                                        }
                                        else if(cn < sn){
                                            cn = cn + 1;
                                            accumlog = accumlog + logratioh[bz][0][cm][cn-1];
                                        }
                                        else if(cn > sn){
                                            cn = cn - 1;
                                            accumlog = accumlog - logratioh[bz][0][cm][cn];
                                        }
                                    }
                                    else{
                                        if(cn < sn){
                                            cn = cn + 1;
                                            accumlog = accumlog + logratioh[bz][0][cm][cn-1];
                                        }
                                        else if(cn > sn){
                                            cn = cn - 1;
                                            accumlog = accumlog - logratioh[bz][0][cm][cn];
                                        }

                                        else if(cm < sm){
                                            cm = cm + 1;
                                            accumlog = accumlog + logratiov[bz][0][cm-1][cn];
                                        }
                                        else if(cm > sm){
                                            cm = cm - 1;
                                            accumlog = accumlog - logratiov[bz][0][cm][cn];
                                        }
                                    }
                                }
                                lossrec[bz][0][m][n] = lossrec[bz][0][m][n] + abs(accumlog - logdepth[bz][0][sm][sn]);
                                accumcount = accumcount + 1;
                            }
                        }
                    }
                    if(accumcount == 0){
                        lossrec[bz][0][m][n] = 0;
                    }
                    else{
                        lossrec[bz][0][m][n] = lossrec[bz][0][m][n] / accumcount;
                    }
                    countsrec[bz][0][m][n] = accumcount;
                }
           }
       }
    }

template <typename scalar_t>
__global__ void inplaceShapeLoss_backward_cuda_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> logdepth,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> logratioh,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> logratiov,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> valindic,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> grad_re,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> gradrech,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> gradrecv,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> countsrec,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> rndseeds,
    const int srw,
    const int srh
    ) {
       int pxnum = logdepth.size(2) * logdepth.size(3);
       int m;
       int n;
       int left;
       int right;
       int up;
       int down;

       int cm;
       int cn;
       float accumlog;
       float unitgrad;
       for(int bz = 0; bz < logdepth.size(0); bz++){
           for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < pxnum; i = i + blockDim.x * gridDim.x){
                m = i / logdepth.size(3);
                n = i - m * logdepth.size(3);

                if((valindic[bz][0][m][n] == 1)&&(countsrec[bz][0][m][n] > 0)){
                    up = m - srh;
                    down = m + srh;
                    left = n - srw;
                    right = n + srw;

                    if(up < 0){
                        up = 0;
                    }
                    if(down >= logdepth.size(2)){
                        down = logdepth.size(2)-1;
                    }
                    if(left < 0){
                        left = 0;
                    }
                    if(right >= logdepth.size(3)){
                        right = logdepth.size(3) -1;
                    }

                    for(int sm = up; sm <= down; sm++){
                        for(int sn = left; sn <= right; sn++){

                            unitgrad = 1.0 / countsrec[bz][0][m][n];
                            unitgrad = unitgrad * grad_re[bz][0][m][n];
                            if((valindic[bz][0][sm][sn] == 1) && ((sm!=m) || (sn!=n))){
                                cm = m;
                                cn = n;
                                accumlog = logdepth[bz][0][m][n];

                                while((cm != sm) || (cn != sn)){
                                    if(rndseeds[bz][0][cm][cn] > 0.5){
                                        if(cm < sm){
                                            cm = cm + 1;
                                            accumlog = accumlog + logratiov[bz][0][cm-1][cn];
                                        }
                                        else if(cm > sm){
                                            cm = cm - 1;
                                            accumlog = accumlog - logratiov[bz][0][cm][cn];
                                        }
                                        else if(cn < sn){
                                            cn = cn + 1;
                                            accumlog = accumlog + logratioh[bz][0][cm][cn-1];
                                        }
                                        else if(cn > sn){
                                            cn = cn - 1;
                                            accumlog = accumlog - logratioh[bz][0][cm][cn];
                                        }
                                    }
                                    else{
                                        if(cn < sn){
                                            cn = cn + 1;
                                            accumlog = accumlog + logratioh[bz][0][cm][cn-1];
                                        }
                                        else if(cn > sn){
                                            cn = cn - 1;
                                            accumlog = accumlog - logratioh[bz][0][cm][cn];
                                        }

                                        else if(cm < sm){
                                            cm = cm + 1;
                                            accumlog = accumlog + logratiov[bz][0][cm-1][cn];
                                        }
                                        else if(cm > sm){
                                            cm = cm - 1;
                                            accumlog = accumlog - logratiov[bz][0][cm][cn];
                                        }
                                    }
                                }

                                if((accumlog - logdepth[bz][0][sm][sn]) > 0){
                                    unitgrad = unitgrad;
                                }
                                else{
                                    unitgrad = -unitgrad;
                                }

                                cm = m;
                                cn = n;

                                while((cm != sm) || (cn != sn)){
                                    if(rndseeds[bz][0][cm][cn] > 0.5){
                                        if(cm < sm){
                                            cm = cm + 1;
                                            atomicAdd((float*)&gradrecv[bz][0][cm-1][cn], unitgrad);
                                        }
                                        else if(cm > sm){
                                            cm = cm - 1;
                                            atomicAdd((float*)&gradrecv[bz][0][cm][cn], -unitgrad);
                                        }
                                        else if(cn < sn){
                                            cn = cn + 1;
                                            atomicAdd((float*)&gradrech[bz][0][cm][cn-1], unitgrad);
                                        }
                                        else if(cn > sn){
                                            cn = cn - 1;
                                            atomicAdd((float*)&gradrech[bz][0][cm][cn], -unitgrad);
                                        }
                                    }
                                    else{
                                        if(cn < sn){
                                            cn = cn + 1;
                                            atomicAdd((float*)&gradrech[bz][0][cm][cn-1], unitgrad);
                                        }
                                        else if(cn > sn){
                                            cn = cn - 1;
                                            atomicAdd((float*)&gradrech[bz][0][cm][cn], -unitgrad);
                                        }
                                        else if(cm < sm){
                                            cm = cm + 1;
                                            atomicAdd((float*)&gradrecv[bz][0][cm-1][cn], unitgrad);
                                        }
                                        else if(cm > sm){
                                            cm = cm - 1;
                                            atomicAdd((float*)&gradrecv[bz][0][cm][cn], -unitgrad);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
           }
       }
    }

template <typename scalar_t>
__global__ void inplaceShapeLoss_integration_cuda_kernel(
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> logdepth,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> logratioh,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> logratiov,
    const torch::PackedTensorAccessor<int,4,torch::RestrictPtrTraits,size_t> valindic,
    torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> lossrec,
    const int srw,
    const int srh
    ) {
       int pxnum = logdepth.size(2) * logdepth.size(3);
       int m;
       int n;
       int left;
       int right;
       int up;
       int down;

       int cm;
       int cn;
       float accumlog;
       bool isaggregated;
       for(int bz = 0; bz < logdepth.size(0); bz++){
           for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < pxnum; i = i + blockDim.x * gridDim.x){
                m = i / logdepth.size(3);
                n = i - m * logdepth.size(3);

                if(valindic[bz][0][m][n] == 1){
                    up = m - srh;
                    down = m + srh;
                    left = n - srw;
                    right = n + srw;

                    if(up < 0){
                        up = 0;
                    }
                    if(down >= logdepth.size(2)){
                        down = logdepth.size(2)-1;
                    }
                    if(left < 0){
                        left = 0;
                    }
                    if(right >= logdepth.size(3)){
                        right = logdepth.size(3) -1;
                    }

                    isaggregated = false;

                    for(int sm = up; sm <= m; sm++){

                        if(isaggregated==true){
                            break;
                        }

                        for(int sn = left; sn <= n; sn++){

                            if(isaggregated==true){
                                break;
                            }

                            if((valindic[bz][0][sm][sn] == 1) && ((sm!=m) || (sn!=n))){
                                cm = m;
                                cn = n;
                                accumlog = logdepth[bz][0][m][n];

                                while((cm != sm) || (cn != sn)){
                                    if(cm < sm){
                                        cm = cm + 1;
                                        accumlog = accumlog + logratiov[bz][0][cm-1][cn];
                                    }
                                    else if(cm > sm){
                                        cm = cm - 1;
                                        accumlog = accumlog - logratiov[bz][0][cm][cn];
                                    }
                                    else if(cn < sn){
                                        cn = cn + 1;
                                        accumlog = accumlog + logratioh[bz][0][cm][cn-1];
                                    }
                                    else if(cn > sn){
                                        cn = cn - 1;
                                        accumlog = accumlog - logratioh[bz][0][cm][cn];
                                    }
                                }
                                lossrec[bz][0][m][n] = lossrec[bz][0][m][n] + accumlog - logdepth[bz][0][sm][sn];
                                isaggregated = true;
                            }
                        }
                    }
                }
           }
       }
    }


}


void inplaceShapeLoss_forward_cuda(
    torch::Tensor logdepth,
    torch::Tensor logratioh,
    torch::Tensor logratiov,
    torch::Tensor valindic,
    torch::Tensor lossrec,
    torch::Tensor countsrec,
    torch::Tensor rndseeds,
    int srw,
    int srh
    ) {
    const int threads = 1024;
    const int blockdimx = 128;
    const dim3 blocks(blockdimx);

    AT_DISPATCH_FLOATING_TYPES(logdepth.type(), "do forward inplace shape loss", ([&] {
        inplaceShapeLoss_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            logdepth.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            logratioh.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            logratiov.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            valindic.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            lossrec.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            countsrec.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            rndseeds.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            srw,
            srh
            );
    }));
    return;
}

void inplaceShapeLoss_backward_cuda(
    torch::Tensor logdepth,
    torch::Tensor logratioh,
    torch::Tensor logratiov,
    torch::Tensor valindic,
    torch::Tensor grad_re,
    torch::Tensor gradrech,
    torch::Tensor gradrecv,
    torch::Tensor countsrec,
    torch::Tensor rndseeds,
    int srw,
    int srh
    ) {
    const int threads = 1024;
    const int blockdimx = 128;
    const dim3 blocks(blockdimx);

    AT_DISPATCH_FLOATING_TYPES(logdepth.type(), "do backward inplace shape loss", ([&] {
        inplaceShapeLoss_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            logdepth.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            logratioh.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            logratiov.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            valindic.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            grad_re.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            gradrech.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            gradrecv.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            countsrec.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            rndseeds.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            srw,
            srh
            );
    }));
    return;
}


void inplaceShapeLoss_integration_cuda(
    torch::Tensor logdepth,
    torch::Tensor logratioh,
    torch::Tensor logratiov,
    torch::Tensor valindic,
    torch::Tensor lossrec,
    int srw,
    int srh
    ) {
    const int threads = 1024;
    const int blockdimx = 128;
    const dim3 blocks(blockdimx);

    AT_DISPATCH_FLOATING_TYPES(logdepth.type(), "do forward inplace shape loss", ([&] {
        inplaceShapeLoss_integration_cuda_kernel<scalar_t><<<blocks, threads>>>(
            logdepth.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            logratioh.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            logratiov.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            valindic.packed_accessor<int,4,torch::RestrictPtrTraits,size_t>(),
            lossrec.packed_accessor<scalar_t,4,torch::RestrictPtrTraits,size_t>(),
            srw,
            srh
            );
    }));
    return;
}