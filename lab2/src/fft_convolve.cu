#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

 __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/*
Warp reduction function which performs max reduction for a warp.
Instructions within a warp are synchronous so don't need to call
__syncthreads()
For loop for reduction has been unrolled.
This saves work since without unrolling, all warps would execute every
iteration of the for loop and if statement
 */
 template <unsigned int blockSize>
 __device__ void warpReduce(volatile float* sdata, int tid) {
     if (blockSize >= 64) sdata[tid] = max(sdata[tid], sdata[tid+32]);
     if (blockSize >= 32) sdata[tid] = max(sdata[tid], sdata[tid+16]);
     if (blockSize >= 16) sdata[tid] = max(sdata[tid], sdata[tid+8]);
     if (blockSize >= 8) sdata[tid] = max(sdata[tid], sdata[tid+4]);
     if (blockSize >= 4) sdata[tid] = max(sdata[tid], sdata[tid+2]);
     if (blockSize >= 2) sdata[tid] = max(sdata[tid], sdata[tid+1]);
 }

__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {


    /* TODO: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response. 

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them. 

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    Please make your implementation
    resilient to varying numbers of threads.

    */

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < padded_length; index += blockDim.x * gridDim.x){
        out_data[index].x = (raw_data[index].x * impulse_v[index].x - raw_data[index].y * impulse_v[index].y) / padded_length;
        out_data[index].y = (raw_data[index].x * impulse_v[index].y + raw_data[index].y * impulse_v[index].x) / padded_length;
    }
}

template <unsigned int blockSize>
__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the maximum-finding.

    There are many ways to do this reduction, and some methods
    have much better performance than others. 

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) Any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)

    */

    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    // Initialize shared mem to 0
    sdata[tid] = 0;

    // Perform two loads and the first step of the reduction as many times
    // as needed. Optimization 1.
    // Step by gridSize each time to maintain coalescing
    for (unsigned int index = blockIdx.x *2* (blockDim.x) + tid; index+blockDim.x < padded_length; index += gridSize){
        sdata[tid] = max(sdata[tid],max(abs(out_data[index].x), abs(out_data[index+blockDim.x].x)));
    }    
    __syncthreads(); // Sync threads are loading in to shared memory

    // Unroll for loop for all possible cases of block size. Optimization 4.
    // Sync threads after step of the reduction.
    // Use sequential addressing to avoid memory bank conflicts. Optimization 2.
    if (blockSize >= 512) {if (tid < 256) { sdata[tid] = max(sdata[tid], sdata[tid+256]);} __syncthreads(); } 
    if (blockSize >= 256) {if (tid < 128) { sdata[tid] = max(sdata[tid], sdata[tid+128]);} __syncthreads(); } 
    if (blockSize >= 128) {if (tid < 64) { sdata[tid] = max(sdata[tid], sdata[tid+64]);} __syncthreads(); } 

    // We're down to a single warp and no longer need to syncthreads
    // Reduce for a single warp
    if (tid < 32) warpReduce<blockSize>(sdata, tid);

    // Use atomic max so after reductions are done, first thread of each block 
    // compares its found max value to global max
    if (tid == 0) atomicMax(max_abs_val, sdata[0]);
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val. 

    This kernel should be quite short.
    */

    for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; index < padded_length; index += blockDim.x * gridDim.x){
        out_data[index].x /= max_abs_val[0];
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        
    /* TODO: Call the element-wise product and scaling kernel. */
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, impulse_v, out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        

    /* TODO 2: Call the max-finding kernel. *

    /* Call the max-finding kernel. */
    // We need to specify the correct blockSize for the kernel call
    // since it's a template parameter but it needs to be a constant. 
    // There are 10 cases of block size so we can call the kernel with
    // the correct blockSize constant for each case
    switch (threadsPerBlock) {
        case 512: cudaMaximumKernel<512><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 256: cudaMaximumKernel<256><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 128: cudaMaximumKernel<128><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 64: cudaMaximumKernel<64><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 32: cudaMaximumKernel<32><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 16: cudaMaximumKernel<16><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 8: cudaMaximumKernel<8><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 4: cudaMaximumKernel<4><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 2: cudaMaximumKernel<2><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
        case 1: cudaMaximumKernel<1><<<blocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(out_data, max_abs_val, padded_length); break;
    }
}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* TODO 2: Call the division kernel. */
    cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val, padded_length);
}
