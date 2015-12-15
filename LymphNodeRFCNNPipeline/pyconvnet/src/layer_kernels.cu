/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>

#include <layer_kernels.cuh>
#include "dropc/dropc_dev.hpp"
#include "dropc/dropc_bit_dev.hpp"

/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * labels:          (1, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 */
__global__ void kLogregCost(float* probs, float* labels, float* maxProbs, float* labelLogProbs, float* correctProbs,
                            const int numCases, const int numOut) {
    const int tx = blockIdx.x * LOGREG_ERR_THREADS_X + threadIdx.x;

    if (tx < numCases) {
        const int label = int(labels[tx]);
        const float maxp = maxProbs[tx];
        const float labelp = probs[label * numCases + tx];
        
        labelLogProbs[tx] = __logf(labelp);
        
        /*
         * Compute the probability of guessing the correct case if you take the most-probable label.
         * 
         * This is done like this:
         * 
         * - If the most probable label is not equal to the true label, then the probability is zero.
         * - Otherwise, the probability is 1 / (number of labels whose probability is equal to the maximum).
         * 
         * This is certainly overkill -- in practice, it's just about impossible for two labels to get assigned
         * maximum probability. But it's a safety measure to prevent over-estimating your accuracy.
         * Though it could never happen in reality. Well it could. But it wouldn't. Cool?
         */
        if (labelp != maxp) {
            correctProbs[tx] = 0;
        } else {
            int numMax = 0;
            for (int i = 0; i < numOut; i++) {
                numMax += probs[i * numCases + tx] == maxp;
            }
            correctProbs[tx] = 1.0f / float(numMax);
        }
    }
}

/*
 * E = -log(y_t)
 * y_l:     (numOut, numCases)
 * labels:  (1, numCases)
 * 
 * dE_dy_l: (numOut, numCases)
 */
template <bool add>
__global__ void kLogregCostGrad(float* y_l, float* labels, float* dE_dy_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        const int label = int(labels[tx]);
        float v = gradCoeff * (label == ty);
        v = __fdividef(v, y_l[tidx]);
        if (add) {
            dE_dy_l[tidx] += v;
        } else {
            dE_dy_l[tidx] = v;
        }
    }
}

/*
 * dE_dy_l: (numOut, numCases)
 * y_l:     (numOut, numCases)
 * 
 * dE_dx_l: (numOut, numCases)
 */
template <bool add>
__global__ void kSoftmaxGrad(float* dE_dy_l, float* y_l, float* dE_dx_l, const int numCases, const int numOut) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        float v = 0;
        for (int j = 0; j < numOut; j++) {
            v += dE_dy_l[j * numCases + tx] * ((j == ty) - y_l[j * numCases + tx]);
        }
        v *= y_l[tidx];
        
        if (add) {
            dE_dx_l[tidx] += v;
        } else {
            dE_dx_l[tidx] = v;
        }
    }
}

/*
 * E = -log(y_t)
 * y_l:     (numOut, numCases)
 * labels:  (1, numCases)
 * 
 * dE_dx_l: (numOut, numCases)
 */
template <bool add>
__global__ void kLogregSoftmaxGrad(float* y_l, float* labels, float* dE_dx_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        const int label = int(labels[tx]);
        float v = gradCoeff * ((label == ty) - y_l[tidx]);
        if (add) {
            dE_dx_l[tidx] += v;
        } else {
            dE_dx_l[tidx] = v;
        }
    }
}

template <int B_X, bool add>
__global__ void kEltwiseMaxGrad(float* actGrad, float* input, float* output, float* target,
                                const int numElements) {
    for (int i = B_X * blockIdx.x + threadIdx.x; i < numElements; i += B_X * gridDim.x) {
        if (add) {
            target[i] += actGrad[i] * (output[i] == input[i]);
        } else {
            target[i] = actGrad[i] * (output[i] == input[i]);
        }
    }
}

void computeEltwiseMaxGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& output, NVMatrix& target, bool add) {
    assert(actGrad.isContiguous());
    assert(output.isContiguous());
    assert(input.isContiguous());
    assert(actGrad.isSameDims(input));
    assert(actGrad.isSameDims(output));
    
    dim3 blocks(DIVUP(actGrad.getNumElements(), 128));
    dim3 threads(128);
    if (add) {
        assert(actGrad.isSameDims(target));
        cudaFuncSetCacheConfig(kEltwiseMaxGrad<128, true>, cudaFuncCachePreferL1);
        kEltwiseMaxGrad<128, true><<<blocks, threads>>>(actGrad.getDevData(), input.getDevData(), output.getDevData(), target.getDevData(), actGrad.getNumElements());
    } else {
        target.resize(actGrad);
        cudaFuncSetCacheConfig(kEltwiseMaxGrad<128, false>, cudaFuncCachePreferL1);
        kEltwiseMaxGrad<128, false><<<blocks, threads>>>(actGrad.getDevData(), input.getDevData(), output.getDevData(), target.getDevData(), actGrad.getNumElements());
    }
    
    cutilCheckMsg("computeEltwiseMaxGrad: Kernel execution failed");
}

/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * labels:          (1, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 */
void computeLogregCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& labelLogProbs_out, NVMatrix& correctProbs_out) {
    int numCases = probs.getNumCols(); 
    int numOut = probs.getNumRows(); 

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    assert(labels.isContiguous());
    assert(probs.isContiguous());
    
    NVMatrix& maxProbs = probs.max(0);
    
    labelLogProbs_out.resize(1, numCases);
    correctProbs_out.resize(1, numCases);
    dim3 threads(LOGREG_ERR_THREADS_X, 1);
    dim3 blocks(DIVUP(numCases, LOGREG_ERR_THREADS_X), 1);
    cudaFuncSetCacheConfig(kLogregCost, cudaFuncCachePreferL1);
    kLogregCost<<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), maxProbs.getDevData(),
                                     labelLogProbs_out.getDevData(), correctProbs_out.getDevData(),
                                     numCases, numOut);
    cutilCheckMsg("computeLogregCost: Kernel execution failed");
//    cudaThreadSynchronize();
    delete &maxProbs;
}

void computeLogregGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff) {
    int numCases = probs.getLeadingDim(); 
    int numOut = probs.getFollowingDim(); 
    assert(labels.getNumElements() == numCases);
    assert(probs.isContiguous());
    assert(target.isContiguous());
    assert(labels.isContiguous());
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(probs);
        kLogregCostGrad<false><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    } else {
        kLogregCostGrad<true><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    }

    cutilCheckMsg("computeLogregGrad: Kernel execution failed");
}

void computeSoftmaxGrad(NVMatrix& acts, NVMatrix& actsGrad, NVMatrix& target, bool add) {
    int numCases = acts.getLeadingDim();
    int numOut = acts.getFollowingDim();

    assert(acts.isSameDims(actsGrad));
    assert(acts.isContiguous());
    assert(actsGrad.isContiguous());
    assert(target.isContiguous());
    assert(acts.isTrans());
    assert(actsGrad.isTrans());

    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(acts);
        kSoftmaxGrad<false><<<blocks, threads>>>(actsGrad.getDevData(), acts.getDevData(), target.getDevData(), numCases, numOut);
    } else {
        kSoftmaxGrad<true><<<blocks, threads>>>(actsGrad.getDevData(), acts.getDevData(), target.getDevData(), numCases, numOut);
    }
    cutilCheckMsg("computeSoftmaxGrad: Kernel execution failed");
}

void computeLogregSoftmaxGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff) {
    int numCases = probs.getLeadingDim(); 
    int numOut = probs.getFollowingDim(); 
    assert(labels.getNumElements() == numCases);
    assert(probs.isContiguous());
    assert(target.isContiguous());
    assert(labels.isContiguous());
    assert(probs.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    if (!add) {
        target.resize(probs);
        kLogregSoftmaxGrad<false><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    } else {
        kLogregSoftmaxGrad<true><<<blocks, threads>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    }

    cutilCheckMsg("computeLogregSoftmaxGrad: Kernel execution failed");
}

//-------------------------------------------------------
//   functions related with dropc
//-------------------------------------------------------
void computeFCDropC_fprop( 
      NVMatrix&  x,         ///<[in]  input matrix x, col major, numData x inDim
      NVMatrix&  w,         ///<[in]  weight matrix w, col major, inDim x outDim
      NVMatrix&  b,         ///<[in]  bias matrix, row major, 1 x outDim
      NVMatrix& mw,         ///<[in]  maskWeights, col major, inDim x (outDimxnumData)
      NVMatrix& mb,         ///<[in]  maskBiases, col major, dataDim x outDim          
      NVMatrix& y           ///<[in,out] target matrix y, col major, dataDim x outDim
      ){
   // pre-condition check
   assert( x.isTrans() );
   int numData = x.getNumRows();
   int inDim = x.getNumCols();

   assert( w.isTrans() );
   assert( w.getNumRows() == inDim );
   int outDim = w.getNumCols();

   assert( !b.isTrans() );
   assert( b.getNumRows() == 1 && b.getNumCols() == outDim );

   assert( mw.isTrans() );
   assert( mw.getNumRows() == inDim && mw.getNumCols() == (outDim*numData) );

   assert( mb.isTrans() );
   assert( mb.getNumRows() == numData&& mb.getNumCols() == outDim );

   assert( y.isTrans() );
   assert( y.getNumRows() == numData && y.getNumCols() == outDim );

   // call dev function
   computeFCDropC_fprop_d(
         x.getDevData(), w.getDevData(), b.getDevData(), // input matrix
         //m, n, d, // dims
         outDim, inDim, numData,
         mw.getDevData(), mb.getDevData(),  // masks
         y.getDevData()        // output
         );

}

void computeFCDropC_bpropActs(
      NVMatrix& v,         ///<[in]  bprop act from previous layer, col major,numData x outDim
      NVMatrix& w,         ///<[in]  weight matrix w, col major, inDim x outDim
      float scale_g,       ///<[in]  input gradient scale
      NVMatrix& mw,        ///<[in]  maskWeights, col major, inDim x (outDimxnumData)
      NVMatrix& da,        ///<[in,out] d-active, col major, numData x inDim              
      float scale_da       ///<[in]  da scale
      ){
   // pre-condition check
   assert( v.isTrans() );
   int numData = v.getNumRows();
   int outDim = v.getNumCols();
   
   assert( w.isTrans() );
   int inDim = w.getNumRows();
   assert( w.getNumCols() == outDim );

   assert( mw.isTrans() );
   assert( mw.getNumRows() == inDim && mw.getNumCols() == (outDim*numData) );

   assert( da.isTrans() );
   assert( da.getNumRows() == numData && da.getNumCols() == inDim );

   // call dev function
   computeFCDropC_bpropActs_d(
         v.getDevData(), w.getDevData(),
         //m, n, d,
         outDim, inDim, numData,
         scale_g,
         mw.getDevData(),
         da.getDevData(),
         scale_da 
         );
}

void computeFCDropC_bpropWeights(
      NVMatrix& a,            ///<[in] prev activation matrix, col major, numData x inDim
      NVMatrix& v,            ///<[in] gradient matrix, col major, numData x outDim
      float scale_g,          ///<[in] inc scale
      NVMatrix& mw,           ///<[in] maskWeights, col major, inDim x (outDimxnumData)
      NVMatrix& dw,           ///<[in,out] w gradient, col major, inDim x outDim
      float scale_dw          ///<[in] gradient scale
      ){
   // pre-condition check
   assert( a.isTrans() );
   int numData = a.getNumRows();
   int inDim = a.getNumCols();

   assert( v.isTrans() );
   assert( v.getNumRows() == numData );
   int outDim = v.getNumCols();

   assert( mw.isTrans() );
   assert( mw.getNumRows() == inDim && mw.getNumCols() == (outDim*numData) );

   assert( dw.isTrans() );
   assert( dw.getNumRows() == inDim && dw.getNumCols() == outDim );

   // call dev function
   computeFCDropC_bpropWeights_d(
         a.getDevData(), v.getDevData(),
         //m, n, d,
         outDim, inDim, numData,
         scale_g,
         mw.getDevData(),
         dw.getDevData(), scale_dw
         );

}

void computeFCDropC_bit_fprop( 
      NVMatrix&  x,         ///<[in]  input matrix x, col major, numData x inDim
      NVMatrix&  w,         ///<[in]  weight matrix w, col major, inDim x outDim
      NVMatrix&  b,         ///<[in]  bias matrix, row major, 1 x outDim
      const MaskWeights& mw,  ///<[in]  maskWeights object
      NVMatrix& mb,         ///<[in]  maskBiases, col major, dataDim x outDim          
      NVMatrix& y           ///<[in,out] target matrix y, col major, dataDim x outDim
      ){
   // pre-condition check
   assert( x.isTrans() );
   int numData = x.getNumRows();
   int inDim = x.getNumCols();

   assert( w.isTrans() );
   assert( w.getNumRows() == inDim );
   int outDim = w.getNumCols();

   assert( !b.isTrans() );
   assert( b.getNumRows() == 1 && b.getNumCols() == outDim );

   assert( mb.isTrans() );
   assert( mb.getNumRows() == numData&& mb.getNumCols() == outDim );

   assert( y.isTrans() );
   assert( y.getNumRows() == numData && y.getNumCols() == outDim );

   // call dev function
   computeFCDropC_bit_fprop_d(
         x.getDevData(), w.getDevData(), b.getDevData(), // input matrix
         //m, n, d, // dims
         outDim, inDim, numData,
         mw, //mask w
         mb.getDevData(),  // mask b
         y.getDevData()        // output
         );

}

void computeFCDropC_bit_bpropActs(
      NVMatrix& v,         ///<[in]  bprop act from previous layer, col major,numData x outDim
      NVMatrix& w,         ///<[in]  weight matrix w, col major, inDim x outDim
      float scale_g,       ///<[in]  input gradient scale
      const MaskWeights& mw,  ///<[in]  maskWeights object
      NVMatrix& da,        ///<[in,out] d-active, col major, numData x inDim              
      float scale_da       ///<[in]  da scale
      ){
   // pre-condition check
   assert( v.isTrans() );
   int numData = v.getNumRows();
   int outDim = v.getNumCols();
   
   assert( w.isTrans() );
   int inDim = w.getNumRows();
   assert( w.getNumCols() == outDim );

   assert( da.isTrans() );
   assert( da.getNumRows() == numData && da.getNumCols() == inDim );

   // call dev function
   computeFCDropC_bit_bpropActs_d(
         v.getDevData(), w.getDevData(),
         //m, n, d,
         outDim, inDim, numData,
         scale_g,
         mw, 
         da.getDevData(),
         scale_da 
         );
}

void computeFCDropC_bit_bpropWeights(
      NVMatrix& a,            ///<[in] prev activation matrix, col major, numData x inDim
      NVMatrix& v,            ///<[in] gradient matrix, col major, numData x outDim
      float scale_g,          ///<[in] inc scale
      const MaskWeights& mw,  ///<[in]  maskWeights object
      NVMatrix& dw,           ///<[in,out] w gradient, col major, inDim x outDim
      float scale_dw          ///<[in] gradient scale
      ){
   // pre-condition check
   assert( a.isTrans() );
   int numData = a.getNumRows();
   int inDim = a.getNumCols();

   assert( v.isTrans() );
   assert( v.getNumRows() == numData );
   int outDim = v.getNumCols();

   assert( dw.isTrans() );
   assert( dw.getNumRows() == inDim && dw.getNumCols() == outDim );

   // call dev function
   computeFCDropC_bit_bpropWeights_d(
         a.getDevData(), v.getDevData(),
         //m, n, d,
         outDim, inDim, numData,
         scale_g,
         mw,
         dw.getDevData(), scale_dw
         );

}

void computeFCDropC_bit_inference(
        NVMatrix& mu,       ///<[in]  mean matrix, col major, dataDim x outDim
        NVMatrix& var,      ///<[in]  var matrix,  col major, dataDim x outDim
        int numSamples,     ///<[in]  number of samples for mc sampling
        NVMatrix& y         ///<[in,out] target matrix y, col major, dataDim x outDim
        ){
    int numData = mu.getNumRows();
    int outDim = mu.getNumCols();
    size_t num_elements = numData * outDim;
    assert( mu.isTrans() );

    assert( var.getNumRows() == numData );
    assert( var.getNumCols() == outDim );
    assert( var.isTrans() );

    assert( y.getNumRows() == numData );
    assert( y.getNumCols() == outDim );
    assert( y.isTrans() );

    // call dev funtion
    computeFCDropC_bit_inference_d( mu.getDevData(), 
        var.getDevData(), num_elements,
        numSamples, y.getDevData());
}
