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

#ifndef LAYER_KERNELS_CUH
#define	LAYER_KERNELS_CUH

#include <cutil_inline.h>
#include <nvmatrix.cuh>

#define LOGREG_GRAD_THREADS_X      32
#define LOGREG_GRAD_THREADS_Y      4

#define LOGREG_ERR_THREADS_X        128
#define LOGREG_ERR_THREADS_Y        1

void computeLogregCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& labelLogProbs_out, NVMatrix& correctProbs_out);
void computeLogregGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff);
void computeSoftmaxGrad(NVMatrix& acts, NVMatrix& actsGrad, NVMatrix& target, bool add);

// Numerical stability optimization: this routine combines computeLogregGrad with computeSoftmaxGrad
// to avoi dividing and then multiplying by quantities that may be near zero.
void computeLogregSoftmaxGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff);
void computeEltwiseMaxGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& output, NVMatrix& target, bool add);

//-------------------------------------------------------
//   functions related with dropc
//-------------------------------------------------------
void computeFCDropC_fprop( 
      NVMatrix&  x,         ///<[in]  input matrix x, col major, numData x inDim
      NVMatrix&  w,         ///<[in]  weight matrix w, col major, inDim x outDim
      NVMatrix&  b,         ///<[in]  bias matrix, row major, 1 x outDim
      NVMatrix& mw,         ///<[in]  maskWeights, col major, inDim x (outDimxdataDim)
      NVMatrix& mb,         ///<[in]  maskBiases, col major, dataDim x outDim          
      NVMatrix& y           ///<[in,out] target matrix y, col major, dataDim x outDim
      );

void computeFCDropC_bpropActs(
      NVMatrix& v,         ///<[in]  bprop act from previous layer, col major,numData x outDim
      NVMatrix& w,         ///<[in]  weight matrix w, col major, inDim x outDim
      float scale_g,       ///<[in]  input gradient scale
      NVMatrix& mw,        ///<[in]  maskWeights, col major, inDim x (outDimxdataDim)
      NVMatrix& da,        ///<[in,out] d-active, col major, numData x inDim              
      float scale_da       ///<[in]  da scale
      );

void computeFCDropC_bpropWeights(
      NVMatrix& a,            ///<[in] prev activation matrix, col major, numData x inDim
      NVMatrix& v,            ///<[in] gradient matrix, col major, numData x outDim
      float scale_g,          ///<[in] inc scale
      NVMatrix& mw,           ///<[in] maskWeights, col major, inDim x (outDimxdataDim)
      NVMatrix& dw,           ///<[in,out] w gradient, col major, inDim x outDim
      float scale_dw          ///<[in] gradient scale
      );

class MaskWeights;
void computeFCDropC_bit_fprop( 
      NVMatrix&  x,         ///<[in]  input matrix x, col major, numData x inDim
      NVMatrix&  w,         ///<[in]  weight matrix w, col major, inDim x outDim
      NVMatrix&  b,         ///<[in]  bias matrix, row major, 1 x outDim
      const MaskWeights& mw,///<[in]  maskWeights object
      NVMatrix& mb,         ///<[in]  maskBiases, col major, dataDim x outDim          
      NVMatrix& y           ///<[in,out] target matrix y, col major, dataDim x outDim
      );

void computeFCDropC_bit_bpropActs(
      NVMatrix& v,            ///<[in]  bprop act from previous layer, col major,numData x outDim
      NVMatrix& w,            ///<[in]  weight matrix w, col major, inDim x outDim
      float scale_g,          ///<[in]  input gradient scale
      const MaskWeights& mw,  ///<[in]  maskWeights object
      NVMatrix& da,           ///<[in,out] d-active, col major, numData x inDim              
      float scale_da          ///<[in]  da scale
      );

void computeFCDropC_bit_bpropWeights(
      NVMatrix& a,            ///<[in] prev activation matrix, col major, numData x inDim
      NVMatrix& v,            ///<[in] gradient matrix, col major, numData x outDim
      float scale_g,          ///<[in] inc scale
      const MaskWeights& mw,  ///<[in]  maskWeights object
      NVMatrix& dw,           ///<[in,out] w gradient, col major, inDim x outDim
      float scale_dw          ///<[in] gradient scale
      );

// only work for relu neurons now
void computeFCDropC_bit_inference(
        NVMatrix& mu,       ///<[in]  mean matrix, col major, dataDim x outDim
        NVMatrix& var,      ///<[in]  var matrix,  col major, dataDim x outDim
        int numSamples,     ///<[in]  number of samples for mc sampling
        NVMatrix& y         ///<[in,out] target matrix y, col major, dataDim x outDim
        );

#endif	/* LAYER_KERNELS_CUH */

