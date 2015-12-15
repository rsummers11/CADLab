/**
 *  dropc function fprop/bprop host declear 
 *  All matrix ptr are col major
 */

#ifndef __DROPC_DEV_HPP__
#define __DROPC_DEV_HPP__

//----------------------------------------------
//           dev code declear
//----------------------------------------------
void computeFCDropC_fprop_d( 
      const float*  x,          ///<[in]  input matrix x, col major, numData x inDim
      const float*  w,          ///<[in]  weight matrix w, col major, inDim x outDim
      const float*  b,          ///<[in]  bias matrix, row major, 1 x outDim
      int outDim,               ///<[in]  output dimension
      int inDim,                ///<[in]  input dimension
      int numData,              ///<[in]  number of data in this batch
      const float * mw,         ///<[in]  maskWeights, col major, inDim x (outDimxdataDim)
      const float * mb,         ///<[in]  maskBiases, col major, dataDim x outDim          
      float * y                 ///<[in,out] target matrix y, col major, dataDim x outDim
      );


void computeFCDropC_bpropActs_d(
      const float*  v,         ///<[in]  bprop act from previous layer, col major,numData x outDim
      const float*  w,         ///<[in]  weight matrix w, col major, inDim x outDim
      int outDim,              ///<[in]  output dimension
      int inDim,               ///<[in]  input dimension
      int numData,             ///<[in]  number of data in this batch
      float scale_g,           ///<[in]  input gradient scale
      const float* mw,         ///<[in]  maskWeights, col major, inDim x (outDimxdataDim)
      float* da,               ///<[in,out] d-active, col major, numData x inDim              
      float scale_da           ///<[in]  da scale
      );

void computeFCDropC_bpropWeights_d(
      const float* a,            ///<[in] prev activation matrix, col major, numData x inDim
      const float* v,            ///<[in] gradient matrix, col major, numData x outDim
      int outDim,                ///<[in]  output dimension              
      int inDim,                 ///<[in]  input dimension
      int numData,               ///<[in]  number of data in this batch
      float scale_g,             ///<[in] inc scale
      const float* mw,           ///<[in] maskWeights, col major, inDim x (outDimxdataDim)
      float* dw,                 ///<[in,out] w gradient, col major, inDim x outDim
      float scale_dw             ///<[in] gradient scale
      );

#endif
