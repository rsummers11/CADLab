/**
 *  dropc function fprop/bprop host declear 
 *  All matrix ptr are col major
 */

#ifndef __DROPC_BIT_DEV_HPP__
#define __DROPC_BIT_DEV_HPP__

#include <cuda.h>
#include <curand_kernel.h>

//----------------------------------------------
//           random maskW bit function
//----------------------------------------------
/**
 * MaskWeights device data structure
 * col major matrix, compressed format
 * store in transposed(row major) 2D matrix
 */
class MaskWeights {
   public:
      /** default cor*/
      explicit MaskWeights( float onProb = 0.5f , unsigned long long seed = 0 );

      /** default dtor*/
      ~MaskWeights();

      /** resize mask weights */
      void resize(
            int outDim,            ///<[in] new outDim
            int inDim,             ///<[in] new inDim
            int numData            ///<[in] new numData
            );

      /** randomize mask weights */
      void randomize();

      /** get width */
      size_t get_width() const;

      /** get height */
      size_t get_height() const; 

      /** get stride */ 
      size_t stride() const {
         return stride_;
      }

      /** get dataptr */
      const unsigned int* data() const {
         return data_;
      }

      /** set onProb */
      void set_onProb( float onProb ); 

      /** get onProb */
      float get_onProb( ) const {
          return onProb_;
      }

   private:
      /** init random seed */
      void initRandomSeed( );

   private:
      int outDim_;  
      int inDim_;
      int numData_;
      size_t stride_;
      unsigned int* data_;
      unsigned long long seed_;
      curandState* devStates_;
      float onProb_;
};

//----------------------------------------------
//           dev code: training
//----------------------------------------------
void computeFCDropC_bit_fprop_d( 
      const float*  x,        ///<[in]  input matrix x, col major, numData x inDim
      const float*  w,        ///<[in]  weight matrix w, col major, inDim x outDim
      const float*  b,        ///<[in]  bias matrix, row major, 1 x outDim
      int outDim,             ///<[in]  output dimension
      int inDim,              ///<[in]  input dimension
      int numData,            ///<[in]  number of data in this batch
      const MaskWeights& mw,  ///<[in]  mask weight object
      const float * mb,       ///<[in]  maskBiases, col major, dataDim x outDim          
      float * y               ///<[in,out] target matrix y, col major, dataDim x outDim
      );


void computeFCDropC_bit_bpropActs_d(
      const float*  v,        ///<[in]  bprop act from previous layer, col major,numData x outDim
      const float*  w,        ///<[in]  weight matrix w, col major, inDim x outDim
      int outDim,             ///<[in]  output dimension
      int inDim,              ///<[in]  input dimension
      int numData,            ///<[in]  number of data in this batch
      float scale_g,          ///<[in]  input gradient scale
      const MaskWeights& mw,  ///<[in]  mask weight object
      float* da,              ///<[in,out] d-active, col major, numData x inDim              
      float scale_da          ///<[in]  da scale
      );

void computeFCDropC_bit_bpropWeights_d(
      const float* a,         ///<[in] prev activation matrix, col major, numData x inDim
      const float* v,         ///<[in] gradient matrix, col major, numData x outDim
      int outDim,             ///<[in]  output dimension              
      int inDim,              ///<[in]  input dimension
      int numData,            ///<[in]  number of data in this batch
      float scale_g,          ///<[in] inc scale
      const MaskWeights& mw,  ///<[in]  mask weight object
      float* dw,              ///<[in,out] w gradient, col major, inDim x outDim
      float scale_dw          ///<[in] gradient scale
      );

//----------------------------------------------
//           dev code: inference
//----------------------------------------------
// only work for relu neurons now
void computeFCDropC_bit_inference_d(
        const float*  mu,       ///<[in]  mean matrix, col major, dataDim x outDim
        const float*  var,      ///<[in]  var matrix,  col major, dataDim x outDim
        size_t n,               ///<[in]  vector length
        int numSamples,         ///<[in]  number of samples for mc sampling
        float * y               ///<[in,out] target matrix y, col major, dataDim x outDim
        );

#endif
