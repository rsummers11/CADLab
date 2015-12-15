/**
 * dropc dev implementation
 *
 * author: Li Wan (ntuwanli@gmail.com)
 */

#include "dropc/dropc_dev.hpp"
#include <cassert>

#include "dropc/cuda_common.hpp"

#define USE_TEXTURE_CACHE
//#define FCDROPC_BLK_SIZE 16
#define FCDROPC_BLK_SIZE 8

// texutre reference
#ifdef USE_TEXTURE_CACHE
texture<float,cudaTextureType1D,cudaReadModeElementType> texMaskWeights;
texture<float,cudaTextureType1D,cudaReadModeElementType> texMaskBiases;
#endif

// FCDrop connection fprop kernel
__global__ void kFCDropC_fprop(
      const float* x,      ///<[in]  input matrix x, col major, numData x inDim
      const float* w,      ///<[in]  weight matrix w, col major, inDim x outDim
      const float* b,      ///<[in]  bias matrix, row major, 1 x outDim
      int m,               ///<[in]  output dimension
      int n,               ///<[in]  input dimension
      int d,               ///<[in]  number of data in this batch
#ifndef USE_TEXTURE_CACHE
      const float* mw,     ///<[in]  maskWeights, col major, inDim x (outDimxdataDim)
      const float* mb,     ///<[in]  maskBiases, col major, dataDim x outDim         
#endif
      float* y             ///<[in,out] target matrix y, col major, dataDim x outDim
      ){
   // bx,by,tx,ty
   int bx = blockIdx.x * FCDROPC_BLK_SIZE;
   int by = blockIdx.y * FCDROPC_BLK_SIZE;
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   // define shared memory
   __shared__ float sA[FCDROPC_BLK_SIZE][FCDROPC_BLK_SIZE];
   __shared__ float sB[FCDROPC_BLK_SIZE][FCDROPC_BLK_SIZE];

   float c = 0;
   if( (bx+tx) < m && (by + ty) < d ) {
      // get old y value 
      c = y[ (bx+tx)*d + (by+ty) ];
   }

   //loop over cols of x and rows of w
   for( int i = 0; i < n; i += FCDROPC_BLK_SIZE ) {
      // load value from x, w into shared memory and sync
      if( (i+tx) < n && (by+ty) < d ) 
         sA[ty][tx] = x[(i+tx)*d + (by+ty)];
      else
         sA[ty][tx] = 0.0f;
      if( (i+ty) < n && (bx+tx) < m )
         sB[ty][tx] = w[(tx+bx)*n + (i+ty)];
      else
         sB[ty][tx] = 0.0f;
      __syncthreads();

      // inc c value
      if( (bx+tx) < m && (by + ty) < d ) {
#pragma unroll
         for( int j = 0; j < FCDROPC_BLK_SIZE; j++ ) {
            float maskW = 0.0f;
            if( (i+j) < n ) {
               // get m row: (i+j), col: (by+ty)^th matrix of col bx+tx
               size_t maskW_index = size_t(m)*n*(by+ty)+(bx+tx)*n + (i+j);
#ifdef USE_TEXTURE_CACHE
               // only cudaArray can use tex1D which is faster than tex1Dfeatch
               //maskW = tex1D( texMaskWeights, maskW_index );
               maskW = tex1Dfetch( texMaskWeights, maskW_index );
#else
               maskW = mw[ maskW_index ];
#endif
            }
            c += sA[ty][j] * sB[j][tx] * maskW;
         }
      }
      __syncthreads(); 
   }

   if( (bx+tx) < m && (by + ty) < d ) {
#ifdef USE_TEXTURE_CACHE 
      float maskB = tex1Dfetch( texMaskBiases, (tx+bx)*d+(ty+by) );
#else
      float maskB = mb[(tx+bx)*d+(ty+by)];
#endif
      // inc c value by bias
      c += b[tx+bx] * maskB;
      y[ (bx+tx)*d + (by+ty) ] = c;
   }
}

// FCDrop connection bprop act kernel
__global__ void kFCDropC_bpropa(
      const float*  v,         ///<[in]  bprop act from previous layer, col major,numData x outDim
      const float*  w,         ///<[in]  weight matrix w, col major, inDim x outDim
      int m,                   ///<[in]  output dimension
      int n,                   ///<[in]  input dimension
      int d,                   ///<[in]  number of data in this batch
      float scale_g,           ///<[in]  input gradient scale
#ifndef USE_TEXTURE_CACHE
      const float* mw,         ///<[in]  maskWeights, col major, inDim x (outDimxdataDim)
#endif
      float* da,               ///<[in,out] d-active, col major, numData x inDim              
      float scale_da           ///<[in]  da scale
      ){
   // bx,by,tx,ty
   int bx = blockIdx.x * FCDROPC_BLK_SIZE;
   int by = blockIdx.y * FCDROPC_BLK_SIZE;
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   // define shared memory
   __shared__ float sA[FCDROPC_BLK_SIZE+1][FCDROPC_BLK_SIZE+1];
   __shared__ float sB[FCDROPC_BLK_SIZE+1][FCDROPC_BLK_SIZE+1];

   float prev_da = 0;
   if( (bx+tx) < n && (by+ty)<d ) {
      // get old value
      prev_da = da[ (bx+tx)*d + (by+ty) ];
   }

   float c = 0;
   //loop over cols of v and rows of w^T(cols of w)
   for( int i = 0; i < m; i+= FCDROPC_BLK_SIZE ) {
      // load value from v and wt into shared memory and sync
      if( (i+tx) < m && (by+ty) < d ) 
         sA[ty][tx] = v[ (i+tx)*d + (by+ty) ];
      else
         sA[ty][tx] = 0.0f;
      if( (i+ty) < m && (tx+bx) < n ){
         // wt row: i+ty col: tx+bx
         sB[ty][tx] = w[ (i+ty)*n + (tx+bx) ];
      }
      else
         sB[ty][tx] = 0.0f;
      __syncthreads();

      // inc c value
      if( (bx+tx) < n && (by+ty)<d ) {
#pragma unroll
         for( int j = 0; j < FCDROPC_BLK_SIZE; j++ ) {
            float maskW = 0.0f;
            if( (i+j) < m ) {
               size_t maskW_index = size_t(m)*n*(ty+by)+(i+j)*n+(tx+bx);
#ifdef USE_TEXTURE_CACHE
               // only cudaArray can use tex1D which is faster than tex1Dfeatch
               //maskW = tex1D( texMaskWeights, maskW_index );
               maskW = tex1Dfetch( texMaskWeights, maskW_index );
#else
               maskW = mw[ maskW_index ];
#endif
            }
            c += sA[ty][j] * sB[j][tx] * maskW;
         }
      }
      __syncthreads();
   }

   // set output data
   if( (bx+tx) < n && (by+ty)<d ) {
      da[ (bx+tx)*d + (by+ty) ] = prev_da * scale_da + scale_g * c;
   }
}

//FCDrop connection bprop weights kernel
__global__ void kFCDropC_bpropw(
      const float* a,            ///<[in] prev activation matrix, col major, numData x inDim
      const float* v,            ///<[in] gradient matrix, col major, numData x outDim
      int m,                     ///<[in]  output dimension              
      int n,                     ///<[in]  input dimension
      int d,                     ///<[in]  number of data in this batch
      float scale_g,             ///<[in] inc scale
#ifndef USE_TEXTURE_CACHE
      const float* mw,           ///<[in] maskWeights, col major, inDim x (outDimxdataDim)
#endif
      float* dw,                 ///<[in,out] w gradient, col major, inDim x outDim
      float scale_dw             ///<[in] gradient scale
      ){
   // bx,by,tx,ty
   int bx = blockIdx.x * FCDROPC_BLK_SIZE;
   int by = blockIdx.y * FCDROPC_BLK_SIZE;
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   // define shared memory
   __shared__ float sA[FCDROPC_BLK_SIZE+1][FCDROPC_BLK_SIZE+1];
   __shared__ float sB[FCDROPC_BLK_SIZE+1][FCDROPC_BLK_SIZE+1];

   float prev_dw = 0;
   if( (bx+tx) < m && (by+ty) < n ) {
      // get the old value
      prev_dw = dw[ (bx+tx)*n + (by+ty) ];
   }

   float c = 0;
   // loop over cols of a^T(rows of a) and rows of v
   for( int i = 0; i < d; i += FCDROPC_BLK_SIZE ) {
      // load value from at and v into shared memory and sync
      if( (ty+by) < n && (i+tx) < d ) 
         sA[ty][tx] = a[ (by+ty)*d + (i+tx) ];
      else
         sA[ty][tx] = 0.0f;

      if( (tx+bx) < m && (i+ty) < d )
         sB[ty][tx] = v[ (bx+tx)*d + (i+ty) ];
      else
         sB[ty][tx] = 0.0f;
      __syncthreads();

      // inc c value
      if( (bx+tx) < m && (by+ty) < n ) {
#pragma unroll
         for( int j = 0; j < FCDROPC_BLK_SIZE; j++ ) {
            float maskW = 0.0f;
            if( (i+j) < d ) {
               size_t maskW_index = size_t(m)*n*(i+j)+(bx+tx)*n+(by+ty);
#ifdef USE_TEXTURE_CACHE
               // only cudaArray can use tex1D which is faster than tex1Dfeatch
               //maskW = tex1D( texMaskWeights, maskW_index );
               maskW = tex1Dfetch( texMaskWeights, maskW_index );
#else
               maskW = mw[ maskW_index ];
#endif
            }
            c += sA[ty][j] * sB[j][tx] * maskW;
         }
      }
      __syncthreads();
   }

   // set output data
   if( (bx+tx) < m && (by+ty) < n ) {
      // get the old value
      dw[ (bx+tx)*n + (by+ty) ] = prev_dw * scale_dw + scale_g * c;
   }
}


//----------------------------------------------------
//  main entry point: dropc_dev.cu
//----------------------------------------------------
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
      ){
#ifdef USE_TEXTURE_CACHE
   // bind texture for maskWeights 
   cudaChannelFormatDesc channelDescFloat = cudaCreateChannelDesc<float>();
   size_t offset_w;
   checkCuda( cudaBindTexture( &offset_w, &texMaskWeights, mw, 
            &channelDescFloat, sizeof(float) * inDim * outDim * numData ) );
   assert( offset_w == 0 );
   // set clamp,point mode
   texMaskWeights.addressMode[0] = cudaAddressModeClamp;
   texMaskWeights.filterMode = cudaFilterModePoint;
   texMaskWeights.normalized= false;

   // bind texture for maskBiases
   size_t offset_b;
   checkCuda( cudaBindTexture( &offset_b, &texMaskBiases, mb, 
            &channelDescFloat, sizeof(float) * numData * outDim ) );
   assert( offset_b == 0 );

   // set clamp,point mode
   texMaskBiases.addressMode[0] = cudaAddressModeClamp;
   texMaskBiases.filterMode = cudaFilterModePoint;
   texMaskBiases.normalized= false;
#else
   checkCuda( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) );
#endif

   // define block/thread info
   dim3 threads( FCDROPC_BLK_SIZE, FCDROPC_BLK_SIZE );
   dim3 blocks( divup(outDim,FCDROPC_BLK_SIZE), divup(numData, FCDROPC_BLK_SIZE) );
   // invoke kernel
   kFCDropC_fprop<<<blocks,threads>>>( x, w, b,
         outDim, inDim, numData, 
#ifndef USE_TEXTURE_CACHE
         mw, mb,
#endif
         y );
   // check error
   checkLastCudaError();

   // unbind texture
#ifdef USE_TEXTURE_CACHE
   checkCuda( cudaUnbindTexture( &texMaskWeights ) );
   checkCuda( cudaUnbindTexture( &texMaskBiases ) );
#endif
}


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
      ){
#ifdef USE_TEXTURE_CACHE
   // bind texture for maskWeights 
   cudaChannelFormatDesc channelDescFloat = cudaCreateChannelDesc<float>();
   size_t offset_w;
   checkCuda( cudaBindTexture( &offset_w, &texMaskWeights, mw, 
            &channelDescFloat, sizeof(float) * inDim * outDim * numData ) );
   assert( offset_w == 0 );
   // set clamp,point mode
   texMaskWeights.addressMode[0] = cudaAddressModeClamp;
   texMaskWeights.filterMode = cudaFilterModePoint;
   texMaskWeights.normalized= false;
#else
   checkCuda( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) );
#endif
   // define block/thread info
   dim3 threads( FCDROPC_BLK_SIZE, FCDROPC_BLK_SIZE );
   dim3 blocks( divup(inDim,FCDROPC_BLK_SIZE), divup(numData, FCDROPC_BLK_SIZE) );

   // invoke kernel
   kFCDropC_bpropa<<<blocks,threads>>>( v, w, 
         outDim, inDim, numData,
         scale_g,
#ifndef USE_TEXTURE_CACHE
         mw,
#endif
         da, scale_da );

   // check error
   checkLastCudaError();

   // unbind texture
#ifdef USE_TEXTURE_CACHE
   checkCuda( cudaUnbindTexture( &texMaskWeights ) );
   checkCuda( cudaUnbindTexture( &texMaskBiases ) );
#endif
}

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
      ){
#ifdef USE_TEXTURE_CACHE
   // bind texture for maskWeights 
   cudaChannelFormatDesc channelDescFloat = cudaCreateChannelDesc<float>();
   size_t offset_w;
   checkCuda( cudaBindTexture( &offset_w, &texMaskWeights, mw, 
            &channelDescFloat, sizeof(float) * inDim * outDim * numData ) );
   assert( offset_w == 0 );
   // set clamp,point mode
   texMaskWeights.addressMode[0] = cudaAddressModeClamp;
   texMaskWeights.filterMode = cudaFilterModePoint;
   texMaskWeights.normalized= false;
#else
   checkCuda( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) );
#endif
   // define block/thread info
   dim3 threads( FCDROPC_BLK_SIZE, FCDROPC_BLK_SIZE );
   dim3 blocks( divup(outDim,FCDROPC_BLK_SIZE), divup(inDim, FCDROPC_BLK_SIZE) );

   // invoke kernel
   kFCDropC_bpropw<<<blocks,threads>>>( a, v, 
         outDim, inDim, numData,
         scale_g,
#ifndef USE_TEXTURE_CACHE
         mw,
#endif
         dw, scale_dw );

   // check error
   checkLastCudaError();

   // unbind texture
#ifdef USE_TEXTURE_CACHE
   checkCuda( cudaUnbindTexture( &texMaskWeights ) );
   checkCuda( cudaUnbindTexture( &texMaskBiases ) );
#endif
}
