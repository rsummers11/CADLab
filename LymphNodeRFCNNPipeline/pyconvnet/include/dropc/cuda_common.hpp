/**
 * cuda common header file
 *
 */

#ifndef __CUDA_COMMON_H__
#define __CUDA_COMMON_H__

#include <iostream>
#include <driver_types.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

//-----------------------------------------
//         cuda utility function
//-----------------------------------------
inline void checkCuda( cudaError_t e ) {
   if( e != cudaSuccess ) {
      std::cerr <<  "CUDA Error: " <<  cudaGetErrorString( e ) << std::endl;
      exit(0);
   }
}

inline void checkCuda( cudaError_enum e ) {
   if( e != CUDA_SUCCESS ) {
      std::cerr <<  "CUDA Error "  << std::endl;
      exit(0);
   }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}

inline int divup( int x, int y ) {
   return (x + y - 1)/y;
}

#endif
