/*
 *  _reg_spline_kernels.cu
 *
 *
 *  Created by Marc Modat on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _reg_spline_KERNELS_CU
#define _reg_spline_KERNELS_CU

#include "_reg_common_gpu.h"

__device__ __constant__ int c_UseBSpline;
__device__ __constant__ int c_VoxelNumber;
__device__ __constant__ int c_ControlPointNumber;
__device__ __constant__ int3 c_ReferenceImageDim;
__device__ __constant__ int3 c_ControlPointImageDim;
__device__ __constant__ float3 c_ControlPointVoxelSpacing;
__device__ __constant__ float3 c_ControlPointSpacing;
__device__ __constant__ float3 c_ReferenceSpacing;
__device__ __constant__ float c_Weight;
__device__ __constant__ float3 c_Weight3;
__device__ __constant__ int c_ActiveVoxelNumber;
__device__ __constant__ bool c_Type;
__device__ __constant__ float3 c_AffineMatrix0;
__device__ __constant__ float3 c_AffineMatrix1;
__device__ __constant__ float3 c_AffineMatrix2;
__device__ __constant__ float4 c_AffineMatrix0b;
__device__ __constant__ float4 c_AffineMatrix1b;
__device__ __constant__ float4 c_AffineMatrix2b;
__device__ __constant__ float4 c_AffineMatrix0c;
__device__ __constant__ float4 c_AffineMatrix1c;
__device__ __constant__ float4 c_AffineMatrix2c;
/* *************************************************************** */
/* *************************************************************** */
texture<float4, 1, cudaReadModeElementType> controlPointTexture;
texture<float4, 1, cudaReadModeElementType> secondDerivativesTexture;
texture<float4,1, cudaReadModeElementType> voxelDeformationTexture;
texture<int, 1, cudaReadModeElementType> maskTexture;
texture<float,1, cudaReadModeElementType> jacobianDeterminantTexture;
texture<float,1, cudaReadModeElementType> jacobianMatricesTexture;
/* *************************************************************** */
/* *************************************************************** */
__device__ float2 operator*(float a, float2 b){
	return make_float2(a*b.x, a*b.y);
}
__device__ float3 operator*(float a, float3 b){
    return make_float3(a*b.x, a*b.y, a*b.z);
}
__device__ float3 operator*(float3 a, float3 b){
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}
__device__ float4 operator*(float4 a, float4 b){
    return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w);
}
__device__ float4 operator*(float a, float4 b){
    return make_float4(a*b.x, a*b.y, a*b.z, 0.0f);
}
/* *************************************************************** */
__device__ float2 operator/(float2 a, float2 b){
	return make_float2(a.x/b.x, a.y/b.y);
}
__device__ float3 operator/(float3 a, float b){
    return make_float3(a.x/b, a.y/b, a.z/b);
}
__device__ float3 operator/(float3 a, float3 b){
    return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);
}
/* *************************************************************** */
__device__ float2 operator+(float2 a, float2 b){
	return make_float2(a.x+b.x, a.y+b.y);
}
__device__ float4 operator+(float4 a, float4 b){
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, 0.0f);
}
__device__ float3 operator+(float3 a, float3 b){
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
/* *************************************************************** */
__device__ float3 operator-(float3 a, float3 b){
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__device__ float4 operator-(float4 a, float4 b){
    return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, 0.f);
}
/* *************************************************************** */
/* *************************************************************** */
__device__ void GetBasisBSplineValues(float basis, float *values)
{
    float FF= basis*basis;
    float FFF= FF*basis;
    float MF=1.f-basis;
    values[0] = (MF)*(MF)*(MF)/(6.f);
    values[1] = (3.f*FFF - 6.f*FF + 4.f)/6.f;
    values[2] = (-3.f*FFF + 3.f*FF + 3.f*basis + 1.f)/6.f;
    values[3] = (FFF/6.f);
}
/* *************************************************************** */
__device__ void GetFirstBSplineValues(float basis, float *values, float *first)
{
    GetBasisBSplineValues(basis, values);
    first[3]= basis * basis / 2.f;
    first[0]= basis - 0.5f - first[3];
    first[2]= 1.f + first[0] - 2.f*first[3];
    first[1]= - first[0] - first[2] - first[3];
}
/* *************************************************************** */
/* *************************************************************** */
__device__ void GetBasisSplineValues(float basis, float *values)
{
    float FF= basis*basis;
    values[0] = (basis * ((2.f-basis)*basis - 1.f))/2.f;
    values[1] = (FF * (3.f*basis-5.f) + 2.f)/2.f;
    values[2] = (basis * ((4.f-3.f*basis)*basis + 1.f))/2.f;
    values[3] = (basis-1.f) * FF/2.f;
}
/* *************************************************************** */
__device__ void GetBasisSplineValuesX(float basis, float4 *values)
{
    float FF= basis*basis;
    values->x = (basis * ((2.f-basis)*basis - 1.f))/2.f;
    values->y = (FF * (3.f*basis-5.f) + 2.f)/2.f;
    values->z = (basis * ((4.f-3.f*basis)*basis + 1.f))/2.f;
    values->w = (basis-1.f) * FF/2.f;
}
/* *************************************************************** */
__device__ void getBSplineBasisValue(float basis, int index, float *value, float *first)
{
    switch(index){
        case 0:
            *value = (1.f-basis)*(1.f-basis)*(1.f-basis)/6.f;
            *first = (2.f*basis - basis*basis - 1.f)/2.f;
            break;
        case 1:
            *value = (3.f*basis*basis*basis - 6.f*basis*basis + 4.f)/6.f;
            *first = (3.f*basis*basis - 4.f*basis)/2.f;
            break;
        case 2:
            *value = (3.f*basis*basis - 3.f*basis*basis*basis + 3.f*basis + 1.f)/6.f;
            *first = (2.f*basis - 3.f*basis*basis + 1.f)/2.f;
            break;
        case 3:
            *value = basis*basis*basis/6.f;
            *first = basis*basis/2.f;
            break;
         default:
            *value = 0.f;
            *first = 0.f;
            break;
    }
}
/* *************************************************************** */
__device__ void GetFirstDerivativeBasisValues2D(int index,
												float *xBasis,
												float *yBasis){
	switch(index){
	case 0: xBasis[0]=-0.0833333f;yBasis[0]=-0.0833333f;break;
	case 1: xBasis[1]=0.f;yBasis[1]=-0.333333f;break;
	case 2: xBasis[2]=0.0833333f;yBasis[2]=-0.0833333f;break;
	case 3: xBasis[3]=-0.333333f;yBasis[3]=0.f;break;
	case 4: xBasis[4]=0.f;yBasis[4]=0.f;break;
	case 5: xBasis[5]=0.333333f;yBasis[5]=0.f;break;
	case 6: xBasis[6]=-0.0833333f;yBasis[6]=0.0833333f;break;
	case 7: xBasis[7]=0.f;yBasis[7]=0.333333f;break;
	case 8: xBasis[8]=0.0833333f;yBasis[8]=0.0833333f;break;
	}
}
/* *************************************************************** */
__device__ void GetFirstDerivativeBasisValues3D(int index,
												float *xBasis,
												float *yBasis,
												float *zBasis){
	switch(index){
	case 0: xBasis[0]=-0.013889f;yBasis[0]=-0.013889f;zBasis[0]=-0.013889f;break;
	case 1: xBasis[1]=0.000000f;yBasis[1]=-0.055556f;zBasis[1]=-0.055556f;break;
	case 2: xBasis[2]=0.013889f;yBasis[2]=-0.013889f;zBasis[2]=-0.013889f;break;
	case 3: xBasis[3]=-0.055556f;yBasis[3]=0.000000f;zBasis[3]=-0.055556f;break;
	case 4: xBasis[4]=0.000000f;yBasis[4]=0.000000f;zBasis[4]=-0.222222f;break;
	case 5: xBasis[5]=0.055556f;yBasis[5]=0.000000f;zBasis[5]=-0.055556f;break;
	case 6: xBasis[6]=-0.013889f;yBasis[6]=0.013889f;zBasis[6]=-0.013889f;break;
	case 7: xBasis[7]=0.000000f;yBasis[7]=0.055556f;zBasis[7]=-0.055556f;break;
	case 8: xBasis[8]=0.013889f;yBasis[8]=0.013889f;zBasis[8]=-0.013889f;break;
	case 9: xBasis[9]=-0.055556f;yBasis[9]=-0.055556f;zBasis[9]=0.000000f;break;
	case 10: xBasis[10]=0.000000f;yBasis[10]=-0.222222f;zBasis[10]=0.000000f;break;
	case 11: xBasis[11]=0.055556f;yBasis[11]=-0.055556f;zBasis[11]=0.000000f;break;
	case 12: xBasis[12]=-0.222222f;yBasis[12]=0.000000f;zBasis[12]=0.000000f;break;
	case 13: xBasis[13]=0.000000f;yBasis[13]=0.000000f;zBasis[13]=0.000000f;break;
	case 14: xBasis[14]=0.222222f;yBasis[14]=0.000000f;zBasis[14]=0.000000f;break;
	case 15: xBasis[15]=-0.055556f;yBasis[15]=0.055556f;zBasis[15]=0.000000f;break;
	case 16: xBasis[16]=0.000000f;yBasis[16]=0.222222f;zBasis[16]=0.000000f;break;
	case 17: xBasis[17]=0.055556f;yBasis[17]=0.055556f;zBasis[17]=0.000000f;break;
	case 18: xBasis[18]=-0.013889f;yBasis[18]=-0.013889f;zBasis[18]=0.013889f;break;
	case 19: xBasis[19]=0.000000f;yBasis[19]=-0.055556f;zBasis[19]=0.055556f;break;
	case 20: xBasis[20]=0.013889f;yBasis[20]=-0.013889f;zBasis[20]=0.013889f;break;
	case 21: xBasis[21]=-0.055556f;yBasis[21]=0.000000f;zBasis[21]=0.055556f;break;
	case 22: xBasis[22]=0.000000f;yBasis[22]=0.000000f;zBasis[22]=0.222222f;break;
	case 23: xBasis[23]=0.055556f;yBasis[23]=0.000000f;zBasis[23]=0.055556f;break;
	case 24: xBasis[24]=-0.013889f;yBasis[24]=0.013889f;zBasis[24]=0.013889f;break;
	case 25: xBasis[25]=0.000000f;yBasis[25]=0.055556f;zBasis[25]=0.055556f;break;
	case 26: xBasis[26]=0.013889f;yBasis[26]=0.013889f;zBasis[26]=0.013889f;break;
	}
}
/* *************************************************************** */
__device__ void GetSecondDerivativeBasisValues2D(int index,
                                                 float *xxBasis,
                                                 float *yyBasis,
                                                 float *xyBasis){
    switch(index){
    case 0:
        xxBasis[0]=0.166667f;yyBasis[0]=0.166667f;xyBasis[0]=0.25f;
        break;
    case 1:
        xxBasis[1]=-0.333333f;yyBasis[1]=0.666667f;xyBasis[1]=-0.f;
        break;
    case 2:
        xxBasis[2]=0.166667f;yyBasis[2]=0.166667f;xyBasis[2]=-0.25f;
        break;
    case 3:
        xxBasis[3]=0.666667f;yyBasis[3]=-0.333333f;xyBasis[3]=-0.f;
        break;
    case 4:
        xxBasis[4]=-1.33333f;yyBasis[4]=-1.33333f;xyBasis[4]=0.f;
        break;
    case 5:
        xxBasis[5]=0.666667f;yyBasis[5]=-0.333333f;xyBasis[5]=0.f;
        break;
    case 6:
        xxBasis[6]=0.166667f;yyBasis[6]=0.166667f;xyBasis[6]=-0.25f;
        break;
    case 7:
        xxBasis[7]=-0.333333f;yyBasis[7]=0.666667f;xyBasis[7]=0.f;
        break;
    case 8:
        xxBasis[8]=0.166667f;yyBasis[8]=0.166667f;xyBasis[8]=0.25f;
        break;
    }
}
/* *************************************************************** */
__device__ void GetSecondDerivativeBasisValues3D(int index,
                                                 float *xxBasis,
                                                 float *yyBasis,
                                                 float *zzBasis,
                                                 float *xyBasis,
                                                 float *yzBasis,
                                                 float *xzBasis){
    switch(index){
        case 0:
            xxBasis[0]=0.027778f;yyBasis[0]=0.027778f;zzBasis[0]=0.027778f;
            xyBasis[0]=0.041667f;yzBasis[0]=0.041667f;xzBasis[0]=0.041667f;
            break;
        case 1:
            xxBasis[1]=-0.055556f;yyBasis[1]=0.111111f;zzBasis[1]=0.111111f;
            xyBasis[1]=-0.000000f;yzBasis[1]=0.166667f;xzBasis[1]=-0.000000f;
            break;
        case 2:
            xxBasis[2]=0.027778f;yyBasis[2]=0.027778f;zzBasis[2]=0.027778f;
            xyBasis[2]=-0.041667f;yzBasis[2]=0.041667f;xzBasis[2]=-0.041667f;
            break;
        case 3:
            xxBasis[3]=0.111111f;yyBasis[3]=-0.055556f;zzBasis[3]=0.111111f;
            xyBasis[3]=-0.000000f;yzBasis[3]=-0.000000f;xzBasis[3]=0.166667f;
            break;
        case 4:
            xxBasis[4]=-0.222222f;yyBasis[4]=-0.222222f;zzBasis[4]=0.444444f;
            xyBasis[4]=0.000000f;yzBasis[4]=-0.000000f;xzBasis[4]=-0.000000f;
            break;
        case 5:
            xxBasis[5]=0.111111f;yyBasis[5]=-0.055556f;zzBasis[5]=0.111111f;
            xyBasis[5]=0.000000f;yzBasis[5]=-0.000000f;xzBasis[5]=-0.166667f;
            break;
        case 6:
            xxBasis[6]=0.027778f;yyBasis[6]=0.027778f;zzBasis[6]=0.027778f;
            xyBasis[6]=-0.041667f;yzBasis[6]=-0.041667f;xzBasis[6]=0.041667f;
            break;
        case 7:
            xxBasis[7]=-0.055556f;yyBasis[7]=0.111111f;zzBasis[7]=0.111111f;
            xyBasis[7]=0.000000f;yzBasis[7]=-0.166667f;xzBasis[7]=-0.000000f;
            break;
        case 8:
            xxBasis[8]=0.027778f;yyBasis[8]=0.027778f;zzBasis[8]=0.027778f;
            xyBasis[8]=0.041667f;yzBasis[8]=-0.041667f;xzBasis[8]=-0.041667f;
            break;
        case 9:
            xxBasis[9]=0.111111f;yyBasis[9]=0.111111f;zzBasis[9]=-0.055556f;
            xyBasis[9]=0.166667f;yzBasis[9]=-0.000000f;xzBasis[9]=-0.000000f;
            break;
        case 10:
            xxBasis[10]=-0.222222f;yyBasis[10]=0.444444f;zzBasis[10]=-0.222222f;
            xyBasis[10]=-0.000000f;yzBasis[10]=-0.000000f;xzBasis[10]=0.000000f;
            break;
        case 11:
            xxBasis[11]=0.111111f;yyBasis[11]=0.111111f;zzBasis[11]=-0.055556f;
            xyBasis[11]=-0.166667f;yzBasis[11]=-0.000000f;xzBasis[11]=0.000000f;
            break;
        case 12:
            xxBasis[12]=0.444444f;yyBasis[12]=-0.222222f;zzBasis[12]=-0.222222f;
            xyBasis[12]=-0.000000f;yzBasis[12]=0.000000f;xzBasis[12]=-0.000000f;
            break;
        case 13:
            xxBasis[13]=-0.888889f;yyBasis[13]=-0.888889f;zzBasis[13]=-0.888889f;
            xyBasis[13]=0.000000f;yzBasis[13]=0.000000f;xzBasis[13]=0.000000f;
            break;
        case 14:
            xxBasis[14]=0.444444f;yyBasis[14]=-0.222222f;zzBasis[14]=-0.222222f;
            xyBasis[14]=0.000000f;yzBasis[14]=0.000000f;xzBasis[14]=0.000000f;
            break;
        case 15:
            xxBasis[15]=0.111111f;yyBasis[15]=0.111111f;zzBasis[15]=-0.055556f;
            xyBasis[15]=-0.166667f;yzBasis[15]=0.000000f;xzBasis[15]=-0.000000f;
            break;
        case 16:
            xxBasis[16]=-0.222222f;yyBasis[16]=0.444444f;zzBasis[16]=-0.222222f;
            xyBasis[16]=0.000000f;yzBasis[16]=0.000000f;xzBasis[16]=0.000000f;
            break;
        case 17:
            xxBasis[17]=0.111111f;yyBasis[17]=0.111111f;zzBasis[17]=-0.055556f;
            xyBasis[17]=0.166667f;yzBasis[17]=0.000000f;xzBasis[17]=0.000000f;
            break;
        case 18:
            xxBasis[18]=0.027778f;yyBasis[18]=0.027778f;zzBasis[18]=0.027778f;
            xyBasis[18]=0.041667f;yzBasis[18]=-0.041667f;xzBasis[18]=-0.041667f;
            break;
        case 19:
            xxBasis[19]=-0.055556f;yyBasis[19]=0.111111f;zzBasis[19]=0.111111f;
            xyBasis[19]=-0.000000f;yzBasis[19]=-0.166667f;xzBasis[19]=0.000000f;
            break;
        case 20:
            xxBasis[20]=0.027778f;yyBasis[20]=0.027778f;zzBasis[20]=0.027778f;
            xyBasis[20]=-0.041667f;yzBasis[20]=-0.041667f;xzBasis[20]=0.041667f;
            break;
        case 21:
            xxBasis[21]=0.111111f;yyBasis[21]=-0.055556f;zzBasis[21]=0.111111f;
            xyBasis[21]=-0.000000f;yzBasis[21]=0.000000f;xzBasis[21]=-0.166667f;
            break;
        case 22:
            xxBasis[22]=-0.222222f;yyBasis[22]=-0.222222f;zzBasis[22]=0.444444f;
            xyBasis[22]=0.000000f;yzBasis[22]=0.000000f;xzBasis[22]=0.000000f;
            break;
        case 23:
            xxBasis[23]=0.111111f;yyBasis[23]=-0.055556f;zzBasis[23]=0.111111f;
            xyBasis[23]=0.000000f;yzBasis[23]=0.000000f;xzBasis[23]=0.166667f;
            break;
        case 24:
            xxBasis[24]=0.027778f;yyBasis[24]=0.027778f;zzBasis[24]=0.027778f;
            xyBasis[24]=-0.041667f;yzBasis[24]=0.041667f;xzBasis[24]=-0.041667f;
            break;
        case 25:
            xxBasis[25]=-0.055556f;yyBasis[25]=0.111111f;zzBasis[25]=0.111111f;
            xyBasis[25]=0.000000f;yzBasis[25]=0.166667f;xzBasis[25]=0.000000f;
            break;
        case 26:
            xxBasis[26]=0.027778f;yyBasis[26]=0.027778f;zzBasis[26]=0.027778f;
            xyBasis[26]=0.041667f;yzBasis[26]=0.041667f;xzBasis[26]=0.041667f;
            break;
    }
}
/* *************************************************************** */
/* *************************************************************** */
__device__ float4 get_SlidedValues_gpu(int x, int y)
{
    int newX=x;
    int newY=y;
    if(x<0){
        newX=0;
    }
    else if(x>=c_ReferenceImageDim.x){
        newX=c_ReferenceImageDim.x-1;
    }
    if(y<0){
        newY=0;
    }
    else if(y>=c_ReferenceImageDim.y){
        newY=c_ReferenceImageDim.y-1;
    }

    x=x-newX;
    y=y-newY;
    float4 slidedValues = make_float4(
                x * c_AffineMatrix0c.x +
                y * c_AffineMatrix0c.y,
                x * c_AffineMatrix1c.x +
                y * c_AffineMatrix1c.y,
                0.f,
                0.f);
    slidedValues = slidedValues +
            tex1Dfetch(voxelDeformationTexture,
                       newY*c_ReferenceImageDim.x+newX);
    return slidedValues;
}
/* *************************************************************** */
/* *************************************************************** */
__device__ float4 get_SlidedValues_gpu(int x, int y, int z)
{
    int newX=x;
    int newY=y;
    int newZ=z;
    if(x<0){
        newX=0;
    }
    else if(x>=c_ReferenceImageDim.x){
        newX=c_ReferenceImageDim.x-1;
    }
    if(y<0){
        newY=0;
    }
    else if(y>=c_ReferenceImageDim.y){
        newY=c_ReferenceImageDim.y-1;
    }
    if(z<0){
        newZ=0;
    }
    else if(z>=c_ReferenceImageDim.z){
        newZ=c_ReferenceImageDim.z-1;
    }

    x=x-newX;
    y=y-newY;
    z=z-newZ;
    float4 slidedValues = make_float4(
                x * c_AffineMatrix0c.x +
                y * c_AffineMatrix0c.y +
                z * c_AffineMatrix0c.z,
                x * c_AffineMatrix1c.x +
                y * c_AffineMatrix1c.y +
                z * c_AffineMatrix1c.z,
                x * c_AffineMatrix2c.x +
                y * c_AffineMatrix2c.y +
                z * c_AffineMatrix2c.z,
                0.f);
    slidedValues = slidedValues +
            tex1Dfetch(voxelDeformationTexture,
                       (newZ*c_ReferenceImageDim.y+newY)*c_ReferenceImageDim.x+newX);
    return slidedValues;
}
/* *************************************************************** */
/* *************************************************************** */
__global__ void reg_spline_getDeformationField3D(float4 *positionField)
{
    const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ActiveVoxelNumber){

        // Allocate the shared memory
        extern __shared__ float yBasis[];
        // Compute the shared memory offset which correspond to four times the number of thread per block
        float *zBasis=&yBasis[4*blockDim.x*blockDim.y*blockDim.z];

        int3 imageSize = c_ReferenceImageDim;

        unsigned int tempIndex=tex1Dfetch(maskTexture,tid);
        const int z = tempIndex/(imageSize.x*imageSize.y);
        tempIndex  -= z*imageSize.x*imageSize.y;
        const int y = tempIndex/imageSize.x;
        const int x = tempIndex - y*imageSize.x;

        // the "nearest previous" node is determined [0,0,0]
        int3 nodeAnte;
        float3 gridVoxelSpacing = c_ControlPointVoxelSpacing;
        nodeAnte.x = (int)floorf((float)x/gridVoxelSpacing.x);
        nodeAnte.y = (int)floorf((float)y/gridVoxelSpacing.y);
        nodeAnte.z = (int)floorf((float)z/gridVoxelSpacing.z);

        const int shareMemIndex = 4*threadIdx.x;

        // Z basis values
        float relative = fabsf((float)z/gridVoxelSpacing.z-(float)nodeAnte.z);
        relative=relative>0?relative:0.f;
        if(c_UseBSpline) GetBasisBSplineValues(relative, &zBasis[shareMemIndex]);
        else GetBasisSplineValues(relative, &zBasis[shareMemIndex]);
        // Y basis values
        relative = fabsf((float)y/gridVoxelSpacing.y-(float)nodeAnte.y);
        relative=relative>0?relative:0.f;
        if(c_UseBSpline) GetBasisBSplineValues(relative, &yBasis[shareMemIndex]);
        else GetBasisSplineValues(relative, &yBasis[shareMemIndex]);
        // X basis values
        float xBasis[4];
        relative = fabsf((float)x/gridVoxelSpacing.x-(float)nodeAnte.x);
        relative=relative>0?relative:0.f;
        if(c_UseBSpline) GetBasisBSplineValues(relative, xBasis);
        else GetBasisSplineValues(relative, xBasis);

        int3 controlPointImageDim = c_ControlPointImageDim;
        float4 displacement=make_float4(0.0f,0.0f,0.0f,0.0f);
        float basis;
        float3 tempDisplacement;

        for(int c=0; c<4; c++){
            tempDisplacement=make_float3(0.0f,0.0f,0.0f);
            int indexYZ= ( (nodeAnte.z + c) * controlPointImageDim.y + nodeAnte.y) * controlPointImageDim.x;
            for(int b=0; b<4; b++){

                int indexXYZ = indexYZ + nodeAnte.x;
                float4 nodeCoefficientA = tex1Dfetch(controlPointTexture,indexXYZ++);
                float4 nodeCoefficientB = tex1Dfetch(controlPointTexture,indexXYZ++);
                float4 nodeCoefficientC = tex1Dfetch(controlPointTexture,indexXYZ++);
                float4 nodeCoefficientD = tex1Dfetch(controlPointTexture,indexXYZ);

                basis=yBasis[shareMemIndex+b];
                tempDisplacement.x += (
                            nodeCoefficientA.x * xBasis[0] +
                            nodeCoefficientB.x * xBasis[1] +
                            nodeCoefficientC.x * xBasis[2] +
                            nodeCoefficientD.x * xBasis[3] ) * basis;

                tempDisplacement.y += (
                            nodeCoefficientA.y * xBasis[0] +
                            nodeCoefficientB.y * xBasis[1] +
                            nodeCoefficientC.y * xBasis[2] +
                            nodeCoefficientD.y * xBasis[3] ) * basis;

                tempDisplacement.z += (
                            nodeCoefficientA.z * xBasis[0] +
                            nodeCoefficientB.z * xBasis[1] +
                            nodeCoefficientC.z * xBasis[2] +
                            nodeCoefficientD.z * xBasis[3] ) * basis;

                indexYZ += controlPointImageDim.x;
            }

            basis = zBasis[shareMemIndex+c];
            displacement.x += tempDisplacement.x * basis;
            displacement.y += tempDisplacement.y * basis;
            displacement.z += tempDisplacement.z * basis;
        }
        positionField[tid] = displacement;
    }
    return;
}
/* *************************************************************** */
/* *************************************************************** */
__global__ void reg_spline_getDeformationField2D(float4 *positionField)
{
    const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ActiveVoxelNumber){

        // Allocate the shared memory
        extern __shared__ float yBasis[];

        int3 imageSize = c_ReferenceImageDim;

        unsigned int tempIndex=tex1Dfetch(maskTexture,tid);
        const int y = tempIndex/imageSize.x;
        const int x = tempIndex - y*imageSize.x;

        // the "nearest previous" node is determined [0,0,0]
        int2 nodeAnte;
        float2 gridVoxelSpacing = make_float2(c_ControlPointVoxelSpacing.x,
                                              c_ControlPointVoxelSpacing.y);
        nodeAnte.x = (int)floorf((float)x/gridVoxelSpacing.x);
        nodeAnte.y = (int)floorf((float)y/gridVoxelSpacing.y);

        const int shareMemIndex = 4*threadIdx.x;

        // Y basis values
        float relative = fabsf((float)y/gridVoxelSpacing.y-(float)nodeAnte.y);
        if(c_UseBSpline) GetBasisBSplineValues(relative, &yBasis[shareMemIndex]);
        else GetBasisSplineValues(relative, &yBasis[shareMemIndex]);
        // X basis values
        float xBasis[4];
        relative = fabsf((float)x/gridVoxelSpacing.x-(float)nodeAnte.x);
        if(c_UseBSpline) GetBasisBSplineValues(relative, xBasis);
        else GetBasisSplineValues(relative, xBasis);

        int2 controlPointImageDim = make_int2(c_ControlPointImageDim.x,
                                              c_ControlPointImageDim.y);
        float4 displacement=make_float4(0.0f,0.0f,0.0f,0.0f);
        float basis;

        for(int b=0; b<4; b++){
            int index =  (nodeAnte.y + b) * controlPointImageDim.x + nodeAnte.x;

            float4 nodeCoefficientA = tex1Dfetch(controlPointTexture,index++);
            float4 nodeCoefficientB = tex1Dfetch(controlPointTexture,index++);
            float4 nodeCoefficientC = tex1Dfetch(controlPointTexture,index++);
            float4 nodeCoefficientD = tex1Dfetch(controlPointTexture,index);

            basis=yBasis[shareMemIndex+b];
            displacement.x += basis * (
                        nodeCoefficientA.x * xBasis[0] +
                        nodeCoefficientB.x * xBasis[1] +
                        nodeCoefficientC.x * xBasis[2] +
                        nodeCoefficientD.x * xBasis[3]);

            displacement.y += basis * (
                        nodeCoefficientA.y * xBasis[0] +
                        nodeCoefficientB.y * xBasis[1] +
                        nodeCoefficientC.y * xBasis[2] +
                        nodeCoefficientD.y * xBasis[3]);

        }
        positionField[tid] = displacement;
    }
    return;
}
/* *************************************************************** */
/* *************************************************************** */
__global__ void reg_spline_getApproxSecondDerivatives2D(float4 *secondDerivativeValues)
{
    __shared__ float xxbasis[9];
    __shared__ float yybasis[9];
    __shared__ float xybasis[9];

    if(threadIdx.x<9)
        GetSecondDerivativeBasisValues2D(threadIdx.x,
                                         xxbasis,
                                         yybasis,
                                         xybasis);
    __syncthreads();

    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ControlPointNumber){

        int3 gridSize = c_ControlPointImageDim;

        const int y =(int)(tid/gridSize.x);
        const int x = int(tid - y*gridSize.x);

        float4 XX = make_float4(0.0f,0.0f,0.0f,0.0f);
        float4 YY = make_float4(0.0f,0.0f,0.0f,0.0f);
        float4 XY = make_float4(0.0f,0.0f,0.0f,0.0f);

        int tempIndex;
        if(0<x && x<gridSize.x-1 &&
           0<y && y<gridSize.y-1){

            tempIndex=0;
            for(int b=y-1; b<y+2; ++b){
                for(int a=x-1; a<x+2; ++a){
                    int indexXY = b*gridSize.x+a;
                    float4 controlPointValues = tex1Dfetch(controlPointTexture,indexXY);
                    XX.x = XX.x + xxbasis[tempIndex] * controlPointValues.x;
                    XX.y = XX.y + xxbasis[tempIndex] * controlPointValues.y;
                    YY.x = YY.x + yybasis[tempIndex] * controlPointValues.x;
                    YY.y = YY.y + yybasis[tempIndex] * controlPointValues.y;
                    XY.x = XY.x + xybasis[tempIndex] * controlPointValues.x;
                    XY.y = XY.y + xybasis[tempIndex] * controlPointValues.y;
                    ++tempIndex;
                }
            }
        }

        tempIndex=3*tid;
        secondDerivativeValues[tempIndex++]=XX;
        secondDerivativeValues[tempIndex++]=YY;
        secondDerivativeValues[tempIndex] = XY;
    }
    return;
}
/* *************************************************************** */
/* *************************************************************** */
__global__ void reg_spline_getApproxSecondDerivatives3D(float4 *secondDerivativeValues)
{
    __shared__ float xxbasis[27];
    __shared__ float yybasis[27];
    __shared__ float zzbasis[27];
    __shared__ float xybasis[27];
    __shared__ float yzbasis[27];
    __shared__ float xzbasis[27];

    if(threadIdx.x<27)
        GetSecondDerivativeBasisValues3D(threadIdx.x,
                                         xxbasis,
                                         yybasis,
                                         zzbasis,
                                         xybasis,
                                         yzbasis,
                                         xzbasis);
    __syncthreads();

    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ControlPointNumber){

        int3 gridSize = c_ControlPointImageDim;

        int tempIndex=tid;
        const int z =(int)(tempIndex/(gridSize.x*gridSize.y));
        tempIndex -= int(z*gridSize.x*gridSize.y);
        const int y =(int)(tempIndex/gridSize.x);
        const int x = int(tempIndex - y*gridSize.x);

        float4 XX = make_float4(0.0f,0.0f,0.0f,0.0f);
        float4 YY = make_float4(0.0f,0.0f,0.0f,0.0f);
        float4 ZZ = make_float4(0.0f,0.0f,0.0f,0.0f);
        float4 XY = make_float4(0.0f,0.0f,0.0f,0.0f);
        float4 YZ = make_float4(0.0f,0.0f,0.0f,0.0f);
        float4 XZ = make_float4(0.0f,0.0f,0.0f,0.0f);

        if(0<x && x<gridSize.x-1 &&
           0<y && y<gridSize.y-1 &&
           0<z && z<gridSize.z-1){

            tempIndex=0;
            for(int c=z-1; c<z+2; ++c){
                for(int b=y-1; b<y+2; ++b){
                    for(int a=x-1; a<x+2; ++a){
                        int indexXYZ = (c*gridSize.y+b)*gridSize.x+a;
                        float4 controlPointValues = tex1Dfetch(controlPointTexture,indexXYZ);
                        XX = XX + xxbasis[tempIndex] * controlPointValues;
                        YY = YY + yybasis[tempIndex] * controlPointValues;
                        ZZ = ZZ + zzbasis[tempIndex] * controlPointValues;
                        XY = XY + xybasis[tempIndex] * controlPointValues;
                        YZ = YZ + yzbasis[tempIndex] * controlPointValues;
                        XZ = XZ + xzbasis[tempIndex] * controlPointValues;
                        tempIndex++;
                    }
                }
            }
        }

        tempIndex=6*tid;
        secondDerivativeValues[tempIndex++]=XX;
        secondDerivativeValues[tempIndex++]=YY;
        secondDerivativeValues[tempIndex++]=ZZ;
        secondDerivativeValues[tempIndex++]=XY;
        secondDerivativeValues[tempIndex++]=YZ;
        secondDerivativeValues[tempIndex] = XZ;
    }
}
/* *************************************************************** */
__global__ void reg_spline_getApproxBendingEnergy2D_kernel(float *penaltyTerm)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ControlPointNumber){
        int index=tid*3;
        float4 XX = tex1Dfetch(secondDerivativesTexture,index++);XX=XX*XX;
        float4 YY = tex1Dfetch(secondDerivativesTexture,index++);YY=YY*YY;
        float4 XY = tex1Dfetch(secondDerivativesTexture,index++);XY=XY*XY;

        penaltyTerm[tid]=
                XX.x + XX.y +
                YY.x + YY.y +
                2.f*(XY.x + XY.y);
    }
    return;
}
/* *************************************************************** */
__global__ void reg_spline_getApproxBendingEnergy3D_kernel(float *penaltyTerm)
{
    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ControlPointNumber){
        int index=tid*6;
        float4 XX = tex1Dfetch(secondDerivativesTexture,index++);XX=XX*XX;
        float4 YY = tex1Dfetch(secondDerivativesTexture,index++);YY=YY*YY;
        float4 ZZ = tex1Dfetch(secondDerivativesTexture,index++);ZZ=ZZ*ZZ;
        float4 XY = tex1Dfetch(secondDerivativesTexture,index++);XY=XY*XY;
        float4 YZ = tex1Dfetch(secondDerivativesTexture,index++);YZ=YZ*YZ;
        float4 XZ = tex1Dfetch(secondDerivativesTexture,index);XZ=XZ*XZ;

        penaltyTerm[tid]=
                XX.x + XX.y + XX.z +
                YY.x + YY.y + YY.z +
                ZZ.x + ZZ.y + ZZ.z +
                2.f*(XY.x + XY.y + XY.z +
                     YZ.x + YZ.y + YZ.z +
                     XZ.x + XZ.y + XZ.z);
    }
    return;
}
/* *************************************************************** */
__global__ void reg_spline_getApproxBendingEnergyGradient2D_kernel(float4 *nodeGradientArray)
{
    __shared__ float xxbasis[9];
    __shared__ float yybasis[9];
    __shared__ float xybasis[9];

    if(threadIdx.x<9)
        GetSecondDerivativeBasisValues2D(threadIdx.x,
                                         xxbasis,
                                         yybasis,
                                         xybasis);
    __syncthreads();

    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ControlPointNumber){

        int3 gridSize = c_ControlPointImageDim;

        const int y = tid/gridSize.x;
        const int x = tid - y*gridSize.x;

        float2 gradientValue=make_float2(0.0f,0.0f);
        float4 secondDerivativeValues;

        int coord=0;
        for(int b=y-1; b<y+2; ++b){
            for(int a=x-1; a<x+2; ++a){
                if(-1<a && -1<b && a<gridSize.x && b<gridSize.y){
                    int indexXY = 3*(b*gridSize.x+a);
                    secondDerivativeValues = tex1Dfetch(secondDerivativesTexture,indexXY++); // XX
                    gradientValue.x += secondDerivativeValues.x * xxbasis[coord];
                    gradientValue.y += secondDerivativeValues.y * xxbasis[coord];
                    secondDerivativeValues = tex1Dfetch(secondDerivativesTexture,indexXY++); // YY
                    gradientValue.x += secondDerivativeValues.x * yybasis[coord];
                    gradientValue.y += secondDerivativeValues.y * yybasis[coord];
                    secondDerivativeValues = 2.f*tex1Dfetch(secondDerivativesTexture,indexXY); // XY
                    gradientValue.x += secondDerivativeValues.x * xybasis[coord];
                    gradientValue.y += secondDerivativeValues.y * xybasis[coord];
                }
                coord++;
            }
        }

        nodeGradientArray[tid].x += c_Weight*gradientValue.x;
        nodeGradientArray[tid].y += c_Weight*gradientValue.y;
    }
}
/* *************************************************************** */
__global__ void reg_spline_getApproxBendingEnergyGradient3D_kernel(float4 *nodeGradientArray)
{
    __shared__ float xxbasis[27];
    __shared__ float yybasis[27];
    __shared__ float zzbasis[27];
    __shared__ float xybasis[27];
    __shared__ float yzbasis[27];
    __shared__ float xzbasis[27];

    if(threadIdx.x<27)
        GetSecondDerivativeBasisValues3D(threadIdx.x,
                                         xxbasis,
                                         yybasis,
                                         zzbasis,
                                         xybasis,
                                         yzbasis,
                                         xzbasis);
    __syncthreads();

    const int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ControlPointNumber){

        int3 gridSize = c_ControlPointImageDim;

        int tempIndex=tid;
        const int z = tempIndex/(gridSize.x*gridSize.y);
        tempIndex  -= z*gridSize.x*gridSize.y;
        const int y = tempIndex/gridSize.x;
        const int x = tempIndex - y*gridSize.x;

        float3 gradientValue=make_float3(0.0f,0.0f,0.0f);
        float4 secondDerivativeValues;

        int coord=0;
        for(int c=z-1; c<z+2; ++c){
            for(int b=y-1; b<y+2; ++b){
                for(int a=x-1; a<x+2; ++a){
                    if(-1<a && -1<b && -1<c && a<gridSize.x && b<gridSize.y && c<gridSize.z){
                        unsigned int indexXYZ = 6*((c*gridSize.y+b)*gridSize.x+a);
                        secondDerivativeValues = tex1Dfetch(secondDerivativesTexture,indexXYZ++); // XX
                        gradientValue.x += secondDerivativeValues.x * xxbasis[coord];
                        gradientValue.y += secondDerivativeValues.y * xxbasis[coord];
                        gradientValue.z += secondDerivativeValues.z * xxbasis[coord];
                        secondDerivativeValues = tex1Dfetch(secondDerivativesTexture,indexXYZ++); // YY
                        gradientValue.x += secondDerivativeValues.x * yybasis[coord];
                        gradientValue.y += secondDerivativeValues.y * yybasis[coord];
                        gradientValue.z += secondDerivativeValues.z * yybasis[coord];
                        secondDerivativeValues = tex1Dfetch(secondDerivativesTexture,indexXYZ++); //ZZ
                        gradientValue.x += secondDerivativeValues.x * zzbasis[coord];
                        gradientValue.y += secondDerivativeValues.y * zzbasis[coord];
                        gradientValue.z += secondDerivativeValues.z * zzbasis[coord];
                        secondDerivativeValues = 2.f*tex1Dfetch(secondDerivativesTexture,indexXYZ++); // XY
                        gradientValue.x += secondDerivativeValues.x * xybasis[coord];
                        gradientValue.y += secondDerivativeValues.y * xybasis[coord];
                        gradientValue.z += secondDerivativeValues.z * xybasis[coord];
                        secondDerivativeValues = 2.f*tex1Dfetch(secondDerivativesTexture,indexXYZ++); // YZ
                        gradientValue.x += secondDerivativeValues.x * yzbasis[coord];
                        gradientValue.y += secondDerivativeValues.y * yzbasis[coord];
                        gradientValue.z += secondDerivativeValues.z * yzbasis[coord];
                        secondDerivativeValues = 2.f*tex1Dfetch(secondDerivativesTexture,indexXYZ); //XZ
                        gradientValue.x += secondDerivativeValues.x * xzbasis[coord];
                        gradientValue.y += secondDerivativeValues.y * xzbasis[coord];
                        gradientValue.z += secondDerivativeValues.z * xzbasis[coord];
                    }
                    coord++;
                }
            }
        }
        gradientValue = c_Weight * gradientValue;

        float4 metricGradientValue;
        metricGradientValue = nodeGradientArray[tid];
        metricGradientValue.x += gradientValue.x;
        metricGradientValue.y += gradientValue.y;
        metricGradientValue.z += gradientValue.z;
        nodeGradientArray[tid]=metricGradientValue;
    }
}
/* *************************************************************** */
/* *************************************************************** */
__global__ void reg_spline_getApproxJacobianValues2D_kernel(float *jacobianMatrices,
															float *jacobianDet)
{
	__shared__ float xbasis[9];
	__shared__ float ybasis[9];

	if(threadIdx.x<9)
		GetFirstDerivativeBasisValues2D(threadIdx.x,
										xbasis,
										ybasis);
	__syncthreads();

	const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if(tid<c_ControlPointNumber){

		int3 gridSize = c_ControlPointImageDim;

		int tempIndex=tid;
		const int y =tempIndex/gridSize.x;
		const int x = tempIndex - y*gridSize.x;

		if(0<x && x<gridSize.x-1 &&
		   0<y && y<gridSize.y-1){

			float Tx_x=0, Tx_y=0;
			float Ty_x=0, Ty_y=0;

			tempIndex=0;
			for(int b=y-1; b<y+2; ++b){
				for(int a=x-1; a<x+2; ++a){
					int indexXY = b * gridSize.x + a;
					float4 controlPointValues = tex1Dfetch(controlPointTexture,indexXY);
					Tx_x += xbasis[tempIndex]*controlPointValues.x;
					Tx_y += ybasis[tempIndex]*controlPointValues.x;
					Ty_x += xbasis[tempIndex]*controlPointValues.y;
					Ty_y += ybasis[tempIndex]*controlPointValues.y;
					tempIndex++;
				}
            }

			// The jacobian matrix is reoriented
			float Tx_x2=c_AffineMatrix0.x*Tx_x + c_AffineMatrix0.y*Ty_x;
			float Tx_y2=c_AffineMatrix0.x*Tx_y + c_AffineMatrix0.y*Ty_y;
			float Ty_x2=c_AffineMatrix1.x*Tx_x + c_AffineMatrix1.y*Ty_x;
			float Ty_y2=c_AffineMatrix1.x*Tx_y + c_AffineMatrix1.y*Ty_y;

			// The Jacobian matrix is stored
			tempIndex=tid*4;
			jacobianMatrices[tempIndex++]=Tx_x2;
			jacobianMatrices[tempIndex++]=Tx_y2;
			jacobianMatrices[tempIndex++]=Ty_x2;
			jacobianMatrices[tempIndex] = Ty_y2;

			// The Jacobian determinant is computed and stored
			jacobianDet[tid]= Tx_x2 * Ty_y2 - Tx_y2 * Ty_x2;
		}
		else{
			tempIndex=tid*4;
			jacobianMatrices[tempIndex++]=1.f;
			jacobianMatrices[tempIndex++]=0.f;
			jacobianMatrices[tempIndex++]=0.f;
			jacobianMatrices[tempIndex]=1.f;
			jacobianDet[tid]= 1.0f;
		}
	}
	return;
}
/* *************************************************************** */
/* *************************************************************** */
__global__ void reg_spline_getApproxJacobianValues3D_kernel(float *jacobianMatrices,
														   float *jacobianDet)
{
	__shared__ float xbasis[27];
	__shared__ float ybasis[27];
	__shared__ float zbasis[27];

	if(threadIdx.x<27)
		GetFirstDerivativeBasisValues3D(threadIdx.x,
									  xbasis,
									  ybasis,
									  zbasis);
	__syncthreads();

	const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if(tid<c_ControlPointNumber){

		int3 gridSize = c_ControlPointImageDim;

		int tempIndex=tid;
		const int z =tempIndex/(gridSize.x*gridSize.y);
		tempIndex -= z*gridSize.x*gridSize.y;
		const int y =tempIndex/gridSize.x;
		const int x = tempIndex - y*gridSize.x;

		if(0<x && x<gridSize.x-1 &&
		   0<y && y<gridSize.y-1 &&
		   0<z && z<gridSize.z-1){

			float Tx_x=0, Tx_y=0, Tx_z=0;
			float Ty_x=0, Ty_y=0, Ty_z=0;
			float Tz_x=0, Tz_y=0, Tz_z=0;

			tempIndex=0;
			for(int c=z-1; c<z+2; ++c){
				for(int b=y-1; b<y+2; ++b){
					for(int a=x-1; a<x+2; ++a){
						int indexXYZ = (c*gridSize.y+b)*gridSize.x+a;
						float4 controlPointValues = tex1Dfetch(controlPointTexture,indexXYZ);
						Tx_x += xbasis[tempIndex]*controlPointValues.x;
						Tx_y += ybasis[tempIndex]*controlPointValues.x;
						Tx_z += zbasis[tempIndex]*controlPointValues.x;
						Ty_x += xbasis[tempIndex]*controlPointValues.y;
						Ty_y += ybasis[tempIndex]*controlPointValues.y;
						Ty_z += zbasis[tempIndex]*controlPointValues.y;
						Tz_x += xbasis[tempIndex]*controlPointValues.z;
						Tz_y += ybasis[tempIndex]*controlPointValues.z;
						Tz_z += zbasis[tempIndex]*controlPointValues.z;
						tempIndex++;
					}
				}
            }

			// The jacobian matrix is reoriented
			float Tx_x2=c_AffineMatrix0.x*Tx_x + c_AffineMatrix0.y*Ty_x + c_AffineMatrix0.z*Tz_x;
			float Tx_y2=c_AffineMatrix0.x*Tx_y + c_AffineMatrix0.y*Ty_y + c_AffineMatrix0.z*Tz_y;
			float Tx_z2=c_AffineMatrix0.x*Tx_z + c_AffineMatrix0.y*Ty_z + c_AffineMatrix0.z*Tz_z;
			float Ty_x2=c_AffineMatrix1.x*Tx_x + c_AffineMatrix1.y*Ty_x + c_AffineMatrix1.z*Tz_x;
			float Ty_y2=c_AffineMatrix1.x*Tx_y + c_AffineMatrix1.y*Ty_y + c_AffineMatrix1.z*Tz_y;
			float Ty_z2=c_AffineMatrix1.x*Tx_z + c_AffineMatrix1.y*Ty_z + c_AffineMatrix1.z*Tz_z;
			float Tz_x2=c_AffineMatrix2.x*Tx_x + c_AffineMatrix2.y*Ty_x + c_AffineMatrix2.z*Tz_x;
			float Tz_y2=c_AffineMatrix2.x*Tx_y + c_AffineMatrix2.y*Ty_y + c_AffineMatrix2.z*Tz_y;
			float Tz_z2=c_AffineMatrix2.x*Tx_z + c_AffineMatrix2.y*Ty_z + c_AffineMatrix2.z*Tz_z;

			// The Jacobian matrix is stored
			tempIndex=tid*9;
			jacobianMatrices[tempIndex++]=Tx_x2;
			jacobianMatrices[tempIndex++]=Tx_y2;
			jacobianMatrices[tempIndex++]=Tx_z2;
			jacobianMatrices[tempIndex++]=Ty_x2;
			jacobianMatrices[tempIndex++]=Ty_y2;
			jacobianMatrices[tempIndex++]=Ty_z2;
			jacobianMatrices[tempIndex++]=Tz_x2;
			jacobianMatrices[tempIndex++]=Tz_y2;
			jacobianMatrices[tempIndex] = Tz_z2;

			// The Jacobian determinant is computed and stored
			jacobianDet[tid]= Tx_x2*Ty_y2*Tz_z2
							+ Tx_y2*Ty_z2*Tz_x2
							+ Tx_z2*Ty_x2*Tz_y2
							- Tx_x2*Ty_z2*Tz_y2
							- Tx_y2*Ty_x2*Tz_z2
							- Tx_z2*Ty_y2*Tz_x2;
		}
		else{
			tempIndex=tid*9;
			jacobianMatrices[tempIndex++]=1.f;
			jacobianMatrices[tempIndex++]=0.f;
			jacobianMatrices[tempIndex++]=0.f;
			jacobianMatrices[tempIndex++]=0.f;
			jacobianMatrices[tempIndex++]=1.f;
			jacobianMatrices[tempIndex++]=0.f;
			jacobianMatrices[tempIndex++]=0.f;
			jacobianMatrices[tempIndex++]=0.f;
			jacobianMatrices[tempIndex]=1.f;
			jacobianDet[tid]= 1.0f;
		}
	}
	return;
}
/* *************************************************************** */
__global__ void reg_spline_getJacobianValues2D_kernel(float *jacobianMatrices,
													 float *jacobianDet)
{
	const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if(tid<c_VoxelNumber){

		int2 imageSize = make_int2(c_ReferenceImageDim.x,c_ReferenceImageDim.y);

		unsigned int tempIndex=tid;
		const int y = tempIndex/imageSize.x;
		const int x = tempIndex - y*imageSize.x;

		// the "nearest previous" node is determined [0,0,0]
		int2 nodeAnte;
		float2 gridVoxelSpacing = make_float2(c_ControlPointVoxelSpacing.x,c_ControlPointVoxelSpacing.y);
		nodeAnte.x = (int)floorf((float)x/gridVoxelSpacing.x);
		nodeAnte.y = (int)floorf((float)y/gridVoxelSpacing.y);

		float xBasis[4], yBasis[4], xFirst[4], yFirst[4], relative;

		relative = fabsf((float)x/gridVoxelSpacing.x-(float)nodeAnte.x);
		GetFirstBSplineValues(relative, xBasis, xFirst);

		relative = fabsf((float)y/gridVoxelSpacing.y-(float)nodeAnte.y);
		GetFirstBSplineValues(relative, yBasis, yFirst);

		int2 controlPointImageDim = make_int2(c_ControlPointImageDim.x,c_ControlPointImageDim.y);
		float2 Tx=make_float2(0.f,0.f);
		float2 Ty=make_float2(0.f,0.f);

		for(int b=0; b<4; ++b){
			int indexXY= ( nodeAnte.y + b) * controlPointImageDim.x + nodeAnte.x;

			float4 nodeCoefficient = tex1Dfetch(controlPointTexture,indexXY++);
			float2 tempBasis = make_float2(xFirst[0]*yBasis[b], xBasis[0]*yFirst[b]);
			Tx = Tx + nodeCoefficient.x * tempBasis;
			Ty = Ty + nodeCoefficient.y * tempBasis;

			nodeCoefficient = tex1Dfetch(controlPointTexture,indexXY++);
			tempBasis = make_float2(xFirst[1]*yBasis[b], xBasis[1]*yFirst[b]);
			Tx = Tx + nodeCoefficient.x * tempBasis;
			Ty = Ty + nodeCoefficient.y * tempBasis;

			nodeCoefficient = tex1Dfetch(controlPointTexture,indexXY++);
			tempBasis = make_float2(xFirst[2]*yBasis[b], xBasis[2]*yFirst[b]);
			Tx = Tx + nodeCoefficient.x * tempBasis;
			Ty = Ty + nodeCoefficient.y * tempBasis;

			nodeCoefficient = tex1Dfetch(controlPointTexture,indexXY);
			tempBasis = make_float2(xFirst[3]*yBasis[b], xBasis[3]*yFirst[b]);
			Tx = Tx + nodeCoefficient.x * tempBasis;
			Ty = Ty + nodeCoefficient.y * tempBasis;
        }

		// The jacobian matrix is reoriented
		float Tx_x2=c_AffineMatrix0.x*Tx.x + c_AffineMatrix0.y*Ty.x;
		float Tx_y2=c_AffineMatrix0.x*Tx.y + c_AffineMatrix0.y*Ty.y;
		float Ty_x2=c_AffineMatrix1.x*Tx.x + c_AffineMatrix1.y*Ty.x;
		float Ty_y2=c_AffineMatrix1.x*Tx.y + c_AffineMatrix1.y*Ty.y;

		// The Jacobian matrix is stored
		tempIndex=tid*4;
		jacobianMatrices[tempIndex++]=Tx_x2;
		jacobianMatrices[tempIndex++]=Tx_y2;
		jacobianMatrices[tempIndex++]=Ty_x2;
		jacobianMatrices[tempIndex] = Ty_y2;

		// The Jacobian determinant is computed and stored
		jacobianDet[tid]= Tx_x2 * Ty_y2 - Tx_y2 * Ty_x2;
	}
}
/* *************************************************************** */
__global__ void reg_spline_getJacobianValues3D_kernel(float *jacobianMatrices,
													 float *jacobianDet)
{
	const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if(tid<c_VoxelNumber){

		int3 imageSize = c_ReferenceImageDim;

		unsigned int tempIndex=tid;
		const int z = tempIndex/(imageSize.x*imageSize.y);
		tempIndex  -= z*imageSize.x*imageSize.y;
		const int y = tempIndex/imageSize.x;
		const int x = tempIndex - y*imageSize.x;

		// the "nearest previous" node is determined [0,0,0]
		int3 nodeAnte;
		float3 gridVoxelSpacing = c_ControlPointVoxelSpacing;
		nodeAnte.x = (int)floorf((float)x/gridVoxelSpacing.x);
		nodeAnte.y = (int)floorf((float)y/gridVoxelSpacing.y);
		nodeAnte.z = (int)floorf((float)z/gridVoxelSpacing.z);

        extern __shared__ float yFirst[];
        float *zFirst=&yFirst[4*blockDim.x*blockDim.y*blockDim.z];

		float xBasis[4], yBasis[4], zBasis[4], xFirst[4], relative;

		const int shareMemIndex = 4*threadIdx.x;

		relative = fabsf((float)x/gridVoxelSpacing.x-(float)nodeAnte.x);
		GetFirstBSplineValues(relative, xBasis, xFirst);

		relative = fabsf((float)y/gridVoxelSpacing.y-(float)nodeAnte.y);
		GetFirstBSplineValues(relative, yBasis, &yFirst[shareMemIndex]);

		relative = fabsf((float)z/gridVoxelSpacing.z-(float)nodeAnte.z);
		GetFirstBSplineValues(relative, zBasis, &zFirst[shareMemIndex]);

		int3 controlPointImageDim = c_ControlPointImageDim;
		float3 Tx=make_float3(0.f,0.f,0.f);
		float3 Ty=make_float3(0.f,0.f,0.f);
		float3 Tz=make_float3(0.f,0.f,0.f);

		for(int c=0; c<4; ++c){
			for(int b=0; b<4; ++b){
				int indexXYZ= ( (nodeAnte.z + c) * controlPointImageDim.y + nodeAnte.y + b) * controlPointImageDim.x + nodeAnte.x;
				float3 tempBasisXY=make_float3(yBasis[b]*zBasis[c],
										yFirst[shareMemIndex+b]*zBasis[c],
										yBasis[b]*zFirst[shareMemIndex+c]);

				float4 nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
				float3 tempBasis = make_float3(xFirst[0],xBasis[0],xBasis[0])*tempBasisXY;
				Tx = Tx + nodeCoefficient.x * tempBasis;
				Ty = Ty + nodeCoefficient.y * tempBasis;
				Tz = Tz + nodeCoefficient.z * tempBasis;

				nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
				tempBasis = make_float3(xFirst[1],xBasis[1],xBasis[1])*tempBasisXY;
				Tx = Tx + nodeCoefficient.x * tempBasis;
				Ty = Ty + nodeCoefficient.y * tempBasis;
				Tz = Tz + nodeCoefficient.z * tempBasis;

				nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ++);
				tempBasis = make_float3(xFirst[2],xBasis[2],xBasis[2])*tempBasisXY;
				Tx = Tx + nodeCoefficient.x * tempBasis;
				Ty = Ty + nodeCoefficient.y * tempBasis;
				Tz = Tz + nodeCoefficient.z * tempBasis;

				nodeCoefficient = tex1Dfetch(controlPointTexture,indexXYZ);
				tempBasis = make_float3(xFirst[3],xBasis[3],xBasis[3])*tempBasisXY;
				Tx = Tx + nodeCoefficient.x * tempBasis;
				Ty = Ty + nodeCoefficient.y * tempBasis;
				Tz = Tz + nodeCoefficient.z * tempBasis;
			}
        }

		// The jacobian matrix is reoriented
		float Tx_x2=c_AffineMatrix0.x*Tx.x + c_AffineMatrix0.y*Ty.x + c_AffineMatrix0.z*Tz.x;
		float Tx_y2=c_AffineMatrix0.x*Tx.y + c_AffineMatrix0.y*Ty.y + c_AffineMatrix0.z*Tz.y;
		float Tx_z2=c_AffineMatrix0.x*Tx.z + c_AffineMatrix0.y*Ty.z + c_AffineMatrix0.z*Tz.z;
		float Ty_x2=c_AffineMatrix1.x*Tx.x + c_AffineMatrix1.y*Ty.x + c_AffineMatrix1.z*Tz.x;
		float Ty_y2=c_AffineMatrix1.x*Tx.y + c_AffineMatrix1.y*Ty.y + c_AffineMatrix1.z*Tz.y;
		float Ty_z2=c_AffineMatrix1.x*Tx.z + c_AffineMatrix1.y*Ty.z + c_AffineMatrix1.z*Tz.z;
		float Tz_x2=c_AffineMatrix2.x*Tx.x + c_AffineMatrix2.y*Ty.x + c_AffineMatrix2.z*Tz.x;
		float Tz_y2=c_AffineMatrix2.x*Tx.y + c_AffineMatrix2.y*Ty.y + c_AffineMatrix2.z*Tz.y;
		float Tz_z2=c_AffineMatrix2.x*Tx.z + c_AffineMatrix2.y*Ty.z + c_AffineMatrix2.z*Tz.z;

		// The Jacobian matrix is stored
		tempIndex=tid*9;
		jacobianMatrices[tempIndex++]=Tx_x2;
		jacobianMatrices[tempIndex++]=Tx_y2;
		jacobianMatrices[tempIndex++]=Tx_z2;
		jacobianMatrices[tempIndex++]=Ty_x2;
		jacobianMatrices[tempIndex++]=Ty_y2;
		jacobianMatrices[tempIndex++]=Ty_z2;
		jacobianMatrices[tempIndex++]=Tz_x2;
		jacobianMatrices[tempIndex++]=Tz_y2;
		jacobianMatrices[tempIndex] = Tz_z2;

		// The Jacobian determinant is computed and stored
		jacobianDet[tid]= Tx_x2*Ty_y2*Tz_z2
						+ Tx_y2*Ty_z2*Tz_x2
						+ Tx_z2*Ty_x2*Tz_y2
						- Tx_x2*Ty_z2*Tz_y2
						- Tx_y2*Ty_x2*Tz_z2
						- Tx_z2*Ty_y2*Tz_x2;
	}
}
/* *************************************************************** */
__global__ void reg_spline_logSquaredValues_kernel(float *det)
{
    const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_VoxelNumber){
        float val = logf(det[tid]);
        det[tid]=val*val;
    }
}
/* *************************************************************** */
__device__ void getJacobianGradientValues2D(float *jacobianMatrix,
											float detJac,
											float basisX,
											float basisY,
											float2 *jacobianConstraint)
{
	jacobianConstraint->x += detJac * (
			basisX * jacobianMatrix[3] -
			basisY * jacobianMatrix[2] );
	jacobianConstraint->y += detJac * (
			basisY * jacobianMatrix[0] -
			basisX * jacobianMatrix[1] );
}
/* *************************************************************** */
__device__ void getJacobianGradientValues3D(float *jacobianMatrix,
											float detJac,
											float basisX,
											float basisY,
											float basisZ,
											float3 *jacobianConstraint)
{
	jacobianConstraint->x += detJac * (
			basisX * (jacobianMatrix[4]*jacobianMatrix[8] - jacobianMatrix[5]*jacobianMatrix[7]) +
			basisY * (jacobianMatrix[5]*jacobianMatrix[6] - jacobianMatrix[3]*jacobianMatrix[8]) +
			basisZ * (jacobianMatrix[3]*jacobianMatrix[7] - jacobianMatrix[4]*jacobianMatrix[6]) );

	jacobianConstraint->y += detJac * (
			basisX * (jacobianMatrix[2]*jacobianMatrix[7] - jacobianMatrix[1]*jacobianMatrix[8]) +
			basisY * (jacobianMatrix[0]*jacobianMatrix[8] - jacobianMatrix[2]*jacobianMatrix[6]) +
			basisZ * (jacobianMatrix[1]*jacobianMatrix[6] - jacobianMatrix[0]*jacobianMatrix[7]) );

	jacobianConstraint->z += detJac * (
			basisX * (jacobianMatrix[1]*jacobianMatrix[5] - jacobianMatrix[2]*jacobianMatrix[4]) +
			basisY * (jacobianMatrix[2]*jacobianMatrix[3] - jacobianMatrix[0]*jacobianMatrix[5]) +
			basisZ * (jacobianMatrix[0]*jacobianMatrix[4] - jacobianMatrix[1]*jacobianMatrix[3]) );
}
/* *************************************************************** */
__global__ void reg_spline_computeApproxJacGradient2D_kernel(float4 *gradient)
{
	__shared__ float xbasis[9];
	__shared__ float ybasis[9];

	if(threadIdx.x<9)
		GetFirstDerivativeBasisValues2D(threadIdx.x,
										xbasis,
										ybasis);
	__syncthreads();

	const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if(tid<c_ControlPointNumber){

		int3 gridSize = c_ControlPointImageDim;

		unsigned int tempIndex=tid;
		const int y =(int)(tempIndex/(gridSize.x));
		const int x = tempIndex - y*(gridSize.x);

		float2 jacobianGradient=make_float2(0.f,0.f);
		tempIndex=8;
		for(int pixelY=(int)(y-1); pixelY<(int)(y+2); ++pixelY){
			if(pixelY>0 && pixelY<gridSize.y-1){

				int jacIndex = pixelY*gridSize.x+x-1;
				for(int pixelX=(int)(x-1); pixelX<(int)(x+2); ++pixelX){
					if(pixelX>0 && pixelX<gridSize.x-1){

						float detJac = tex1Dfetch(jacobianDeterminantTexture,jacIndex);

						if(detJac>0.f){
							detJac = 2.f*logf(detJac) / detJac;
							float jacobianMatrix[4];
							jacobianMatrix[0] = tex1Dfetch(jacobianMatricesTexture,jacIndex*4);
							jacobianMatrix[1] = tex1Dfetch(jacobianMatricesTexture,jacIndex*4+1);
							jacobianMatrix[2] = tex1Dfetch(jacobianMatricesTexture,jacIndex*4+2);
							jacobianMatrix[3] = tex1Dfetch(jacobianMatricesTexture,jacIndex*4+3);

							getJacobianGradientValues2D(jacobianMatrix,
														detJac,
														xbasis[tempIndex],
														ybasis[tempIndex],
														&jacobianGradient);
						}
					}
					jacIndex++;
					tempIndex--;
				}
			}
			else tempIndex-=3;
		}
		gradient[tid] = gradient[tid] + make_float4(c_Weight3.x
													* (c_AffineMatrix0.x * jacobianGradient.x
													   + c_AffineMatrix0.y * jacobianGradient.y),
													c_Weight3.y
													* (c_AffineMatrix1.x * jacobianGradient.x
													   + c_AffineMatrix1.y * jacobianGradient.y),
													0.f,
													0.f);

	}
}
/* *************************************************************** */
__global__ void reg_spline_computeApproxJacGradient3D_kernel(float4 *gradient)
{
	__shared__ float xbasis[27];
	__shared__ float ybasis[27];
	__shared__ float zbasis[27];

	if(threadIdx.x<27)
		GetFirstDerivativeBasisValues3D(threadIdx.x,
									  xbasis,
									  ybasis,
									  zbasis);
	__syncthreads();

	const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if(tid<c_ControlPointNumber){

		int3 gridSize = c_ControlPointImageDim;

		unsigned int tempIndex=tid;
		const int z =(int)(tempIndex/(gridSize.x*gridSize.y));
		tempIndex -= z*(gridSize.x)*(gridSize.y);
		const int y =(int)(tempIndex/(gridSize.x));
		const int x = tempIndex - y*(gridSize.x);

		float3 jacobianGradient=make_float3(0.f,0.f,0.f);
		tempIndex=26;
		for(int pixelZ=(int)(z-1); pixelZ<(int)(z+2); ++pixelZ){
			if(pixelZ>0 && pixelZ<gridSize.z-1){

				for(int pixelY=(int)(y-1); pixelY<(int)(y+2); ++pixelY){
					if(pixelY>0 && pixelY<gridSize.y-1){

						int jacIndex = (pixelZ*gridSize.y+pixelY)*gridSize.x+x-1;
						for(int pixelX=(int)(x-1); pixelX<(int)(x+2); ++pixelX){
							if(pixelX>0 && pixelX<gridSize.x-1){

								float detJac = tex1Dfetch(jacobianDeterminantTexture,jacIndex);

								if(detJac>0.f){
									detJac = 2.f*logf(detJac) / detJac;
									float jacobianMatrix[9];
									jacobianMatrix[0] = tex1Dfetch(jacobianMatricesTexture,jacIndex*9);
									jacobianMatrix[1] = tex1Dfetch(jacobianMatricesTexture,jacIndex*9+1);
									jacobianMatrix[2] = tex1Dfetch(jacobianMatricesTexture,jacIndex*9+2);
									jacobianMatrix[3] = tex1Dfetch(jacobianMatricesTexture,jacIndex*9+3);
									jacobianMatrix[4] = tex1Dfetch(jacobianMatricesTexture,jacIndex*9+4);
									jacobianMatrix[5] = tex1Dfetch(jacobianMatricesTexture,jacIndex*9+5);
									jacobianMatrix[6] = tex1Dfetch(jacobianMatricesTexture,jacIndex*9+6);
									jacobianMatrix[7] = tex1Dfetch(jacobianMatricesTexture,jacIndex*9+7);
									jacobianMatrix[8] = tex1Dfetch(jacobianMatricesTexture,jacIndex*9+8);

									getJacobianGradientValues3D(jacobianMatrix,
															  detJac,
															  xbasis[tempIndex],
															  ybasis[tempIndex],
															  zbasis[tempIndex],
															  &jacobianGradient);
								}
							}
							jacIndex++;
							tempIndex--;
						}
					}
					else tempIndex-=3;
				}
			}
			else tempIndex-=9;
		}
		gradient[tid] = gradient[tid] + make_float4(c_Weight3.x
													* (c_AffineMatrix0.x * jacobianGradient.x
													   + c_AffineMatrix0.y * jacobianGradient.y
													   + c_AffineMatrix0.z * jacobianGradient.z),
													c_Weight3.y
													* (c_AffineMatrix1.x * jacobianGradient.x
													   + c_AffineMatrix1.y * jacobianGradient.y
													   + c_AffineMatrix1.z * jacobianGradient.z),
													c_Weight3.z
													* (c_AffineMatrix2.x * jacobianGradient.x
													   + c_AffineMatrix2.y * jacobianGradient.y
													   + c_AffineMatrix2.z * jacobianGradient.z),
													0.f);

	}
}
/* *************************************************************** */
__global__ void reg_spline_computeJacGradient2D_kernel(float4 *gradient)
{
	const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if(tid<c_ControlPointNumber){

		int3 gridSize = c_ControlPointImageDim;

		int tempIndex=tid;
		const int y = tempIndex/gridSize.x;
		const int x = tempIndex - y*gridSize.x;

		float2 jacobianGradient=make_float2(0.f,0.f);

		float3 spacingVoxel = c_ControlPointVoxelSpacing;

		for(int pixelY=(int)ceilf((y-3)*spacingVoxel.y);
			pixelY<=(int)ceilf((y+1)*spacingVoxel.y);
			++pixelY){
			if(pixelY>-1 && pixelY<c_ReferenceImageDim.y){

				int yPre = (int)((float)pixelY/spacingVoxel.y);
				float basis = (float)pixelY/spacingVoxel.y - (float)yPre;
				float yBasis, yFirst;
				getBSplineBasisValue(basis,y-yPre,&yBasis,&yFirst);

				for(int pixelX=(int)ceilf((x-3)*spacingVoxel.x);
					pixelX<=(int)ceilf((x+1)*spacingVoxel.x);
					++pixelX){
					if(pixelX>-1 && pixelX<c_ReferenceImageDim.x && (yFirst!=0.f || yBasis!=0.f)){

						int xPre = (int)((float)pixelX/spacingVoxel.x);
						basis = (float)pixelX/spacingVoxel.x - (float)xPre;
						float xBasis, xFirst;
						getBSplineBasisValue(basis,x-xPre,&xBasis,&xFirst);

						int jacIndex = pixelY*c_ReferenceImageDim.x + pixelX;

						float detJac = tex1Dfetch(jacobianDeterminantTexture,jacIndex);

						if(detJac>0.f && (xFirst!=0.f || xBasis!=0.f)){
							detJac = 2.f*logf(detJac) / detJac;
							float jacobianMatrix[4];
							jacIndex *= 4;
							jacobianMatrix[0] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
							jacobianMatrix[1] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
							jacobianMatrix[2] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
							jacobianMatrix[3] = tex1Dfetch(jacobianMatricesTexture,jacIndex);

							float2 basisValues = make_float2(
										xFirst*yBasis,
										xBasis*yFirst);
							getJacobianGradientValues2D(jacobianMatrix,
														detJac,
														basisValues.x,
														basisValues.y,
														&jacobianGradient);
						}
					}
				}
			}
		}
		gradient[tid] = gradient[tid] + make_float4(
						c_Weight3.x
						* (c_AffineMatrix0.x * jacobianGradient.x
						   + c_AffineMatrix0.y * jacobianGradient.y),
						c_Weight3.y
						* (c_AffineMatrix1.x * jacobianGradient.x
						   + c_AffineMatrix1.y * jacobianGradient.y),
						0.f,
						0.f);
   }
}
/* *************************************************************** */
__global__ void reg_spline_computeJacGradient3D_kernel(float4 *gradient)
{
	const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
	if(tid<c_ControlPointNumber){

		int3 gridSize = c_ControlPointImageDim;

		int tempIndex=tid;
		const int z = tempIndex/(gridSize.x*gridSize.y);
		tempIndex  -= z*gridSize.x*gridSize.y;
		const int y = tempIndex/gridSize.x;
		const int x = tempIndex - y*gridSize.x;

		float3 jacobianGradient=make_float3(0.f,0.f,0.f);

		float3 spacingVoxel = c_ControlPointVoxelSpacing;

		for(int pixelZ=(int)ceilf((z-3)*spacingVoxel.z);
			pixelZ<=(int)ceilf((z+1)*spacingVoxel.z);
			++pixelZ){
			if(pixelZ>-1 && pixelZ<c_ReferenceImageDim.z){

				int zPre = (int)((float)pixelZ/spacingVoxel.z);
				float basis = (float)pixelZ/spacingVoxel.z - (float)zPre;
				float zBasis, zFirst;
				getBSplineBasisValue(basis,z-zPre,&zBasis,&zFirst);

				for(int pixelY=(int)ceilf((y-3)*spacingVoxel.y);
					pixelY<=(int)ceilf((y+1)*spacingVoxel.y);
					++pixelY){
					if(pixelY>-1 && pixelY<c_ReferenceImageDim.y && (zFirst!=0.f || zBasis!=0.f)){

						int yPre = (int)((float)pixelY/spacingVoxel.y);
						basis = (float)pixelY/spacingVoxel.y - (float)yPre;
						float yBasis, yFirst;
						getBSplineBasisValue(basis,y-yPre,&yBasis,&yFirst);

						for(int pixelX=(int)ceilf((x-3)*spacingVoxel.x);
							pixelX<=(int)ceilf((x+1)*spacingVoxel.x);
							++pixelX){
							if(pixelX>-1 && pixelX<c_ReferenceImageDim.x && (yFirst!=0.f || yBasis!=0.f)){

								int xPre = (int)((float)pixelX/spacingVoxel.x);
								basis = (float)pixelX/spacingVoxel.x - (float)xPre;
								float xBasis, xFirst;
								getBSplineBasisValue(basis,x-xPre,&xBasis,&xFirst);

								int jacIndex = (pixelZ*c_ReferenceImageDim.y+pixelY)*c_ReferenceImageDim.x + pixelX;

								float detJac = tex1Dfetch(jacobianDeterminantTexture,jacIndex);

								if(detJac>0.f && (xFirst!=0.f || xBasis!=0.f)){
									detJac = 2.f*logf(detJac) / detJac;
									float jacobianMatrix[9];
									jacIndex *= 9;
									jacobianMatrix[0] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
									jacobianMatrix[1] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
									jacobianMatrix[2] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
									jacobianMatrix[3] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
									jacobianMatrix[4] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
									jacobianMatrix[5] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
									jacobianMatrix[6] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
									jacobianMatrix[7] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
									jacobianMatrix[8] = tex1Dfetch(jacobianMatricesTexture,jacIndex);

									float3 basisValues = make_float3(
											xFirst*yBasis*zBasis,
											xBasis*yFirst*zBasis,
											xBasis*yBasis*zFirst);
									getJacobianGradientValues3D(jacobianMatrix,
															  detJac,
															  basisValues.x,
															  basisValues.y,
															  basisValues.z,
															  &jacobianGradient);
								}
							}
						}
					}
				}
			}
		}
		gradient[tid] = gradient[tid] + make_float4(
						c_Weight3.x
						* (c_AffineMatrix0.x * jacobianGradient.x
						   + c_AffineMatrix0.y * jacobianGradient.y
						   + c_AffineMatrix0.z * jacobianGradient.z),
						c_Weight3.y
						* (c_AffineMatrix1.x * jacobianGradient.x
						   + c_AffineMatrix1.y * jacobianGradient.y
						   + c_AffineMatrix1.z * jacobianGradient.z),
						c_Weight3.z
						* (c_AffineMatrix2.x * jacobianGradient.x
						   + c_AffineMatrix2.y * jacobianGradient.y
						   + c_AffineMatrix2.z * jacobianGradient.z),
						0.f);
   }
}
/* *************************************************************** */
__global__ void reg_spline_approxCorrectFolding3D_kernel(float4 *controlPointGrid_d)
{
    const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ControlPointNumber){

        int3 gridSize = c_ControlPointImageDim;

        unsigned int tempIndex=tid;
        const int z = tempIndex/(gridSize.x*gridSize.y);
        tempIndex  -= z*gridSize.x*gridSize.y;
        const int y = tempIndex/gridSize.x;
        const int x = tempIndex - y*gridSize.x;

        float3 foldingCorrection=make_float3(0.f,0.f,0.f);
        for(int pixelZ=(int)(z-1); pixelZ<(int)(z+2); ++pixelZ){
            if(pixelZ>0 && pixelZ<gridSize.z-1){

                for(int pixelY=(int)(y-1); pixelY<(int)(y+2); ++pixelY){
                    if(pixelY>0 && pixelY<gridSize.y-1){

                        for(int pixelX=(int)(x-1); pixelX<(int)(x+2); ++pixelX){
                            if(pixelX>0 && pixelX<gridSize.x-1){

                                int jacIndex = (pixelZ*gridSize.y+pixelY)*gridSize.x+pixelX;
                                float detJac = tex1Dfetch(jacobianDeterminantTexture,jacIndex);

                                if(detJac<=0.f){

                                    float jacobianMatrix[9];
                                    jacIndex*=9;
                                    jacobianMatrix[0] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
                                    jacobianMatrix[1] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
                                    jacobianMatrix[2] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
                                    jacobianMatrix[3] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
                                    jacobianMatrix[4] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
                                    jacobianMatrix[5] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
                                    jacobianMatrix[6] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
                                    jacobianMatrix[7] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
                                    jacobianMatrix[8] = tex1Dfetch(jacobianMatricesTexture,jacIndex);

                                    float xBasis, xFirst, yBasis, yFirst, zBasis, zFirst;
                                    getBSplineBasisValue(0.f,x-pixelX+1,&xBasis,&xFirst);
                                    getBSplineBasisValue(0.f,y-pixelY+1,&yBasis,&yFirst);
                                    getBSplineBasisValue(0.f,z-pixelZ+1,&zBasis,&zFirst);

                                    float3 basisValue = make_float3(
                                            xFirst*yBasis*zBasis,
                                            xBasis*yFirst*zBasis,
                                            xBasis*yBasis*zFirst);

                                    getJacobianGradientValues3D(jacobianMatrix,
                                                              1.f,
                                                              basisValue.x,
                                                              basisValue.y,
                                                              basisValue.z,
                                                              &foldingCorrection);
                                }
                            }
                        }
                    }
                }
            }
        }
        if(foldingCorrection.x!=0.f && foldingCorrection.y!=0.f && foldingCorrection.z!=0.f){
            float3 gradient = make_float3(
                c_AffineMatrix0.x * foldingCorrection.x
                + c_AffineMatrix0.y * foldingCorrection.y
                + c_AffineMatrix0.z * foldingCorrection.z,
                c_AffineMatrix1.x * foldingCorrection.x
               + c_AffineMatrix1.y * foldingCorrection.y
               + c_AffineMatrix1.z * foldingCorrection.z,
               c_AffineMatrix2.x * foldingCorrection.x
               + c_AffineMatrix2.y * foldingCorrection.y
               + c_AffineMatrix2.z * foldingCorrection.z);

            float norm = 5.f * sqrtf(gradient.x*gradient.x
                                     + gradient.y*gradient.y
                                     + gradient.z*gradient.z);
            controlPointGrid_d[tid] = controlPointGrid_d[tid] +
                                      make_float4(gradient.x*c_ControlPointSpacing.x/norm,
                                                  gradient.y*c_ControlPointSpacing.y/norm,
                                                  gradient.z*c_ControlPointSpacing.z/norm,
                                                  0.f);
        }
    }
}
/* *************************************************************** */
__global__ void reg_spline_correctFolding3D_kernel(float4 *controlPointGrid_d)
{
    const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_ControlPointNumber){

        int3 gridSize = c_ControlPointImageDim;

        unsigned int tempIndex=tid;
        const int z = tempIndex/(gridSize.x*gridSize.y);
        tempIndex  -= z*gridSize.x*gridSize.y;
        const int y = tempIndex/gridSize.x;
        const int x = tempIndex - y*gridSize.x;

        float3 spacingVoxel = c_ControlPointVoxelSpacing;
        float3 foldingCorrection=make_float3(0.f,0.f,0.f);

        for(int pixelZ=(int)ceilf((z-3)*spacingVoxel.z);
            pixelZ<(int)ceilf((z+1)*spacingVoxel.z);
            ++pixelZ){
            if(pixelZ>-1 && pixelZ<c_ReferenceImageDim.z){

                for(int pixelY=(int)ceilf((y-3)*spacingVoxel.y);
                    pixelY<(int)ceilf((y+1)*spacingVoxel.y);
                    ++pixelY){
                    if(pixelY>-1 && pixelY<c_ReferenceImageDim.y){

                        for(int pixelX=(int)ceilf((x-3)*spacingVoxel.x);
                            pixelX<(int)ceilf((x+1)*spacingVoxel.x);
                            ++pixelX){
                            if(pixelX>-1 && pixelX<c_ReferenceImageDim.x){

                                int jacIndex = (pixelZ*c_ReferenceImageDim.y+pixelY)*c_ReferenceImageDim.x+pixelX;
                                float detJac = tex1Dfetch(jacobianDeterminantTexture,jacIndex);

                                if(detJac<=0.f){

                                    float jacobianMatrix[9];
                                    jacIndex*=9;
                                    jacobianMatrix[0] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
                                    jacobianMatrix[1] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
                                    jacobianMatrix[2] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
                                    jacobianMatrix[3] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
                                    jacobianMatrix[4] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
                                    jacobianMatrix[5] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
                                    jacobianMatrix[6] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
                                    jacobianMatrix[7] = tex1Dfetch(jacobianMatricesTexture,jacIndex++);
                                    jacobianMatrix[8] = tex1Dfetch(jacobianMatricesTexture,jacIndex);

                                    float xBasis, xFirst, yBasis, yFirst, zBasis, zFirst;
                                    int pre=(int)((float)pixelX/spacingVoxel.x);
                                    float basis=(float)pixelX/spacingVoxel.x-(float)pre;
                                    getBSplineBasisValue(basis,x-pre,&xBasis,&xFirst);
                                    pre=(int)((float)pixelY/spacingVoxel.y);
                                    basis=(float)pixelY/spacingVoxel.y-(float)pre;
                                    getBSplineBasisValue(basis,y-pre,&yBasis,&yFirst);
                                    pre=(int)((float)pixelZ/spacingVoxel.z);
                                    basis=(float)pixelZ/spacingVoxel.z-(float)pre;
                                    getBSplineBasisValue(basis,z-pre,&zBasis,&zFirst);

                                    float3 basisValue = make_float3(
                                            xFirst*yBasis*zBasis,
                                            xBasis*yFirst*zBasis,
                                            xBasis*yBasis*zFirst);

                                    getJacobianGradientValues3D(jacobianMatrix,
                                                              1.f,
                                                              basisValue.x,
                                                              basisValue.y,
                                                              basisValue.z,
                                                              &foldingCorrection);
                                }
                            }
                        }
                    }
                }
            }
        }
        if(foldingCorrection.x!=0.f && foldingCorrection.y!=0.f && foldingCorrection.z!=0.f){
            float3 gradient = make_float3(
                c_AffineMatrix0.x * foldingCorrection.x
                + c_AffineMatrix0.y * foldingCorrection.y
                + c_AffineMatrix0.z * foldingCorrection.z,
                c_AffineMatrix1.x * foldingCorrection.x
               + c_AffineMatrix1.y * foldingCorrection.y
               + c_AffineMatrix1.z * foldingCorrection.z,
               c_AffineMatrix2.x * foldingCorrection.x
               + c_AffineMatrix2.y * foldingCorrection.y
               + c_AffineMatrix2.z * foldingCorrection.z);

            float norm = 5.f * sqrtf(gradient.x*gradient.x
                                     + gradient.y*gradient.y
                                     + gradient.z*gradient.z);
            controlPointGrid_d[tid] = controlPointGrid_d[tid] +
                                      make_float4(gradient.x*c_ControlPointSpacing.x/norm,
                                                  gradient.y*c_ControlPointSpacing.y/norm,
                                                  gradient.z*c_ControlPointSpacing.z/norm,
                                                  0.f);
        }
    }
}
/* *************************************************************** */
__global__ void reg_getDeformationFromDisplacement3D_kernel(float4 *imageArray_d)
{
    const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_VoxelNumber){

        int3 imageSize = c_ReferenceImageDim;

        unsigned int tempIndex=tid;
        const int z = tempIndex/(imageSize.x*imageSize.y);
        tempIndex  -= z*imageSize.x*imageSize.y;
        const int y = tempIndex/imageSize.x;
        const int x = tempIndex - y*imageSize.x;

        float4 initialPosition;
        initialPosition.x=x*c_AffineMatrix0b.x + y*c_AffineMatrix0b.y + z*c_AffineMatrix0b.z + c_AffineMatrix0b.w;
        initialPosition.y=x*c_AffineMatrix1b.x + y*c_AffineMatrix1b.y + z*c_AffineMatrix1b.z + c_AffineMatrix1b.w;
        initialPosition.z=x*c_AffineMatrix2b.x + y*c_AffineMatrix2b.y + z*c_AffineMatrix2b.z + c_AffineMatrix2b.w;
        initialPosition.w=0.f;

        imageArray_d[tid] = imageArray_d[tid] + initialPosition;
    }
}
/* *************************************************************** */
__global__ void reg_getDisplacementFromDeformation3D_kernel(float4 *imageArray_d)
{
    const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_VoxelNumber){

        int3 imageSize = c_ReferenceImageDim;

        unsigned int tempIndex=tid;
        const int z = tempIndex/(imageSize.x*imageSize.y);
        tempIndex  -= z*imageSize.x*imageSize.y;
        const int y = tempIndex/imageSize.x;
        const int x = tempIndex - y*imageSize.x;

        float4 initialPosition;
        initialPosition.x=x*c_AffineMatrix0b.x + y*c_AffineMatrix0b.y + z*c_AffineMatrix0b.z + c_AffineMatrix0b.w;
        initialPosition.y=x*c_AffineMatrix1b.x + y*c_AffineMatrix1b.y + z*c_AffineMatrix1b.z + c_AffineMatrix1b.w;
        initialPosition.z=x*c_AffineMatrix2b.x + y*c_AffineMatrix2b.y + z*c_AffineMatrix2b.z + c_AffineMatrix2b.w;
        initialPosition.w=0.f;

        imageArray_d[tid] = imageArray_d[tid] - initialPosition;
    }
}
/* *************************************************************** */
__global__ void reg_defField_compose2D_kernel(float4 *outDef)
{
    const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_VoxelNumber){

        // Extract the original voxel position
        float4 position=outDef[tid];

        // Conversion from real position to voxel coordinate
        float4 voxelPosition;
        voxelPosition.x=
                position.x*c_AffineMatrix0b.x +
                position.y*c_AffineMatrix0b.y +
                c_AffineMatrix0b.w;
        voxelPosition.y=
                position.x*c_AffineMatrix1b.x +
                position.y*c_AffineMatrix1b.y +
                c_AffineMatrix1b.w;
        voxelPosition.z=0.f;
        voxelPosition.w=0.f;

        // linear interpolation
        int2 ante=make_int2(floorf(voxelPosition.x),
                            floorf(voxelPosition.y));

        float relX[2], relY[2];
        relX[1]=voxelPosition.x-(float)ante.x;relX[0]=1.f-relX[1];
        relY[1]=voxelPosition.y-(float)ante.y;relY[0]=1.f-relY[1];

        position=make_float4(0.f,0.f,0.f,0.f);

        for(int b=0;b<2;++b){
            for(int a=0;a<2;++a){
                unsigned int index=(ante.y+b)*c_ReferenceImageDim.x+ante.x+a;
                float4 deformation;
                if((ante.x+a)>-1 && (ante.y+b)>-1 &&
                   (ante.x+a)<c_ReferenceImageDim.x &&
                   (ante.y+b)<c_ReferenceImageDim.y){
                    deformation=tex1Dfetch(voxelDeformationTexture,index);
                }
                else{
                    deformation = get_SlidedValues_gpu((ante.x+a),
                                                       (ante.y+b));
                }
                float basis=relX[a]*relY[b];
                position=position+basis*deformation;
            }
        }
        outDef[tid]=position;
    }
}
/* *************************************************************** */
__global__ void reg_defField_compose3D_kernel(float4 *outDef)
{
    const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_VoxelNumber){

        // Extract the original voxel position
        float4 position=outDef[tid];

        // Conversion from real position to voxel coordinate
        float4 voxelPosition;
        voxelPosition.x=
                position.x*c_AffineMatrix0b.x +
                position.y*c_AffineMatrix0b.y +
                position.z*c_AffineMatrix0b.z +
                c_AffineMatrix0b.w;
        voxelPosition.y=
                position.x*c_AffineMatrix1b.x +
                position.y*c_AffineMatrix1b.y +
                position.z*c_AffineMatrix1b.z +
                c_AffineMatrix1b.w;
        voxelPosition.z=
                position.x*c_AffineMatrix2b.x +
                position.y*c_AffineMatrix2b.y +
                position.z*c_AffineMatrix2b.z +
                c_AffineMatrix2b.w;
        voxelPosition.w=0.f;

        // linear interpolation
        int3 ante=make_int3(floorf(voxelPosition.x),
                            floorf(voxelPosition.y),
                            floorf(voxelPosition.z));

        float relX[2], relY[2], relZ[2];
        relX[1]=voxelPosition.x-(float)ante.x;relX[0]=1.f-relX[1];
        relY[1]=voxelPosition.y-(float)ante.y;relY[0]=1.f-relY[1];
        relZ[1]=voxelPosition.z-(float)ante.z;relZ[0]=1.f-relZ[1];

        position=make_float4(0.f,0.f,0.f,0.f);

        for(int c=0;c<2;++c){
            for(int b=0;b<2;++b){
                for(int a=0;a<2;++a){
                    unsigned int index=((ante.z+c)*c_ReferenceImageDim.y+ante.y+b)*c_ReferenceImageDim.x+ante.x+a;
                    float4 deformation;
                    if((ante.x+a)>-1 && (ante.y+b)>-1 && (ante.z+c)>-1 &&
                       (ante.x+a)<c_ReferenceImageDim.x &&
                       (ante.y+b)<c_ReferenceImageDim.y &&
                       (ante.z+c)<c_ReferenceImageDim.z){
                        deformation=tex1Dfetch(voxelDeformationTexture,index);
                    }
                    else{
                        deformation = get_SlidedValues_gpu((ante.x+a),
                                                           (ante.y+b),
                                                           (ante.z+c));
                    }
                    float basis=relX[a]*relY[b]*relZ[c];
                    position=position+basis*deformation;
                }
            }
        }
        outDef[tid]=position;
    }
}
/* *************************************************************** */
__global__ void reg_defField_getJacobianMatrix3D_kernel(float *jacobianMatrices)
{
    const unsigned int tid= (blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x+threadIdx.x;
    if(tid<c_VoxelNumber){

        int3 imageSize = c_ReferenceImageDim;

        unsigned int tempIndex=tid;
        const int z = tempIndex/(imageSize.x*imageSize.y);
        tempIndex  -= z*imageSize.x*imageSize.y;
        const int y = tempIndex/imageSize.x;
        const int x = tempIndex - y*imageSize.x;

        if(x==imageSize.x-1 ||
           y==imageSize.y-1 ||
           z==imageSize.z-1 ){
            int index=tid*9;
            jacobianMatrices[index++]=1.0;
            jacobianMatrices[index++]=0.0;
            jacobianMatrices[index++]=0.0;
            jacobianMatrices[index++]=0.0;
            jacobianMatrices[index++]=1.0;
            jacobianMatrices[index++]=0.0;
            jacobianMatrices[index++]=0.0;
            jacobianMatrices[index++]=0.0;
            jacobianMatrices[index]=1.0;
            return;
        }

        float matrix[9];
        int index=(z*imageSize.y+y)*imageSize.x+x;
        float4 deformation = tex1Dfetch(voxelDeformationTexture,index);
        matrix[0] = deformation.x * -1.f;
        matrix[1] = deformation.x * -1.f;
        matrix[2] = deformation.x * -1.f;
        matrix[3] = deformation.y * -1.f;
        matrix[4] = deformation.y * -1.f;
        matrix[5] = deformation.y * -1.f;
        matrix[6] = deformation.z * -1.f;
        matrix[7] = deformation.z * -1.f;
        matrix[8] = deformation.z * -1.f;
        deformation = tex1Dfetch(voxelDeformationTexture,index+1);
        matrix[0] += deformation.x * 1.f;
        matrix[3] += deformation.y * 1.f;
        matrix[6] += deformation.z * 1.f;
        index=(z*imageSize.y+y+1)*imageSize.x+x;
        deformation = tex1Dfetch(voxelDeformationTexture,index);
        matrix[1] += deformation.x * 1.f;
        matrix[4] += deformation.y * 1.f;
        matrix[7] += deformation.z * 1.f;
        index=((z+1)*imageSize.y+y)*imageSize.x+x;
        deformation = tex1Dfetch(voxelDeformationTexture,index);
        matrix[2] += deformation.x * 1.f;
        matrix[5] += deformation.y * 1.f;
        matrix[8] += deformation.z * 1.f;

        index=tid*9;
        jacobianMatrices[index++]=c_AffineMatrix0.x*matrix[0] + c_AffineMatrix0.y*matrix[3] + c_AffineMatrix0.z*matrix[6];
        jacobianMatrices[index++]=c_AffineMatrix0.x*matrix[1] + c_AffineMatrix0.y*matrix[4] + c_AffineMatrix0.z*matrix[7];
        jacobianMatrices[index++]=c_AffineMatrix0.x*matrix[2] + c_AffineMatrix0.y*matrix[5] + c_AffineMatrix0.z*matrix[8];
        jacobianMatrices[index++]=c_AffineMatrix1.x*matrix[0] + c_AffineMatrix1.y*matrix[3] + c_AffineMatrix1.z*matrix[6];
        jacobianMatrices[index++]=c_AffineMatrix1.x*matrix[1] + c_AffineMatrix1.y*matrix[4] + c_AffineMatrix1.z*matrix[7];
        jacobianMatrices[index++]=c_AffineMatrix1.x*matrix[2] + c_AffineMatrix1.y*matrix[5] + c_AffineMatrix1.z*matrix[8];
        jacobianMatrices[index++]=c_AffineMatrix2.x*matrix[0] + c_AffineMatrix2.y*matrix[3] + c_AffineMatrix2.z*matrix[6];
        jacobianMatrices[index++]=c_AffineMatrix2.x*matrix[1] + c_AffineMatrix2.y*matrix[4] + c_AffineMatrix2.z*matrix[7];
        jacobianMatrices[index] = c_AffineMatrix2.x*matrix[2] + c_AffineMatrix2.y*matrix[5] + c_AffineMatrix2.z*matrix[8];
    }
}
/* *************************************************************** */
/* *************************************************************** */
/* *************************************************************** */
#endif
