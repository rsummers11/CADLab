/*
 *  _reg_femTransformation_gpu.h
 *
 *
 *  Created by Marc Modat on 02/11/2011.
 *  Copyright (c) 2011, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_FEMTRANSFORMATION_CPP
#define _REG_FEMTRANSFORMATION_CPP

#include "_reg_femTransformation.h"

float reg_getTetrahedronVolume(float *node1,float *node2,float *node3,float *node4)
{
   mat33 matrix;
   matrix.m[0][0]=node2[0]-node1[0];
   matrix.m[0][1]=node2[1]-node1[1];
   matrix.m[0][2]=node2[2]-node1[2];

   matrix.m[1][0]=node3[0]-node2[0];
   matrix.m[1][1]=node3[1]-node2[1];
   matrix.m[1][2]=node3[2]-node2[2];

   matrix.m[2][0]=node4[0]-node3[0];
   matrix.m[2][1]=node4[1]-node3[1];
   matrix.m[2][2]=node4[2]-node3[2];
   return fabs(nifti_mat33_determ(matrix))/6.f;
}

void reg_fem_InitialiseTransformation(int *elementNodes,
                                      unsigned int elementNumber,
                                      float *nodePositions,
                                      nifti_image *deformationFieldImage,
                                      unsigned int *closestNodes,
                                      float *femInterpolationWeight
                                     )
{
   // Set all the closest nodes and coefficients to zero
   for(int i=0; i<4*deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz; ++i)
   {
      closestNodes[i]=0;
      femInterpolationWeight[i]=0.f;
   }

   mat44 *realToVoxel;
   if(deformationFieldImage->sform_code>0)
   {
      realToVoxel=&(deformationFieldImage->sto_ijk);
   }
   else realToVoxel=&(deformationFieldImage->qto_ijk);

   int currentNodes[4];
   float nodeRealPosition[3];
   float nodeVoxelIndices[4][3];
   float voxel[3];
   float fullVolume;
   float subVolume[4];

   for(unsigned int element=0; element<elementNumber; ++element)
   {
      // Compute the element bounding box in voxel coordinate
      for(unsigned int i=0; i<4; ++i)
      {
         currentNodes[i]=elementNodes[4*element+i];
         nodeRealPosition[0]=nodePositions[3*currentNodes[i]];
         nodeRealPosition[1]=nodePositions[3*currentNodes[i]+1];
         nodeRealPosition[2]=nodePositions[3*currentNodes[i]+2];
         reg_mat44_mul(realToVoxel, nodeRealPosition, nodeVoxelIndices[i]);
      }

      int xRange[2]= {(int)reg_ceil(nodeVoxelIndices[0][0]), (int)reg_floor(nodeVoxelIndices[0][0])};
      int yRange[2]= {(int)reg_ceil(nodeVoxelIndices[0][1]), (int)reg_floor(nodeVoxelIndices[0][1])};
      int zRange[2]= {(int)reg_ceil(nodeVoxelIndices[0][2]), (int)reg_floor(nodeVoxelIndices[0][2])};
      for(unsigned int i=1; i<4; ++i)
      {
         xRange[0]=xRange[0]<(int)reg_ceil(nodeVoxelIndices[i][0])?xRange[0]:(int)reg_ceil(nodeVoxelIndices[i][0]);
         xRange[1]=xRange[1]>(int)reg_floor(nodeVoxelIndices[i][0])?xRange[1]:(int)reg_floor(nodeVoxelIndices[i][0]);
         yRange[0]=yRange[0]<(int)reg_ceil(nodeVoxelIndices[i][1])?yRange[0]:(int)reg_ceil(nodeVoxelIndices[i][1]);
         yRange[1]=yRange[1]>(int)reg_floor(nodeVoxelIndices[i][1])?yRange[1]:(int)reg_floor(nodeVoxelIndices[i][1]);
         zRange[0]=zRange[0]<(int)reg_ceil(nodeVoxelIndices[i][2])?zRange[0]:(int)reg_ceil(nodeVoxelIndices[i][2]);
         zRange[1]=zRange[1]>(int)reg_floor(nodeVoxelIndices[i][2])?zRange[1]:(int)reg_floor(nodeVoxelIndices[i][2]);
      }

      xRange[0]=xRange[0]<0?0:xRange[0];
      yRange[0]=yRange[0]<0?0:yRange[0];
      zRange[0]=zRange[0]<0?0:zRange[0];
      xRange[1]=xRange[1]<deformationFieldImage->nx?xRange[1]:deformationFieldImage->nx-1;
      yRange[1]=yRange[1]<deformationFieldImage->ny?yRange[1]:deformationFieldImage->ny-1;
      zRange[1]=zRange[1]<deformationFieldImage->nz?zRange[1]:deformationFieldImage->nz-1;

      fullVolume=reg_getTetrahedronVolume(nodeVoxelIndices[0],
                                          nodeVoxelIndices[1],
                                          nodeVoxelIndices[2],
                                          nodeVoxelIndices[3]);
      for(int z=zRange[0]; z<=zRange[1]; ++z)
      {
         voxel[2]=z;
         for(int y=yRange[0]; y<=yRange[1]; ++y)
         {
            voxel[1]=y;
            for(int x=xRange[0]; x<=xRange[1]; ++x)
            {
               voxel[0]=x;
               subVolume[0]=reg_getTetrahedronVolume(voxel,
                                                     nodeVoxelIndices[1],
                                                     nodeVoxelIndices[2],
                                                     nodeVoxelIndices[3]);

               subVolume[1]=reg_getTetrahedronVolume(nodeVoxelIndices[0],
                                                     voxel,
                                                     nodeVoxelIndices[2],
                                                     nodeVoxelIndices[3]);

               subVolume[2]=reg_getTetrahedronVolume(nodeVoxelIndices[0],
                                                     nodeVoxelIndices[1],
                                                     voxel,
                                                     nodeVoxelIndices[3]);

               subVolume[3]=reg_getTetrahedronVolume(nodeVoxelIndices[0],
                                                     nodeVoxelIndices[1],
                                                     nodeVoxelIndices[2],
                                                     voxel);

               // Check if the voxel is in the element
               if(fabs(fullVolume/(subVolume[0]+subVolume[1]+subVolume[2]+subVolume[3])-1.f)<.000001f)
               {
                  int index=(z*deformationFieldImage->ny+y)*deformationFieldImage->nx+x;
                  for(unsigned int i=0; i<4; ++i)
                  {
                     closestNodes[4*index+i]=currentNodes[i];
                     femInterpolationWeight[4*index+i]=subVolume[i]/fullVolume;
                  }
               }// voxel in element check
            }//x bounding box
         }//y bounding box
      }//z bounding box
   }// element loop
   return;
}// reg_fem_InitialiseTransformation


void reg_fem_getDeformationField(float *nodePositions,
                                 nifti_image *deformationFieldImage,
                                 unsigned int *closestNodes,
                                 float *femInterpolationWeight
                                )
{
#ifdef _WIN32
   long voxel;
   long voxelNumber=(long)deformationFieldImage->nx*
      deformationFieldImage->ny*deformationFieldImage->nz;
#else
   size_t voxel;
   size_t voxelNumber=(size_t)deformationFieldImage->nx*
      deformationFieldImage->ny*deformationFieldImage->nz;
#endif
   float *defPtrX = static_cast<float *>(deformationFieldImage->data);
   float *defPtrY = &defPtrX[voxelNumber];
   float *defPtrZ = &defPtrY[voxelNumber];

   float coefficients[4];
   float positionA[3], positionB[3], positionC[3], positionD[3];
#if defined (_OPENMP)
   #pragma omp parallel for default(none) \
   shared(defPtrX, defPtrY, defPtrZ, femInterpolationWeight, \
          nodePositions, closestNodes, voxelNumber) \
   private(voxel, coefficients, positionA, positionB, positionC, positionD)
#endif
   for(voxel=0; voxel<voxelNumber; ++voxel)
   {
      coefficients[0]=femInterpolationWeight[4*voxel];
      coefficients[1]=femInterpolationWeight[4*voxel+1];
      coefficients[2]= femInterpolationWeight[4*voxel+2];
      coefficients[3]=femInterpolationWeight[4*voxel+3];

      positionA[0]=nodePositions[3*closestNodes[4*voxel]];
      positionA[1]=nodePositions[3*closestNodes[4*voxel]+1];
      positionA[2]=nodePositions[3*closestNodes[4*voxel]+2];

      positionB[0]=nodePositions[3*closestNodes[4*voxel+1]];
      positionB[1]=nodePositions[3*closestNodes[4*voxel+1]+1];
      positionB[2]=nodePositions[3*closestNodes[4*voxel+1]+2];

      positionC[0]=nodePositions[3*closestNodes[4*voxel+2]];
      positionC[1]=nodePositions[3*closestNodes[4*voxel+2]+1];
      positionC[2]=nodePositions[3*closestNodes[4*voxel+2]+2];

      positionD[0]=nodePositions[3*closestNodes[4*voxel+3]];
      positionD[1]=nodePositions[3*closestNodes[4*voxel+3]+1];
      positionD[2]=nodePositions[3*closestNodes[4*voxel+3]+2];

      defPtrX[voxel]=positionA[0]*coefficients[0] +
                     positionB[0]*coefficients[1] +
                     positionC[0]*coefficients[2] +
                     positionD[0]*coefficients[3];

      defPtrY[voxel]=positionA[1]*coefficients[0] +
                     positionB[1]*coefficients[1] +
                     positionC[1]*coefficients[2] +
                     positionD[1]*coefficients[3];

      defPtrZ[voxel]=positionA[2]*coefficients[0] +
                     positionB[2]*coefficients[1] +
                     positionC[2]*coefficients[2] +
                     positionD[2]*coefficients[3];
   }
   return;
}// reg_fem_getDeformationField

void reg_fem_voxelToNodeGradient(nifti_image *voxelBasedGradient,
                                 unsigned int *closestNodes,
                                 float *femInterpolationWeight,
                                 unsigned int nodeNumber,
                                 float *femBasedGradient)
{
   unsigned int voxelNumber = voxelBasedGradient->nx *
                              voxelBasedGradient->ny *
                              voxelBasedGradient->nz;
   float *voxGradPtrX = static_cast<float *>(voxelBasedGradient->data);
   float *voxGradPtrY = &voxGradPtrX[voxelNumber];
   float *voxGradPtrZ = &voxGradPtrY[voxelNumber];

   for(unsigned int node=0; node<3*nodeNumber; ++node)
      femBasedGradient[node]=0.f;

   unsigned int currentNodes[4], voxel;
   float currentGradient[3];
   float coefficients[4];
   for(voxel=0; voxel<voxelNumber; ++voxel)
   {
      currentNodes[0]=closestNodes[4*voxel];
      currentNodes[1]=closestNodes[4*voxel+1];
      currentNodes[2]=closestNodes[4*voxel+2];
      currentNodes[3]=closestNodes[4*voxel+3];

      coefficients[0]=femInterpolationWeight[4*voxel];
      coefficients[1]=femInterpolationWeight[4*voxel+1];
      coefficients[2]=femInterpolationWeight[4*voxel+2];
      coefficients[3]=femInterpolationWeight[4*voxel+3];

      currentGradient[0]=voxGradPtrX[voxel];
      currentGradient[1]=voxGradPtrY[voxel];
      currentGradient[2]=voxGradPtrZ[voxel];

      for(unsigned int i=0; i<4; ++i)
      {
         femBasedGradient[3*currentNodes[i]  ] += currentGradient[0]*coefficients[i];
         femBasedGradient[3*currentNodes[i]+1] += currentGradient[1]*coefficients[i];
         femBasedGradient[3*currentNodes[i]+2] += currentGradient[2]*coefficients[i];
      }
   }// voxel

   return;
}// reg_fem_voxelToNodeGradient

#endif
