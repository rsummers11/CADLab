/*
 *  _reg_bspline.cpp
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TRANSFORMATION_CPP
#define _REG_TRANSFORMATION_CPP

#include "_reg_localTransformation.h"

/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void get_BSplineBasisValues(DTYPE basis, DTYPE *values)
{
   DTYPE FF= basis*basis;
   DTYPE FFF= FF*basis;
   DTYPE MF=static_cast<DTYPE>(1.0-basis);
   values[0] = static_cast<DTYPE>((MF)*(MF)*(MF)/(6.0));
   values[1] = static_cast<DTYPE>((3.0*FFF - 6.0*FF + 4.0)/6.0);
   values[2] = static_cast<DTYPE>((-3.0*FFF + 3.0*FF + 3.0*basis + 1.0)/6.0);
   values[3] = static_cast<DTYPE>(FFF/6.0);
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void get_BSplineBasisValues(DTYPE basis, DTYPE *values, DTYPE *first)
{
   get_BSplineBasisValues<DTYPE>(basis, values);
   first[3]= static_cast<DTYPE>(basis * basis / 2.0);
   first[0]= static_cast<DTYPE>(basis - 1.0/2.0 - first[3]);
   first[2]= static_cast<DTYPE>(1.0 + first[0] - 2.0*first[3]);
   first[1]= - first[0] - first[2] - first[3];
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void get_BSplineBasisValues(DTYPE basis, DTYPE *values, DTYPE *first, DTYPE *second)
{
   get_BSplineBasisValues<DTYPE>(basis, values, first);
   second[3]= basis;
   second[0]= static_cast<DTYPE>(1.0 - second[3]);
   second[2]= static_cast<DTYPE>(second[0] - 2.0*second[3]);
   second[1]= - second[0] - second[2] - second[3];
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void Get_SplineBasisValues(DTYPE basis, DTYPE *values)
{
   DTYPE FF= basis*basis;
   values[0] = static_cast<DTYPE>((basis * ((2.0-basis)*basis - 1.0))/2.0);
   values[1] = static_cast<DTYPE>((FF * (3.0*basis-5.0) + 2.0)/2.0);
   values[2] = static_cast<DTYPE>((basis * ((4.0-3.0*basis)*basis + 1.0))/2.0);
   values[3] = static_cast<DTYPE>((basis-1.0) * FF/2.0);
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void Get_SplineBasisValues(DTYPE basis, DTYPE *values, DTYPE *first)
{
   Get_SplineBasisValues<DTYPE>(basis,values);
   DTYPE FF= basis*basis;
   first[0] = static_cast<DTYPE>((4.0*basis - 3.0*FF - 1.0)/2.0);
   first[1] = static_cast<DTYPE>((9.0*basis - 10.0) * basis/2.0);
   first[2] = static_cast<DTYPE>((8.0*basis - 9.0*FF + 1)/2.0);
   first[3] = static_cast<DTYPE>((3.0*basis - 2.0) * basis/2.0);
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void Get_SplineBasisValues(DTYPE basis, DTYPE *values, DTYPE *first, DTYPE *second)
{
   Get_SplineBasisValues<DTYPE>(basis, values, first);
   second[0] = static_cast<DTYPE>(2.0 - 3.0*basis);
   second[1] = static_cast<DTYPE>(9.0*basis - 5.0);
   second[2] = static_cast<DTYPE>(4.0 - 9.0*basis);
   second[3] = static_cast<DTYPE>(3.0*basis - 1.0);
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void get_SlidedValues(DTYPE &defX,
                      DTYPE &defY,
                      int X,
                      int Y,
                      DTYPE *defPtrX,
                      DTYPE *defPtrY,
                      mat44 *df_voxel2Real,
                      int *dim,
                      bool displacement)
{
   int newX=X;
   int newY=Y;
   if(X<0)
   {
      newX=0;
   }
   else if(X>=dim[1])
   {
      newX=dim[1]-1;
   }
   if(Y<0)
   {
      newY=0;
   }
   else if(Y>=dim[2])
   {
      newY=dim[2]-1;
   }
   DTYPE shiftValueX = 0;
   DTYPE shiftValueY = 0;
   if(!displacement)
   {
      int shiftIndexX=X-newX;
      int shiftIndexY=Y-newY;
      shiftValueX = shiftIndexX * df_voxel2Real->m[0][0] +
            shiftIndexY * df_voxel2Real->m[0][1];
      shiftValueY = shiftIndexX * df_voxel2Real->m[1][0] +
            shiftIndexY * df_voxel2Real->m[1][1];
   }
   size_t index=newY*dim[1]+newX;
   defX = defPtrX[index] + shiftValueX;
   defY = defPtrY[index] + shiftValueY;
}
/* *************************************************************** */
template <class DTYPE>
void get_SlidedValues(DTYPE &defX,
                      DTYPE &defY,
                      DTYPE &defZ,
                      int X,
                      int Y,
                      int Z,
                      DTYPE *defPtrX,
                      DTYPE *defPtrY,
                      DTYPE *defPtrZ,
                      mat44 *df_voxel2Real,
                      int *dim,
                      bool displacement)
{
   int newX=X;
   int newY=Y;
   int newZ=Z;
   if(X<0)
   {
      newX=0;
   }
   else if(X>=dim[1])
   {
      newX=dim[1]-1;
   }
   if(Y<0)
   {
      newY=0;
   }
   else if(Y>=dim[2])
   {
      newY=dim[2]-1;
   }
   if(Z<0)
   {
      newZ=0;
   }
   else if(Z>=dim[3])
   {
      newZ=dim[3]-1;
   }
   DTYPE shiftValueX=0;
   DTYPE shiftValueY=0;
   DTYPE shiftValueZ=0;
   if(!displacement)
   {
      int shiftIndexX=X-newX;
      int shiftIndexY=Y-newY;
      int shiftIndexZ=Z-newZ;
      shiftValueX =
            shiftIndexX * df_voxel2Real->m[0][0] +
            shiftIndexY * df_voxel2Real->m[0][1] +
            shiftIndexZ * df_voxel2Real->m[0][2];
      shiftValueY =
            shiftIndexX * df_voxel2Real->m[1][0] +
            shiftIndexY * df_voxel2Real->m[1][1] +
            shiftIndexZ * df_voxel2Real->m[1][2];
      shiftValueZ =
            shiftIndexX * df_voxel2Real->m[2][0] +
            shiftIndexY * df_voxel2Real->m[2][1] +
            shiftIndexZ * df_voxel2Real->m[2][2];
   }
   size_t index=(newZ*dim[2]+newY)*dim[1]+newX;
   defX = defPtrX[index] + shiftValueX;
   defY = defPtrY[index] + shiftValueY;
   defZ = defPtrZ[index] + shiftValueZ;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void get_GridValues(int startX,
                    int startY,
                    nifti_image *splineControlPoint,
                    DTYPE *splineX,
                    DTYPE *splineY,
                    DTYPE *dispX,
                    DTYPE *dispY,
                    bool approx,
                    bool displacement)

{
   int range=4;
   if(approx) range=3;

   size_t index;
   size_t coord=0;
   DTYPE *xxPtr=NULL, *yyPtr=NULL;

   mat44 *voxel2realMatrix=NULL;
   if(splineControlPoint->sform_code>0)
      voxel2realMatrix=&(splineControlPoint->sto_xyz);
   else voxel2realMatrix=&(splineControlPoint->qto_xyz);

   for(int Y=startY; Y<startY+range; Y++)
   {
      bool out=false;
      if(Y>-1 && Y<splineControlPoint->ny)
      {
         index = Y*splineControlPoint->nx;
         xxPtr = &splineX[index];
         yyPtr = &splineY[index];
      }
      else out=true;
      for(int X=startX; X<startX+range; X++)
      {
         if(X>-1 && X<splineControlPoint->nx && out==false)
         {
            dispX[coord] = xxPtr[X];
            dispY[coord] = yyPtr[X];
         }
         else
         {
            get_SlidedValues<DTYPE>(dispX[coord],
                                    dispY[coord],
                                    X,
                                    Y,
                                    splineX,
                                    splineY,
                                    voxel2realMatrix,
                                    splineControlPoint->dim,
                                    displacement);
         }
         coord++;
      }
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void get_GridValues(int startX,
                    int startY,
                    int startZ,
                    nifti_image *splineControlPoint,
                    DTYPE *splineX,
                    DTYPE *splineY,
                    DTYPE *splineZ,
                    DTYPE *dispX,
                    DTYPE *dispY,
                    DTYPE *dispZ,
                    bool approx,
                    bool displacement)
{
   int range=4;
   if(approx==true)
      range=3;

   size_t index;
   size_t coord=0;
   DTYPE *xPtr=NULL, *yPtr=NULL, *zPtr=NULL;
   DTYPE *xxPtr=NULL, *yyPtr=NULL, *zzPtr=NULL;

   mat44 *voxel2realMatrix=NULL;
   if(splineControlPoint->sform_code>0)
      voxel2realMatrix=&(splineControlPoint->sto_xyz);
   else voxel2realMatrix=&(splineControlPoint->qto_xyz);

   for(int Z=startZ; Z<startZ+range; Z++)
   {
      bool out=false;
      if(Z>-1 && Z<splineControlPoint->nz)
      {
         index=Z*splineControlPoint->nx*splineControlPoint->ny;
         xPtr = &splineX[index];
         yPtr = &splineY[index];
         zPtr = &splineZ[index];
      }
      else out=true;
      for(int Y=startY; Y<startY+range; Y++)
      {
         if(Y>-1 && Y<splineControlPoint->ny && out==false)
         {
            index = Y*splineControlPoint->nx;
            xxPtr = &xPtr[index];
            yyPtr = &yPtr[index];
            zzPtr = &zPtr[index];
         }
         else out=true;
         for(int X=startX; X<startX+range; X++)
         {
            if(X>-1 && X<splineControlPoint->nx && out==false)
            {
               dispX[coord] = xxPtr[X];
               dispY[coord] = yyPtr[X];
               dispZ[coord] = zzPtr[X];
            }
            else
            {
               get_SlidedValues<DTYPE>(dispX[coord],
                                       dispY[coord],
                                       dispZ[coord],
                                       X,
                                       Y,
                                       Z,
                                       splineX,
                                       splineY,
                                       splineZ,
                                       voxel2realMatrix,
                                       splineControlPoint->dim,
                                       displacement);
            }
            coord++;
         } // X
      } // Y
   } // Z
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_createControlPointGrid(nifti_image **controlPointGridImage,
                                nifti_image *referenceImage,
                                float *spacingMillimeter)
{
   // Define the control point grid dimension
   int dim_cpp[8];
   dim_cpp[0]=5;
   dim_cpp[1]=static_cast<int>(reg_ceil(referenceImage->nx*referenceImage->dx/spacingMillimeter[0])+3.f);
   dim_cpp[2]=static_cast<int>(reg_ceil(referenceImage->ny*referenceImage->dy/spacingMillimeter[1])+3.f);
   dim_cpp[3]=1;
   dim_cpp[5]=2;
   if(referenceImage->nz>1)
   {
      dim_cpp[3]=static_cast<int>(reg_ceil(referenceImage->nz*referenceImage->dz/spacingMillimeter[2])+3.f);
      dim_cpp[5]=3;
   }
   dim_cpp[4]=dim_cpp[6]=dim_cpp[7]=1;

   // Create the new control point grid image and allocate its space
   if(sizeof(DTYPE)==4)
      *controlPointGridImage = nifti_make_new_nim(dim_cpp, NIFTI_TYPE_FLOAT32, true);
   else *controlPointGridImage = nifti_make_new_nim(dim_cpp, NIFTI_TYPE_FLOAT64, true);

   // Fill the header information
   (*controlPointGridImage)->cal_min=0;
   (*controlPointGridImage)->cal_max=0;
   (*controlPointGridImage)->pixdim[0]=1.0f;
   (*controlPointGridImage)->pixdim[1]=(*controlPointGridImage)->dx=spacingMillimeter[0];
   (*controlPointGridImage)->pixdim[2]=(*controlPointGridImage)->dy=spacingMillimeter[1];
   if(referenceImage->nz==1)
   {
      (*controlPointGridImage)->pixdim[3]=(*controlPointGridImage)->dz=1.0f;
   }
   else (*controlPointGridImage)->pixdim[3]=(*controlPointGridImage)->dz=spacingMillimeter[2];
   (*controlPointGridImage)->pixdim[4]=(*controlPointGridImage)->dt=1.0f;
   (*controlPointGridImage)->pixdim[5]=(*controlPointGridImage)->du=1.0f;
   (*controlPointGridImage)->pixdim[6]=(*controlPointGridImage)->dv=1.0f;
   (*controlPointGridImage)->pixdim[7]=(*controlPointGridImage)->dw=1.0f;

   // Reproduce the orientation of the reference image and add a one voxel shift
   if(referenceImage->qform_code+referenceImage->sform_code>0)
   {
      (*controlPointGridImage)->qform_code=referenceImage->qform_code;
      (*controlPointGridImage)->sform_code=referenceImage->sform_code;
   }
   else
   {
      (*controlPointGridImage)->qform_code=1;
      (*controlPointGridImage)->sform_code=0;
   }

   // The qform (and sform) are set for the control point position image
   (*controlPointGridImage)->quatern_b=referenceImage->quatern_b;
   (*controlPointGridImage)->quatern_c=referenceImage->quatern_c;
   (*controlPointGridImage)->quatern_d=referenceImage->quatern_d;
   (*controlPointGridImage)->qoffset_x=referenceImage->qoffset_x;
   (*controlPointGridImage)->qoffset_y=referenceImage->qoffset_y;
   (*controlPointGridImage)->qoffset_z=referenceImage->qoffset_z;
   (*controlPointGridImage)->qfac=referenceImage->qfac;
   (*controlPointGridImage)->qto_xyz = nifti_quatern_to_mat44((*controlPointGridImage)->quatern_b,
                                                              (*controlPointGridImage)->quatern_c,
                                                              (*controlPointGridImage)->quatern_d,
                                                              (*controlPointGridImage)->qoffset_x,
                                                              (*controlPointGridImage)->qoffset_y,
                                                              (*controlPointGridImage)->qoffset_z,
                                                              (*controlPointGridImage)->dx,
                                                              (*controlPointGridImage)->dy,
                                                              (*controlPointGridImage)->dz,
                                                              (*controlPointGridImage)->qfac);

   // Origin is shifted from 1 control point in the qform
   float originIndex[3];
   float originReal[3];
   originIndex[0] = -1.0f;
   originIndex[1] = -1.0f;
   originIndex[2] = 0.0f;
   if(referenceImage->nz>1) originIndex[2] = -1.0f;
   reg_mat44_mul(&((*controlPointGridImage)->qto_xyz), originIndex, originReal);
   (*controlPointGridImage)->qto_xyz.m[0][3] = (*controlPointGridImage)->qoffset_x = originReal[0];
   (*controlPointGridImage)->qto_xyz.m[1][3] = (*controlPointGridImage)->qoffset_y = originReal[1];
   (*controlPointGridImage)->qto_xyz.m[2][3] = (*controlPointGridImage)->qoffset_z = originReal[2];

   (*controlPointGridImage)->qto_ijk = nifti_mat44_inverse((*controlPointGridImage)->qto_xyz);

   // Update the sform if required
   if((*controlPointGridImage)->sform_code>0)
   {
      float scalingRatio[3];
      scalingRatio[0]= (*controlPointGridImage)->dx / referenceImage->dx;
      scalingRatio[1]= (*controlPointGridImage)->dy / referenceImage->dy;
      scalingRatio[2]= (*controlPointGridImage)->dz / referenceImage->dz;

      (*controlPointGridImage)->sto_xyz.m[0][0]=referenceImage->sto_xyz.m[0][0] * scalingRatio[0];
      (*controlPointGridImage)->sto_xyz.m[1][0]=referenceImage->sto_xyz.m[1][0] * scalingRatio[0];
      (*controlPointGridImage)->sto_xyz.m[2][0]=referenceImage->sto_xyz.m[2][0] * scalingRatio[0];
      (*controlPointGridImage)->sto_xyz.m[3][0]=referenceImage->sto_xyz.m[3][0];
      (*controlPointGridImage)->sto_xyz.m[0][1]=referenceImage->sto_xyz.m[0][1] * scalingRatio[1];
      (*controlPointGridImage)->sto_xyz.m[1][1]=referenceImage->sto_xyz.m[1][1] * scalingRatio[1];
      (*controlPointGridImage)->sto_xyz.m[2][1]=referenceImage->sto_xyz.m[2][1] * scalingRatio[1];
      (*controlPointGridImage)->sto_xyz.m[3][1]=referenceImage->sto_xyz.m[3][1];
      (*controlPointGridImage)->sto_xyz.m[0][2]=referenceImage->sto_xyz.m[0][2] * scalingRatio[2];
      (*controlPointGridImage)->sto_xyz.m[1][2]=referenceImage->sto_xyz.m[1][2] * scalingRatio[2];
      (*controlPointGridImage)->sto_xyz.m[2][2]=referenceImage->sto_xyz.m[2][2] * scalingRatio[2];
      (*controlPointGridImage)->sto_xyz.m[3][2]=referenceImage->sto_xyz.m[3][2];
      (*controlPointGridImage)->sto_xyz.m[0][3]=referenceImage->sto_xyz.m[0][3];
      (*controlPointGridImage)->sto_xyz.m[1][3]=referenceImage->sto_xyz.m[1][3];
      (*controlPointGridImage)->sto_xyz.m[2][3]=referenceImage->sto_xyz.m[2][3];
      (*controlPointGridImage)->sto_xyz.m[3][3]=referenceImage->sto_xyz.m[3][3];

      // Origin is shifted from 1 control point in the sform
      reg_mat44_mul(&((*controlPointGridImage)->sto_xyz), originIndex, originReal);
      (*controlPointGridImage)->sto_xyz.m[0][3] = originReal[0];
      (*controlPointGridImage)->sto_xyz.m[1][3] = originReal[1];
      (*controlPointGridImage)->sto_xyz.m[2][3] = originReal[2];
      (*controlPointGridImage)->sto_ijk = nifti_mat44_inverse((*controlPointGridImage)->sto_xyz);
   }

   (*controlPointGridImage)->intent_code=NIFTI_INTENT_VECTOR;
   memset((*controlPointGridImage)->intent_name, 0, 16);
   strcpy((*controlPointGridImage)->intent_name,"NREG_TRANS");
   (*controlPointGridImage)->intent_p1=SPLINE_GRID;
}
template void reg_createControlPointGrid<float>(nifti_image **, nifti_image *, float *);
template void reg_createControlPointGrid<double>(nifti_image **, nifti_image *, float *);
/* *************************************************************** */
template <class DTYPE>
void reg_createSymmetricControlPointGrids(nifti_image **forwardGridImage,
                                          nifti_image **backwardGridImage,
                                          nifti_image *referenceImage,
                                          nifti_image *floatingImage,
                                          mat44 *forwardAffineTrans,
                                          float *spacing)
{
   // Delete the grid if they are already initialised
   if(*forwardGridImage!=NULL)
      nifti_image_free(*forwardGridImage);
   *forwardGridImage=NULL;
   if(*backwardGridImage!=NULL)
      nifti_image_free(*backwardGridImage);
   *backwardGridImage=NULL;
   // We specified a space which is in-between both input images
   // // Get the reference image space
   mat44 referenceImageSpace = referenceImage->qto_xyz;
   if(referenceImage->sform_code>0)
      referenceImageSpace = referenceImage->sto_xyz;
#ifndef NDEBUG
   reg_mat44_disp(&referenceImageSpace,(char *)"[NiftyReg DEBUG] Input reference image orientation");
#endif
   // // Get the floating image space
   mat44 floatingImageSpace = floatingImage->qto_xyz;
   if(floatingImage->sform_code>0)
      floatingImageSpace = floatingImage->sto_xyz;
#ifndef NDEBUG
   reg_mat44_disp(&floatingImageSpace,(char *)"[NiftyReg DEBUG] Input floating image orientation");
#endif
   // Check if an affine transformation is specified
   mat44 halfForwardAffine, halfBackwardAffine;
   if(forwardAffineTrans!=NULL)
   {
      // Compute half of the affine transformation - ref to flo
      halfForwardAffine = reg_mat44_logm(forwardAffineTrans);
      halfForwardAffine = reg_mat44_mul(&halfForwardAffine,.5f);
      halfForwardAffine = reg_mat44_expm(&halfForwardAffine);
      // Compute half of the affine transformation - flo to ref
      // Note that this is done twice for symmetry consideration
      halfBackwardAffine = nifti_mat44_inverse(*forwardAffineTrans);
      halfBackwardAffine = reg_mat44_logm(&halfBackwardAffine);
      halfBackwardAffine = reg_mat44_mul(&halfBackwardAffine,.5f);
      halfBackwardAffine = reg_mat44_expm(&halfBackwardAffine);
      reg_print_msg_warn("Note that the symmetry of the registration is affected by the input affine transformation");
   }
   else
   {
      reg_mat44_eye(&halfForwardAffine);
      reg_mat44_eye(&halfBackwardAffine);
   }

   // Update the reference and floating transformation to propagate to a mid space
   referenceImageSpace = reg_mat44_mul(&halfForwardAffine,&referenceImageSpace);
   floatingImageSpace = reg_mat44_mul(&halfBackwardAffine,&floatingImageSpace);

   // Define the largest field of view in the mid space
   float minPosition[3]={0,0,0}, maxPosition[3]={0,0,0};
   if(referenceImage->nz>1)  // 3D
   {
      float referenceImageCorners[8][3]=
      {
         {0,0,0},
         {float(referenceImage->nx),0,0},
         {0,float(referenceImage->ny),0},
         {float(referenceImage->nx),float(referenceImage->ny),0},
         {0,0,float(referenceImage->nz)},
         {float(referenceImage->nx),0,float(referenceImage->nz)},
         {0,float(referenceImage->ny),float(referenceImage->nz)},
         {float(referenceImage->nx),float(referenceImage->ny),float(referenceImage->nz)}
      };
      float floatingImageCorners[8][3]=
      {
         {0,0,0},
         {float(floatingImage->nx),0,0},
         {0,float(floatingImage->ny),0},
         {float(floatingImage->nx),float(floatingImage->ny),0},
         {0,0,float(floatingImage->nz)},
         {float(floatingImage->nx),0,float(floatingImage->nz)},
         {0,float(floatingImage->ny),float(floatingImage->nz)},
         {float(floatingImage->nx),float(floatingImage->ny),float(floatingImage->nz)}
      };
      float out[3];
      for(int c=0; c<8; ++c)
      {
         reg_mat44_mul(&referenceImageSpace,referenceImageCorners[c],out);
         referenceImageCorners[c][0]=out[0];
         referenceImageCorners[c][1]=out[1];
         referenceImageCorners[c][2]=out[2];
         reg_mat44_mul(&floatingImageSpace,floatingImageCorners[c],out);
         floatingImageCorners[c][0]=out[0];
         floatingImageCorners[c][1]=out[1];
         floatingImageCorners[c][2]=out[2];

      }
      minPosition[0]=referenceImageCorners[0][0]<floatingImageCorners[0][0]?referenceImageCorners[0][0]:floatingImageCorners[0][0];
      minPosition[1]=referenceImageCorners[0][1]<floatingImageCorners[0][1]?referenceImageCorners[0][1]:floatingImageCorners[0][1];
      minPosition[2]=referenceImageCorners[0][2]<floatingImageCorners[0][2]?referenceImageCorners[0][2]:floatingImageCorners[0][2];
      maxPosition[0]=referenceImageCorners[0][0]>floatingImageCorners[0][0]?referenceImageCorners[0][0]:floatingImageCorners[0][0];
      maxPosition[1]=referenceImageCorners[0][1]>floatingImageCorners[0][1]?referenceImageCorners[0][1]:floatingImageCorners[0][1];
      maxPosition[2]=referenceImageCorners[0][2]>floatingImageCorners[0][2]?referenceImageCorners[0][2]:floatingImageCorners[0][2];
      for(int c=1; c<8; ++c)
      {
         minPosition[0]=minPosition[0]<referenceImageCorners[c][0]?minPosition[0]:referenceImageCorners[c][0];
         minPosition[0]=minPosition[0]<floatingImageCorners[c][0]?minPosition[0]:floatingImageCorners[c][0];
         minPosition[1]=minPosition[1]<referenceImageCorners[c][1]?minPosition[1]:referenceImageCorners[c][1];
         minPosition[1]=minPosition[1]<floatingImageCorners[c][1]?minPosition[1]:floatingImageCorners[c][1];
         minPosition[2]=minPosition[2]<referenceImageCorners[c][2]?minPosition[2]:referenceImageCorners[c][2];
         minPosition[2]=minPosition[2]<floatingImageCorners[c][2]?minPosition[2]:floatingImageCorners[c][2];
         maxPosition[0]=maxPosition[0]>referenceImageCorners[c][0]?maxPosition[0]:referenceImageCorners[c][0];
         maxPosition[0]=maxPosition[0]>floatingImageCorners[c][0]?maxPosition[0]:floatingImageCorners[c][0];
         maxPosition[1]=maxPosition[1]>referenceImageCorners[c][1]?maxPosition[1]:referenceImageCorners[c][1];
         maxPosition[1]=maxPosition[1]>floatingImageCorners[c][1]?maxPosition[1]:floatingImageCorners[c][1];
         maxPosition[2]=maxPosition[2]>referenceImageCorners[c][2]?maxPosition[2]:referenceImageCorners[c][2];
         maxPosition[2]=maxPosition[2]>floatingImageCorners[c][2]?maxPosition[2]:floatingImageCorners[c][2];
      }
   }
   else  // 2D
   {
      float referenceImageCorners[4][2]=
      {
         {0,0},
         {float(referenceImage->nx),0},
         {0,float(referenceImage->ny)},
         {float(referenceImage->nx),float(referenceImage->ny)}
      };
      float floatingImageCorners[4][2]=
      {
         {0,0},
         {float(floatingImage->nx),0},
         {0,float(floatingImage->ny)},
         {float(floatingImage->nx),float(floatingImage->ny)}
      };
      float out[2];
      for(int c=0; c<4; ++c)
      {
         out[0]= referenceImageCorners[c][0] * referenceImageSpace.m[0][0]
               +referenceImageCorners[c][1] * referenceImageSpace.m[0][1]
               + referenceImageSpace.m[0][3];
         out[1]= referenceImageCorners[c][0] * referenceImageSpace.m[1][0]
               +referenceImageCorners[c][1] * referenceImageSpace.m[1][1]
               + referenceImageSpace.m[1][3];
         referenceImageCorners[c][0]=out[0];
         referenceImageCorners[c][1]=out[1];
         out[0]= floatingImageCorners[c][0] * floatingImageSpace.m[0][0]
               +floatingImageCorners[c][1] * floatingImageSpace.m[0][1]
               + floatingImageSpace.m[0][3];
         out[1]= floatingImageCorners[c][0] * floatingImageSpace.m[1][0]
               +floatingImageCorners[c][1] * floatingImageSpace.m[1][1]
               + floatingImageSpace.m[1][3];
         floatingImageCorners[c][0]=out[0];
         floatingImageCorners[c][1]=out[1];

      }
      minPosition[0]=referenceImageCorners[0][0]<floatingImageCorners[0][0]?referenceImageCorners[0][0]:floatingImageCorners[0][0];
      minPosition[1]=referenceImageCorners[0][1]<floatingImageCorners[0][1]?referenceImageCorners[0][1]:floatingImageCorners[0][1];
      maxPosition[0]=referenceImageCorners[0][0]>floatingImageCorners[0][0]?referenceImageCorners[0][0]:floatingImageCorners[0][0];
      maxPosition[1]=referenceImageCorners[0][1]>floatingImageCorners[0][1]?referenceImageCorners[0][1]:floatingImageCorners[0][1];
      for(int c=1; c<4; ++c)
      {
         minPosition[0]=minPosition[0]<referenceImageCorners[c][0]?minPosition[0]:referenceImageCorners[c][0];
         minPosition[0]=minPosition[0]<floatingImageCorners[c][0]?minPosition[0]:floatingImageCorners[c][0];
         minPosition[1]=minPosition[1]<referenceImageCorners[c][1]?minPosition[1]:referenceImageCorners[c][1];
         minPosition[1]=minPosition[1]<floatingImageCorners[c][1]?minPosition[1]:floatingImageCorners[c][1];
         maxPosition[0]=maxPosition[0]>referenceImageCorners[c][0]?maxPosition[0]:referenceImageCorners[c][0];
         maxPosition[0]=maxPosition[0]>floatingImageCorners[c][0]?maxPosition[0]:floatingImageCorners[c][0];
         maxPosition[1]=maxPosition[1]>referenceImageCorners[c][1]?maxPosition[1]:referenceImageCorners[c][1];
         maxPosition[1]=maxPosition[1]>floatingImageCorners[c][1]?maxPosition[1]:floatingImageCorners[c][1];
      }
   }

   // Compute the dimension of the control point grids
   const int dim[8]= {5,
                      static_cast<int>(reg_ceil((maxPosition[0]-minPosition[0])/spacing[0])+3),
                      static_cast<int>(reg_ceil((maxPosition[1]-minPosition[1])/spacing[1])+3),
                      referenceImage->nz>1?static_cast<int>(reg_ceil((maxPosition[2]-minPosition[2])/spacing[2])+3):1,
                      1,
                      referenceImage->nz>1?3:2,
                      1,
                      1
                     };

   // Create the control point grid image
   if(sizeof(DTYPE)==sizeof(float))
   {
      (*forwardGridImage)=nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32,true);
      (*backwardGridImage)=nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT32,true);
   }
   else
   {
      (*forwardGridImage)=nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT64,true);
      (*backwardGridImage)=nifti_make_new_nim(dim, NIFTI_TYPE_FLOAT64,true);
   }
   // Set the control point grid spacing
   (*forwardGridImage)->pixdim[1]=(*forwardGridImage)->dx=(*backwardGridImage)->pixdim[1]=(*backwardGridImage)->dx=spacing[0];
   (*forwardGridImage)->pixdim[2]=(*forwardGridImage)->dy=(*backwardGridImage)->pixdim[2]=(*backwardGridImage)->dy=spacing[1];
   if(referenceImage->nz>1)
      (*forwardGridImage)->pixdim[3]=(*forwardGridImage)->dz=(*backwardGridImage)->pixdim[3]=(*backwardGridImage)->dz=spacing[2];
   // Set the control point grid image orientation
   (*forwardGridImage)->qform_code=(*backwardGridImage)->qform_code=0;
   (*forwardGridImage)->sform_code=(*backwardGridImage)->sform_code=1;
   reg_mat44_eye(&(*forwardGridImage)->sto_xyz);
   reg_mat44_eye(&(*backwardGridImage)->sto_xyz);
   reg_mat44_eye(&(*forwardGridImage)->sto_ijk);
   reg_mat44_eye(&(*backwardGridImage)->sto_ijk);
   for(unsigned int i=0; i<3; ++i)
   {
      if(referenceImage->nz>1 || i<2)
      {
         (*forwardGridImage)->sto_xyz.m[i][i]=(*backwardGridImage)->sto_xyz.m[i][i]=spacing[i];
         (*forwardGridImage)->sto_xyz.m[i][3]=(*backwardGridImage)->sto_xyz.m[i][3]=minPosition[i]-spacing[i];
      }
      else
      {
         (*forwardGridImage)->sto_xyz.m[i][i]=(*backwardGridImage)->sto_xyz.m[i][i]=1.f;
         (*forwardGridImage)->sto_xyz.m[i][3]=(*backwardGridImage)->sto_xyz.m[i][3]=0.f;
      }
   }
   (*forwardGridImage)->sto_ijk=(*backwardGridImage)->sto_ijk=nifti_mat44_inverse((*forwardGridImage)->sto_xyz);
   // Set the intent type
   (*forwardGridImage)->intent_code=(*backwardGridImage)->intent_code=NIFTI_INTENT_VECTOR;
   memset((*forwardGridImage)->intent_name, 0, 16);
   memset((*backwardGridImage)->intent_name, 0, 16);
   strcpy((*forwardGridImage)->intent_name,"NREG_TRANS");
   strcpy((*backwardGridImage)->intent_name,"NREG_TRANS");
   (*forwardGridImage)->intent_p1=(*backwardGridImage)->intent_p1=SPLINE_GRID;
   // Set the affine matrices
   mat44 identity;
   reg_mat44_eye(&identity);
   if((*forwardGridImage)->ext_list!=NULL)
      free((*forwardGridImage)->ext_list);
   if((*backwardGridImage)->ext_list!=NULL)
      free((*backwardGridImage)->ext_list);
   (*forwardGridImage)->num_ext=0;
   (*backwardGridImage)->num_ext=0;
   if(identity!=halfForwardAffine && identity!=halfBackwardAffine)
   {
      // Create extensions to store the affine parametrisations for the forward transformation
      (*forwardGridImage)->num_ext=2;
      (*forwardGridImage)->ext_list=(nifti1_extension *)malloc(2*sizeof(nifti1_extension));
      (*forwardGridImage)->ext_list[0].esize=16*sizeof(float)+16;
      (*forwardGridImage)->ext_list[1].esize=16*sizeof(float)+16;
      (*forwardGridImage)->ext_list[0].ecode=NIFTI_ECODE_IGNORE;
      (*forwardGridImage)->ext_list[1].ecode=NIFTI_ECODE_IGNORE;
      (*forwardGridImage)->ext_list[0].edata=(char *)calloc((*forwardGridImage)->ext_list[0].esize-8,sizeof(float));
      (*forwardGridImage)->ext_list[1].edata=(char *)calloc((*forwardGridImage)->ext_list[1].esize-8,sizeof(float));
      memcpy((*forwardGridImage)->ext_list[0].edata, &halfForwardAffine, sizeof(mat44));
      memcpy((*forwardGridImage)->ext_list[1].edata, &halfForwardAffine, sizeof(mat44));
#ifndef NDEBUG
      reg_mat44_disp(&halfForwardAffine,(char *)"[NiftyReg DEBUG] Forward transformation half-affine");
#endif
      // Create extensions to store the affine parametrisations for the backward transformation
      (*backwardGridImage)->num_ext=2;
      (*backwardGridImage)->ext_list=(nifti1_extension *)malloc(2*sizeof(nifti1_extension));
      (*backwardGridImage)->ext_list[0].esize=16*sizeof(float)+16;
      (*backwardGridImage)->ext_list[1].esize=16*sizeof(float)+16;
      (*backwardGridImage)->ext_list[0].ecode=NIFTI_ECODE_IGNORE;
      (*backwardGridImage)->ext_list[1].ecode=NIFTI_ECODE_IGNORE;
      (*backwardGridImage)->ext_list[0].edata=(char *)calloc((*backwardGridImage)->ext_list[0].esize-8,sizeof(float));
      (*backwardGridImage)->ext_list[1].edata=(char *)calloc((*backwardGridImage)->ext_list[1].esize-8,sizeof(float));
      memcpy((*backwardGridImage)->ext_list[0].edata, &halfBackwardAffine, sizeof(mat44));
      memcpy((*backwardGridImage)->ext_list[1].edata, &halfBackwardAffine, sizeof(mat44));
#ifndef NDEBUG
      reg_mat44_disp(&halfBackwardAffine,(char *)"[NiftyReg DEBUG] Backward transformation half-affine");
#endif
   }
   // Initialise the grid with identity transformations
   reg_tools_multiplyValueToImage(*forwardGridImage,*forwardGridImage,0.f);
   reg_tools_multiplyValueToImage(*backwardGridImage,*backwardGridImage,0.f);
   // Convert the parametrisations into deformation fields
   reg_getDeformationFromDisplacement(*forwardGridImage);
   reg_getDeformationFromDisplacement(*backwardGridImage);
}
/* *************************************************************** */
template void reg_createSymmetricControlPointGrids<float>
(nifti_image **,nifti_image **,nifti_image *,nifti_image *,mat44 *,float *);
template void reg_createSymmetricControlPointGrids<double>
(nifti_image **,nifti_image **,nifti_image *,nifti_image *,mat44 *,float *);
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_spline_getDeformationField2D(nifti_image *splineControlPoint,
                                      nifti_image *deformationField,
                                      int *mask,
                                      bool composition,
                                      bool bspline)
{

#if _USE_SSE
   union
   {
      __m128 m;
      float f[4];
   } val;
   __m128 tempCurrent, tempX, tempY;
#ifdef _WIN32
   __declspec(align(16)) DTYPE temp[4];
   __declspec(align(16)) DTYPE yBasis[4];
   union
   {
      __m128 m[16];
      __declspec(align(16)) DTYPE f[16];
   } xControlPointCoordinates;
   union
   {
      __m128 m[16];
      __declspec(align(16)) DTYPE f[16];
   } yControlPointCoordinates;
   union u1
   {
      __m128 m[4];
      __declspec(align(16)) DTYPE f[16];
   } xyBasis;
#else // _WIN32
   DTYPE temp[4] __attribute__((aligned(16)));
   DTYPE yBasis[4] __attribute__((aligned(16)));
   union
   {
      __m128 m[16];
      DTYPE f[16] __attribute__((aligned(16)));
   } xControlPointCoordinates;
   union
   {
      __m128 m[16];
      DTYPE f[16] __attribute__((aligned(16)));
   } yControlPointCoordinates;
   union u1
   {
      __m128 m[4];
      DTYPE f[16] __attribute__((aligned(16)));
   } xyBasis;
#endif // _WIN32
#else // _USE_SSE
   DTYPE temp[4];
   DTYPE yBasis[4];
   DTYPE xyBasis[16];
   DTYPE xControlPointCoordinates[16];
   DTYPE yControlPointCoordinates[16];
#endif // _USE_SSE


   DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny];

   DTYPE *fieldPtrX=static_cast<DTYPE *>(deformationField->data);
   DTYPE *fieldPtrY=&fieldPtrX[deformationField->nx*deformationField->ny*deformationField->nz];

   DTYPE gridVoxelSpacing[2];
   gridVoxelSpacing[0] = splineControlPoint->dx / deformationField->dx;
   gridVoxelSpacing[1] = splineControlPoint->dy / deformationField->dy;

   DTYPE basis, xReal, yReal, xVoxel, yVoxel;
   int x, y, a, b, xPre, yPre, oldXpre, oldYpre;
   size_t index, coord;

   if(composition)  // Composition of deformation fields
   {

      // read the ijk sform or qform, as appropriate
      mat44 *referenceMatrix_real_to_voxel;
      if(splineControlPoint->sform_code>0)
         referenceMatrix_real_to_voxel=&(splineControlPoint->sto_ijk);
      else referenceMatrix_real_to_voxel=&(splineControlPoint->qto_ijk);

      for(y=0; y<deformationField->ny; y++)
      {
         index=y*deformationField->nx;
         oldXpre=oldYpre=99999999;
         for(x=0; x<deformationField->nx; x++)
         {

            // The previous position at the current pixel position is read
            xReal = (DTYPE)(fieldPtrX[index]);
            yReal = (DTYPE)(fieldPtrY[index]);

            // From real to pixel position in the CPP
            xVoxel = referenceMatrix_real_to_voxel->m[0][0]*xReal
                  + referenceMatrix_real_to_voxel->m[0][1]*yReal
                  + referenceMatrix_real_to_voxel->m[0][3];
            yVoxel = referenceMatrix_real_to_voxel->m[1][0]*xReal
                  + referenceMatrix_real_to_voxel->m[1][1]*yReal
                  + referenceMatrix_real_to_voxel->m[1][3];

            // The spline coefficients are computed
            xPre=(int)reg_floor(xVoxel);
            basis=xVoxel-(DTYPE)xPre;
            --xPre;
            if(basis<0.0) basis=0.0; //rounding error
            if(bspline) get_BSplineBasisValues<DTYPE>(basis, temp);
            else Get_SplineBasisValues<DTYPE>(basis, temp);

            yPre=(int)reg_floor(yVoxel);
            basis=yVoxel-(DTYPE)yPre;
            --yPre;
            if(basis<0.0) basis=0.0; //rounding error
            if(bspline) get_BSplineBasisValues<DTYPE>(basis, yBasis);
            else Get_SplineBasisValues<DTYPE>(basis, yBasis);


            if(xVoxel>=0 && xVoxel<=deformationField->nx-1 &&
                  yVoxel>=0 && yVoxel<=deformationField->ny-1)
            {

               // The control point postions are extracted
               if(oldXpre!=xPre || oldYpre!=yPre)
               {
#ifdef _USE_SSE
                  get_GridValues<DTYPE>(xPre,
                                        yPre,
                                        splineControlPoint,
                                        controlPointPtrX,
                                        controlPointPtrY,
                                        xControlPointCoordinates.f,
                                        yControlPointCoordinates.f,
                                        false, // no approximation
                                        false // not a displacement field
                                        );
#else // _USE_SSE
                  get_GridValues<DTYPE>(xPre,
                                        yPre,
                                        splineControlPoint,
                                        controlPointPtrX,
                                        controlPointPtrY,
                                        xControlPointCoordinates,
                                        yControlPointCoordinates,
                                        false, // no approximation
                                        false // not a displacement field
                                        );
#endif // _USE_SSE
                  oldXpre=xPre;
                  oldYpre=yPre;
               }
               xReal=0.0;
               yReal=0.0;

               if(mask[index]>-1)
               {
#if _USE_SSE
                  coord=0;
                  for(b=0; b<4; b++)
                  {
                     for(a=0; a<4; a++)
                     {
                        xyBasis.f[coord++] = temp[a] * yBasis[b];
                     }
                  }

                  tempX =  _mm_set_ps1(0.0);
                  tempY =  _mm_set_ps1(0.0);
                  //addition and multiplication of the 16 basis value and CP position for each axis
                  for(a=0; a<4; a++)
                  {
                     tempX = _mm_add_ps(_mm_mul_ps(xyBasis.m[a], xControlPointCoordinates.m[a]), tempX );
                     tempY = _mm_add_ps(_mm_mul_ps(xyBasis.m[a], yControlPointCoordinates.m[a]), tempY );
                  }
                  //the values stored in SSE variables are transfered to normal float
                  val.m = tempX;
                  xReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempY;
                  yReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                  for(b=0; b<4; b++)
                  {
                     for(a=0; a<4; a++)
                     {
                        DTYPE tempValue = temp[a] * yBasis[b];
                        xReal += xControlPointCoordinates[b*4+a] * tempValue;
                        yReal += yControlPointCoordinates[b*4+a] * tempValue;
                     }
                  }
#endif
               }

               fieldPtrX[index] = (DTYPE)xReal;
               fieldPtrY[index] = (DTYPE)yReal;
            }
            index++;
         }
      }
   }
   else  // starting deformation field is blank - !composition
   {

#if defined (_OPENMP)
#ifdef _USE_SSE
#pragma  omp parallel for default(none) \
   shared(deformationField, gridVoxelSpacing, splineControlPoint, controlPointPtrX, \
   controlPointPtrY, mask, fieldPtrX, fieldPtrY, bspline) \
   private(x, y, a, xPre, yPre, oldXpre, oldYpre, index, xReal, yReal, basis, \
   val, temp, yBasis, tempCurrent, xyBasis, tempX, tempY, \
   xControlPointCoordinates, yControlPointCoordinates)
#else // _USE_SSE
#pragma  omp parallel for default(none) \
   shared(deformationField, gridVoxelSpacing, splineControlPoint, controlPointPtrX, \
   controlPointPtrY, mask, fieldPtrX, fieldPtrY, bspline) \
   private(x, y, a, xPre, yPre, oldXpre, oldYpre, index, xReal, yReal, basis, coord, \
   temp, yBasis, xyBasis, xControlPointCoordinates, yControlPointCoordinates)
#endif // _USE_SEE
#endif // _OPENMP
      for( y=0; y<deformationField->ny; y++)
      {
         index=y*deformationField->nx;
         oldXpre=oldYpre=9999999;

         yPre=(int)((DTYPE)y/gridVoxelSpacing[1]);
         basis=(DTYPE)y/gridVoxelSpacing[1]-(DTYPE)yPre;
         if(basis<0.0) basis=0.0; //rounding error
         if(bspline) get_BSplineBasisValues<DTYPE>(basis, yBasis);
         else Get_SplineBasisValues<DTYPE>(basis, yBasis);

         for(x=0; x<deformationField->nx; x++)
         {

            xPre=(int)((DTYPE)x/gridVoxelSpacing[0]);
            basis=(DTYPE)x/gridVoxelSpacing[0]-(DTYPE)xPre;
            if(basis<0.0) basis=0.0; //rounding error
            if(bspline) get_BSplineBasisValues<DTYPE>(basis, temp);
            else Get_SplineBasisValues<DTYPE>(basis, temp);
#if _USE_SSE
            val.f[0] = temp[0];
            val.f[1] = temp[1];
            val.f[2] = temp[2];
            val.f[3] = temp[3];
            tempCurrent=val.m;
            for(a=0; a<4; a++)
            {
               val.m=_mm_set_ps1(yBasis[a]);
               xyBasis.m[a]=_mm_mul_ps(tempCurrent,val.m);
            }
#else
            coord=0;
            for(a=0; a<4; a++)
            {
               xyBasis[coord++]=temp[0]*yBasis[a];
               xyBasis[coord++]=temp[1]*yBasis[a];
               xyBasis[coord++]=temp[2]*yBasis[a];
               xyBasis[coord++]=temp[3]*yBasis[a];
            }
#endif
            if(oldXpre!=xPre || oldYpre!=yPre)
            {
#ifdef _USE_SSE
               get_GridValues<DTYPE>(xPre,
                                     yPre,
                                     splineControlPoint,
                                     controlPointPtrX,
                                     controlPointPtrY,
                                     xControlPointCoordinates.f,
                                     yControlPointCoordinates.f,
                                     false, // no approximation
                                     false // not a deformation field
                                     );
#else // _USE_SSE
               get_GridValues<DTYPE>(xPre,
                                     yPre,
                                     splineControlPoint,
                                     controlPointPtrX,
                                     controlPointPtrY,
                                     xControlPointCoordinates,
                                     yControlPointCoordinates,
                                     false, // no approximation
                                     false // not a deformation field
                                     );
#endif // _USE_SSE
               oldXpre=xPre;
               oldYpre=yPre;
            }

            xReal=0.0;
            yReal=0.0;

            if(mask[index]>-1)
            {
#if _USE_SSE
               tempX =  _mm_set_ps1(0.0);
               tempY =  _mm_set_ps1(0.0);
               //addition and multiplication of the 64 basis value and CP displacement for each axis
               for(a=0; a<4; a++)
               {
                  tempX = _mm_add_ps(_mm_mul_ps(xyBasis.m[a], xControlPointCoordinates.m[a]), tempX );
                  tempY = _mm_add_ps(_mm_mul_ps(xyBasis.m[a], yControlPointCoordinates.m[a]), tempY );
               }
               //the values stored in SSE variables are transfered to normal float
               val.m=tempX;
               xReal=val.f[0]+val.f[1]+val.f[2]+val.f[3];
               val.m=tempY;
               yReal= val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
               for(a=0; a<16; a++)
               {
                  xReal += xControlPointCoordinates[a] * xyBasis[a];
                  yReal += yControlPointCoordinates[a] * xyBasis[a];
               }
#endif
            }// mask
            fieldPtrX[index] = (DTYPE)xReal;
            fieldPtrY[index] = (DTYPE)yReal;
            index++;
         } // x
      } // y
   } // composition

   return;
}
/* *************************************************************** */
template<class DTYPE>
void reg_spline_getDeformationField3D(nifti_image *splineControlPoint,
                                      nifti_image *deformationField,
                                      int *mask,
                                      bool composition,
                                      bool bspline
                                      )
{
#if _USE_SSE
   union
   {
      __m128 m;
      float f[4];
   } val;
   __m128 tempX, tempY, tempZ, tempCurrent;
   __m128 xBasis_sse, yBasis_sse, zBasis_sse, temp_basis_sse, basis_sse;

#ifdef _WIN32
   __declspec(align(16)) DTYPE temp[4];
   __declspec(align(16)) DTYPE zBasis[4];
   union
   {
      __m128 m[16];
      __declspec(align(16)) DTYPE f[16];
   } xControlPointCoordinates;
   union
   {
      __m128 m[16];
      __declspec(align(16)) DTYPE f[16];
   } yControlPointCoordinates;
   union
   {
      __m128 m[16];
      __declspec(align(16)) DTYPE f[16];
   } zControlPointCoordinates;
#else // _WIN32
   DTYPE temp[4] __attribute__((aligned(16)));
   DTYPE zBasis[4] __attribute__((aligned(16)));
   union
   {
      __m128 m[16];
      DTYPE f[16] __attribute__((aligned(16)));
   } xControlPointCoordinates;
   union
   {
      __m128 m[16];
      DTYPE f[16] __attribute__((aligned(16)));
   } yControlPointCoordinates;
   union
   {
      __m128 m[16];
      DTYPE f[16] __attribute__((aligned(16)));
   } zControlPointCoordinates;
#endif // _WIN32
#else // _USE_SSE
   DTYPE temp[4];
   DTYPE zBasis[4];
   DTYPE xControlPointCoordinates[64];
   DTYPE yControlPointCoordinates[64];
   DTYPE zControlPointCoordinates[64];
   int coord;
#endif // _USE_SSE


   DTYPE *controlPointPtrX = static_cast<DTYPE *>(splineControlPoint->data);
   DTYPE *controlPointPtrY = &controlPointPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
   DTYPE *controlPointPtrZ = &controlPointPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];

   DTYPE *fieldPtrX=static_cast<DTYPE *>(deformationField->data);
   DTYPE *fieldPtrY=&fieldPtrX[deformationField->nx*deformationField->ny*deformationField->nz];
   DTYPE *fieldPtrZ=&fieldPtrY[deformationField->nx*deformationField->ny*deformationField->nz];

   DTYPE basis, oldBasis=(DTYPE)(1.1);

   int x, y, z, a, b, c, oldPreX, oldPreY, oldPreZ, xPre, yPre, zPre, index;
   DTYPE real[3];

   if(composition)  // Composition of deformation fields
   {

      // read the ijk sform or qform, as appropriate
      mat44 referenceMatrix_real_to_voxel;
      if(splineControlPoint->sform_code>0)
         referenceMatrix_real_to_voxel=(splineControlPoint->sto_ijk);
      else referenceMatrix_real_to_voxel=(splineControlPoint->qto_ijk);
#ifdef _USE_SSE
#ifdef _WIN32
      __declspec(align(16)) DTYPE xBasis[4];
      __declspec(align(16)) DTYPE yBasis[4];
#else
      DTYPE xBasis[4] __attribute__((aligned(16)));
      DTYPE yBasis[4] __attribute__((aligned(16)));
#endif
#else // _USE_SSE
      DTYPE xBasis[4], yBasis[4];
#endif // _USE_SSE

      DTYPE voxel[3];

#if defined (_OPENMP)
#ifdef _USE_SSE
#pragma omp parallel for default(none) \
   private(x, y, z, a, b, c, oldPreX, oldPreY, oldPreZ, xPre, yPre, zPre, real, \
   index, voxel, basis, xBasis, yBasis, zBasis, xControlPointCoordinates, \
   yControlPointCoordinates, zControlPointCoordinates,  \
   tempX, tempY, tempZ, xBasis_sse, yBasis_sse, zBasis_sse, \
   temp_basis_sse, basis_sse, val) \
   shared(deformationField, fieldPtrX, fieldPtrY, fieldPtrZ, referenceMatrix_real_to_voxel, \
   bspline, controlPointPtrX, controlPointPtrY, controlPointPtrZ, \
   splineControlPoint, mask)
#else
#pragma omp parallel for default(none) \
   private(x, y, z, a, b, c, oldPreX, oldPreY, oldPreZ, xPre, yPre, zPre, real, \
   index, voxel, basis, xBasis, yBasis, zBasis, xControlPointCoordinates, \
   yControlPointCoordinates, zControlPointCoordinates, coord) \
   shared(deformationField, fieldPtrX, fieldPtrY, fieldPtrZ, referenceMatrix_real_to_voxel, \
   bspline, controlPointPtrX, controlPointPtrY, controlPointPtrZ, \
   splineControlPoint, mask)
#endif // _USE_SSE
#endif // _OPENMP
      for(z=0; z<deformationField->nz; z++)
      {

         index=z*deformationField->nx*deformationField->ny;
         oldPreX=-99;
         oldPreY=-99;
         oldPreZ=-99;
         for(y=0; y<deformationField->ny; y++)
         {
            for(x=0; x<deformationField->nx; x++)
            {

               if(mask[index]>-1)
               {
                  // The previous position at the current pixel position is read
                  real[0] = fieldPtrX[index];
                  real[1] = fieldPtrY[index];
                  real[2] = fieldPtrZ[index];

                  // From real to pixel position in the control point space
                  voxel[0] =
                        referenceMatrix_real_to_voxel.m[0][0] * real[0] +
                        referenceMatrix_real_to_voxel.m[0][1] * real[1] +
                        referenceMatrix_real_to_voxel.m[0][2] * real[2] +
                        referenceMatrix_real_to_voxel.m[0][3] ;
                  voxel[1] =
                        referenceMatrix_real_to_voxel.m[1][0] * real[0] +
                        referenceMatrix_real_to_voxel.m[1][1] * real[1] +
                        referenceMatrix_real_to_voxel.m[1][2] * real[2] +
                        referenceMatrix_real_to_voxel.m[1][3] ;
                  voxel[2] =
                        referenceMatrix_real_to_voxel.m[2][0] * real[0] +
                        referenceMatrix_real_to_voxel.m[2][1] * real[1] +
                        referenceMatrix_real_to_voxel.m[2][2] * real[2] +
                        referenceMatrix_real_to_voxel.m[2][3] ;
                  //                        reg_mat44_mul(referenceMatrix_real_to_voxel, real, voxel);

                  // The spline coefficients are computed
                  xPre=(int)reg_floor(voxel[0]);
                  basis=voxel[0]-static_cast<DTYPE>(xPre);
                  --xPre;
                  if(basis<0.0) basis=0.0; //rounding error
                  if(bspline) get_BSplineBasisValues<DTYPE>(basis, xBasis);
                  else Get_SplineBasisValues<DTYPE>(basis, xBasis);

                  yPre=(int)reg_floor(voxel[1]);
                  basis=voxel[1]-static_cast<DTYPE>(yPre);
                  --yPre;
                  if(basis<0.0) basis=0.0; //rounding error
                  if(bspline) get_BSplineBasisValues<DTYPE>(basis, yBasis);
                  else Get_SplineBasisValues<DTYPE>(basis, yBasis);

                  zPre=(int)reg_floor(voxel[2]);
                  basis=voxel[2]-static_cast<DTYPE>(zPre);
                  --zPre;
                  if(basis<0.0) basis=0.0; //rounding error
                  if(bspline) get_BSplineBasisValues<DTYPE>(basis, zBasis);
                  else Get_SplineBasisValues<DTYPE>(basis, zBasis);

                  // The control point postions are extracted
                  if(xPre!=oldPreX || yPre!=oldPreY || zPre!=oldPreZ)
                  {
#ifdef _USE_SSE
                     get_GridValues<DTYPE>(xPre,
                                           yPre,
                                           zPre,
                                           splineControlPoint,
                                           controlPointPtrX,
                                           controlPointPtrY,
                                           controlPointPtrZ,
                                           xControlPointCoordinates.f,
                                           yControlPointCoordinates.f,
                                           zControlPointCoordinates.f,
                                           false, // no approximation
                                           false // not a deformation field
                                           );
#else // _USE_SSE
                     get_GridValues<DTYPE>(xPre,
                                           yPre,
                                           zPre,
                                           splineControlPoint,
                                           controlPointPtrX,
                                           controlPointPtrY,
                                           controlPointPtrZ,
                                           xControlPointCoordinates,
                                           yControlPointCoordinates,
                                           zControlPointCoordinates,
                                           false, // no approximation
                                           false // not a deformation field
                                           );
#endif // _USE_SSE
                     oldPreX=xPre;
                     oldPreY=yPre;
                     oldPreZ=zPre;
                  }

#if _USE_SSE
                  tempX =  _mm_set_ps1(0.0);
                  tempY =  _mm_set_ps1(0.0);
                  tempZ =  _mm_set_ps1(0.0);
                  val.f[0] = xBasis[0];
                  val.f[1] = xBasis[1];
                  val.f[2] = xBasis[2];
                  val.f[3] = xBasis[3];
                  xBasis_sse = val.m;

                  //addition and multiplication of the 16 basis value and CP position for each axis
                  for(c=0; c<4; c++)
                  {
                     for(b=0; b<4; b++)
                     {
                        yBasis_sse  = _mm_set_ps1(yBasis[b]);
                        zBasis_sse  = _mm_set_ps1(zBasis[c]);
                        temp_basis_sse = _mm_mul_ps(yBasis_sse, zBasis_sse);
                        basis_sse = _mm_mul_ps(temp_basis_sse, xBasis_sse);

                        tempX = _mm_add_ps(_mm_mul_ps(basis_sse, xControlPointCoordinates.m[c*4+b]), tempX );
                        tempY = _mm_add_ps(_mm_mul_ps(basis_sse, yControlPointCoordinates.m[c*4+b]), tempY );
                        tempZ = _mm_add_ps(_mm_mul_ps(basis_sse, zControlPointCoordinates.m[c*4+b]), tempZ );
                     }
                  }
                  //the values stored in SSE variables are transfered to normal float
                  val.m = tempX;
                  real[0] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempY;
                  real[1] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m = tempZ;
                  real[2] = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                  real[0]=0.0;
                  real[1]=0.0;
                  real[2]=0.0;
                  coord=0;
                  for(c=0; c<4; c++)
                  {
                     for(b=0; b<4; b++)
                     {
                        for(a=0; a<4; a++)
                        {
                           DTYPE tempValue = xBasis[a] * yBasis[b] * zBasis[c];
                           real[0] += xControlPointCoordinates[coord] * tempValue;
                           real[1] += yControlPointCoordinates[coord] * tempValue;
                           real[2] += zControlPointCoordinates[coord] * tempValue;
                           coord++;
                        }
                     }
                  }
#endif
                  fieldPtrX[index] = real[0];
                  fieldPtrY[index] = real[1];
                  fieldPtrZ[index] = real[2];
               }
               index++;
            }
         }
      }
   }//Composition of deformation
   else  // !composition
   {
      DTYPE gridVoxelSpacing[3];
      gridVoxelSpacing[0] = splineControlPoint->dx / deformationField->dx;
      gridVoxelSpacing[1] = splineControlPoint->dy / deformationField->dy;
      gridVoxelSpacing[2] = splineControlPoint->dz / deformationField->dz;
#ifdef _USE_SSE
#ifdef _WIN32
      union u1
      {
         __m128 m[4];
         __declspec(align(16)) DTYPE f[16];
      } yzBasis;
      union u2
      {
         __m128 m[16];
         __declspec(align(16)) DTYPE f[64];
      } xyzBasis;
#else // _WIN32
      union
      {
         __m128 m[4];
         DTYPE f[16] __attribute__((aligned(16)));
      } yzBasis;
      union
      {
         __m128 m[16];
         DTYPE f[64] __attribute__((aligned(16)));
      } xyzBasis;
#endif // _WIN32
#else // _USE_SSE
      DTYPE yzBasis[16], xyzBasis[64];
#endif // _USE_SSE

#if defined (_OPENMP)
#ifdef _USE_SSE
#pragma omp parallel for default(none) \
   private(x, y, z, a, b, c, oldPreX, oldPreY, oldPreZ, xPre, yPre, zPre, real, \
   index, basis, xyzBasis, yzBasis, zBasis, temp, xControlPointCoordinates, \
   yControlPointCoordinates, zControlPointCoordinates, oldBasis, \
   tempX, tempY, tempZ, xBasis_sse, yBasis_sse, zBasis_sse, \
   temp_basis_sse, basis_sse, val, tempCurrent) \
   shared(deformationField, fieldPtrX, fieldPtrY, fieldPtrZ, splineControlPoint, mask, \
   gridVoxelSpacing, bspline, controlPointPtrX, controlPointPtrY, controlPointPtrZ)
#else //  _USE_SSE
#pragma omp parallel for default(none) \
   private(x, y, z, a, b, c, oldPreX, oldPreY, oldPreZ, xPre, yPre, zPre, real, \
   index, basis, xyzBasis, yzBasis, zBasis, temp, xControlPointCoordinates, \
   yControlPointCoordinates, zControlPointCoordinates, oldBasis, coord) \
   shared(deformationField, fieldPtrX, fieldPtrY, fieldPtrZ, splineControlPoint, mask, \
   gridVoxelSpacing, bspline, controlPointPtrX, controlPointPtrY, controlPointPtrZ)
#endif // _USE_SSE
#endif // _OPENMP
      for(z=0; z<deformationField->nz; z++)
      {

         index=z*deformationField->nx*deformationField->ny;
         oldBasis=1.1;

         zPre=static_cast<int>(static_cast<DTYPE>(z)/gridVoxelSpacing[2]);
         basis=static_cast<DTYPE>(z)/gridVoxelSpacing[2]-static_cast<DTYPE>(zPre);
         if(basis<0.0) basis=0.0; //rounding error
         if(bspline) get_BSplineBasisValues<DTYPE>(basis, zBasis);
         else Get_SplineBasisValues<DTYPE>(basis, zBasis);

         for(y=0; y<deformationField->ny; y++)
         {

            yPre=static_cast<int>(static_cast<DTYPE>(y)/gridVoxelSpacing[1]);
            basis=static_cast<DTYPE>(y)/gridVoxelSpacing[1]-static_cast<DTYPE>(yPre);
            if(basis<0.0) basis=0.0; //rounding error
            if(bspline) get_BSplineBasisValues<DTYPE>(basis, temp);
            else Get_SplineBasisValues<DTYPE>(basis, temp);
#if _USE_SSE
            val.f[0] = temp[0];
            val.f[1] = temp[1];
            val.f[2] = temp[2];
            val.f[3] = temp[3];
            tempCurrent=val.m;
            for(a=0; a<4; a++)
            {
               val.m=_mm_set_ps1(zBasis[a]);
               yzBasis.m[a] = _mm_mul_ps(tempCurrent,val.m);
            }
#else
            coord=0;
            for(a=0; a<4; a++)
            {
               yzBasis[coord++]=temp[0]*zBasis[a];
               yzBasis[coord++]=temp[1]*zBasis[a];
               yzBasis[coord++]=temp[2]*zBasis[a];
               yzBasis[coord++]=temp[3]*zBasis[a];
            }
#endif

            for(x=0; x<deformationField->nx; x++)
            {

               xPre=static_cast<int>(static_cast<DTYPE>(x)/gridVoxelSpacing[0]);
               basis=static_cast<DTYPE>(x)/gridVoxelSpacing[0]-static_cast<DTYPE>(xPre);
               if(basis<0.0) basis=0.0; //rounding error
               if(bspline) get_BSplineBasisValues<DTYPE>(basis, temp);
               else Get_SplineBasisValues<DTYPE>(basis, temp);
#if _USE_SSE

               val.f[0] = temp[0];
               val.f[1] = temp[1];
               val.f[2] = temp[2];
               val.f[3] = temp[3];
               tempCurrent=val.m;
               for(a=0; a<16; ++a)
               {
                  val.m=_mm_set_ps1(yzBasis.f[a]);
                  xyzBasis.m[a]=_mm_mul_ps(tempCurrent,val.m);
               }
#else
               coord=0;
               for(a=0; a<16; a++)
               {
                  xyzBasis[coord++]=temp[0]*yzBasis[a];
                  xyzBasis[coord++]=temp[1]*yzBasis[a];
                  xyzBasis[coord++]=temp[2]*yzBasis[a];
                  xyzBasis[coord++]=temp[3]*yzBasis[a];
               }
#endif
               if(basis<=oldBasis || x==0)
               {
#ifdef _USE_SSE
                  get_GridValues<DTYPE>(xPre,
                                        yPre,
                                        zPre,
                                        splineControlPoint,
                                        controlPointPtrX,
                                        controlPointPtrY,
                                        controlPointPtrZ,
                                        xControlPointCoordinates.f,
                                        yControlPointCoordinates.f,
                                        zControlPointCoordinates.f,
                                        false, // no approximation
                                        false // not a deformation field
                                        );
#else // _USE_SSE
                  get_GridValues<DTYPE>(xPre,
                                        yPre,
                                        zPre,
                                        splineControlPoint,
                                        controlPointPtrX,
                                        controlPointPtrY,
                                        controlPointPtrZ,
                                        xControlPointCoordinates,
                                        yControlPointCoordinates,
                                        zControlPointCoordinates,
                                        false, // no approximation
                                        false // not a deformation field
                                        );
#endif // _USE_SSE
               }
               oldBasis=basis;

               real[0]=0.0;
               real[1]=0.0;
               real[2]=0.0;

               if(mask[index]>-1)
               {
#if _USE_SSE
                  tempX =  _mm_set_ps1(0.0);
                  tempY =  _mm_set_ps1(0.0);
                  tempZ =  _mm_set_ps1(0.0);
                  //addition and multiplication of the 64 basis value and CP displacement for each axis
                  for(a=0; a<16; a++)
                  {
                     tempX = _mm_add_ps(_mm_mul_ps(xyzBasis.m[a], xControlPointCoordinates.m[a]), tempX );
                     tempY = _mm_add_ps(_mm_mul_ps(xyzBasis.m[a], yControlPointCoordinates.m[a]), tempY );
                     tempZ = _mm_add_ps(_mm_mul_ps(xyzBasis.m[a], zControlPointCoordinates.m[a]), tempZ );
                  }
                  //the values stored in SSE variables are transfered to normal float
                  val.m=tempX;
                  real[0]=val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m=tempY;
                  real[1]= val.f[0]+val.f[1]+val.f[2]+val.f[3];
                  val.m=tempZ;
                  real[2]= val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
                  for(a=0; a<64; a++)
                  {
                     real[0] += xControlPointCoordinates[a] * xyzBasis[a];
                     real[1] += yControlPointCoordinates[a] * xyzBasis[a];
                     real[2] += zControlPointCoordinates[a] * xyzBasis[a];
                  }
#endif
               }// mask
               fieldPtrX[index] = real[0];
               fieldPtrY[index] = real[1];
               fieldPtrZ[index] = real[2];
               index++;
            } // x
         } // y
      } // z
   }// from a deformation field

   return;
}
/* *************************************************************** */
void reg_spline_getDeformationField(nifti_image *splineControlPoint,
                                    nifti_image *deformationField,
                                    int *mask,
                                    bool composition,
                                    bool bspline)
{
   if(splineControlPoint->datatype != deformationField->datatype)
   {
      reg_print_fct_error("reg_spline_getDeformationField");
      reg_print_msg_error("The spline control point image and the deformation field image are expected to be the same type");
      reg_exit(1);
   }

#if _USE_SSE
   if(splineControlPoint->datatype != NIFTI_TYPE_FLOAT32)
   {
      fprintf(stderr,"[NiftyReg ERROR] SSE computation has only been implemented for single precision.\n");
      fprintf(stderr,"[NiftyReg ERROR] The deformation field is not computed\n");
      reg_exit(1);
   }
#endif

   bool MrPropre=false;
   if(mask==NULL)
   {
      // Active voxel are all superior to -1, 0 thus will do !
      MrPropre=true;
      mask=(int *)calloc(deformationField->nx*deformationField->ny*deformationField->nz, sizeof(int));
   }

   // Check if an affine initialisation is required
   if(splineControlPoint->num_ext>0)
   {
      if(splineControlPoint->ext_list[0].edata!=NULL)
      {
         reg_affine_getDeformationField(reinterpret_cast<mat44 *>(splineControlPoint->ext_list[0].edata),
               deformationField,
               composition,
               mask);
         composition=true;
      }
   }

   if(splineControlPoint->nz==1)
   {
      switch(deformationField->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_spline_getDeformationField2D<float>(splineControlPoint, deformationField, mask, composition, bspline);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_spline_getDeformationField2D<double>(splineControlPoint, deformationField, mask, composition, bspline);
         break;
      default:
         reg_print_fct_error("reg_spline_getDeformationField");
         reg_print_msg_error("Only single or double precision is implemented for deformation field");
         reg_exit(1);
      }
   }
   else
   {
      switch(deformationField->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_spline_getDeformationField3D<float>(splineControlPoint, deformationField, mask, composition, bspline);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_spline_getDeformationField3D<double>(splineControlPoint, deformationField, mask, composition, bspline);
         break;
      default:
         reg_print_fct_error("reg_spline_getDeformationField");
         reg_print_msg_error("Only single or double precision is implemented for deformation field");
         reg_exit(1);
      }
   }

   if(splineControlPoint->num_ext>1)
   {
      if(splineControlPoint->ext_list[1].edata!=NULL)
      {
         reg_affine_getDeformationField(reinterpret_cast<mat44 *>(splineControlPoint->ext_list[1].edata),
               deformationField,
               true, //composition
               mask);
      }
   }
   if(MrPropre==true)
   {
      free(mask);
      mask=NULL;
   }

   return;
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_voxelCentric2NodeCentric_core(nifti_image *nodeImage,
                                       nifti_image *voxelImage,
                                       float weight,
                                       bool update,
                                       mat44 *voxelToMillimeter
                                       )
{
   size_t nodeNumber = (size_t)nodeImage->nx*nodeImage->ny*nodeImage->nz;
   size_t voxelNumber = (size_t)voxelImage->nx*voxelImage->ny*voxelImage->nz;
   DTYPE *nodePtrX = static_cast<DTYPE *>(nodeImage->data);
   DTYPE *nodePtrY = &nodePtrX[nodeNumber];
   DTYPE *nodePtrZ = NULL;

   DTYPE *voxelPtrX = static_cast<DTYPE *>(voxelImage->data);
   DTYPE *voxelPtrY = &voxelPtrX[voxelNumber];
   DTYPE *voxelPtrZ = NULL;

   if(nodeImage->nz>1)
   {
      nodePtrZ = &nodePtrY[nodeNumber];
      voxelPtrZ= &voxelPtrY[voxelNumber];
   }

   // The transformation between the image and the grid is used
   mat44 transformation;
   // voxel to millimeter in the grid image
   if(nodeImage->sform_code>0)
      transformation=nodeImage->sto_xyz;
   else transformation=nodeImage->qto_xyz;
   // Affine transformation between the grid and the reference image
   if(nodeImage->num_ext>0)
   {
      if(nodeImage->ext_list[0].edata!=NULL)
      {
         mat44 temp=*(reinterpret_cast<mat44 *>(nodeImage->ext_list[0].edata));
         temp=nifti_mat44_inverse(temp);
         transformation = reg_mat44_mul(&temp,&transformation);
      }
   }
   // millimeter to voxel in the reference image
   if(voxelImage->sform_code>0)
      transformation = reg_mat44_mul(&voxelImage->sto_ijk,&transformation);
   else transformation = reg_mat44_mul(&voxelImage->qto_ijk,&transformation);

   // The information has to be reoriented
   mat33 reorientation;
   // Voxel to millimeter contains the orientation of the image that is used
   // to compute the spatial gradient (floating image)
   if(voxelToMillimeter!=NULL)
   {
      reorientation=reg_mat44_to_mat33(voxelToMillimeter);
      if(nodeImage->num_ext>0)
      {
         if(nodeImage->ext_list[0].edata!=NULL)
         {
            mat33 temp = reg_mat44_to_mat33(reinterpret_cast<mat44 *>(nodeImage->ext_list[0].edata));
            temp=nifti_mat33_inverse(temp);
            reorientation = nifti_mat33_mul(temp,reorientation);
         }
      }
   }
   else reg_mat33_eye(&reorientation);
   // The information has to be weighted
   float ratio[3]= {nodeImage->dx,nodeImage->dy,nodeImage->dz};
   for(int i=0; i<(nodeImage->nz>1?3:2); ++i)
   {
      if(nodeImage->sform_code>0)
      {
         ratio[i] = sqrt(
                  reg_pow2(nodeImage->sto_xyz.m[i][0]) +
               reg_pow2(nodeImage->sto_xyz.m[i][1]) +
               reg_pow2(nodeImage->sto_xyz.m[i][2]) );
      }
      ratio[i] /= voxelImage->pixdim[i+1];
      weight *= ratio[i];
   }
   // For each node, the corresponding voxel is computed
   float nodeCoord[3];
   float voxelCoord[3];
   for(int z=0; z<nodeImage->nz; z++)
   {
      nodeCoord[2]=z;
      for(int y=0; y<nodeImage->ny; y++)
      {
         nodeCoord[1]=y;
         for(int x=0; x<nodeImage->nx; x++)
         {
            nodeCoord[0]=x;
            reg_mat44_mul(&transformation,nodeCoord,voxelCoord);
            // linear interpolation is performed
            DTYPE basisX[2], basisY[2], basisZ[2]={0,0};
            int pre[3]=
            {
               static_cast<int>(reg_floor(voxelCoord[0])),
               static_cast<int>(reg_floor(voxelCoord[1])),
               static_cast<int>(reg_floor(voxelCoord[2]))
            };
            basisX[1]=voxelCoord[0]-static_cast<DTYPE>(pre[0]);
            basisX[0]=static_cast<DTYPE>(1) - basisX[1];
            basisY[1]=voxelCoord[1]-static_cast<DTYPE>(pre[1]);
            basisY[0]=static_cast<DTYPE>(1) - basisY[1];
            if(voxelPtrZ!=NULL)
            {
               basisZ[1]=voxelCoord[2]-static_cast<DTYPE>(pre[2]);
               basisZ[0]=static_cast<DTYPE>(1) - basisZ[1];
            }
            DTYPE interpolatedValue[3]= {0,0,0};
            for(int c=0; c<2; ++c)
            {
               int indexZ=pre[2]+c;
               if(indexZ>-1 && indexZ<voxelImage->nz)
               {
                  for(int b=0; b<2; ++b)
                  {
                     int indexY=pre[1]+b;
                     if(indexY>-1 && indexY<voxelImage->ny)
                     {
                        for(int a=0; a<2; ++a)
                        {
                           int indexX=pre[0]+a;
                           if(indexX>-1 && indexX<voxelImage->nx)
                           {
                              size_t index=(indexZ*voxelImage->ny+indexY) *
                                    voxelImage->nx+indexX;
                              DTYPE linearWeight = basisX[a] * basisY[b];
                              if(voxelPtrZ!=NULL) linearWeight *= basisZ[c];
                              interpolatedValue[0] += linearWeight * voxelPtrX[index];
                              interpolatedValue[1] += linearWeight * voxelPtrY[index];
                              if(voxelPtrZ!=NULL)
                                 interpolatedValue[2] += linearWeight * voxelPtrZ[index];
                           }
                        }
                     }
                  }
               }
            }
            DTYPE reorientedValue[3]={0,0,0};
            reorientedValue[0] =
                  reorientation.m[0][0] * interpolatedValue[0] +
                  reorientation.m[1][0] * interpolatedValue[1] +
                  reorientation.m[2][0] * interpolatedValue[2] ;
            reorientedValue[1] =
                  reorientation.m[0][1] * interpolatedValue[0] +
                  reorientation.m[1][1] * interpolatedValue[1] +
                  reorientation.m[2][1] * interpolatedValue[2] ;
            if(voxelPtrZ!=NULL)
               reorientedValue[2] =
                     reorientation.m[0][2] * interpolatedValue[0] +
                     reorientation.m[1][2] * interpolatedValue[1] +
                     reorientation.m[2][2] * interpolatedValue[2] ;
            if(update)
            {
               *nodePtrX += reorientedValue[0]*static_cast<DTYPE>(weight);
               *nodePtrY += reorientedValue[1]*static_cast<DTYPE>(weight);
               if(voxelPtrZ!=NULL)
                  *nodePtrZ += reorientedValue[2]*static_cast<DTYPE>(weight);
            }
            else
            {
               *nodePtrX = reorientedValue[0]*static_cast<DTYPE>(weight);
               *nodePtrY = reorientedValue[1]*static_cast<DTYPE>(weight);
               if(voxelPtrZ!=NULL)
                  *nodePtrZ = reorientedValue[2]*static_cast<DTYPE>(weight);
            }
            ++nodePtrX;
            ++nodePtrY;
            if(voxelPtrZ!=NULL)
               ++nodePtrZ;
         } // loop over
      } // loop over y
   } // loop over z
}
/* *************************************************************** */
extern "C++"
void reg_voxelCentric2NodeCentric(nifti_image *nodeImage,
                                  nifti_image *voxelImage,
                                  float weight,
                                  bool update,
                                  mat44 *voxelToMillimeter
                                  )
{
   if(nodeImage->datatype!=voxelImage->datatype)
   {
      reg_print_fct_error("reg_voxelCentric2NodeCentric");
      reg_print_msg_error("Both input images do not have the same type");
      reg_exit(1);
   }

   switch(nodeImage->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      reg_voxelCentric2NodeCentric_core<float>
            (nodeImage, voxelImage, weight, update, voxelToMillimeter);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_voxelCentric2NodeCentric_core<double>
            (nodeImage, voxelImage, weight, update, voxelToMillimeter);
      break;
   default:
      reg_print_fct_error("reg_voxelCentric2NodeCentric");
      reg_print_msg_error("Data type not supported");
      reg_exit(1);
   }
}
/* *************************************************************** */
/* *************************************************************** */
template<class SplineTYPE>
SplineTYPE GetValue(SplineTYPE *array, int *dim, int x, int y, int z)
{
   if(x<0 || x>= dim[1] || y<0 || y>= dim[2] || z<0 || z>= dim[3])
      return 0.0;
   return array[(z*dim[2]+y)*dim[1]+x];
}
/* *************************************************************** */
template<class SplineTYPE>
void SetValue(SplineTYPE *array, int *dim, int x, int y, int z, SplineTYPE value)
{
   if(x<0 || x>= dim[1] || y<0 || y>= dim[2] || z<0 || z>= dim[3])
      return;
   array[(z*dim[2]+y)*dim[1]+x] = value;
}
/* *************************************************************** */
template<class SplineTYPE>
void reg_spline_refineControlPointGrid2D(nifti_image *splineControlPoint,
                                         nifti_image *referenceImage)
{
   // The input grid is first saved
   SplineTYPE *oldGrid = (SplineTYPE *)malloc(splineControlPoint->nvox*splineControlPoint->nbyper);
   SplineTYPE *gridPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
   memcpy(oldGrid, gridPtrX, splineControlPoint->nvox*splineControlPoint->nbyper);
   if(splineControlPoint->data!=NULL) free(splineControlPoint->data);
   int oldDim[4];
   oldDim[0]=splineControlPoint->dim[0];
   oldDim[1]=splineControlPoint->dim[1];
   oldDim[2]=splineControlPoint->dim[2];
   oldDim[3]=splineControlPoint->dim[3];

   splineControlPoint->dx = splineControlPoint->pixdim[1] = splineControlPoint->dx / 2.0f;
   splineControlPoint->dy = splineControlPoint->pixdim[2] = splineControlPoint->dy / 2.0f;
   splineControlPoint->dz = 1.0f;
   if(referenceImage!=NULL)
   {
      splineControlPoint->dim[1]=splineControlPoint->nx=static_cast<int>(reg_ceil(referenceImage->nx*referenceImage->dx/splineControlPoint->dx)+3.f);
      splineControlPoint->dim[2]=splineControlPoint->ny=static_cast<int>(reg_ceil(referenceImage->ny*referenceImage->dy/splineControlPoint->dy)+3.f);
   }
   else
   {
      splineControlPoint->dim[1]=splineControlPoint->nx=(oldDim[1]-3)*2+3;
      splineControlPoint->dim[2]=splineControlPoint->ny=(oldDim[2]-3)*2+3;
   }
   splineControlPoint->dim[3]=splineControlPoint->nz=1;

   splineControlPoint->nvox =
         (size_t)splineControlPoint->nx*
         (size_t)splineControlPoint->ny*
         (size_t)splineControlPoint->nz*
         (size_t)splineControlPoint->nt*
         (size_t)splineControlPoint->nu;

   splineControlPoint->data = (void *)calloc(splineControlPoint->nvox, splineControlPoint->nbyper);
   gridPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
   SplineTYPE *gridPtrY = &gridPtrX[splineControlPoint->nx*splineControlPoint->ny];
   SplineTYPE *oldGridPtrX = &oldGrid[0];
   SplineTYPE *oldGridPtrY = &oldGridPtrX[oldDim[1]*oldDim[2]];

   for(int y=0; y<oldDim[2]; y++)
   {
      int Y=2*y-1;
      if(Y<splineControlPoint->ny)
      {
         for(int x=0; x<oldDim[1]; x++)
         {
            int X=2*x-1;
            if(X<splineControlPoint->nx)
            {

               /* X Axis */
               // 0 0
               SetValue(gridPtrX, splineControlPoint->dim, X, Y, 0,
                        (GetValue(oldGridPtrX,oldDim,x-1,y-1,0) + GetValue(oldGridPtrX,oldDim,x+1,y-1,0) +
                         GetValue(oldGridPtrX,oldDim,x-1,y+1,0) + GetValue(oldGridPtrX,oldDim,x+1,y+1,0)
                         + 6.0f * (GetValue(oldGridPtrX,oldDim,x-1,y,0) + GetValue(oldGridPtrX,oldDim,x+1,y,0) +
                                   GetValue(oldGridPtrX,oldDim,x,y-1,0) + GetValue(oldGridPtrX,oldDim,x,y+1,0) )
                         + 36.0f * GetValue(oldGridPtrX,oldDim,x,y,0) ) / 64.0f);
               // 1 0
               SetValue(gridPtrX, splineControlPoint->dim, X+1, Y, 0,
                        (GetValue(oldGridPtrX,oldDim,x,y-1,0) + GetValue(oldGridPtrX,oldDim,x+1,y-1,0) +
                         GetValue(oldGridPtrX,oldDim,x,y+1,0) + GetValue(oldGridPtrX,oldDim,x+1,y+1,0)
                         + 6.0f * ( GetValue(oldGridPtrX,oldDim,x,y,0) + GetValue(oldGridPtrX,oldDim,x+1,y,0) ) ) / 16.0f);
               // 0 1
               SetValue(gridPtrX, splineControlPoint->dim, X, Y+1, 0,
                        (GetValue(oldGridPtrX,oldDim,x-1,y,0) + GetValue(oldGridPtrX,oldDim,x-1,y+1,0) +
                         GetValue(oldGridPtrX,oldDim,x+1,y,0) + GetValue(oldGridPtrX,oldDim,x+1,y+1,0)
                         + 6.0f * ( GetValue(oldGridPtrX,oldDim,x,y,0) + GetValue(oldGridPtrX,oldDim,x,y+1,0) ) ) / 16.0f);
               // 1 1
               SetValue(gridPtrX, splineControlPoint->dim, X+1, Y+1, 0,
                        (GetValue(oldGridPtrX,oldDim,x,y,0) + GetValue(oldGridPtrX,oldDim,x+1,y,0) +
                         GetValue(oldGridPtrX,oldDim,x,y+1,0) + GetValue(oldGridPtrX,oldDim,x+1,y+1,0) ) / 4.0f);

               /* Y Axis */
               // 0 0
               SetValue(gridPtrY, splineControlPoint->dim, X, Y, 0,
                        (GetValue(oldGridPtrY,oldDim,x-1,y-1,0) + GetValue(oldGridPtrY,oldDim,x+1,y-1,0) +
                         GetValue(oldGridPtrY,oldDim,x-1,y+1,0) + GetValue(oldGridPtrY,oldDim,x+1,y+1,0)
                         + 6.0f * (GetValue(oldGridPtrY,oldDim,x-1,y,0) + GetValue(oldGridPtrY,oldDim,x+1,y,0) +
                                   GetValue(oldGridPtrY,oldDim,x,y-1,0) + GetValue(oldGridPtrY,oldDim,x,y+1,0) )
                         + 36.0f * GetValue(oldGridPtrY,oldDim,x,y,0) ) / 64.0f);
               // 1 0
               SetValue(gridPtrY, splineControlPoint->dim, X+1, Y, 0,
                        (GetValue(oldGridPtrY,oldDim,x,y-1,0) + GetValue(oldGridPtrY,oldDim,x+1,y-1,0) +
                         GetValue(oldGridPtrY,oldDim,x,y+1,0) + GetValue(oldGridPtrY,oldDim,x+1,y+1,0)
                         + 6.0f * ( GetValue(oldGridPtrY,oldDim,x,y,0) + GetValue(oldGridPtrY,oldDim,x+1,y,0) ) ) / 16.0f);
               // 0 1
               SetValue(gridPtrY, splineControlPoint->dim, X, Y+1, 0,
                        (GetValue(oldGridPtrY,oldDim,x-1,y,0) + GetValue(oldGridPtrY,oldDim,x-1,y+1,0) +
                         GetValue(oldGridPtrY,oldDim,x+1,y,0) + GetValue(oldGridPtrY,oldDim,x+1,y+1,0)
                         + 6.0f * ( GetValue(oldGridPtrY,oldDim,x,y,0) + GetValue(oldGridPtrY,oldDim,x,y+1,0) ) ) / 16.0f);
               // 1 1
               SetValue(gridPtrY, splineControlPoint->dim, X+1, Y+1, 0,
                        (GetValue(oldGridPtrY,oldDim,x,y,0) + GetValue(oldGridPtrY,oldDim,x+1,y,0) +
                         GetValue(oldGridPtrY,oldDim,x,y+1,0) + GetValue(oldGridPtrY,oldDim,x+1,y+1,0) ) / 4.0f);

            }
         }
      }
   }

   free(oldGrid);
}
/* *************************************************************** */
template<class SplineTYPE>
void reg_spline_refineControlPointGrid3D(nifti_image *splineControlPoint, nifti_image *referenceImage)
{

   // The input grid is first saved
   SplineTYPE *oldGrid = (SplineTYPE *)malloc(splineControlPoint->nvox*splineControlPoint->nbyper);
   SplineTYPE *gridPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
   memcpy(oldGrid, gridPtrX, splineControlPoint->nvox*splineControlPoint->nbyper);
   if(splineControlPoint->data!=NULL) free(splineControlPoint->data);
   int oldDim[4];
   oldDim[0]=splineControlPoint->dim[0];
   oldDim[1]=splineControlPoint->dim[1];
   oldDim[2]=splineControlPoint->dim[2];
   oldDim[3]=splineControlPoint->dim[3];

   splineControlPoint->dx = splineControlPoint->pixdim[1] = splineControlPoint->dx / 2.0f;
   splineControlPoint->dy = splineControlPoint->pixdim[2] = splineControlPoint->dy / 2.0f;
   splineControlPoint->dz = splineControlPoint->pixdim[3] = splineControlPoint->dz / 2.0f;

   if(referenceImage!=NULL)
   {
      splineControlPoint->dim[1]=splineControlPoint->nx=static_cast<int>(reg_ceil(referenceImage->nx*referenceImage->dx/splineControlPoint->dx)+3.f);
      splineControlPoint->dim[2]=splineControlPoint->ny=static_cast<int>(reg_ceil(referenceImage->ny*referenceImage->dy/splineControlPoint->dy)+3.f);
      splineControlPoint->dim[3]=splineControlPoint->nz=static_cast<int>(reg_ceil(referenceImage->nz*referenceImage->dz/splineControlPoint->dz)+3.f);
   }
   else
   {
      splineControlPoint->dim[1]=splineControlPoint->nx=(oldDim[1]-3)*2+3;
      splineControlPoint->dim[2]=splineControlPoint->ny=(oldDim[2]-3)*2+3;
      splineControlPoint->dim[3]=splineControlPoint->nz=(oldDim[3]-3)*2+3;
   }
   splineControlPoint->nvox =
         (size_t)splineControlPoint->nx*
         (size_t)splineControlPoint->ny*
         (size_t)splineControlPoint->nz*
         (size_t)splineControlPoint->nt*
         (size_t)splineControlPoint->nu;
   splineControlPoint->data = (void *)calloc(splineControlPoint->nvox, splineControlPoint->nbyper);

   gridPtrX = static_cast<SplineTYPE *>(splineControlPoint->data);
   SplineTYPE *gridPtrY = &gridPtrX[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
   SplineTYPE *gridPtrZ = &gridPtrY[splineControlPoint->nx*splineControlPoint->ny*splineControlPoint->nz];
   SplineTYPE *oldGridPtrX = &oldGrid[0];
   SplineTYPE *oldGridPtrY = &oldGridPtrX[oldDim[1]*oldDim[2]*oldDim[3]];
   SplineTYPE *oldGridPtrZ = &oldGridPtrY[oldDim[1]*oldDim[2]*oldDim[3]];


   for(int z=0; z<oldDim[3]; z++)
   {
      int Z=2*z-1;
      if(Z<splineControlPoint->nz)
      {
         for(int y=0; y<oldDim[2]; y++)
         {
            int Y=2*y-1;
            if(Y<splineControlPoint->ny)
            {
               for(int x=0; x<oldDim[1]; x++)
               {
                  int X=2*x-1;
                  if(X<splineControlPoint->nx)
                  {

                     /* X Axis */
                     // 0 0 0
                     SetValue(gridPtrX, splineControlPoint->dim, X, Y, Z,
                              (GetValue(oldGridPtrX,oldDim,x-1,y-1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z-1) +
                               GetValue(oldGridPtrX,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z-1) +
                               GetValue(oldGridPtrX,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z+1)+
                               GetValue(oldGridPtrX,oldDim,x-1,y+1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1)
                               + 6.0f * (GetValue(oldGridPtrX,oldDim,x-1,y-1,z) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z) +
                                         GetValue(oldGridPtrX,oldDim,x+1,y-1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                         GetValue(oldGridPtrX,oldDim,x-1,y,z-1) + GetValue(oldGridPtrX,oldDim,x-1,y,z+1) +
                                         GetValue(oldGridPtrX,oldDim,x+1,y,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                         GetValue(oldGridPtrX,oldDim,x,y-1,z-1) + GetValue(oldGridPtrX,oldDim,x,y-1,z+1) +
                                         GetValue(oldGridPtrX,oldDim,x,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1) )
                               + 36.0f * (GetValue(oldGridPtrX,oldDim,x-1,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                          GetValue(oldGridPtrX,oldDim,x,y-1,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z) +
                                          GetValue(oldGridPtrX,oldDim,x,y,z-1) + GetValue(oldGridPtrX,oldDim,x,y,z+1) )
                               + 216.0f * GetValue(oldGridPtrX,oldDim,x,y,z) ) / 512.0f);

                     // 1 0 0
                     SetValue(gridPtrX, splineControlPoint->dim, X+1, Y, Z,
                              ( GetValue(oldGridPtrX,oldDim,x,y-1,z-1) + GetValue(oldGridPtrX,oldDim,x,y-1,z+1) +
                                GetValue(oldGridPtrX,oldDim,x,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1) +
                                GetValue(oldGridPtrX,oldDim,x+1,y-1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z+1) +
                                GetValue(oldGridPtrX,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrX,oldDim,x,y-1,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z) +
                                        GetValue(oldGridPtrX,oldDim,x,y,z-1) + GetValue(oldGridPtrX,oldDim,x,y,z+1) +
                                        GetValue(oldGridPtrX,oldDim,x+1,y-1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                        GetValue(oldGridPtrX,oldDim,x+1,y,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1)) +
                                36.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z)) ) / 128.0f);

                     // 0 1 0
                     SetValue(gridPtrX, splineControlPoint->dim, X, Y+1, Z,
                              ( GetValue(oldGridPtrX,oldDim,x-1,y,z-1) + GetValue(oldGridPtrX,oldDim,x-1,y,z+1) +
                                GetValue(oldGridPtrX,oldDim,x+1,y,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrX,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z+1) +
                                GetValue(oldGridPtrX,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrX,oldDim,x-1,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                        GetValue(oldGridPtrX,oldDim,x,y,z-1) + GetValue(oldGridPtrX,oldDim,x,y,z+1) +
                                        GetValue(oldGridPtrX,oldDim,x-1,y+1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                        GetValue(oldGridPtrX,oldDim,x,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1)) +
                                36.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z)) ) / 128.0f);

                     // 1 1 0
                     SetValue(gridPtrX, splineControlPoint->dim, X+1, Y+1, Z,
                              (GetValue(oldGridPtrX,oldDim,x,y,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y,z-1) +
                               GetValue(oldGridPtrX,oldDim,x,y+1,z-1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z-1) +
                               GetValue(oldGridPtrX,oldDim,x,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                               GetValue(oldGridPtrX,oldDim,x,y+1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                               6.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                       GetValue(oldGridPtrX,oldDim,x,y+1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) ) ) / 32.0f);

                     // 0 0 1
                     SetValue(gridPtrX, splineControlPoint->dim, X, Y, Z+1,
                              ( GetValue(oldGridPtrX,oldDim,x-1,y-1,z) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z) +
                                GetValue(oldGridPtrX,oldDim,x+1,y-1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrX,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z+1) +
                                GetValue(oldGridPtrX,oldDim,x+1,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrX,oldDim,x-1,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                        GetValue(oldGridPtrX,oldDim,x,y-1,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z) +
                                        GetValue(oldGridPtrX,oldDim,x-1,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                                        GetValue(oldGridPtrX,oldDim,x,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1)) +
                                36.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x,y,z+1)) ) / 128.0f);

                     // 1 0 1
                     SetValue(gridPtrX, splineControlPoint->dim, X+1, Y, Z+1,
                              (GetValue(oldGridPtrX,oldDim,x,y-1,z) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z) +
                               GetValue(oldGridPtrX,oldDim,x,y-1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y-1,z+1) +
                               GetValue(oldGridPtrX,oldDim,x,y+1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                               GetValue(oldGridPtrX,oldDim,x,y+1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                               6.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                                       GetValue(oldGridPtrX,oldDim,x,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) ) ) / 32.0f);

                     // 0 1 1
                     SetValue(gridPtrX, splineControlPoint->dim, X, Y+1, Z+1,
                              (GetValue(oldGridPtrX,oldDim,x-1,y,z) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z) +
                               GetValue(oldGridPtrX,oldDim,x-1,y,z+1) + GetValue(oldGridPtrX,oldDim,x-1,y+1,z+1) +
                               GetValue(oldGridPtrX,oldDim,x+1,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                               GetValue(oldGridPtrX,oldDim,x+1,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1) +
                               6.0f * (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x,y+1,z) +
                                       GetValue(oldGridPtrX,oldDim,x,y,z+1) + GetValue(oldGridPtrX,oldDim,x,y+1,z+1) ) ) / 32.0f);

                     // 1 1 1
                     SetValue(gridPtrX, splineControlPoint->dim, X+1, Y+1, Z+1,
                              (GetValue(oldGridPtrX,oldDim,x,y,z) + GetValue(oldGridPtrX,oldDim,x+1,y,z) +
                               GetValue(oldGridPtrX,oldDim,x,y+1,z) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z) +
                               GetValue(oldGridPtrX,oldDim,x,y,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y,z+1) +
                               GetValue(oldGridPtrX,oldDim,x,y+1,z+1) + GetValue(oldGridPtrX,oldDim,x+1,y+1,z+1)) / 8.0f);


                     /* Y Axis */
                     // 0 0 0
                     SetValue(gridPtrY, splineControlPoint->dim, X, Y, Z,
                              (GetValue(oldGridPtrY,oldDim,x-1,y-1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z-1) +
                               GetValue(oldGridPtrY,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z-1) +
                               GetValue(oldGridPtrY,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z+1)+
                               GetValue(oldGridPtrY,oldDim,x-1,y+1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1)
                               + 6.0f * (GetValue(oldGridPtrY,oldDim,x-1,y-1,z) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z) +
                                         GetValue(oldGridPtrY,oldDim,x+1,y-1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                         GetValue(oldGridPtrY,oldDim,x-1,y,z-1) + GetValue(oldGridPtrY,oldDim,x-1,y,z+1) +
                                         GetValue(oldGridPtrY,oldDim,x+1,y,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                         GetValue(oldGridPtrY,oldDim,x,y-1,z-1) + GetValue(oldGridPtrY,oldDim,x,y-1,z+1) +
                                         GetValue(oldGridPtrY,oldDim,x,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1) )
                               + 36.0f * (GetValue(oldGridPtrY,oldDim,x-1,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                          GetValue(oldGridPtrY,oldDim,x,y-1,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z) +
                                          GetValue(oldGridPtrY,oldDim,x,y,z-1) + GetValue(oldGridPtrY,oldDim,x,y,z+1) )
                               + 216.0f * GetValue(oldGridPtrY,oldDim,x,y,z) ) / 512.0f);

                     // 1 0 0
                     SetValue(gridPtrY, splineControlPoint->dim, X+1, Y, Z,
                              ( GetValue(oldGridPtrY,oldDim,x,y-1,z-1) + GetValue(oldGridPtrY,oldDim,x,y-1,z+1) +
                                GetValue(oldGridPtrY,oldDim,x,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1) +
                                GetValue(oldGridPtrY,oldDim,x+1,y-1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z+1) +
                                GetValue(oldGridPtrY,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrY,oldDim,x,y-1,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z) +
                                        GetValue(oldGridPtrY,oldDim,x,y,z-1) + GetValue(oldGridPtrY,oldDim,x,y,z+1) +
                                        GetValue(oldGridPtrY,oldDim,x+1,y-1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                        GetValue(oldGridPtrY,oldDim,x+1,y,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1)) +
                                36.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z)) ) / 128.0f);

                     // 0 1 0
                     SetValue(gridPtrY, splineControlPoint->dim, X, Y+1, Z,
                              ( GetValue(oldGridPtrY,oldDim,x-1,y,z-1) + GetValue(oldGridPtrY,oldDim,x-1,y,z+1) +
                                GetValue(oldGridPtrY,oldDim,x+1,y,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrY,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z+1) +
                                GetValue(oldGridPtrY,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrY,oldDim,x-1,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                        GetValue(oldGridPtrY,oldDim,x,y,z-1) + GetValue(oldGridPtrY,oldDim,x,y,z+1) +
                                        GetValue(oldGridPtrY,oldDim,x-1,y+1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                        GetValue(oldGridPtrY,oldDim,x,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1)) +
                                36.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z)) ) / 128.0f);

                     // 1 1 0
                     SetValue(gridPtrY, splineControlPoint->dim, X+1, Y+1, Z,
                              (GetValue(oldGridPtrY,oldDim,x,y,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y,z-1) +
                               GetValue(oldGridPtrY,oldDim,x,y+1,z-1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z-1) +
                               GetValue(oldGridPtrY,oldDim,x,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                               GetValue(oldGridPtrY,oldDim,x,y+1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                               6.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                       GetValue(oldGridPtrY,oldDim,x,y+1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) ) ) / 32.0f);

                     // 0 0 1
                     SetValue(gridPtrY, splineControlPoint->dim, X, Y, Z+1,
                              ( GetValue(oldGridPtrY,oldDim,x-1,y-1,z) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z) +
                                GetValue(oldGridPtrY,oldDim,x+1,y-1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrY,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z+1) +
                                GetValue(oldGridPtrY,oldDim,x+1,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrY,oldDim,x-1,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                        GetValue(oldGridPtrY,oldDim,x,y-1,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z) +
                                        GetValue(oldGridPtrY,oldDim,x-1,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                                        GetValue(oldGridPtrY,oldDim,x,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1)) +
                                36.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x,y,z+1)) ) / 128.0f);

                     // 1 0 1
                     SetValue(gridPtrY, splineControlPoint->dim, X+1, Y, Z+1,
                              (GetValue(oldGridPtrY,oldDim,x,y-1,z) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z) +
                               GetValue(oldGridPtrY,oldDim,x,y-1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y-1,z+1) +
                               GetValue(oldGridPtrY,oldDim,x,y+1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                               GetValue(oldGridPtrY,oldDim,x,y+1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                               6.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                                       GetValue(oldGridPtrY,oldDim,x,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) ) ) / 32.0f);

                     // 0 1 1
                     SetValue(gridPtrY, splineControlPoint->dim, X, Y+1, Z+1,
                              (GetValue(oldGridPtrY,oldDim,x-1,y,z) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z) +
                               GetValue(oldGridPtrY,oldDim,x-1,y,z+1) + GetValue(oldGridPtrY,oldDim,x-1,y+1,z+1) +
                               GetValue(oldGridPtrY,oldDim,x+1,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                               GetValue(oldGridPtrY,oldDim,x+1,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1) +
                               6.0f * (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x,y+1,z) +
                                       GetValue(oldGridPtrY,oldDim,x,y,z+1) + GetValue(oldGridPtrY,oldDim,x,y+1,z+1) ) ) / 32.0f);

                     // 1 1 1
                     SetValue(gridPtrY, splineControlPoint->dim, X+1, Y+1, Z+1,
                              (GetValue(oldGridPtrY,oldDim,x,y,z) + GetValue(oldGridPtrY,oldDim,x+1,y,z) +
                               GetValue(oldGridPtrY,oldDim,x,y+1,z) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z) +
                               GetValue(oldGridPtrY,oldDim,x,y,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y,z+1) +
                               GetValue(oldGridPtrY,oldDim,x,y+1,z+1) + GetValue(oldGridPtrY,oldDim,x+1,y+1,z+1)) / 8.0f);

                     /* Z Axis */
                     // 0 0 0
                     SetValue(gridPtrZ, splineControlPoint->dim, X, Y, Z,
                              (GetValue(oldGridPtrZ,oldDim,x-1,y-1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z-1) +
                               GetValue(oldGridPtrZ,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z-1) +
                               GetValue(oldGridPtrZ,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z+1)+
                               GetValue(oldGridPtrZ,oldDim,x-1,y+1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1)
                               + 6.0f * (GetValue(oldGridPtrZ,oldDim,x-1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z) +
                                         GetValue(oldGridPtrZ,oldDim,x+1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                         GetValue(oldGridPtrZ,oldDim,x-1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x-1,y,z+1) +
                                         GetValue(oldGridPtrZ,oldDim,x+1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                         GetValue(oldGridPtrZ,oldDim,x,y-1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y-1,z+1) +
                                         GetValue(oldGridPtrZ,oldDim,x,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) )
                               + 36.0f * (GetValue(oldGridPtrZ,oldDim,x-1,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                          GetValue(oldGridPtrZ,oldDim,x,y-1,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z) +
                                          GetValue(oldGridPtrZ,oldDim,x,y,z-1) + GetValue(oldGridPtrZ,oldDim,x,y,z+1) )
                               + 216.0f * GetValue(oldGridPtrZ,oldDim,x,y,z) ) / 512.0f);

                     // 1 0 0
                     SetValue(gridPtrZ, splineControlPoint->dim, X+1, Y, Z,
                              ( GetValue(oldGridPtrZ,oldDim,x,y-1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y-1,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y-1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrZ,oldDim,x,y-1,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z) +
                                        GetValue(oldGridPtrZ,oldDim,x,y,z-1) + GetValue(oldGridPtrZ,oldDim,x,y,z+1) +
                                        GetValue(oldGridPtrZ,oldDim,x+1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                        GetValue(oldGridPtrZ,oldDim,x+1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1)) +
                                36.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z)) ) / 128.0f);

                     // 0 1 0
                     SetValue(gridPtrZ, splineControlPoint->dim, X, Y+1, Z,
                              ( GetValue(oldGridPtrZ,oldDim,x-1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x-1,y,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x-1,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrZ,oldDim,x-1,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                        GetValue(oldGridPtrZ,oldDim,x,y,z-1) + GetValue(oldGridPtrZ,oldDim,x,y,z+1) +
                                        GetValue(oldGridPtrZ,oldDim,x-1,y+1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                        GetValue(oldGridPtrZ,oldDim,x,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1)) +
                                36.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z)) ) / 128.0f);

                     // 1 1 0
                     SetValue(gridPtrZ, splineControlPoint->dim, X+1, Y+1, Z,
                              (GetValue(oldGridPtrZ,oldDim,x,y,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z-1) +
                               GetValue(oldGridPtrZ,oldDim,x,y+1,z-1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z-1) +
                               GetValue(oldGridPtrZ,oldDim,x,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                               GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                               6.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                       GetValue(oldGridPtrZ,oldDim,x,y+1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) ) ) / 32.0f);

                     // 0 0 1
                     SetValue(gridPtrZ, splineControlPoint->dim, X, Y, Z+1,
                              ( GetValue(oldGridPtrZ,oldDim,x-1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y-1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                                GetValue(oldGridPtrZ,oldDim,x-1,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z+1) +
                                GetValue(oldGridPtrZ,oldDim,x+1,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                                6.0f * (GetValue(oldGridPtrZ,oldDim,x-1,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                        GetValue(oldGridPtrZ,oldDim,x,y-1,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z) +
                                        GetValue(oldGridPtrZ,oldDim,x-1,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                                        GetValue(oldGridPtrZ,oldDim,x,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1)) +
                                36.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x,y,z+1)) ) / 128.0f);

                     // 1 0 1
                     SetValue(gridPtrZ, splineControlPoint->dim, X+1, Y, Z+1,
                              (GetValue(oldGridPtrZ,oldDim,x,y-1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z) +
                               GetValue(oldGridPtrZ,oldDim,x,y-1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y-1,z+1) +
                               GetValue(oldGridPtrZ,oldDim,x,y+1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                               GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                               6.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                                       GetValue(oldGridPtrZ,oldDim,x,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) ) ) / 32.0f);

                     // 0 1 1
                     SetValue(gridPtrZ, splineControlPoint->dim, X, Y+1, Z+1,
                              (GetValue(oldGridPtrZ,oldDim,x-1,y,z) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z) +
                               GetValue(oldGridPtrZ,oldDim,x-1,y,z+1) + GetValue(oldGridPtrZ,oldDim,x-1,y+1,z+1) +
                               GetValue(oldGridPtrZ,oldDim,x+1,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                               GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1) +
                               6.0f * (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x,y+1,z) +
                                       GetValue(oldGridPtrZ,oldDim,x,y,z+1) + GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) ) ) / 32.0f);

                     // 1 1 1
                     SetValue(gridPtrZ, splineControlPoint->dim, X+1, Y+1, Z+1,
                              (GetValue(oldGridPtrZ,oldDim,x,y,z) + GetValue(oldGridPtrZ,oldDim,x+1,y,z) +
                               GetValue(oldGridPtrZ,oldDim,x,y+1,z) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z) +
                               GetValue(oldGridPtrZ,oldDim,x,y,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y,z+1) +
                               GetValue(oldGridPtrZ,oldDim,x,y+1,z+1) + GetValue(oldGridPtrZ,oldDim,x+1,y+1,z+1)) / 8.0f);
                  }
               }
            }
         }
      }
   }
   free(oldGrid);
}
/* *************************************************************** */
extern "C++"
void reg_spline_refineControlPointGrid(nifti_image *controlPointGrid,
                                       nifti_image *referenceImage)
{
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] Starting the refine the control point grid\n");
#endif
   if(controlPointGrid->nz==1)
   {
      switch(controlPointGrid->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_spline_refineControlPointGrid2D<float>(controlPointGrid,referenceImage);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_spline_refineControlPointGrid2D<double>(controlPointGrid,referenceImage);
         break;
      default:
         reg_print_fct_error("reg_spline_refineControlPointGrid");
         reg_print_msg_error("Only single or double precision is implemented for the bending energy gradient");
         reg_exit(1);
      }
   }
   else
   {
      switch(controlPointGrid->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_spline_refineControlPointGrid3D<float>(controlPointGrid,referenceImage);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_spline_refineControlPointGrid3D<double>(controlPointGrid,referenceImage);
         break;
      default:
         reg_print_fct_error("reg_spline_refineControlPointGrid");
         reg_print_msg_error("Only single or double precision is implemented for the bending energy gradient");
         reg_exit(1);
      }
   }
   if(referenceImage!=NULL)
   {
      // Compute the new control point header
      // The qform (and sform) are set for the control point position image
      controlPointGrid->quatern_b=referenceImage->quatern_b;
      controlPointGrid->quatern_c=referenceImage->quatern_c;
      controlPointGrid->quatern_d=referenceImage->quatern_d;
      controlPointGrid->qoffset_x=referenceImage->qoffset_x;
      controlPointGrid->qoffset_y=referenceImage->qoffset_y;
      controlPointGrid->qoffset_z=referenceImage->qoffset_z;
      controlPointGrid->qfac=referenceImage->qfac;
      controlPointGrid->qto_xyz = nifti_quatern_to_mat44(controlPointGrid->quatern_b,
                                                         controlPointGrid->quatern_c,
                                                         controlPointGrid->quatern_d,
                                                         controlPointGrid->qoffset_x,
                                                         controlPointGrid->qoffset_y,
                                                         controlPointGrid->qoffset_z,
                                                         controlPointGrid->dx,
                                                         controlPointGrid->dy,
                                                         controlPointGrid->dz,
                                                         controlPointGrid->qfac);

      // Origin is shifted from 1 control point in the qform
      float originIndex[3];
      float originReal[3];
      originIndex[0] = -1.0f;
      originIndex[1] = -1.0f;
      originIndex[2] = 0.0f;
      if(referenceImage->nz>1) originIndex[2] = -1.0f;
      reg_mat44_mul(&(controlPointGrid->qto_xyz), originIndex, originReal);
      if(controlPointGrid->qform_code==0 && controlPointGrid->sform_code==0)
         controlPointGrid->qform_code=1;
      controlPointGrid->qto_xyz.m[0][3] = controlPointGrid->qoffset_x = originReal[0];
      controlPointGrid->qto_xyz.m[1][3] = controlPointGrid->qoffset_y = originReal[1];
      controlPointGrid->qto_xyz.m[2][3] = controlPointGrid->qoffset_z = originReal[2];

      controlPointGrid->qto_ijk = nifti_mat44_inverse(controlPointGrid->qto_xyz);

      if(controlPointGrid->sform_code>0)
      {
         float scalingRatio[3];
         scalingRatio[0]= controlPointGrid->dx / referenceImage->dx;
         scalingRatio[1]= controlPointGrid->dy / referenceImage->dy;
         scalingRatio[2] = 1.f;
         if(controlPointGrid->nz>1)
            scalingRatio[2]= controlPointGrid->dz / referenceImage->dz;

         controlPointGrid->sto_xyz.m[0][0]=referenceImage->sto_xyz.m[0][0] * scalingRatio[0];
         controlPointGrid->sto_xyz.m[1][0]=referenceImage->sto_xyz.m[1][0] * scalingRatio[0];
         controlPointGrid->sto_xyz.m[2][0]=referenceImage->sto_xyz.m[2][0] * scalingRatio[0];
         controlPointGrid->sto_xyz.m[3][0]=0.f;
         controlPointGrid->sto_xyz.m[0][1]=referenceImage->sto_xyz.m[0][1] * scalingRatio[1];
         controlPointGrid->sto_xyz.m[1][1]=referenceImage->sto_xyz.m[1][1] * scalingRatio[1];
         controlPointGrid->sto_xyz.m[2][1]=referenceImage->sto_xyz.m[2][1] * scalingRatio[1];
         controlPointGrid->sto_xyz.m[3][1]=0.f;
         controlPointGrid->sto_xyz.m[0][2]=referenceImage->sto_xyz.m[0][2] * scalingRatio[2];
         controlPointGrid->sto_xyz.m[1][2]=referenceImage->sto_xyz.m[1][2] * scalingRatio[2];
         controlPointGrid->sto_xyz.m[2][2]=referenceImage->sto_xyz.m[2][2] * scalingRatio[2];
         controlPointGrid->sto_xyz.m[3][2]=0.f;
         controlPointGrid->sto_xyz.m[0][3]=referenceImage->sto_xyz.m[0][3];
         controlPointGrid->sto_xyz.m[1][3]=referenceImage->sto_xyz.m[1][3];
         controlPointGrid->sto_xyz.m[2][3]=referenceImage->sto_xyz.m[2][3];
         controlPointGrid->sto_xyz.m[3][3]=1.f;

         // The origin is shifted by one compare to the reference image
         float originIndex[3];
         originIndex[0]=originIndex[1]=originIndex[2]=-1;
         if(referenceImage->nz<=1) originIndex[2]=0;
         reg_mat44_mul(&(controlPointGrid->sto_xyz), originIndex, originReal);
         controlPointGrid->sto_xyz.m[0][3] = originReal[0];
         controlPointGrid->sto_xyz.m[1][3] = originReal[1];
         controlPointGrid->sto_xyz.m[2][3] = originReal[2];
         controlPointGrid->sto_ijk = nifti_mat44_inverse(controlPointGrid->sto_xyz);
      }
   }
   else
   {
      // The voxel spacing is reduced by two
      for(unsigned int i=0; i<3; ++i)
      {
         controlPointGrid->sto_xyz.m[0][i] /= 2.f;
         controlPointGrid->sto_xyz.m[1][i] /= 2.f;
         if(controlPointGrid->nz>1)
            controlPointGrid->sto_xyz.m[2][i] /= 2.f;
      }
      // The origin is shifted by one node when compared to the previous origin
      float nodeCoord[3]= {1,1,1};
      float newOrigin[3];
      reg_mat44_mul(&controlPointGrid->sto_xyz, nodeCoord, newOrigin);
      controlPointGrid->sto_xyz.m[0][3]=newOrigin[0];
      controlPointGrid->sto_xyz.m[1][3]=newOrigin[1];
      if(controlPointGrid->nz>1)
         controlPointGrid->sto_xyz.m[2][3]=newOrigin[2];
      controlPointGrid->sto_ijk = nifti_mat44_inverse(controlPointGrid->sto_xyz);
   }
#ifndef NDEBUG
   printf("[NiftyReg DEBUG] The control point grid has been refined\n");
#endif
   return;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void reg_defField_compose2D(nifti_image *deformationField,
                            nifti_image *dfToUpdate,
                            int *mask)
{
   size_t DFVoxelNumber=(size_t)deformationField->nx*deformationField->ny;
#ifdef _WIN32
   long i;
   long warVoxelNumber=(size_t)dfToUpdate->nx*dfToUpdate->ny;
#else
   size_t i;
   size_t warVoxelNumber=(size_t)dfToUpdate->nx*dfToUpdate->ny;
#endif
   DTYPE *defPtrX = static_cast<DTYPE *>(deformationField->data);
   DTYPE *defPtrY = &defPtrX[DFVoxelNumber];

   DTYPE *resPtrX = static_cast<DTYPE *>(dfToUpdate->data);
   DTYPE *resPtrY = &resPtrX[warVoxelNumber];

   mat44 *df_real2Voxel=NULL;
   mat44 *df_voxel2Real=NULL;
   if(deformationField->sform_code>0)
   {
      df_real2Voxel=&(dfToUpdate->sto_ijk);
      df_voxel2Real=&(deformationField->sto_xyz);
   }
   else
   {
      df_real2Voxel=&(dfToUpdate->qto_ijk);
      df_voxel2Real=&(deformationField->qto_xyz);
   }

   size_t index;
   int a, b, pre[2];
   DTYPE realDefX, realDefY, voxelX, voxelY;
   DTYPE defX, defY, relX[2], relY[2], basis;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(warVoxelNumber, mask, df_real2Voxel, df_voxel2Real, \
   deformationField, defPtrX, defPtrY, resPtrX, resPtrY) \
   private(i, a, b, index, pre,realDefX, realDefY, voxelX, voxelY, \
   defX, defY, relX, relY, basis)
#endif
   for(i=0; i<warVoxelNumber; ++i)
   {
      if(mask[i]>-1)
      {
         realDefX = resPtrX[i];
         realDefY = resPtrY[i];

         // Conversion from real to voxel in the deformation field
         voxelX = realDefX * df_real2Voxel->m[0][0]
               + realDefY * df_real2Voxel->m[0][1]
               + df_real2Voxel->m[0][3];
         voxelY = realDefX * df_real2Voxel->m[1][0]
               + realDefY * df_real2Voxel->m[1][1]
               + df_real2Voxel->m[1][3];

         // Linear interpolation to compute the new deformation
         pre[0]=(int)reg_floor(voxelX);
         pre[1]=(int)reg_floor(voxelY);
         relX[1]=voxelX-(DTYPE)pre[0];
         relX[0]=1.f-relX[1];
         relY[1]=voxelY-(DTYPE)pre[1];
         relY[0]=1.f-relY[1];
         realDefX=realDefY=0.f;
         for(b=0; b<2; ++b)
         {
            for(a=0; a<2; ++a)
            {
               basis = relX[a] * relY[b];
               if(pre[0]+a>-1 && pre[0]+a<deformationField->nx &&
                     pre[1]+b>-1 && pre[1]+b<deformationField->ny)
               {
                  // Uses the deformation field if voxel is in its space
                  index=(pre[1]+b)*deformationField->nx+pre[0]+a;
                  defX = defPtrX[index];
                  defY = defPtrY[index];
               }
               else
               {
                  // Uses a sliding effect
                  get_SlidedValues<DTYPE>(defX,
                                          defY,
                                          pre[0]+a,
                        pre[1]+b,
                        defPtrX,
                        defPtrY,
                        df_voxel2Real,
                        deformationField->dim,
                        false // not a deformation field
                        );
               }
               realDefX += defX * basis;
               realDefY += defY * basis;
            }
         }
         resPtrX[i]=realDefX;
         resPtrY[i]=realDefY;
      }// mask
   }// loop over every voxel
}
/* *************************************************************** */
template <class DTYPE>
void reg_defField_compose3D(nifti_image *deformationField,
                            nifti_image *dfToUpdate,
                            int *mask)
{
   const int DefFieldDim[3]= {deformationField->nx,deformationField->ny,deformationField->nz};
   const size_t DFVoxelNumber=(size_t)DefFieldDim[0]*DefFieldDim[1]*DefFieldDim[2];
#ifdef _WIN32
   long i;
   long warVoxelNumber=(size_t)dfToUpdate->nx*
         dfToUpdate->ny*dfToUpdate->nz;
#else
   size_t i;
   size_t warVoxelNumber=(size_t)dfToUpdate->nx*
         dfToUpdate->ny*dfToUpdate->nz;
#endif

   DTYPE *defPtrX = static_cast<DTYPE *>(deformationField->data);
   DTYPE *defPtrY = &defPtrX[DFVoxelNumber];
   DTYPE *defPtrZ = &defPtrY[DFVoxelNumber];

   DTYPE *resPtrX = static_cast<DTYPE *>(dfToUpdate->data);
   DTYPE *resPtrY = &resPtrX[warVoxelNumber];
   DTYPE *resPtrZ = &resPtrY[warVoxelNumber];

#ifdef _WIN32
   __declspec(align(16))mat44 df_real2Voxel;
#else
   mat44 df_real2Voxel __attribute__((aligned(16)));
#endif
   mat44 *df_voxel2Real=NULL;
   if(deformationField->sform_code>0)
   {
      df_real2Voxel=deformationField->sto_ijk;
      df_voxel2Real=&deformationField->sto_xyz;
   }
   else
   {
      df_real2Voxel=deformationField->qto_ijk;
      df_voxel2Real=&deformationField->qto_xyz;
   }

   size_t tempIndex, index;
   int a, b, c, currentX, currentY, currentZ, pre[3];
   DTYPE realDef[3], voxel[3], basis, tempBasis;
   DTYPE defX, defY, defZ, relX[2], relY[2], relZ[2];
   bool inY, inZ;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(warVoxelNumber, mask, df_real2Voxel, df_voxel2Real, DefFieldDim, \
   defPtrX, defPtrY, defPtrZ, resPtrX, resPtrY, resPtrZ, deformationField) \
   private(i, a, b, c, currentX, currentY, currentZ, index, tempIndex, pre, \
   realDef, voxel, tempBasis, defX, defY, defZ, relX, relY, relZ, basis, inY, inZ)
#endif
   for(i=0; i<warVoxelNumber; ++i)
   {
      if(mask[i]>-1)
      {
         // Conversion from real to voxel in the deformation field
         realDef[0] = resPtrX[i];
         realDef[1] = resPtrY[i];
         realDef[2] = resPtrZ[i];
         voxel[0] =
               df_real2Voxel.m[0][0] * realDef[0] +
               df_real2Voxel.m[0][1] * realDef[1] +
               df_real2Voxel.m[0][2] * realDef[2] +
               df_real2Voxel.m[0][3] ;
         voxel[1] =
               df_real2Voxel.m[1][0] * realDef[0] +
               df_real2Voxel.m[1][1] * realDef[1] +
               df_real2Voxel.m[1][2] * realDef[2] +
               df_real2Voxel.m[1][3] ;
         voxel[2] =
               df_real2Voxel.m[2][0] * realDef[0] +
               df_real2Voxel.m[2][1] * realDef[1] +
               df_real2Voxel.m[2][2] * realDef[2] +
               df_real2Voxel.m[2][3] ;
         //reg_mat44_mul(df_real2Voxel, realDef, voxel);

         // Linear interpolation to compute the new deformation
         pre[0]=static_cast<int>reg_floor(voxel[0]);
         pre[1]=static_cast<int>reg_floor(voxel[1]);
         pre[2]=static_cast<int>reg_floor(voxel[2]);
         relX[1]=voxel[0]-static_cast<DTYPE>(pre[0]);
         relX[0]=1.-relX[1];
         relY[1]=voxel[1]-static_cast<DTYPE>(pre[1]);
         relY[0]=1.-relY[1];
         relZ[1]=voxel[2]-static_cast<DTYPE>(pre[2]);
         relZ[0]=1.-relZ[1];
         realDef[0]=realDef[1]=realDef[2]=0.;
         for(c=0; c<2; ++c)
         {
            currentZ = pre[2]+c;
            tempIndex=currentZ*DefFieldDim[0]*DefFieldDim[1];
            if(currentZ>-1 && currentZ<DefFieldDim[2]) inZ=true;
            else inZ=false;
            for(b=0; b<2; ++b)
            {
               currentY = pre[1]+b;
               index=tempIndex+currentY*DefFieldDim[0] + pre[0];
               tempBasis= relY[b] * relZ[c];
               if(currentY>-1 && currentY<DefFieldDim[1]) inY=true;
               else inY=false;
               for(a=0; a<2; ++a)
               {
                  currentX = pre[0]+a;
                  if(currentX>-1 && currentX<DefFieldDim[0] && inY && inZ)
                  {
                     // Uses the deformation field if voxel is in its space
                     defX = defPtrX[index];
                     defY = defPtrY[index];
                     defZ = defPtrZ[index];
                  }
                  else
                  {
                     // Uses a sliding effect
                     get_SlidedValues<DTYPE>(defX,
                                             defY,
                                             defZ,
                                             currentX,
                                             currentY,
                                             currentZ,
                                             defPtrX,
                                             defPtrY,
                                             defPtrZ,
                                             df_voxel2Real,
                                             deformationField->dim,
                                             false // not a displacement field
                                             );
                  }
                  ++index;
                  basis = relX[a] * tempBasis;
                  realDef[0] += defX * basis;
                  realDef[1] += defY * basis;
                  realDef[2] += defZ * basis;
               } // a loop
            } // b loop
         } // c loop
         resPtrX[i] = realDef[0];
         resPtrY[i] = realDef[1];
         resPtrZ[i] = realDef[2];
      }// mask
   }// loop over every voxel
}
/* *************************************************************** */
void reg_defField_compose(nifti_image *deformationField,
                          nifti_image *dfToUpdate,
                          int *mask)
{
   if(deformationField->datatype != dfToUpdate->datatype)
   {
      reg_print_fct_error("reg_defField_compose");
      reg_print_msg_error("Both deformation fields are expected to have the same type");
      reg_exit(1);
   }

   bool freeMask=false;
   if(mask==NULL)
   {
      mask=(int *)calloc(dfToUpdate->nx*
                         dfToUpdate->ny*
                         dfToUpdate->nz,
                         sizeof(int));
      freeMask=true;
   }

   if(dfToUpdate->nu==2)
   {
      switch(deformationField->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_defField_compose2D<float>(deformationField,dfToUpdate,mask);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_defField_compose2D<double>(deformationField,dfToUpdate,mask);
         break;
      default:
         reg_print_fct_error("reg_defField_compose");
         reg_print_msg_error("Deformation field pixel type unsupported");
         reg_exit(1);
      }
   }
   else
   {
      switch(deformationField->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_defField_compose3D<float>(deformationField,dfToUpdate,mask);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_defField_compose3D<double>(deformationField,dfToUpdate,mask);
         break;
      default:
         reg_print_fct_error("reg_defField_compose");
         reg_print_msg_error("Deformation field pixel type unsupported");
         reg_exit(1);
      }
   }

   if(freeMask==true) free(mask);
}
/* *************************************************************** */
/* *************************************************************** */
/* Internal data structure to pass user data into optimizer that get passed to cost_function */
struct ddata
{
   nifti_image *deformationField;
   double gx, gy, gz;
   double *arrayy[4];
   double values[4];
};

/* ************************************************************************** */
/* internal routine : deform one point(x, y, x) according to deformationField */
/* returns ERROR when the input point falls outside the deformation field     */
/* ************************************************************************** */

template<class FieldTYPE>
static int inline FastWarp(double x, double y, double z, nifti_image *deformationField, double *px, double *py, double *pz)
{
   double wax, wbx, wcx, wdx, wex, wfx, wgx, whx, wf3x;
   FieldTYPE *wpx;
   double way, wby, wcy, wdy, wey, wfy, wgy, why, wf3y;
   FieldTYPE *wpy;
   double waz, wbz, wcz, wdz, wez, wfz, wgz, whz, wf3z;
   FieldTYPE *wpz;
   int   xw, yw, zw, dxw, dyw, dxyw, dxyzw;
   double wxf, wyf, wzf, wyzf;
   double world[4], position[4];

   FieldTYPE *warpdata = static_cast<FieldTYPE *>(deformationField->data);

   mat44 *deformationFieldIJKMatrix;
   if(deformationField->sform_code>0)
      deformationFieldIJKMatrix=&(deformationField->sto_ijk);
   else deformationFieldIJKMatrix=&(deformationField->qto_ijk);

   dxw = deformationField->nx;
   dyw = deformationField->ny;
   dxyw = dxw * dyw;
   dxyzw = dxw * dyw * deformationField->nz;

   // first guess
   *px = x;
   *py = y;
   *pz = z;

   // detect NAN input
   if (x!=x || y!=y || z!=z) return EXIT_FAILURE;

   // convert x, y,z to indices in deformationField
   world[0] = x;
   world[1] = y;
   world[2] = z;
   world[3] = 1;
   reg_mat44_mul(deformationFieldIJKMatrix, world, position);
   x = position[0];
   y = position[1];
   z = position[2];

   xw = (int)x;        /* get indices into DVF */
   yw = (int)y;
   zw = (int)z;

   // if you block out the next three lines the routine will extrapolate indefinitively
#if 0
   if (x<0 || x>=deformationField->nx-1) return ERROR;
   if (y<0 || y>=deformationField->ny-1) return ERROR;
   if (z<0 || z>=deformationField->nz-1) return ERROR;
#else
   if (xw<0) xw=0;     /* clip */
   if (yw<0) yw=0;
   if (zw<0) zw=0;
   if (xw>deformationField->nx-2) xw = deformationField->nx-2;
   if (yw>deformationField->ny-2) yw = deformationField->ny-2;
   if (zw>deformationField->nz-2) zw = deformationField->nz-2;
#endif

   wxf = x-xw;                  /* fractional coordinates */
   wyf = y-yw;
   wzf = z-zw;

   /* cornerstone for warp coordinates */
   wpx = warpdata + zw*dxyw + yw*dxw + xw;
   wpy = wpx+dxyzw;
   wpz = wpy+dxyzw;

   wf3x = wpx[dxw+1];
   wax  = wpx[0];
   wbx  = wpx[1]      - wax;
   wcx  = wpx[dxw]    - wax;
   wdx  = wpx[dxyw]   - wax;
   wex  = wpx[dxyw + dxw] - wax - wcx - wdx;
   wfx  = wpx[dxyw + 1 ]  - wax - wbx - wdx;
   wgx  = wf3x            - wax - wbx - wcx;
   whx  = wpx[dxyw + dxw + 1] - wf3x - wdx - wex - wfx;

   wf3y = wpy[dxw+1];
   way  = wpy[0];
   wby  = wpy[1]      - way;
   wcy  = wpy[dxw]    - way;
   wdy  = wpy[dxyw]   - way;
   wey  = wpy[dxyw + dxw] - way - wcy - wdy;
   wfy  = wpy[dxyw + 1 ]  - way - wby - wdy;
   wgy  = wf3y            - way - wby - wcy;
   why  = wpy[dxyw + dxw + 1] - wf3y - wdy - wey - wfy;

   wf3z = wpz[dxw+1];
   waz  = wpz[0];
   wbz  = wpz[1]      - waz;
   wcz  = wpz[dxw]    - waz;
   wdz  = wpz[dxyw]   - waz;
   wez  = wpz[dxyw + dxw] - waz - wcz - wdz;
   wfz  = wpz[dxyw + 1 ]  - waz - wbz - wdz;
   wgz  = wf3z            - waz - wbz - wcz;
   whz  = wpz[dxyw + dxw + 1] - wf3z - wdz - wez - wfz;

   wyzf = wyf * wzf;                   /* common term in interpolation     */

   /* trilinear interpolation formulae  */
   *px = wax + wbx*wxf + wcx*wyf + wdx*wzf + wex*wyzf + wfx*wxf*wzf + wgx*wxf*wyf + whx*wxf*wyzf;
   *py = way + wby*wxf + wcy*wyf + wdy*wzf + wey*wyzf + wfy*wxf*wzf + wgy*wxf*wyf + why*wxf*wyzf;
   *pz = waz + wbz*wxf + wcz*wyf + wdz*wzf + wez*wyzf + wfz*wxf*wzf + wgz*wxf*wyf + whz*wxf*wyzf;

   return EXIT_SUCCESS;
}

/* Internal square distance cost function; supports NIFTI_TYPE_FLOAT32 and NIFTI_TYPE_FLOAT64 */
static double cost_function(const double *vector, const void *data)
{
   struct ddata *dat = (struct ddata*) data;
   double x, y, z;
   if (dat->deformationField->datatype == NIFTI_TYPE_FLOAT64)
      FastWarp<double>(vector[0], vector[1], vector[2], dat->deformationField, &x, &y, &z);
   else
      FastWarp<float>(vector[0], vector[1], vector[2], dat->deformationField, &x, &y, &z);

   return (x-dat->gx)*(x-dat->gx) + (y-dat->gy)*(y-dat->gy) + (z-dat->gz)*(z-dat->gz);
}

/* multimin/simplex.c
 *
 * Copyright (C) 2002 Tuomo Keskitalo, Ivo Alxneit
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/*
   - Originally written by Tuomo Keskitalo <tuomo.keskitalo@iki.fi>
   - Corrections to nmsimplex_iterate and other functions
     by Ivo Alxneit <ivo.alxneit@psi.ch>
   - Additional help by Brian Gough <bjg@network-theory.co.uk>

   Modified version by mvh to make it work standalone of GSL
*/

/* The Simplex method of Nelder and Mead,
   also known as the polytope search alogorithm. Ref:
   Nelder, J.A., Mead, R., Computer Journal 7 (1965) pp. 308-313.

   This implementation uses 4 corner points in the simplex for a 3D search.
*/

typedef struct
{
   double x1[12];              /* simplex corner points nsimplex*nvec */
   double y1[4];               /* function value at corner points */
   double ws1[3];              /* workspace 1 for algorithm */
   double ws2[3];              /* workspace 2 for algorithm */
   int    nvec;
   int    nsimplex;
}
nmsimplex_state_t;

typedef double gsl_multimin_function(const double *, const void *);

static double
nmsimplex_move_corner (const double coeff, nmsimplex_state_t *state,
                       size_t corner, double *xc,
                       gsl_multimin_function *f, void *fdata)
{
   /* moves a simplex corner scaled by coeff (negative value represents
    mirroring by the middle point of the "other" corner points)
    and gives new corner in xc and function value at xc as a
    return value
   */

   double *x1 = state->x1;

   size_t i, j;
   double newval, mp;

   for (j = 0; j < (size_t)state->nvec; j++)
   {
      mp = 0.0;
      for (i = 0; i < (size_t)state->nsimplex; i++)
      {
         if (i != corner)
         {
            mp += x1[i*state->nvec + j];
         }
      }
      mp /= (double) (state->nsimplex - 1);
      newval = mp - coeff * (mp - x1[corner*state->nvec + j]);
      xc[j] = newval;
   }

   newval = f(xc, fdata);

   return newval;
}

static void
nmsimplex_contract_by_best (nmsimplex_state_t *state, size_t best,
                            double *xc, gsl_multimin_function *f, void *fdata)
{

   /* Function contracts the simplex in respect to
    best valued corner. That is, all corners besides the
    best corner are moved. */

   /* the xc vector is simply work space here */

   double *x1 = state->x1;
   double *y1 = state->y1;

   size_t i, j;
   double newval;

   for (i = 0; i < (size_t)state->nsimplex; i++)
   {
      if (i != best)
      {
         for (j = 0; j < (size_t)state->nvec; j++)
         {
            newval = 0.5 * (x1[i*state->nvec + j] + x1[best*state->nvec + j]);
            x1[i*state->nvec +  j] = newval;
         }

         /* evaluate function in the new point */

         xc = x1 + i*state->nvec;
         newval = f(xc, fdata);
         y1[i] = newval;
      }
   }
}

static void
nmsimplex_calc_center (const nmsimplex_state_t *state, double *mp)
{
   /* calculates the center of the simplex to mp */

   const double *x1 = state->x1;

   size_t i, j;
   double val;

   for (j = 0; j < (size_t)state->nvec; j++)
   {
      val = 0.0;
      for (i = 0; i < (size_t)state->nsimplex; i++)
      {
         val += x1[i*state->nvec + j];
      }
      val /= state->nsimplex;
      mp[j] = val;
   }
}

static double
nmsimplex_size (nmsimplex_state_t *state)
{
   /* calculates simplex size as average sum of length of vectors
    from simplex center to corner points:

    ( sum ( || y - y_middlepoint || ) ) / n
   */

   double *s = state->ws1;
   double *mp = state->ws2;
   double *x1 = state->x1;

   size_t i, j;

   double t, ss = 0.0;

   /* Calculate middle point */
   nmsimplex_calc_center (state, mp);

   for (i = 0; i < (size_t)state->nsimplex; i++)
   {
      for (j=0; j<(size_t)state->nvec; j++) s[j] = x1[i*state->nvec + j] - mp[j];
      t = 0;
      for (j=0; j<(size_t)state->nvec; j++) t += s[j]*s[j];
      ss += sqrt(t);
   }

   return ss / (double) (state->nsimplex);
}

static void
nmsimplex_set (void *vstate, gsl_multimin_function *f,
               const double *x,
               double *size, const double *step_size, void *fdata)
{
   size_t i, j;
   double val;

   nmsimplex_state_t *state = (nmsimplex_state_t *) vstate;

   double *xtemp = state->ws1;

   /* first point is the original x0 */

   val = f(x, fdata);
   for (j=0; j<(size_t)state->nvec; j++) state->x1[j] = x[j];
   state->y1[0] = val;

   /* following points are initialized to x0 + step_size */

   for (i = 0; i < (size_t)state->nvec; i++)
   {
      for (j=0; j<(size_t)state->nvec; j++) xtemp[j] = x[j];

      val = xtemp[i] + step_size[i];
      xtemp[i] = val;
      val = f(xtemp, fdata);
      for (j=0; j<(size_t)state->nvec; j++)
         state->x1[(i + 1)*state->nvec + j] = xtemp[j];
      state->y1[i + 1] = val;
   }

   /* Initialize simplex size */

   *size = nmsimplex_size (state);
}

static void
nmsimplex_iterate (void *vstate, gsl_multimin_function *f,
                   double *x, double *size, double *fval, void *fdata)
{

   /* Simplex iteration tries to minimize function f value */
   /* Includes corrections from Ivo Alxneit <ivo.alxneit@psi.ch> */

   nmsimplex_state_t *state = (nmsimplex_state_t *) vstate;

   /* xc and xc2 vectors store tried corner point coordinates */

   double *xc = state->ws1;
   double *xc2 = state->ws2;
   double *y1 = state->y1;
   double *x1 = state->x1;

   size_t n = state->nsimplex;
   size_t i, j;
   size_t hi = 0, s_hi = 0, lo = 0;
   double dhi, ds_hi, dlo;
   double val, val2;

   /* get index of highest, second highest and lowest point */

   dhi = ds_hi = dlo = y1[0];

   for (i = 1; i < n; i++)
   {
      val = y1[i];
      if (val < dlo)
      {
         dlo = val;
         lo = i;
      }
      else if (val > dhi)
      {
         ds_hi = dhi;
         s_hi = hi;
         dhi = val;
         hi = i;
      }
      else if (val > ds_hi)
      {
         ds_hi = val;
         s_hi = i;
      }
   }

   /* reflect the highest value */

   val = nmsimplex_move_corner (-1.0, state, hi, xc, f, fdata);

   if (val < y1[lo])
   {

      /* reflected point becomes lowest point, try expansion */

      val2 = nmsimplex_move_corner (-2.0, state, hi, xc2, f, fdata);

      if (val2 < y1[lo])
      {
         for (j=0; j<(size_t)state->nvec; j++) x1[hi*state->nvec+j] = xc2[j];
         y1[hi] = val2;
      }
      else
      {
         for (j=0; j<(size_t)state->nvec; j++) x1[hi*state->nvec+j] = xc[j];
         y1[hi] = val;
      }
   }

   /* reflection does not improve things enough */

   else if (val > y1[s_hi])
   {
      if (val <= y1[hi])
      {

         /* if trial point is better than highest point, replace
          highest point */

         for (j=0; j<(size_t)state->nvec; j++) x1[hi*state->nvec+j] = xc[j];
         y1[hi] = val;
      }

      /* try one dimensional contraction */

      val2 = nmsimplex_move_corner (0.5, state, hi, xc2, f, fdata);

      if (val2 <= y1[hi])
      {
         for (j=0; j<(size_t)state->nvec; j++) x1[hi*state->nvec+j] = xc2[j];
         y1[hi] = val2;
      }

      else
      {
         /* contract the whole simplex in respect to the best point */
         nmsimplex_contract_by_best (state, lo, xc, f, fdata);
      }
   }
   else
   {

      /* trial point is better than second highest point.
       Replace highest point by it */

      for (j=0; j<(size_t)state->nvec; j++) x1[hi*state->nvec+j] = xc[j];
      y1[hi] = val;
   }

   /* return lowest point of simplex as x */

   lo=0;
   val=y1[0];
   for (j=1; j<(size_t)state->nsimplex; j++) if (y1[j]<val) lo=j, val=y1[j];
   for (j=0; j<(size_t)state->nvec; j++) x[j] = x1[lo*state->nvec+j];
   *fval = y1[lo];


   /* Update simplex size */

   *size = nmsimplex_size (state);
}

/* Internal wrapper for nmsimplex_iterate */
static void optimize(gsl_multimin_function *f, double *start, void *data, double tol)
{
   nmsimplex_state_t t;
   double fval[4];
   double offset[3] = {10, 10, 10};
   double size;
   int n=0;
   t.nvec = 3;
   t.nsimplex = 4;
   nmsimplex_set (&t, f, start, &size, offset, data);
   while (size>tol && n<300)
   {
      nmsimplex_iterate (&t, f, start, &size, fval, data);
      n++;
   }
   nmsimplex_calc_center (&t, start);
}
/* *************************************************************** */
template <class DTYPE>
void reg_defFieldInvert3D(nifti_image *inputDeformationField,
                          nifti_image *outputDeformationField,
                          float tolerance)
{
   int outputVoxelNumber = outputDeformationField->nx *
         outputDeformationField->ny *
         outputDeformationField->nz;

   mat44 *OutXYZMatrix;
   if(outputDeformationField->sform_code>0)
      OutXYZMatrix=&(outputDeformationField->sto_xyz);
   else OutXYZMatrix=&(outputDeformationField->qto_xyz);

   // added:
   mat44 *InXYZMatrix;
   if(inputDeformationField->sform_code>0)
      InXYZMatrix=&(inputDeformationField->sto_xyz);
   else InXYZMatrix=&(inputDeformationField->qto_xyz);
   float center[4], center2[4];
   double centerout[4], delta[4];
   center[0] = inputDeformationField->nx / 2;
   center[1] = inputDeformationField->ny / 2;
   center[2] = inputDeformationField->nz / 2;
   center[3] = 1;
   reg_mat44_mul(InXYZMatrix, center, center2);
   FastWarp<float>(center2[0], center2[1], center2[2], inputDeformationField, &centerout[0], &centerout[1], &centerout[2]);
   delta[0] = center2[0]-centerout[0];
   delta[1] = center2[1]-centerout[1];
   delta[2] = center2[2]-centerout[2];
   // end added


   int i,x,y,z;
   double position[4], pars[4], arrayy[4][3];
   struct ddata dat;
   DTYPE *outData;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(outputDeformationField,tolerance,outputVoxelNumber, \
   inputDeformationField, OutXYZMatrix, delta) \
   private(i,x,y,z,dat,outData,position,pars,arrayy)
#endif
   for (z=0; z<outputDeformationField->nz; ++z)
   {
      dat.deformationField = inputDeformationField;
      for(i=0; i<4; ++i)              /* set up 2D array pointers */
         dat.arrayy[i]= arrayy[i];

      outData = (DTYPE *)(outputDeformationField->data) +
            outputDeformationField->nx * outputDeformationField->ny * z;

      for(y=0; y<outputDeformationField->ny; ++y)
      {
         for(x=0; x<outputDeformationField->nx; ++x)
         {

            // convert x, y,z to world coordinates
            position[0] = x;
            position[1] = y;
            position[2] = z;
            position[3] = 1;
            reg_mat44_mul(OutXYZMatrix, position, pars);
            dat.gx = pars[0];
            dat.gy = pars[1];
            dat.gz = pars[2];

            // added
            pars[0] += delta[0];
            pars[1] += delta[1];
            pars[2] += delta[2];
            // end added

            optimize(cost_function, pars, (void *)&dat, tolerance);
            // output = (warp-1)(input);

            outData[0]        = pars[0];
            outData[outputVoxelNumber]   = pars[1];
            outData[outputVoxelNumber*2] = pars[2];
            ++outData;
         }
      }
   }
}
/* *************************************************************** */
void reg_defFieldInvert(nifti_image *inputDeformationField,
                        nifti_image *outputDeformationField,
                        float tolerance)
{
   // Check the input image data types
   if(inputDeformationField->datatype!=outputDeformationField->datatype)
   {
      reg_print_fct_error("reg_defFieldInvert");
      reg_print_msg_error("Both deformation fields are expected to have the same data type");
      reg_exit(1);
   }

   if(inputDeformationField->nu!=3)
   {
      reg_print_fct_error("reg_defFieldInvert");
      reg_print_msg_error("The function has only been implemented for 3D deformation field yet");
      reg_exit(1);
   }

   switch(inputDeformationField->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      reg_defFieldInvert3D<float>
            (inputDeformationField,outputDeformationField,tolerance);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_defFieldInvert3D<double>
            (inputDeformationField,outputDeformationField,tolerance);
   default:
      reg_print_fct_error("reg_defFieldInvert");
      reg_print_msg_error("Deformation field pixel type unsupported");
      reg_exit(1);
   }
}
/* *************************************************************** */
/* *************************************************************** */
template<class DTYPE>
void reg_spline_cppComposition_2D(nifti_image *grid1,
                                  nifti_image *grid2,
                                  bool displacement1,
                                  bool displacement2,
                                  bool bspline)
{
   // REMINDER Grid2(x)=Grid1(Grid2(x))

#if _USE_SSE
   union
   {
      __m128 m;
      float f[4];
   } val;
#endif // _USE_SSE

   DTYPE *outCPPPtrX = static_cast<DTYPE *>(grid2->data);
   DTYPE *outCPPPtrY = &outCPPPtrX[grid2->nx*grid2->ny];

   DTYPE *controlPointPtrX = static_cast<DTYPE *>(grid1->data);
   DTYPE *controlPointPtrY = &controlPointPtrX[grid1->nx*grid1->ny];

   DTYPE basis;

#ifdef _WIN32
   __declspec(align(16)) DTYPE xBasis[4];
   __declspec(align(16)) DTYPE yBasis[4];
#if _USE_SSE
   __declspec(align(16)) DTYPE xyBasis[16];
#endif  //_USE_SSE

   __declspec(align(16)) DTYPE xControlPointCoordinates[16];
   __declspec(align(16)) DTYPE yControlPointCoordinates[16];
#else // _WIN32
   DTYPE xBasis[4] __attribute__((aligned(16)));
   DTYPE yBasis[4] __attribute__((aligned(16)));
#if _USE_SSE
   DTYPE xyBasis[16] __attribute__((aligned(16)));
#endif  //_USE_SSE

   DTYPE xControlPointCoordinates[16] __attribute__((aligned(16)));
   DTYPE yControlPointCoordinates[16] __attribute__((aligned(16)));
#endif // _WIN32

   size_t coord;

   // read the xyz/ijk sform or qform, as appropriate
   mat44 *matrix_real_to_voxel1=NULL;
   mat44 *matrix_voxel_to_real2=NULL;
   if(grid1->sform_code>0)
      matrix_real_to_voxel1=&(grid1->sto_ijk);
   else matrix_real_to_voxel1=&(grid1->qto_ijk);
   if(grid2->sform_code>0)
      matrix_voxel_to_real2=&(grid2->sto_xyz);
   else matrix_voxel_to_real2=&(grid2->qto_xyz);

   for(int y=0; y<grid2->ny; y++)
   {
      for(int x=0; x<grid2->nx; x++)
      {

         // Get the control point actual position
         DTYPE xReal = *outCPPPtrX;
         DTYPE yReal = *outCPPPtrY;
         DTYPE initialX=xReal;
         DTYPE initialY=yReal;
         if(displacement2)
         {
            xReal +=
                  matrix_voxel_to_real2->m[0][0]*x
                  + matrix_voxel_to_real2->m[0][1]*y
                  + matrix_voxel_to_real2->m[0][3];
            yReal +=
                  matrix_voxel_to_real2->m[1][0]*x
                  + matrix_voxel_to_real2->m[1][1]*y
                  + matrix_voxel_to_real2->m[1][3];
         }

         // Get the voxel based control point position in grid1
         DTYPE xVoxel = matrix_real_to_voxel1->m[0][0]*xReal
               + matrix_real_to_voxel1->m[0][1]*yReal
               + matrix_real_to_voxel1->m[0][3];
         DTYPE yVoxel = matrix_real_to_voxel1->m[1][0]*xReal
               + matrix_real_to_voxel1->m[1][1]*yReal
               + matrix_real_to_voxel1->m[1][3];

         // The spline coefficients are computed
         int xPre=(int)(reg_floor(xVoxel));
         basis=(DTYPE)xVoxel-(DTYPE)xPre;
         xPre--;
         if(basis<0.0) basis=0.0; //rounding error
         if(bspline) get_BSplineBasisValues<DTYPE>(basis, xBasis);
         else Get_SplineBasisValues<DTYPE>(basis, xBasis);

         int yPre=(int)(reg_floor(yVoxel));
         basis=(DTYPE)yVoxel-(DTYPE)yPre;
         yPre--;
         if(basis<0.0) basis=0.0; //rounding error
         if(bspline) get_BSplineBasisValues<DTYPE>(basis, yBasis);
         else Get_SplineBasisValues<DTYPE>(basis, yBasis);

         // The control points are stored
         get_GridValues<DTYPE>(xPre,
                               yPre,
                               grid1,
                               controlPointPtrX,
                               controlPointPtrY,
                               xControlPointCoordinates,
                               yControlPointCoordinates,
                               false, // no approximation
                               displacement1 // displacement field?
                               );
         xReal=0.0;
         yReal=0.0;
#if _USE_SSE
         coord=0;
         for(unsigned int b=0; b<4; b++)
         {
            for(unsigned int a=0; a<4; a++)
            {
               xyBasis[coord++] = xBasis[a] * yBasis[b];
            }
         }

         __m128 tempX =  _mm_set_ps1(0.0);
         __m128 tempY =  _mm_set_ps1(0.0);
         __m128 *ptrX = (__m128 *) &xControlPointCoordinates[0];
         __m128 *ptrY = (__m128 *) &yControlPointCoordinates[0];
         __m128 *ptrBasis   = (__m128 *) &xyBasis[0];
         //addition and multiplication of the 16 basis value and CP position for each axis
         for(unsigned int a=0; a<4; a++)
         {
            tempX = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrX), tempX );
            tempY = _mm_add_ps(_mm_mul_ps(*ptrBasis, *ptrY), tempY );
            ptrBasis++;
            ptrX++;
            ptrY++;
         }
         //the values stored in SSE variables are transfered to normal float
         val.m = tempX;
         xReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
         val.m = tempY;
         yReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
         coord=0;
         for(unsigned int b=0; b<4; b++)
         {
            for(unsigned int a=0; a<4; a++)
            {
               DTYPE tempValue = xBasis[a] * yBasis[b];
               xReal += xControlPointCoordinates[coord] * tempValue;
               yReal += yControlPointCoordinates[coord] * tempValue;
               coord++;
            }
         }
#endif
         if(displacement1)
         {
            xReal += initialX;
            yReal += initialY;
         }
         *outCPPPtrX++ = xReal;
         *outCPPPtrY++ = yReal;
      }
   }
   return;
}
/* *************************************************************** */
template<class DTYPE>
void reg_spline_cppComposition_3D(nifti_image *grid1,
                                  nifti_image *grid2,
                                  bool displacement1,
                                  bool displacement2,
                                  bool bspline)
{
   // REMINDER Grid2(x)=Grid1(Grid2(x))
#if _USE_SSE
   union
   {
      __m128 m;
      float f[4];
   } val;
   __m128 _xBasis_sse;
   __m128 tempX;
   __m128 tempY;
   __m128 tempZ;
   __m128 *ptrX;
   __m128 *ptrY;
   __m128 *ptrZ;
   __m128 _yBasis_sse;
   __m128 _zBasis_sse;
   __m128 _temp_basis;
   __m128 _basis;
#else
   int a, b, c;
   size_t coord;
   DTYPE tempValue;
#endif

   DTYPE *outCPPPtrX = static_cast<DTYPE *>(grid2->data);
   DTYPE *outCPPPtrY = &outCPPPtrX[grid2->nx*grid2->ny*grid2->nz];
   DTYPE *outCPPPtrZ = &outCPPPtrY[grid2->nx*grid2->ny*grid2->nz];

   DTYPE *controlPointPtrX = static_cast<DTYPE *>(grid1->data);
   DTYPE *controlPointPtrY = &controlPointPtrX[grid1->nx*grid1->ny*grid1->nz];
   DTYPE *controlPointPtrZ = &controlPointPtrY[grid1->nx*grid1->ny*grid1->nz];

   DTYPE basis;

#ifdef _WIN32
   __declspec(align(16)) DTYPE xBasis[4];
   __declspec(align(16)) DTYPE yBasis[4];
   __declspec(align(16)) DTYPE zBasis[4];
   __declspec(align(16)) DTYPE xControlPointCoordinates[64];
   __declspec(align(16)) DTYPE yControlPointCoordinates[64];
   __declspec(align(16)) DTYPE zControlPointCoordinates[64];
#else
   DTYPE xBasis[4] __attribute__((aligned(16)));
   DTYPE yBasis[4] __attribute__((aligned(16)));
   DTYPE zBasis[4] __attribute__((aligned(16)));
   DTYPE xControlPointCoordinates[64] __attribute__((aligned(16)));
   DTYPE yControlPointCoordinates[64] __attribute__((aligned(16)));
   DTYPE zControlPointCoordinates[64] __attribute__((aligned(16)));
#endif

   int xPre, xPreOld, yPre, yPreOld, zPre, zPreOld;
   int x, y, z;
   size_t index;
   DTYPE xReal, yReal, zReal, initialPositionX, initialPositionY, initialPositionZ;
   DTYPE xVoxel, yVoxel, zVoxel;

   // read the xyz/ijk sform or qform, as appropriate
   mat44 *matrix_real_to_voxel1=NULL;
   mat44 *matrix_voxel_to_real2=NULL;
   if(grid1->sform_code>0)
      matrix_real_to_voxel1=&(grid1->sto_ijk);
   else matrix_real_to_voxel1=&(grid1->qto_ijk);
   if(grid2->sform_code>0)
      matrix_voxel_to_real2=&(grid2->sto_xyz);
   else matrix_voxel_to_real2=&(grid2->qto_xyz);

#if defined (_OPENMP)
#ifdef _USE_SSE
#pragma omp parallel for default(none) \
   shared(grid1, grid2, displacement1, displacement2, matrix_voxel_to_real2, matrix_real_to_voxel1, \
   outCPPPtrX, outCPPPtrY, outCPPPtrZ, controlPointPtrX, controlPointPtrY, controlPointPtrZ, bspline) \
   private(xPre, xPreOld, yPre, yPreOld, zPre, zPreOld, val, index, \
   x, y, z, xVoxel, yVoxel, zVoxel, basis, xBasis, yBasis, zBasis, \
   xReal, yReal, zReal, initialPositionX, initialPositionY, initialPositionZ, \
   _xBasis_sse, tempX, tempY, tempZ, ptrX, ptrY, ptrZ, _yBasis_sse, _zBasis_sse, _temp_basis, _basis, \
   xControlPointCoordinates, yControlPointCoordinates, zControlPointCoordinates)
#else
#pragma omp parallel for default(none) \
   shared(grid1, grid2, displacement1, displacement2, matrix_voxel_to_real2, matrix_real_to_voxel1, \
   outCPPPtrX, outCPPPtrY, outCPPPtrZ, controlPointPtrX, controlPointPtrY, controlPointPtrZ, bspline) \
   private(xPre, xPreOld, yPre, yPreOld, zPre, zPreOld, index, \
   x, y, z, xVoxel, yVoxel, zVoxel, a, b, c, coord, basis, tempValue, xBasis, yBasis, zBasis, \
   xReal, yReal, zReal, initialPositionX, initialPositionY, initialPositionZ, \
   xControlPointCoordinates, yControlPointCoordinates, zControlPointCoordinates)
#endif
#endif
   for(z=0; z<grid2->nz; z++)
   {
      xPreOld=99999;
      yPreOld=99999;
      zPreOld=99999;
      index=z*grid2->nx*grid2->ny;
      for(y=0; y<grid2->ny; y++)
      {
         for(x=0; x<grid2->nx; x++)
         {
            // Get the control point actual position
            xReal = outCPPPtrX[index];
            yReal = outCPPPtrY[index];
            zReal = outCPPPtrZ[index];
            initialPositionX=0;
            initialPositionY=0;
            initialPositionZ=0;
            if(displacement2)
            {
               xReal += initialPositionX =
                     matrix_voxel_to_real2->m[0][0]*x
                     + matrix_voxel_to_real2->m[0][1]*y
                     + matrix_voxel_to_real2->m[0][2]*z
                     + matrix_voxel_to_real2->m[0][3];
               yReal += initialPositionY =
                     matrix_voxel_to_real2->m[1][0]*x
                     + matrix_voxel_to_real2->m[1][1]*y
                     + matrix_voxel_to_real2->m[1][2]*z
                     + matrix_voxel_to_real2->m[1][3];
               zReal += initialPositionZ =
                     matrix_voxel_to_real2->m[2][0]*x
                     + matrix_voxel_to_real2->m[2][1]*y
                     + matrix_voxel_to_real2->m[2][2]*z
                     + matrix_voxel_to_real2->m[2][3];
            }

            // Get the voxel based control point position in grid1
            xVoxel =
                  matrix_real_to_voxel1->m[0][0]*xReal
                  + matrix_real_to_voxel1->m[0][1]*yReal
                  + matrix_real_to_voxel1->m[0][2]*zReal
                  + matrix_real_to_voxel1->m[0][3];
            yVoxel =
                  matrix_real_to_voxel1->m[1][0]*xReal
                  + matrix_real_to_voxel1->m[1][1]*yReal
                  + matrix_real_to_voxel1->m[1][2]*zReal
                  + matrix_real_to_voxel1->m[1][3];
            zVoxel =
                  matrix_real_to_voxel1->m[2][0]*xReal
                  + matrix_real_to_voxel1->m[2][1]*yReal
                  + matrix_real_to_voxel1->m[2][2]*zReal
                  + matrix_real_to_voxel1->m[2][3];

            // The spline coefficients are computed
            xPre=(int)(reg_floor(xVoxel));
            basis=(DTYPE)xVoxel-(DTYPE)xPre;
            if(basis<0.0) basis=0.0; //rounding error
            if(bspline) get_BSplineBasisValues<DTYPE>(basis, xBasis);
            else Get_SplineBasisValues<DTYPE>(basis, xBasis);

            yPre=(int)(reg_floor(yVoxel));
            basis=(DTYPE)yVoxel-(DTYPE)yPre;
            if(basis<0.0) basis=0.0; //rounding error
            if(bspline) get_BSplineBasisValues<DTYPE>(basis, yBasis);
            else Get_SplineBasisValues<DTYPE>(basis, yBasis);

            zPre=(int)(reg_floor(zVoxel));
            basis=(DTYPE)zVoxel-(DTYPE)zPre;
            if(basis<0.0) basis=0.0; //rounding error
            if(bspline) get_BSplineBasisValues<DTYPE>(basis, zBasis);
            else Get_SplineBasisValues<DTYPE>(basis, zBasis);

            --xPre;
            --yPre;
            --zPre;

            // The control points are stored
            if(xPre!=xPreOld || yPre!=yPreOld || zPre!=zPreOld)
            {
               get_GridValues(xPre,
                              yPre,
                              zPre,
                              grid1,
                              controlPointPtrX,
                              controlPointPtrY,
                              controlPointPtrZ,
                              xControlPointCoordinates,
                              yControlPointCoordinates,
                              zControlPointCoordinates,
                              false, // no approximation
                              displacement1 // a displacement field?
                              );
               xPreOld=xPre;
               yPreOld=yPre;
               zPreOld=zPre;
            }
            xReal=0.0;
            yReal=0.0;
            zReal=0.0;
#if _USE_SSE
            val.f[0] = xBasis[0];
            val.f[1] = xBasis[1];
            val.f[2] = xBasis[2];
            val.f[3] = xBasis[3];
            _xBasis_sse = val.m;

            tempX =  _mm_set_ps1(0.0);
            tempY =  _mm_set_ps1(0.0);
            tempZ =  _mm_set_ps1(0.0);
            ptrX = (__m128 *) &xControlPointCoordinates[0];
            ptrY = (__m128 *) &yControlPointCoordinates[0];
            ptrZ = (__m128 *) &zControlPointCoordinates[0];

            for(unsigned int c=0; c<4; c++)
            {
               for(unsigned int b=0; b<4; b++)
               {
                  _yBasis_sse  = _mm_set_ps1(yBasis[b]);
                  _zBasis_sse  = _mm_set_ps1(zBasis[c]);
                  _temp_basis   = _mm_mul_ps(_yBasis_sse, _zBasis_sse);
                  _basis       = _mm_mul_ps(_temp_basis, _xBasis_sse);
                  tempX = _mm_add_ps(_mm_mul_ps(_basis, *ptrX), tempX );
                  tempY = _mm_add_ps(_mm_mul_ps(_basis, *ptrY), tempY );
                  tempZ = _mm_add_ps(_mm_mul_ps(_basis, *ptrZ), tempZ );
                  ptrX++;
                  ptrY++;
                  ptrZ++;
               }
            }
            //the values stored in SSE variables are transfered to normal float
            val.m = tempX;
            xReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
            val.m = tempY;
            yReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
            val.m = tempZ;
            zReal = val.f[0]+val.f[1]+val.f[2]+val.f[3];
#else
            coord=0;
            for(c=0; c<4; c++)
            {
               for(b=0; b<4; b++)
               {
                  for(a=0; a<4; a++)
                  {
                     tempValue = xBasis[a] * yBasis[b] * zBasis[c];
                     xReal += xControlPointCoordinates[coord] * tempValue;
                     yReal += yControlPointCoordinates[coord] * tempValue;
                     zReal += zControlPointCoordinates[coord] * tempValue;
                     coord++;
                  }
               }
            }
#endif
            if(displacement2)
            {
               xReal -= initialPositionX;
               yReal -= initialPositionY;
               zReal -= initialPositionZ;
            }
            outCPPPtrX[index] = xReal;
            outCPPPtrY[index] = yReal;
            outCPPPtrZ[index] = zReal;
            index++;
         }
      }
   }
   return;
}
/* *************************************************************** */
int reg_spline_cppComposition(nifti_image *grid1,
                              nifti_image *grid2,
                              bool displacement1,
                              bool displacement2,
                              bool bspline)
{
   // REMINDER Grid2(x)=Grid1(Grid2(x))

   if(grid1->datatype != grid2->datatype)
   {
      fprintf(stderr,"[NiftyReg ERROR] reg_spline_cppComposition\n");
      fprintf(stderr,"[NiftyReg ERROR] Both input images do not have the same type\n");
      reg_exit(1);
   }

#if _USE_SSE
   if(grid1->datatype != NIFTI_TYPE_FLOAT32)
   {
      fprintf(stderr,"[NiftyReg ERROR] SSE computation has only been implemented for single precision.\n");
      fprintf(stderr,"[NiftyReg ERROR] The deformation field is not computed\n");
      reg_exit(1);
   }
#endif

   if(grid1->nz>1)
   {
      switch(grid1->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_spline_cppComposition_3D<float>
               (grid1, grid2, displacement1, displacement2, bspline);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_spline_cppComposition_3D<double>
               (grid1, grid2, displacement1, displacement2, bspline);
         break;
      default:
         fprintf(stderr,"[NiftyReg ERROR] reg_spline_cppComposition 3D\n");
         fprintf(stderr,"[NiftyReg ERROR] Only implemented for single or double floating images\n");
         return 1;
      }
   }
   else
   {
      switch(grid1->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_spline_cppComposition_2D<float>
               (grid1, grid2, displacement1, displacement2, bspline);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_spline_cppComposition_2D<double>
               (grid1, grid2, displacement1, displacement2, bspline);
         break;
      default:
         fprintf(stderr,"[NiftyReg ERROR] reg_spline_cppComposition 2D\n");
         fprintf(stderr,"[NiftyReg ERROR] Only implemented for single or double precision images\n");
         return 1;
      }
   }
   return 0;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_spline_getFlowFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                             nifti_image *flowField)
{
   // Check first if the velocity field is actually a velocity field
   if(velocityFieldGrid->intent_p1 != SPLINE_VEL_GRID)
   {
      fprintf(stderr, "[NiftyReg ERROR] reg_spline_getFlowFieldFromVelocityGrid`n");
      fprintf(stderr, "[NiftyReg ERROR] The provide grid is not a velocity field\n");
      reg_exit(1);
   }

   // Initialise the flow field with an identity transformation
   reg_tools_multiplyValueToImage(flowField, flowField, 0.f);
   flowField->intent_p1=DISP_VEL_FIELD;
   reg_getDeformationFromDisplacement(flowField);

   // fake the number of extension here to avoid the second half of the affine
   int oldNumExt = velocityFieldGrid->num_ext;
   if(oldNumExt>1)
      velocityFieldGrid->num_ext=1;


   // Copy over the number of required squaring steps
   flowField->intent_p2=velocityFieldGrid->intent_p2;
   // The initial flow field is generated using cubic B-Spline interpolation/approximation
   reg_spline_getDeformationField(velocityFieldGrid,
                                  flowField,
                                  NULL, // mask
                                  true, //composition
                                  true // bspline
                                  );

   velocityFieldGrid->num_ext=oldNumExt;
}
/* *************************************************************** */
void reg_defField_getDeformationFieldFromFlowField(nifti_image *flowFieldImage,
                                                   nifti_image *deformationFieldImage,
                                                   bool updateStepNumber)
{
   // Check first if the velocity field is actually a velocity field
   if(flowFieldImage->intent_p1 != DEF_VEL_FIELD)
   {
      reg_print_fct_error("reg_defField_getDeformationFieldFromFlowField");
      reg_print_msg_error("The provide field is not a velocity field");
      reg_exit(1);
   }

   // Remove the affine component from the flow field
   nifti_image *affineOnly=NULL;
   if(flowFieldImage->num_ext>0)
   {
      if(flowFieldImage->ext_list[0].edata!=NULL)
      {
         // Create a field that contains the affine component only
         affineOnly = nifti_copy_nim_info(deformationFieldImage);
         affineOnly->data = (void *)calloc(affineOnly->nvox,affineOnly->nbyper);
         reg_affine_getDeformationField(reinterpret_cast<mat44 *>(flowFieldImage->ext_list[0].edata),
               affineOnly,
               false);
         reg_tools_substractImageToImage(flowFieldImage,affineOnly,flowFieldImage);
      }
   }
   else reg_getDisplacementFromDeformation(flowFieldImage);

   // Compute the number of scaling value to ensure unfolded transformation
   int squaringNumber = 1;
   if(updateStepNumber || flowFieldImage->intent_p2==0)
   {
      // Check the largest value
      float extrema = fabsf(reg_tools_getMinValue(flowFieldImage));
      float temp = reg_tools_getMaxValue(flowFieldImage);
      extrema=extrema>temp?extrema:temp;
      // Check the values for scaling purpose
      float maxLength;
      if(deformationFieldImage->nz>1)
         // 0.2888675 = sqrt(0.5^2/3)
         maxLength=0.28;
      // 0.3535533 = sqrt(0.5^2/2)
      else maxLength=0.35;
      while(true)
      {
         if( (extrema/pow(2.0f,squaringNumber)) >= maxLength)
            squaringNumber++;
         else break;
      }
      // The minimal number of step is set to 6 by default
      squaringNumber=squaringNumber<6?6:squaringNumber;
      // Set the number of squaring step in the flow field
      if(fabs(flowFieldImage->intent_p2)!=squaringNumber)
      {
         printf("[NiftyReg] Changing from %i to %i squaring step (equivalent to scaling down by %i)\n",
                static_cast<int>(reg_round(fabs(flowFieldImage->intent_p2))),
                abs(squaringNumber),
                (int)pow(2.0f,squaringNumber));
      }
      // Update the number of squaring step required
      if(flowFieldImage->intent_p2>=0)
         flowFieldImage->intent_p2 = squaringNumber;
      else flowFieldImage->intent_p2 = -squaringNumber;
   }
   else squaringNumber=static_cast<int>(fabsf(flowFieldImage->intent_p2));

   // The displacement field is scaled
   float scalingValue = pow(2.0f,fabs(squaringNumber));
   if(flowFieldImage->intent_p2<0)
      // backward deformation field is scaled down
      reg_tools_divideValueToImage(flowFieldImage,
                                   flowFieldImage,
                                   -scalingValue); // (/-scalingValue)
   else
      // forward deformation field is scaled down
      reg_tools_divideValueToImage(flowFieldImage,
                                   flowFieldImage,
                                   scalingValue); // (/scalingValue)

   // Conversion from displacement to deformation
   reg_getDeformationFromDisplacement(flowFieldImage);

   // The computed scaled deformation field is copied over
   memcpy(deformationFieldImage->data, flowFieldImage->data,
          deformationFieldImage->nvox*deformationFieldImage->nbyper);

   // The deformation field is squared
   for(unsigned short i=0; i<squaringNumber; ++i)
   {
      // The deformation field is applied to itself
      reg_defField_compose(deformationFieldImage,
                           flowFieldImage,
                           NULL);
      // The computed scaled deformation field is copied over
      memcpy(deformationFieldImage->data, flowFieldImage->data,
             deformationFieldImage->nvox*deformationFieldImage->nbyper);
#ifndef NDEBUG
      printf("[NiftyReg DEBUG] Squaring (composition) step %u/%u\n", i+1, squaringNumber);
#endif
   }
   // The affine conponent of the transformation is restored
   if(affineOnly!=NULL)
   {
      reg_getDisplacementFromDeformation(deformationFieldImage);
      reg_tools_addImageToImage(deformationFieldImage,affineOnly,deformationFieldImage);
      nifti_image_free(affineOnly);
      affineOnly=NULL;
   }
   deformationFieldImage->intent_p1=DEF_FIELD;
   deformationFieldImage->intent_p2=0;
   // If required an affine component is composed
   if(flowFieldImage->num_ext>1)
   {
      reg_affine_getDeformationField(reinterpret_cast<mat44 *>(flowFieldImage->ext_list[1].edata),
            deformationFieldImage,
            true);
   }
}
/* *************************************************************** */
void reg_spline_getDefFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                            nifti_image *deformationFieldImage,
                                            bool updateStepNumber)
{
   // Check if the velocity field is actually a velocity field
   if(velocityFieldGrid->intent_p1 == SPLINE_GRID)
   {
      // Use the spline approximation to generate the deformation field
      reg_spline_getDeformationField(velocityFieldGrid,
                                     deformationFieldImage,
                                     NULL,
                                     false, // composition
                                     true // bspline
                                     );
   }
   else if(velocityFieldGrid->intent_p1 == SPLINE_VEL_GRID)
   {
      // Create an image to store the flow field
      nifti_image *flowField = nifti_copy_nim_info(deformationFieldImage);
      flowField->data = (void *)calloc(flowField->nvox,flowField->nbyper);
      flowField->intent_code=NIFTI_INTENT_VECTOR;
      memset(flowField->intent_name, 0, 16);
      strcpy(flowField->intent_name,"NREG_TRANS");
      flowField->intent_p1=DEF_VEL_FIELD;
      flowField->intent_p2=velocityFieldGrid->intent_p2;
      if(velocityFieldGrid->num_ext>0 && flowField->ext_list==NULL)
         nifti_copy_extensions(flowField, velocityFieldGrid);

      // Generate the velocity field
      reg_spline_getFlowFieldFromVelocityGrid(velocityFieldGrid,
                                              flowField);
      // Exponentiate the flow field
      reg_defField_getDeformationFieldFromFlowField(flowField,
                                                    deformationFieldImage,
                                                    updateStepNumber);
      // Update the number of step required. No action otherwise
      velocityFieldGrid->intent_p2=flowField->intent_p2;
      // Clear the allocated flow field
      nifti_image_free(flowField);
   }
   else
   {
      reg_print_fct_error("reg_spline_getDeformationFieldFromVelocityGrid");
      reg_print_msg_error("The provided input image is not a spline parametrised transformation");
      reg_exit(1);
   }
   return;
}
/* *************************************************************** */
/* *************************************************************** */
void reg_spline_getIntermediateDefFieldFromVelGrid(nifti_image *velocityFieldGrid,
                                                   nifti_image **deformationFieldImage)
{
   reg_exit(1);// Needs to be updated
   // Check first if the velocity field is actually a velocity field
   if( velocityFieldGrid->intent_p1!=SPLINE_VEL_GRID)
   {
      reg_print_fct_error("reg_spline_getIntermediateDefFieldFromVelGrid");
      reg_print_msg_error("The provided grid is not a velocity field");
      reg_exit(1);
   }
   // Set the initial deformation field to an identity transformation
   memset(deformationFieldImage[0]->data,0,
         deformationFieldImage[0]->nvox* deformationFieldImage[0]->nbyper); // (*0)
   reg_getDeformationFromDisplacement(deformationFieldImage[0]);
   // The initial deformation is generated using cubic B-Spline parametrisation
   reg_spline_getDeformationField(velocityFieldGrid,
                                  deformationFieldImage[0],
         NULL, // mask
         true, //composition
         true // bspline
         );

   // The deformation field is converted from deformation field to displacement field
   reg_getDisplacementFromDeformation(deformationFieldImage[0]);

   // The deformation field is scaled
   float scalingValue = pow(2.0f,fabs(velocityFieldGrid->intent_p2));
   if(velocityFieldGrid->intent_p1<0)
      // backward deformation field is scaled down
      reg_tools_divideValueToImage(deformationFieldImage[0],
            deformationFieldImage[0],
            -scalingValue);
   else
      // forward deformation field is scaled down
      reg_tools_divideValueToImage(deformationFieldImage[0],
            deformationFieldImage[0],
            scalingValue);

   // The displacement field is converted back into a deformation field
   reg_getDeformationFromDisplacement(deformationFieldImage[0]);

   // The deformation field is squared
   unsigned int squaringNumber = (unsigned int)fabs(velocityFieldGrid->intent_p2);
   for(unsigned int i=0; i<squaringNumber; ++i)
   {
      // The computed scaled deformation field is copied over
      memcpy(deformationFieldImage[i+1]->data, deformationFieldImage[i]->data,
            deformationFieldImage[i]->nvox*deformationFieldImage[i]->nbyper);
      // The deformation field is applied to itself
      reg_defField_compose(deformationFieldImage[i], // to apply
                           deformationFieldImage[i+1], // to update
            NULL);
#ifndef NDEBUG
      printf("[NiftyReg DEBUG] Squaring (composition) step %u/%u\n", i+1, squaringNumber);
#endif
   }
   return;
}
/* *************************************************************** */
/* *************************************************************** */
template <class DTYPE>
void compute_lie_bracket(nifti_image *img1,
                         nifti_image *img2,
                         nifti_image *res,
                         bool use_jac
                         )
{
   reg_exit(1); // to update
#ifdef _WIN32
   long voxNumber=(long)img1->nx*img1->ny*img1->nz;
#else
   size_t voxNumber=(size_t)img1->nx*img1->ny*img1->nz;
#endif
   // Lie bracket using Jacobian for testing
   if(use_jac)
   {
      mat33 *jacImg1=(mat33 *)malloc(voxNumber*sizeof(mat33));
      mat33 *jacImg2=(mat33 *)malloc(voxNumber*sizeof(mat33));

      reg_getDeformationFromDisplacement(img1);
      reg_getDeformationFromDisplacement(img2);
      // HERE TO DO
      reg_exit(1);
      //        reg_spline_GetJacobianMatrixFull(img1,img1,jacImg1);
      //        reg_spline_GetJacobianMatrixFull(img2,img2,jacImg2);
      reg_getDisplacementFromDeformation(img1);
      reg_getDisplacementFromDeformation(img2);

      DTYPE *resPtrX=static_cast<DTYPE *>(res->data);
      DTYPE *resPtrY=&resPtrX[voxNumber];
      DTYPE *img1DispPtrX=static_cast<DTYPE *>(img1->data);
      DTYPE *img1DispPtrY=&img1DispPtrX[voxNumber];
      DTYPE *img2DispPtrX=static_cast<DTYPE *>(img2->data);
      DTYPE *img2DispPtrY=&img1DispPtrX[voxNumber];
      if(img1->nz>1)
      {
         DTYPE *resPtrZ=&resPtrY[voxNumber];
         DTYPE *img1DispPtrZ=&img1DispPtrY[voxNumber];
         DTYPE *img2DispPtrZ=&img1DispPtrY[voxNumber];

         for(size_t i=0; i<voxNumber; ++i)
         {
            resPtrX[i]=
                  (jacImg2[i].m[0][0]*img1DispPtrX[i] +
                  jacImg2[i].m[0][1]*img1DispPtrY[i] +
                  jacImg2[i].m[0][2]*img1DispPtrZ[i] )
                  -
                  (jacImg1[i].m[0][0]*img2DispPtrX[i] +
                  jacImg1[i].m[0][1]*img2DispPtrY[i] +
                  jacImg1[i].m[0][2]*img2DispPtrZ[i] );
            resPtrY[i]=
                  (jacImg2[i].m[1][0]*img1DispPtrX[i] +
                  jacImg2[i].m[1][1]*img1DispPtrY[i] +
                  jacImg2[i].m[1][2]*img1DispPtrZ[i] )
                  -
                  (jacImg1[i].m[1][0]*img2DispPtrX[i] +
                  jacImg1[i].m[1][1]*img2DispPtrY[i] +
                  jacImg1[i].m[1][2]*img2DispPtrZ[i] );
            resPtrZ[i]=
                  (jacImg2[i].m[2][0]*img1DispPtrX[i] +
                  jacImg2[i].m[2][1]*img1DispPtrY[i] +
                  jacImg2[i].m[2][2]*img1DispPtrZ[i] )
                  -
                  (jacImg1[i].m[2][0]*img2DispPtrX[i] +
                  jacImg1[i].m[2][1]*img2DispPtrY[i] +
                  jacImg1[i].m[2][2]*img2DispPtrZ[i] );
         }
      }
      else
      {
         for(size_t i=0; i<voxNumber; ++i)
         {
            resPtrX[i]=
                  (jacImg2[i].m[0][0]*img1DispPtrX[i] +
                  jacImg2[i].m[0][1]*img1DispPtrY[i] )
                  -
                  (jacImg1[i].m[0][0]*img2DispPtrX[i] +
                  jacImg1[i].m[0][1]*img2DispPtrY[i] );
            resPtrY[i]=
                  (jacImg2[i].m[1][0]*img1DispPtrX[i] +
                  jacImg2[i].m[1][1]*img1DispPtrY[i] )
                  -
                  (jacImg1[i].m[1][0]*img2DispPtrX[i] +
                  jacImg1[i].m[1][1]*img2DispPtrY[i] );
         }
      }
      free(jacImg1);
      free(jacImg2);
      return;
   }


   // Allocate two temporary nifti images
   nifti_image *one_two = nifti_copy_nim_info(img2);
   nifti_image *two_one = nifti_copy_nim_info(img1);
   // Set the temporary images to zero displacement
   one_two->data=(void *)calloc(one_two->nvox, one_two->nbyper);
   two_one->data=(void *)calloc(two_one->nvox, two_one->nbyper);
   // Compute the displacement from img1
   reg_spline_cppComposition(img1,
                             two_one,
                             true, // displacement1?
                             true, // displacement2?
                             true // bspline?
                             );
   // Compute the displacement from img2
   reg_spline_cppComposition(img2,
                             one_two,
                             true, // displacement1?
                             true, // displacement2?
                             true // bspline?
                             );
   // Compose both transformations
   reg_spline_cppComposition(img1,
                             one_two,
                             true, // displacement1?
                             true, // displacement2?
                             true // bspline?
                             );
   // Compose both transformations
   reg_spline_cppComposition(img2,
                             two_one,
                             true, // displacement1?
                             true, // displacement2?
                             true // bspline?
                             );
   // Create the data pointers
   DTYPE *resPtr=static_cast<DTYPE *>(res->data);
   DTYPE *one_twoPtr=static_cast<DTYPE *>(one_two->data);
   DTYPE *two_onePtr=static_cast<DTYPE *>(two_one->data);
   // Compute the lie bracket value using difference of composition

#ifdef _WIN32
   long i;
   voxNumber=(long)res->nvox;
#else
   size_t i;
   voxNumber=res->nvox;
#endif

#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(voxNumber, resPtr, one_twoPtr, two_onePtr) \
   private(i)
#endif
   for(i=0; i<voxNumber; ++i)
      resPtr[i]=two_onePtr[i]-one_twoPtr[i];
   // Free the temporary nifti images
   nifti_image_free(one_two);
   nifti_image_free(two_one);
   //    reg_spline_GetDeconvolvedCoefficents(res);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class DTYPE>
void compute_BCH_update1(nifti_image *img1, // current field
                         nifti_image *img2, // gradient
                         int type)
{
   DTYPE *res=(DTYPE *)malloc(img1->nvox*sizeof(DTYPE));

#ifdef _WIN32
   long i;
   long voxelNumber=(long)img1->nvox;
#else
   size_t i;
   size_t voxelNumber=img1->nvox;
#endif

   bool use_jac=false;

   // r <- 2 + 1
   DTYPE *img1Ptr=static_cast<DTYPE *>(img1->data);
   DTYPE *img2Ptr=static_cast<DTYPE *>(img2->data);
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(voxelNumber,img1Ptr,img2Ptr, res) \
   private(i)
#endif
   for(i=0; i<voxelNumber; ++i)
      res[i] = img1Ptr[i] + img2Ptr[i];

   if(type>0)
   {
      // Convert the deformation field into a displacement field
      reg_getDisplacementFromDeformation(img1);

      // r <- 2 + 1 + 0.5[2,1]
      nifti_image *lie_bracket_img2_img1=nifti_copy_nim_info(img1);
      lie_bracket_img2_img1->data=(void *)malloc(lie_bracket_img2_img1->nvox*lie_bracket_img2_img1->nbyper);
      compute_lie_bracket<DTYPE>(img2, img1, lie_bracket_img2_img1, use_jac);
      DTYPE *lie_bracket_img2_img1Ptr=static_cast<DTYPE *>(lie_bracket_img2_img1->data);
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(voxelNumber, res, lie_bracket_img2_img1Ptr) \
   private(i)
#endif
      for(i=0; i<voxelNumber; ++i)
         res[i] += 0.5 * lie_bracket_img2_img1Ptr[i];

      if(type>1)
      {
         // r <- 2 + 1 + 0.5[2,1] + [2,[2,1]]/12
         nifti_image *lie_bracket_img2_lie1=nifti_copy_nim_info(lie_bracket_img2_img1);
         lie_bracket_img2_lie1->data=(void *)malloc(lie_bracket_img2_lie1->nvox*lie_bracket_img2_lie1->nbyper);
         compute_lie_bracket<DTYPE>(img2, lie_bracket_img2_img1, lie_bracket_img2_lie1, use_jac);
         DTYPE *lie_bracket_img2_lie1Ptr=static_cast<DTYPE *>(lie_bracket_img2_lie1->data);
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(voxelNumber, res, lie_bracket_img2_lie1Ptr) \
   private(i)
#endif
         for(i=0; i<voxelNumber; ++i)
            res[i] += lie_bracket_img2_lie1Ptr[i]/12.0;

         if(type>2)
         {
            // r <- 2 + 1 + 0.5[2,1] + [2,[2,1]]/12 - [1,[2,1]]/12
            nifti_image *lie_bracket_img1_lie1=nifti_copy_nim_info(lie_bracket_img2_img1);
            lie_bracket_img1_lie1->data=(void *)malloc(lie_bracket_img1_lie1->nvox*lie_bracket_img1_lie1->nbyper);
            compute_lie_bracket<DTYPE>(img1, lie_bracket_img2_img1, lie_bracket_img1_lie1, use_jac);
            DTYPE *lie_bracket_img1_lie1Ptr=static_cast<DTYPE *>(lie_bracket_img1_lie1->data);
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(voxelNumber, res, lie_bracket_img1_lie1Ptr) \
   private(i)
#endif
            for(i=0; i<voxelNumber; ++i)
               res[i] -= lie_bracket_img1_lie1Ptr[i]/12.0;
            nifti_image_free(lie_bracket_img1_lie1);

            if(type>3)
            {
               // r <- 2 + 1 + 0.5[2,1] + [2,[2,1]]/12 - [1,[2,1]]/12 - [1,[2,[2,1]]]/24
               nifti_image *lie_bracket_img1_lie2=nifti_copy_nim_info(lie_bracket_img2_lie1);
               lie_bracket_img1_lie2->data=(void *)malloc(lie_bracket_img1_lie2->nvox*lie_bracket_img1_lie2->nbyper);
               compute_lie_bracket<DTYPE>(img1, lie_bracket_img2_lie1, lie_bracket_img1_lie2, use_jac);
               DTYPE *lie_bracket_img1_lie2Ptr=static_cast<DTYPE *>(lie_bracket_img1_lie2->data);
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(voxelNumber, res, lie_bracket_img1_lie2Ptr) \
   private(i)
#endif
               for(i=0; i<voxelNumber; ++i)
                  res[i] -= lie_bracket_img1_lie2Ptr[i]/24.0;
               nifti_image_free(lie_bracket_img1_lie2);
            }// >3
         }// >2
         nifti_image_free(lie_bracket_img2_lie1);
      }// >1
      nifti_image_free(lie_bracket_img2_img1);
   }// >0

   // update the deformation field
   memcpy(img1->data, res, img1->nvox*img1->nbyper);
   free(res);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void compute_BCH_update(nifti_image *img1, // current field
                        nifti_image *img2, // gradient
                        int type)
{
   if(img1->datatype!=img2->datatype)
   {
      fprintf(stderr,"[NiftyReg ERROR] compute_BCH_update\n");
      fprintf(stderr,"[NiftyReg ERROR] Both input images are expected to be of similar type\n");
      reg_exit(1);
   }
   switch(img1->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      compute_BCH_update1<float>(img1, img2, type);
      break;
   case NIFTI_TYPE_FLOAT64:
      compute_BCH_update1<double>(img1, img2, type);
      break;
   default:
      fprintf(stderr,"[NiftyReg ERROR] compute_BCH_update\n");
      fprintf(stderr,"[NiftyReg ERROR] Only implemented for single or double precision images\n");
      reg_exit(1);
   }
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class DTYPE>
void extractLine(int start, int end, int increment,const DTYPE *image, DTYPE *values)
{
   size_t index = 0;
   for(int i=start; i<end; i+=increment) values[index++] = image[i];
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class DTYPE>
void restoreLine(int start, int end, int increment, DTYPE *image, const DTYPE *values)
{
   size_t index = 0;
   for(int i=start; i<end; i+=increment) image[i] = values[index++];
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class DTYPE>
void intensitiesToSplineCoefficients(DTYPE *values, int number)
{
   // Border are set to zero
   DTYPE pole = sqrt(3.0) - 2.0;
   DTYPE currentPole = pole;
   DTYPE currentOpposite = pow(pole,(DTYPE)(2.0*(DTYPE)number-1.0));
   DTYPE sum=0.0;
   for(int i=1; i<number; i++)
   {
      sum += (currentPole - currentOpposite) * values[i];
      currentPole *= pole;
      currentOpposite /= pole;
   }
   values[0] = (DTYPE)((values[0] - pole*pole*(values[0] + sum)) / (1.0 - pow(pole,(DTYPE)(2.0*(double)number+2.0))));

   //other values forward
   for(int i=1; i<number; i++)
   {
      values[i] += pole * values[i-1];
   }

   DTYPE ipp=(DTYPE)(1.0-pole);
   ipp*=ipp;

   //last value
   values[number-1] = ipp * values[number-1];

   //other values backward
   for(int i=number-2; 0<=i; i--)
   {
      values[i] = pole * values[i+1] + ipp*values[i];
   }
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class DTYPE>
void reg_spline_GetDeconvolvedCoefficents1(nifti_image *img)
{
   double *coeff=(double *)malloc(img->nvox*sizeof(double));
   DTYPE *imgPtr=static_cast<DTYPE *>(img->data);
   for(size_t i=0; i<img->nvox; ++i)
      coeff[i]=imgPtr[i];
   for(int u=0; u<img->nu; ++u)
   {
      for(int t=0; t<img->nt; ++t)
      {
         double *coeffPtr=&coeff[(u*img->nt+t)*img->nx*img->ny*img->nz];

         // Along the X axis
         int number = img->nx;
         double *values=new double[number];
         int increment = 1;
         for(int i=0; i<img->ny*img->nz; i++)
         {
            int start = i*img->nx;
            int end = start + img->nx;
            extractLine<double>(start,end,increment,coeffPtr,values);
            intensitiesToSplineCoefficients<double>(values, number);
            restoreLine<double>(start,end,increment,coeffPtr,values);
         }
         delete[] values;
         values=NULL;

         // Along the Y axis
         number = img->ny;
         values=new double[number];
         increment = img->nx;
         for(int i=0; i<img->nx*img->nz; i++)
         {
            int start = i + i/img->nx * img->nx * (img->ny - 1);
            int end = start + img->nx*img->ny;
            extractLine<double>(start,end,increment,coeffPtr,values);
            intensitiesToSplineCoefficients<double>(values, number);
            restoreLine<double>(start,end,increment,coeffPtr,values);
         }
         delete[] values;
         values=NULL;

         // Along the Y axis
         if(img->nz>1)
         {
            number = img->nz;
            values=new double[number];
            increment = img->nx*img->ny;
            for(int i=0; i<img->nx*img->ny; i++)
            {
               int start = i;
               int end = start + img->nx*img->ny*img->nz;
               extractLine<double>(start,end,increment,coeffPtr,values);
               intensitiesToSplineCoefficients<double>(values, number);
               restoreLine<double>(start,end,increment,coeffPtr,values);
            }
            delete[] values;
            values=NULL;
         }
      }//t
   }//u

   for(size_t i=0; i<img->nvox; ++i)
      imgPtr[i]=coeff[i];
   free(coeff);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
void reg_spline_GetDeconvolvedCoefficents(nifti_image *img)
{

   switch(img->datatype)
   {
   case NIFTI_TYPE_FLOAT32:
      reg_spline_GetDeconvolvedCoefficents1<float>(img);
      break;
   case NIFTI_TYPE_FLOAT64:
      reg_spline_GetDeconvolvedCoefficents1<double>(img);
      break;
   default:
      fprintf(stderr,"[NiftyReg ERROR] reg_spline_GetDeconvolvedCoefficents1\n");
      fprintf(stderr,"[NiftyReg ERROR] Only implemented for single or double precision images\n");
      reg_exit(1);
   }
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

#include "_reg_localTransformation_jac.cpp"
#include "_reg_localTransformation_be.cpp"

#endif
