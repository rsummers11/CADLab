/*
 *  _reg_thinPlateSpline.cpp
 *
 *
 *  Created by Marc Modat on 22/02/2011.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_THINPLATESPLINE_CPP
#define _REG_THINPLATESPLINE_CPP

#include "_reg_thinPlateSpline.h"

/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_tps<T>::reg_tps(size_t d, size_t n)
{
   this->dim=d;
   this->number=n;
   this->positionX=(T*)calloc(this->number,sizeof(T));
   this->positionY=(T*)calloc(this->number,sizeof(T));
   this->coefficientX=(T*)calloc(this->number+this->dim+1,sizeof(T));
   this->coefficientY=(T*)calloc(this->number+this->dim+1,sizeof(T));
   if(this->dim==3)
   {
      this->positionZ=(T*)calloc(this->number,sizeof(T));
      this->coefficientZ=(T*)calloc(this->number+this->dim+1,sizeof(T));
   }
   else
   {
      this->positionZ=NULL;
      this->coefficientZ=NULL;
   }
   this->initialised=false;
   this->approxInter=0.;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_tps<T>::~reg_tps()
{
   if(this->positionX!=NULL) free(this->positionX);
   this->positionX=NULL;
   if(this->positionY!=NULL) free(this->positionY);
   this->positionY=NULL;
   if(this->positionZ!=NULL) free(this->positionZ);
   this->positionZ=NULL;
   if(this->coefficientX!=NULL) free(this->coefficientX);
   this->coefficientX=NULL;
   if(this->coefficientY!=NULL) free(this->coefficientY);
   this->coefficientY=NULL;
   if(this->coefficientZ!=NULL) free(this->coefficientZ);
   this->coefficientZ=NULL;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_tps<T>::SetPosition(T *px, T *py, T *pz, T *cx,T *cy, T *cz)
{
   memcpy(this->positionX,px,this->number*sizeof(T));
   memcpy(this->positionY,py,this->number*sizeof(T));
   memcpy(this->positionZ,pz,this->number*sizeof(T));
   memcpy(this->coefficientX,cx,this->number*sizeof(T));
   memcpy(this->coefficientY,cy,this->number*sizeof(T));
   memcpy(this->coefficientZ,cz,this->number*sizeof(T));
   for(size_t i=this->number; i<this->number+this->dim+1; ++i)
   {
      this->coefficientX[i]=0;
      this->coefficientY[i]=0;
      this->coefficientZ[i]=0;
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_tps<T>::SetPosition(T *px, T *py, T *cx,T *cy)
{
   memcpy(this->positionX,px,this->number*sizeof(T));
   memcpy(this->positionY,py,this->number*sizeof(T));
   memcpy(this->coefficientX,cx,this->number*sizeof(T));
   memcpy(this->coefficientY,cy,this->number*sizeof(T));
   for(size_t i=this->number; i<this->number+this->dim+1; ++i)
   {
      this->coefficientX[i]=0;
      this->coefficientY[i]=0;
   }
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_tps<T>::SetAproxInter(T v)
{
   this->approxInter=v;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
T reg_tps<T>::GetTPSEuclideanDistance(size_t i, size_t j)
{
   T temp = this->positionX[i] - this->positionX[j];
   T dist = temp*temp;
   temp = this->positionY[i] - this->positionY[j];
   dist += temp*temp;
   if(this->dim==3)
   {
      temp = this->positionZ[i] - this->positionZ[j];
      dist += temp*temp;
   }
   return sqrt(dist);
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
T reg_tps<T>::GetTPSEuclideanDistance(size_t i, T *p)
{
   T temp = this->positionX[i] - p[0];
   T dist = temp*temp;
   temp = this->positionY[i] - p[1];
   dist += temp*temp;
   if(this->dim==3)
   {
      temp = this->positionZ[i] - p[2];
      dist += temp*temp;
   }
   return sqrt(dist);
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
T reg_tps<T>::GetTPSweight(T dist)
{
   if(dist==0) return 0;
   return dist*dist*log(dist);
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_tps<T>::InitialiseTPS()
{
   size_t matrix_side=this->number + this->dim + 1;
   T *matrixL=(T *)calloc(matrix_side*matrix_side,sizeof(T));
   if(matrixL==NULL)
   {
      fprintf(stderr,"[NiftyReg] ERROR Calloc failed, the TPS distance matrix is too large\n");
      fprintf(stderr,"[NiftyReg] ERROR Size should be %g GB (%i x %i) ... Exit\n",
              (T)(matrix_side*matrix_side)*sizeof(T)/1000000000.f,
              (int)matrix_side,(int)matrix_side);
      reg_exit(1);
   }

   // Distance matrix is computed
   double a=0.;
   for(size_t i=0; i<this->number; ++i)
   {
      for(size_t j=i+1; j<this->number; ++j)
      {
         T distance = this->GetTPSEuclideanDistance(i,j);
         a += distance * 2.;
         distance = this->GetTPSweight(distance);
         matrixL[i*matrix_side+j]=matrixL[j*matrix_side+i]=distance;
      }
   }
   a/=(double)(this->number*this->number);
   a=(double)this->approxInter*a*a;
   printf("a=%f\n",a);
   for(size_t i=0; i<this->number; ++i)
   {
      matrixL[i*matrix_side+i]=a;
   }
   for(size_t i=0; i<this->number; ++i)
   {
      matrixL[i*matrix_side+this->number]=matrixL[(this->number)*matrix_side+i]=1;
      matrixL[i*matrix_side+this->number+1]=matrixL[(this->number+1)*matrix_side+i]=this->positionX[i];
      matrixL[i*matrix_side+this->number+2]=matrixL[(this->number+2)*matrix_side+i]=this->positionY[i];
      if(this->dim==3)
         matrixL[i*matrix_side+this->number+3]=matrixL[(this->number+3)*matrix_side+i]=this->positionZ[i];

   }
   for(size_t i=this->number; i<matrix_side; ++i)
   {
      for(size_t j=this->number; j<matrix_side; ++j)
      {
         matrixL[i*matrix_side+j]=0;
      }
   }
   printf("[NiftyReg] TPS Distance matrix initialised\n");

   // Run the LU decomposition
   size_t *index=(size_t *)calloc(matrix_side,sizeof(size_t));
   reg_LUdecomposition<T>(matrixL, matrix_side, index);
   printf("[NiftyReg] TPS LU decomposition done\n");

   // Perform the multiplications
   reg_matrixInvertMultiply<T>(matrixL, matrix_side, index, this->coefficientX);
   printf("[NiftyReg] TPS x-axis coefficients computed\n");
   reg_matrixInvertMultiply<T>(matrixL, matrix_side, index, this->coefficientY);
   printf("[NiftyReg] TPS y-axis coefficients computed\n");
   if(this->dim==3)
   {
      reg_matrixInvertMultiply<T>(matrixL, matrix_side, index, this->coefficientZ);
      printf("[NiftyReg] TPS z-axis coefficients computed\n");
   }

   free(index);
   free(matrixL);
   this->initialised=true;
   return;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_tps<T>::FillDeformationField(nifti_image *deformationField)
{
   if(this->initialised==false)
      this->InitialiseTPS();

   size_t voxelNumber = deformationField->nx*deformationField->ny*deformationField->nz;
   T *defX=static_cast<T *>(deformationField->data);
   T *defY=&defX[voxelNumber];
   T *defZ=NULL;
   if(this->dim==3)
      defZ=&defY[voxelNumber];

   mat44 *voxel2realDF=NULL;
   if(deformationField->sform_code>0)
      voxel2realDF=&(deformationField->sto_xyz);
   else voxel2realDF=&(deformationField->qto_xyz);

   T position[3];

   int index=0;
   for(int z=0; z<deformationField->nz; ++z)
   {
      for(int y=0; y<deformationField->ny; ++y)
      {
         for(int x=0; x<deformationField->nx; ++x)
         {

            // Compute the voxel position in mm
            position[0]=x * voxel2realDF->m[0][0] +
                        y * voxel2realDF->m[0][1] +
                        z * voxel2realDF->m[0][2] +
                        voxel2realDF->m[0][3];
            position[1]=x * voxel2realDF->m[1][0] +
                        y * voxel2realDF->m[1][1] +
                        z * voxel2realDF->m[1][2] +
                        voxel2realDF->m[1][3];
            position[2]=x * voxel2realDF->m[2][0] +
                        y * voxel2realDF->m[2][1] +
                        z * voxel2realDF->m[2][2] +
                        voxel2realDF->m[2][3];

            T finalPositionX=0;
            T finalPositionY=0;
            T finalPositionZ=0;
            if(this->dim==3)
            {
               finalPositionX=this->coefficientX[this->number]+
                              this->coefficientX[this->number+1]*position[0]+
                              this->coefficientX[this->number+2]*position[1]+
                              this->coefficientX[this->number+3]*position[2];

               finalPositionY=this->coefficientY[this->number]+
                              this->coefficientY[this->number+1]*position[0]+
                              this->coefficientY[this->number+2]*position[1]+
                              this->coefficientY[this->number+3]*position[2];

               finalPositionZ=this->coefficientZ[this->number]+
                              this->coefficientZ[this->number+1]*position[0]+
                              this->coefficientZ[this->number+2]*position[1]+
                              this->coefficientZ[this->number+3]*position[2];
            }
            else
            {
               finalPositionX=this->coefficientX[this->number] +
                              this->coefficientX[this->number+1]*position[0]+
                              this->coefficientX[this->number+2]*position[1];

               finalPositionY=this->coefficientY[this->number] +
                              this->coefficientY[this->number+1]*position[0]+
                              this->coefficientY[this->number+2]*position[1];
            }

            // Compute the displacement
            for(size_t i=0; i<this->number; ++i)
            {
               T distance=GetTPSweight(GetTPSEuclideanDistance(i,position));
               finalPositionX += this->coefficientX[i]*distance;
               finalPositionY += this->coefficientY[i]*distance;
               if(this->dim==3)
                  finalPositionZ += this->coefficientZ[i]*distance;
            }
            defX[index]=finalPositionX+position[0];
            defY[index]=finalPositionY+position[1];
            if(this->dim==3)
               defZ[index]=finalPositionZ+position[2];
            index++;
         }
      }
   }

}
/* *************************************************************** */
/* *************************************************************** */

#endif // _REG_THINPLATESPLINE_CPP
