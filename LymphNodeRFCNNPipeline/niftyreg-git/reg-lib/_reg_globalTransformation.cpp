/*
 *  _reg_affineTransformation.cpp
 *
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_AFFINETRANSFORMATION_CPP
#define _REG_AFFINETRANSFORMATION_CPP

#include "_reg_globalTransformation.h"
#include "_reg_maths.h"

/* *************************************************************** */
/* *************************************************************** */
template <class FieldTYPE>
void reg_affine_deformationField2D(mat44 *affineTransformation,
                                   nifti_image *deformationFieldImage,
                                   bool compose,
                                   int *mask)
{
   FieldTYPE *deformationFieldPtr = static_cast<FieldTYPE *>(deformationFieldImage->data);

   size_t deformationFieldIndX=0;
   size_t deformationFieldIndY=deformationFieldImage->nx*deformationFieldImage->ny;

   mat44 *targetMatrix;
   if(deformationFieldImage->sform_code>0)
   {
      targetMatrix=&(deformationFieldImage->sto_xyz);
   }
   else targetMatrix=&(deformationFieldImage->qto_xyz);

   mat44 transformationMatrix;
   if(compose==true)
      transformationMatrix = *affineTransformation;
   else transformationMatrix = reg_mat44_mul(affineTransformation, targetMatrix);

   float voxel[3]; voxel[2]=0;
   float position[3];
   size_t index=0;
   for(int y=0; y<deformationFieldImage->ny; y++)
   {
      voxel[1]=(float)y;
      for(int x=0; x<deformationFieldImage->nx; x++)
      {
         voxel[0]=(float)x;

         if(mask[index++]>-1)
         {
            if(compose==true)
            {
               voxel[0]=deformationFieldPtr[deformationFieldIndX];
               voxel[1]=deformationFieldPtr[deformationFieldIndY];
            }
            reg_mat44_mul(&transformationMatrix, voxel, position);

            /* the deformation field (real coordinates) is stored */
            deformationFieldPtr[deformationFieldIndX++] = position[0];
            deformationFieldPtr[deformationFieldIndY++] = position[1];
         }
      }
   }
}
/* *************************************************************** */
template <class FieldTYPE>
void reg_affine_deformationField3D(mat44 *affineTransformation,
                                   nifti_image *deformationFieldImage,
                                   bool composition,
                                   int *mask)
{
   size_t voxelNumber=deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz;
   FieldTYPE *deformationFieldPtrX = static_cast<FieldTYPE *>(deformationFieldImage->data);
   FieldTYPE *deformationFieldPtrY = &deformationFieldPtrX[voxelNumber];
   FieldTYPE *deformationFieldPtrZ = &deformationFieldPtrY[voxelNumber];

   mat44 *targetMatrix;
   if(deformationFieldImage->sform_code>0)
   {
      targetMatrix=&(deformationFieldImage->sto_xyz);
   }
   else targetMatrix=&(deformationFieldImage->qto_xyz);

   mat44 transformationMatrix;
   if(composition==true)
      transformationMatrix = *affineTransformation;
   else transformationMatrix = reg_mat44_mul(affineTransformation, targetMatrix);

   float voxel[3], position[3];
   int x, y, z;
   size_t index;
#if defined (_OPENMP)
#pragma omp parallel for default(none) \
   shared(deformationFieldImage, transformationMatrix, deformationFieldPtrX, \
   deformationFieldPtrY, deformationFieldPtrZ, mask, composition) \
   private(voxel, position, x, y, z, index)
#endif
   for(z=0; z<deformationFieldImage->nz; z++)
   {
      index=z*deformationFieldImage->nx*deformationFieldImage->ny;
      voxel[2]=(float)z;
      for(y=0; y<deformationFieldImage->ny; y++)
      {
         voxel[1]=(float)y;
         for(x=0; x<deformationFieldImage->nx; x++)
         {
            voxel[0]=(float)x;
            if(mask[index]>-1)
            {
               if(composition==true)
               {
                  voxel[0]=deformationFieldPtrX[index];
                  voxel[1]=deformationFieldPtrY[index];
                  voxel[2]=deformationFieldPtrZ[index];
               }
               position[0] =
                     transformationMatrix.m[0][0] * voxel[0] +
                     transformationMatrix.m[0][1] * voxel[1] +
                     transformationMatrix.m[0][2] * voxel[2] +
                     transformationMatrix.m[0][3] ;
               position[1] =
                     transformationMatrix.m[1][0] * voxel[0] +
                     transformationMatrix.m[1][1] * voxel[1] +
                     transformationMatrix.m[1][2] * voxel[2] +
                     transformationMatrix.m[1][3] ;
               position[2] =
                     transformationMatrix.m[2][0] * voxel[0] +
                     transformationMatrix.m[2][1] * voxel[1] +
                     transformationMatrix.m[2][2] * voxel[2] +
                     transformationMatrix.m[2][3] ;
               //                    reg_mat44_mul(&transformationMatrix, voxel, position);

               /* the deformation field (real coordinates) is stored */
               deformationFieldPtrX[index] = position[0];
               deformationFieldPtrY[index] = position[1];
               deformationFieldPtrZ[index] = position[2];
            }
            index++;
         }
      }
   }
}
/* *************************************************************** */
void reg_affine_getDeformationField(mat44 *affineTransformation,
                                    nifti_image *deformationField,
                                    bool compose,
                                    int *mask)
{
   int *tempMask=mask;
   if(mask==NULL)
   {
      tempMask=(int *)calloc(deformationField->nx*
                             deformationField->ny*
                             deformationField->nz,
                             sizeof(int));
   }
   if(deformationField->nz==1)
   {
      switch(deformationField->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_affine_deformationField2D<float>(affineTransformation, deformationField, compose, tempMask);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_affine_deformationField2D<double>(affineTransformation, deformationField, compose, tempMask);
         break;
      default:
         fprintf(stderr,"[NiftyReg ERROR] reg_affine_deformationField\tThe deformation field data type is not supported\n");
         return;
      }
   }
   else
   {
      switch(deformationField->datatype)
      {
      case NIFTI_TYPE_FLOAT32:
         reg_affine_deformationField3D<float>(affineTransformation, deformationField, compose, tempMask);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_affine_deformationField3D<double>(affineTransformation, deformationField, compose, tempMask);
         break;
      default:
         fprintf(stderr,"[NiftyReg ERROR] reg_affine_deformationField: The deformation field data type is not supported\n");
         return;
      }
   }
   if(mask==NULL)
      free(tempMask);
}
/* *************************************************************** */
/* *************************************************************** */
void reg_tool_ReadAffineFile(mat44 *mat,
                             nifti_image* target,
                             nifti_image* source,
                             char *fileName,
                             bool flirtFile)
{
   std::ifstream affineFile;
   affineFile.open(fileName);
   if(affineFile.is_open())
   {
      int i=0;
      float value1,value2,value3,value4;
      while(!affineFile.eof())
      {
         affineFile >> value1 >> value2 >> value3 >> value4;
         mat->m[i][0] = value1;
         mat->m[i][1] = value2;
         mat->m[i][2] = value3;
         mat->m[i][3] = value4;
         i++;
         if(i>3) break;
      }
   }
   else
   {
      fprintf(stderr, "[NiftyReg ERROR] The affine file can not be read: %s\n", fileName);
      reg_exit(1);
   }
   affineFile.close();

#ifndef NDEBUG
   reg_mat44_disp(mat, (char *)"[NiftyReg DEBUG] Read affine transformation");
#endif

   if(flirtFile)
   {
      mat44 absoluteTarget;
      mat44 absoluteSource;
      for(int i=0; i<4; i++)
      {
         for(int j=0; j<4; j++)
         {
            absoluteTarget.m[i][j]=absoluteSource.m[i][j]=0.0;
         }
      }
      //If the target sform is defined, it is used; qform otherwise;
      mat44 *targetMatrix;
      if(target->sform_code > 0)
      {
         targetMatrix = &(target->sto_xyz);
#ifndef NDEBUG
         printf("[NiftyReg DEBUG] The target sform matrix is defined and used\n");
#endif
      }
      else targetMatrix = &(target->qto_xyz);
      //If the source sform is defined, it is used; qform otherwise;
      mat44 *sourceMatrix;
      if(source->sform_code > 0)
      {
#ifndef NDEBUG
         printf("[NiftyReg DEBUG]  The source sform matrix is defined and used\n");
#endif
         sourceMatrix = &(source->sto_xyz);
      }
      else sourceMatrix = &(source->qto_xyz);

      for(int i=0; i<3; i++)
      {
         absoluteTarget.m[i][i]=sqrt(targetMatrix->m[0][i]*targetMatrix->m[0][i]
               + targetMatrix->m[1][i]*targetMatrix->m[1][i]
               + targetMatrix->m[2][i]*targetMatrix->m[2][i]);
         absoluteSource.m[i][i]=sqrt(sourceMatrix->m[0][i]*sourceMatrix->m[0][i]
               + sourceMatrix->m[1][i]*sourceMatrix->m[1][i]
               + sourceMatrix->m[2][i]*sourceMatrix->m[2][i]);
      }
      absoluteTarget.m[3][3]=absoluteSource.m[3][3]=1.0;
#ifndef NDEBUG
      printf("[NiftyReg DEBUG] An flirt affine file is assumed and is converted to a real word affine matrix\n");
      reg_mat44_disp(mat, (char *)"[DEBUG] Matrix read from the input file");
      reg_mat44_disp(targetMatrix, (char *)"[DEBUG] Target Matrix");
      reg_mat44_disp(sourceMatrix, (char *)"[DEBUG] Source Matrix");
      reg_mat44_disp(&(absoluteTarget), (char *)"[DEBUG] Target absolute Matrix");
      reg_mat44_disp(&(absoluteSource), (char *)"[DEBUG] Source absolute Matrix");
#endif

      absoluteSource = nifti_mat44_inverse(absoluteSource);
      *mat = nifti_mat44_inverse(*mat);

      *mat = reg_mat44_mul(&absoluteSource,mat);
      *mat = reg_mat44_mul(mat, &absoluteTarget);
      *mat = reg_mat44_mul(sourceMatrix,mat);
      mat44 tmp = nifti_mat44_inverse(*targetMatrix);
      *mat = reg_mat44_mul(mat, &tmp);
   }

#ifndef NDEBUG
   reg_mat44_disp(mat, (char *)"[DEBUG] Affine matrix");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
void reg_tool_ReadAffineFile(mat44 *mat,
                             char *fileName)
{
   std::ifstream affineFile;
   affineFile.open(fileName);
   if(affineFile.is_open())
   {
      int i=0;
      float value1,value2,value3,value4;
      while(!affineFile.eof())
      {
         affineFile >> value1 >> value2 >> value3 >> value4;
         mat->m[i][0] = value1;
         mat->m[i][1] = value2;
         mat->m[i][2] = value3;
         mat->m[i][3] = value4;
         i++;
         if(i>3) break;
      }
   }
   else
   {
      fprintf(stderr, "[NiftyReg ERROR] The affine file can not be read: %s\n", fileName);
      reg_exit(1);
   }
   affineFile.close();

#ifndef NDEBUG
   reg_mat44_disp(mat, (char *)"[DEBUG] Affine matrix");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
void reg_tool_WriteAffineFile(mat44 *mat,
                              const char *fileName)
{
   FILE *affineFile;
   affineFile=fopen(fileName, "w");
   for(int i=0; i<4; i++)
      fprintf(affineFile, "%.7g %.7g %.7g %.7g\n", mat->m[i][0], mat->m[i][1], mat->m[i][2], mat->m[i][3]);
   fclose(affineFile);
}
/* *************************************************************** */
/* *************************************************************** */

#endif
