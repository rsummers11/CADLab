/**
 * @file reg_jacobian.cpp
 * @author Marc Modat
 * @date 15/11/2010
 * @brief Executable use to generate Jacobian matrices and determinant
 * images.
 *
 *  Created by Marc Modat on 15/11/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _MM_JACOBIAN_CPP
#define _MM_JACOBIAN_CPP

#include "_reg_ReadWriteImage.h"
#include "_reg_globalTransformation.h"
#include "_reg_localTransformation.h"
#include "_reg_tools.h"
#include "_reg_resampling.h"
#include "reg_jacobian.h"

typedef struct
{
   char *refImageName;
   char *inputTransName;
   char *outputJacDetName;
   char *outputJacMatName;
   char *outputLogDetName;
} PARAM;
typedef struct
{
   bool refImageFlag;
   bool inputTransFlag;
   bool outputJacDetFlag;
   bool outputJacMatFlag;
   bool outputLogDetFlag;
} FLAG;

template <class DTYPE>
void reg_jacobian_computeLog(nifti_image *image)
{
   DTYPE *imgPtr=static_cast<DTYPE *>(image->data);
   for(size_t i=0; i<image->nvox;++i){
      *imgPtr = static_cast<DTYPE>(log(*imgPtr));
      ++imgPtr;
   }
   return;
}

template <class DTYPE>
void reg_jacobian_convertMat33ToNii(mat33 *array, nifti_image *image)
{
   size_t voxelNumber=image->nx*image->ny*image->nz;
   DTYPE *ptrXX=static_cast<DTYPE *>(image->data);
   if(image->nz>1)
   {
      DTYPE *ptrXY=&ptrXX[voxelNumber];
      DTYPE *ptrXZ=&ptrXY[voxelNumber];
      DTYPE *ptrYX=&ptrXZ[voxelNumber];
      DTYPE *ptrYY=&ptrYX[voxelNumber];
      DTYPE *ptrYZ=&ptrYY[voxelNumber];
      DTYPE *ptrZX=&ptrYZ[voxelNumber];
      DTYPE *ptrZY=&ptrZX[voxelNumber];
      DTYPE *ptrZZ=&ptrZY[voxelNumber];
      for(size_t voxel=0; voxel<voxelNumber; ++voxel)
      {
         mat33 matrix=array[voxel];
         ptrXX[voxel]=matrix.m[0][0];
         ptrXY[voxel]=matrix.m[0][1];
         ptrXZ[voxel]=matrix.m[0][2];
         ptrYX[voxel]=matrix.m[1][0];
         ptrYY[voxel]=matrix.m[1][1];
         ptrYZ[voxel]=matrix.m[1][2];
         ptrZX[voxel]=matrix.m[2][0];
         ptrZY[voxel]=matrix.m[2][1];
         ptrZZ[voxel]=matrix.m[2][2];
      }
   }
   else
   {
      DTYPE *ptrXY=&ptrXX[voxelNumber];
      DTYPE *ptrYX=&ptrXY[voxelNumber];
      DTYPE *ptrYY=&ptrYX[voxelNumber];
      for(size_t voxel=0; voxel<voxelNumber; ++voxel)
      {
         mat33 matrix=array[voxel];
         ptrXX[voxel]=matrix.m[0][0];
         ptrXY[voxel]=matrix.m[0][1];
         ptrYX[voxel]=matrix.m[1][0];
         ptrYY[voxel]=matrix.m[1][1];
      }

   }
}

void PetitUsage(char *exec)
{
   fprintf(stderr,"Usage:\t%s -ref <referenceImage> [OPTIONS].\n",exec);
   fprintf(stderr,"\tSee the help for more details (-h).\n");
   return;
}
void Usage(char *exec)
{
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   printf("Usage:\t%s [OPTIONS].\n",exec);
   printf("* * INPUT * *\n");
   printf("\t-trans <filename>\n");
   printf("\t\tFilename of the file containing the transformation (mandatory).\n");
   printf("\t-ref <filename>\n");
   printf("\t\tFilename of the reference image (required if the transformation is a spline parametrisation)\n");
   printf("\n* * OUTPUT * *\n");
   printf("\t-jac <filename>\n");
   printf("\t\tFilename of the Jacobian determinant map.\n");
   printf("\t-jacM <filename>\n");
   printf("\t\tFilename of the Jacobian matrix map. (9 or 4 values are stored as a 5D nifti).\n");
   printf("\t-jacL <filename>\n");
   printf("\t\tFilename of the Log of the Jacobian determinant map.\n");
#ifdef _GIT_HASH
   printf("\n\t--version\t\tPrint current source code git hash key and exit\n\t\t\t\t(%s)\n",_GIT_HASH);
#endif
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   return;
}

int main(int argc, char **argv)
{
   if(argc==1){
      PetitUsage(argv[0]);
      return 1;
   }

   PARAM *param = (PARAM *)calloc(1,sizeof(PARAM));
   FLAG *flag = (FLAG *)calloc(1,sizeof(FLAG));

   // read the input parameters
   for(int i=1; i<argc; i++)
   {
      if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
            strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
            strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0)
      {
         Usage(argv[0]);
         return 0;
      }
      else if(strcmp(argv[i], "--xml")==0)
      {
         printf("%s",xml_jacobian);
         return 0;
      }
#ifdef _GIT_HASH
      else if( strcmp(argv[i], "-version")==0 ||
          strcmp(argv[i], "-Version")==0 ||
          strcmp(argv[i], "-V")==0 ||
          strcmp(argv[i], "-v")==0 ||
          strcmp(argv[i], "--v")==0 ||
          strcmp(argv[i], "--version")==0)
      {
         printf("%s\n",_GIT_HASH);
         return EXIT_SUCCESS;
      }
#endif
      else if((strcmp(argv[i],"-ref")==0) || (strcmp(argv[i],"-target")==0) ||
              (strcmp(argv[i],"--ref")==0))
      {
         param->refImageName=argv[++i];
         flag->refImageFlag=1;
      }
      else if(strcmp(argv[i], "-trans") == 0 ||
              (strcmp(argv[i],"--trans")==0))
      {
         param->inputTransName=argv[++i];
         flag->inputTransFlag=1;
      }
      else if(strcmp(argv[i], "-jac") == 0 ||
              (strcmp(argv[i],"--jac")==0))
      {
         param->outputJacDetName=argv[++i];
         flag->outputJacDetFlag=1;
      }
      else if(strcmp(argv[i], "-jacM") == 0 ||
              (strcmp(argv[i],"--jacM")==0))
      {
         param->outputJacMatName=argv[++i];
         flag->outputJacMatFlag=1;
      }
      else if(strcmp(argv[i], "-jacL") == 0 ||
              (strcmp(argv[i],"--jacL")==0))
      {
         param->outputLogDetName=argv[++i];
         flag->outputLogDetFlag=1;
      }
      else
      {
         fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
         PetitUsage(argv[0]);
         return 1;
      }
   }

   /* ******************* */
   /* READ TRANSFORMATION */
   /* ******************* */
   nifti_image *inputTransformation=NULL;
   if(flag->inputTransFlag)
   {
      // Check of the input transformation is an affine
      if(!reg_isAnImageFileName(param->inputTransName)){
         mat44 *affineTransformation=(mat44 *)malloc(sizeof(mat44));
         reg_tool_ReadAffineFile(affineTransformation,param->inputTransName);
         printf("%g\n", reg_mat44_det(affineTransformation));
         return EXIT_SUCCESS;
      }

      inputTransformation = reg_io_ReadImageFile(param->inputTransName);
      if(inputTransformation == NULL)
      {
         fprintf(stderr,"** ERROR Error when reading the transformation image: %s\n",param->inputTransName);
         return 1;
      }
      reg_checkAndCorrectDimension(inputTransformation);
   }
   else
   {
      fprintf(stderr, "No transformation has been provided.\n");
      return 1;
   }

   /* *************************** */
   /* COMPUTE JACOBIAN MAT OR DET */
   /* *************************** */
   // Create a deformation field if needed
   nifti_image *referenceImage=NULL;
   if(inputTransformation->intent_p1==SPLINE_GRID ||
         inputTransformation->intent_p1==SPLINE_VEL_GRID){
      if(!flag->refImageFlag){
         reg_print_msg_error("A reference image has to be specified with a spline parametrisation.");
         reg_exit(1);
      }
      // Read the reference image
      referenceImage = reg_io_ReadImageHeader(param->refImageName);
      if(referenceImage == NULL)
      {
         reg_print_msg_error("Error when reading the reference image.");
         reg_exit(1);
      }
      reg_checkAndCorrectDimension(referenceImage);
   }

   if(flag->outputJacDetFlag || flag->outputLogDetFlag){
      // Compute the map of Jacobian determinant
      // Create the Jacobian image
      nifti_image *jacobianImage=NULL;
      if(referenceImage!=NULL){
         jacobianImage=nifti_copy_nim_info(referenceImage);
         nifti_image_free(referenceImage);referenceImage=NULL;
      }
      else jacobianImage=nifti_copy_nim_info(inputTransformation);
      jacobianImage->ndim=jacobianImage->dim[0]=jacobianImage->nz>1?3:2;
      jacobianImage->nu=jacobianImage->dim[5]=1;
      jacobianImage->nt=jacobianImage->dim[4]=1;
      jacobianImage->nvox=(size_t)jacobianImage->nx *jacobianImage->ny*
            jacobianImage->nz*jacobianImage->nt*jacobianImage->nu;
      jacobianImage->datatype = inputTransformation->datatype;
      jacobianImage->nbyper = inputTransformation->nbyper;
      jacobianImage->cal_min=0;
      jacobianImage->cal_max=0;
      jacobianImage->scl_slope = 1.0f;
      jacobianImage->scl_inter = 0.0f;
      jacobianImage->data = (void *)calloc(jacobianImage->nvox, jacobianImage->nbyper);

      switch((int)inputTransformation->intent_p1){
      case DISP_FIELD:
         reg_getDeformationFromDisplacement(inputTransformation);
      case DEF_FIELD:
         reg_defField_getJacobianMap(inputTransformation,jacobianImage);
         break;
      case DISP_VEL_FIELD:
         reg_getDeformationFromDisplacement(inputTransformation);
      case DEF_VEL_FIELD:
         reg_defField_GetJacobianDetFromFlowField(jacobianImage,inputTransformation);
         break;
      case SPLINE_GRID:
         reg_spline_GetJacobianMap(inputTransformation,jacobianImage);
         break;
      case SPLINE_VEL_GRID:
         reg_spline_GetJacobianDetFromVelocityGrid(jacobianImage,inputTransformation);
         break;
      }
      if(flag->outputJacDetFlag)
         reg_io_WriteImageFile(jacobianImage,param->outputJacDetName);
      if(flag->outputLogDetFlag){
         switch(jacobianImage->datatype){
         case NIFTI_TYPE_FLOAT32:
            reg_jacobian_computeLog<float>(jacobianImage);
            break;
         case NIFTI_TYPE_FLOAT64:
            reg_jacobian_computeLog<double>(jacobianImage);
            break;
         }
         reg_io_WriteImageFile(jacobianImage,param->outputLogDetName);
      }
      nifti_image_free(jacobianImage);jacobianImage=NULL;
   }
   if(flag->outputJacMatFlag){

      nifti_image *jacobianImage=NULL;
      if(referenceImage!=NULL){
         jacobianImage=nifti_copy_nim_info(referenceImage);
         nifti_image_free(referenceImage);referenceImage=NULL;
      }
      else jacobianImage=nifti_copy_nim_info(inputTransformation);
      jacobianImage->ndim=jacobianImage->dim[0]=5;
      jacobianImage->nu=jacobianImage->dim[5]=jacobianImage->nz>1?9:4;
      jacobianImage->nt=jacobianImage->dim[4]=1;
      jacobianImage->nvox=(size_t)jacobianImage->nx *jacobianImage->ny*
            jacobianImage->nz*jacobianImage->nt*jacobianImage->nu;
      jacobianImage->datatype = inputTransformation->datatype;
      jacobianImage->nbyper = inputTransformation->nbyper;
      jacobianImage->cal_min=0;
      jacobianImage->cal_max=0;
      jacobianImage->scl_slope = 1.0f;
      jacobianImage->scl_inter = 0.0f;
      jacobianImage->data = (void *)calloc(jacobianImage->nvox, jacobianImage->nbyper);

      mat33 *jacobianMatriceArray=(mat33 *)malloc(jacobianImage->nx*jacobianImage->ny*jacobianImage->nz*sizeof(mat33));
      // Compute the map of Jacobian matrices
      switch((int)inputTransformation->intent_p1){
      case DISP_FIELD:
         reg_getDeformationFromDisplacement(inputTransformation);
      case DEF_FIELD:
         reg_defField_getJacobianMatrix(inputTransformation,jacobianMatriceArray);
         break;
      case DISP_VEL_FIELD:
         reg_getDeformationFromDisplacement(inputTransformation);
      case DEF_VEL_FIELD:
         reg_defField_GetJacobianMatFromFlowField(jacobianMatriceArray,inputTransformation);
         break;
      case SPLINE_GRID:
         reg_spline_GetJacobianMatrix(jacobianImage,inputTransformation,jacobianMatriceArray);
         break;
      case SPLINE_VEL_GRID:
         reg_spline_GetJacobianMatFromVelocityGrid(jacobianMatriceArray,inputTransformation,jacobianImage);
         break;
      }
      switch(jacobianImage->datatype){
      case NIFTI_TYPE_FLOAT32:
         reg_jacobian_convertMat33ToNii<float>(jacobianMatriceArray,jacobianImage);
         break;
      case NIFTI_TYPE_FLOAT64:
         reg_jacobian_convertMat33ToNii<double>(jacobianMatriceArray,jacobianImage);
         break;
      }
      free(jacobianMatriceArray);jacobianMatriceArray=NULL;
      reg_io_WriteImageFile(jacobianImage,param->outputJacMatName);
      nifti_image_free(jacobianImage);jacobianImage=NULL;
   }

   // Free the allocated image
   nifti_image_free(inputTransformation);inputTransformation=NULL;

   return EXIT_SUCCESS;
}

#endif
