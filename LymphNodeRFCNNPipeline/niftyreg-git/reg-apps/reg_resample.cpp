/**
 * @file reg_resample.cpp
 * @author Marc Modat
 * @date 18/05/2009
 *
 *  Created by Marc Modat on 18/05/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _MM_RESAMPLE_CPP
#define _MM_RESAMPLE_CPP

#include <limits>

#include "_reg_ReadWriteImage.h"
#include "_reg_resampling.h"
#include "_reg_globalTransformation.h"
#include "_reg_localTransformation.h"
#include "_reg_tools.h"
#include "reg_resample.h"

typedef struct
{
   char *referenceImageName;
   char *floatingImageName;
   char *inputTransName;
   char *outputResultName;
   char *outputBlankName;
   float sourceBGValue;
   int interpolation;
   float paddingValue;
} PARAM;
typedef struct
{
   bool referenceImageFlag;
   bool floatingImageFlag;
   bool inputTransFlag;
   bool outputResultFlag;
   bool outputBlankFlag;
   bool outputBlankXYFlag;
   bool outputBlankYZFlag;
   bool outputBlankXZFlag;
   bool isTensor;
   bool usePSF;
} FLAG;


void PetitUsage(char *exec)
{
   fprintf(stderr,"Usage:\t%s -ref <referenceImageName> -flo <floatingImageName> [OPTIONS].\n",exec);
   fprintf(stderr,"\tSee the help for more details (-h).\n");
   return;
}
void Usage(char *exec)
{
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   printf("Usage:\t%s -ref <filename> -flo <filename> [OPTIONS].\n",exec);
   printf("\t-ref <filename>\n\t\tFilename of the reference image (mandatory)\n");
   printf("\t-flo <filename>\n\t\tFilename of the floating image (mandatory)\n\n");
   printf("* * OPTIONS * *\n");
   printf("\t-trans <filename>\n\t\tFilename of the file containing the transformation parametrisation (from reg_aladin, reg_f3d or reg_transform)\n");
   printf("\t-res <filename>\n\t\tFilename of the resampled image [none]\n");
   printf("\t-blank <filename>\n\t\tFilename of the resampled blank grid [none]\n");
   printf("\t-inter <int>\n\t\tInterpolation order (0, 1, 3, 4)[3] (0=NN, 1=LIN; 3=CUB, 4=SINC)\n");
   printf("\t-pad <int>\n\t\tInterpolation padding value [0]\n");
   printf("\t-tensor\n\t\tThe last six timepoints of the floating image are considered to be tensor order as XX, XY, YY, XZ, YZ, ZZ [off]\n");
   printf("\t-psf\n\t\tPerform the resampling in two steps to resample an image to a lower resolution [off]\n");
   printf("\t-voff\n\t\tTurns verbose off [on]\n");
#ifdef _GIT_HASH
   printf("\n\t--version\n\t\tPrint current source code git hash key and exit\n\t\t\t\t(%s)\n",_GIT_HASH);
#endif
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   return;
}

int main(int argc, char **argv)
{
   PARAM *param = (PARAM *)calloc(1,sizeof(PARAM));
   FLAG *flag = (FLAG *)calloc(1,sizeof(FLAG));

   param->interpolation=3; // Cubic spline interpolation used by default
   param->paddingValue=0;
   bool verbose=true;

   /* read the input parameter */
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
         printf("%s",xml_resample);
         return 0;
      }
      else if(strcmp(argv[i], "-voff")==0)
      {
         verbose=false;
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
         param->referenceImageName=argv[++i];
         flag->referenceImageFlag=1;
      }
      else if((strcmp(argv[i],"-flo")==0) || (strcmp(argv[i],"-source")==0) ||
              (strcmp(argv[i],"--flo")==0))
      {
         param->floatingImageName=argv[++i];
         flag->floatingImageFlag=1;
      }
      else if((strcmp(argv[i],"-res")==0) || (strcmp(argv[i],"-result")==0) ||
              (strcmp(argv[i],"--res")==0))
      {
         param->outputResultName=argv[++i];
         flag->outputResultFlag=1;
      }
      else if(strcmp(argv[i], "-trans") == 0 ||
              strcmp(argv[i],"--trans")==0 ||
              strcmp(argv[i],"-aff")==0 || // added for backward compatibility
              strcmp(argv[i],"-def")==0 || // added for backward compatibility
              strcmp(argv[i],"-cpp")==0 )  // added for backward compatibility
      {
         param->inputTransName=argv[++i];
         flag->inputTransFlag=1;
      }
      else if(strcmp(argv[i], "-inter") == 0 ||
              (strcmp(argv[i],"--inter")==0))
      {
         param->interpolation=atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-NN") == 0)
      {
         param->interpolation=0;
      }
      else if(strcmp(argv[i], "-LIN") == 0 ||
              (strcmp(argv[i],"-TRI")==0))
      {
         param->interpolation=1;
      }
      else if(strcmp(argv[i], "-CUB") == 0 ||
              (strcmp(argv[i],"-SPL")==0))
      {
         param->interpolation=3;
      }
      else if(strcmp(argv[i], "-SINC") == 0)
      {
         param->interpolation=4;
      }
      else if(strcmp(argv[i], "-pad") == 0 ||
              (strcmp(argv[i],"--pad")==0))
      {
         param->paddingValue=(float)atof(argv[++i]);
      }
      else if(strcmp(argv[i], "-blank") == 0 ||
              (strcmp(argv[i],"--blank")==0))
      {
         param->outputBlankName=argv[++i];
         flag->outputBlankFlag=1;
      }
      else if(strcmp(argv[i], "-blankXY") == 0 ||
              (strcmp(argv[i],"--blankXY")==0))
      {
         param->outputBlankName=argv[++i];
         flag->outputBlankXYFlag=1;
      }
      else if(strcmp(argv[i], "-blankYZ") == 0 ||
              (strcmp(argv[i],"--blankYZ")==0))
      {
         param->outputBlankName=argv[++i];
         flag->outputBlankYZFlag=1;
      }
      else if(strcmp(argv[i], "-blankXZ") == 0 ||
              (strcmp(argv[i],"--blankXZ")==0))
      {
         param->outputBlankName=argv[++i];
         flag->outputBlankXZFlag=1;
      }
      else if(strcmp(argv[i], "-tensor") == 0 ||
              (strcmp(argv[i],"--tensor")==0))
      {
         flag->isTensor=true;
      }
      else if(strcmp(argv[i], "-psf") == 0 ||
              (strcmp(argv[i],"--psf")==0))
      {
         flag->usePSF=true;
      }
      else
      {
         fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
         PetitUsage(argv[0]);
         return 1;
      }
   }

   if(!flag->referenceImageFlag || !flag->floatingImageFlag)
   {
      fprintf(stderr,"[NiftyReg ERROR] The reference and the floating image have both to be defined.\n");
      PetitUsage(argv[0]);
      return 1;
   }

   /* Read the reference image */
   nifti_image *referenceImage = reg_io_ReadImageHeader(param->referenceImageName);
   if(referenceImage == NULL)
   {
      fprintf(stderr,"[NiftyReg ERROR] Error when reading the reference image: %s\n",
              param->referenceImageName);
      return 1;
   }
   reg_checkAndCorrectDimension(referenceImage);

   /* Read the floating image */
   nifti_image *floatingImage = reg_io_ReadImageFile(param->floatingImageName);
   if(floatingImage == NULL)
   {
      fprintf(stderr,"[NiftyReg ERROR] Error when reading the floating image: %s\n",
              param->floatingImageName);
      return 1;
   }
   reg_checkAndCorrectDimension(floatingImage);



   /* *********************************** */
   /* DISPLAY THE RESAMPLING PARAMETERS */
   /* *********************************** */
   if(verbose){
      printf("\n* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
      printf("Command line:\n");
      for(int i=0; i<argc; i++) printf(" %s", argv[i]);
      printf("\n\n");
      printf("Parameters\n");
      printf("Reference image name: %s\n",referenceImage->fname);
      printf("\t%ix%ix%i voxels, %i volumes\n",referenceImage->nx,referenceImage->ny,referenceImage->nz,referenceImage->nt);
      printf("\t%gx%gx%g mm\n",referenceImage->dx,referenceImage->dy,referenceImage->dz);
      printf("Floating image name: %s\n",floatingImage->fname);
      printf("\t%ix%ix%i voxels, %i volumes\n",floatingImage->nx,floatingImage->ny,floatingImage->nz,floatingImage->nt);
      printf("\t%gx%gx%g mm\n",floatingImage->dx,floatingImage->dy,floatingImage->dz);
      printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n\n");
   }


   // Define a higher resolution image of the reference image if required (-psf)
   if(flag->usePSF)
   {
      // Check if the resolution of the input images is relevant
      if(referenceImage->dx<=floatingImage->dx &&
         referenceImage->dy<=floatingImage->dy &&
         referenceImage->dz<=floatingImage->dz){
         reg_print_msg_warn("The -psf argument expect the reference image to be of lower resolution than the floating image.");
         reg_print_msg_warn("Your input images have the reverse charateristics.");
      }
      // First compute the resolution ration between the two input images
      float resolutionRatio[3]={
         referenceImage->dx/floatingImage->dx,
         referenceImage->dy/floatingImage->dy,
         referenceImage->dz/floatingImage->dz
      };
      // Compute the new reference image dimension
      int refNewDim[3]={
         static_cast<int>(reg_ceil(resolutionRatio[0]*static_cast<float>(referenceImage->nx))),
         static_cast<int>(reg_ceil(resolutionRatio[1]*static_cast<float>(referenceImage->ny))),
         static_cast<int>(reg_ceil(resolutionRatio[2]*static_cast<float>(referenceImage->nz)))
      };
      // Update the reference image header
      referenceImage->nvox=(size_t)refNewDim[0]*
            refNewDim[1]*refNewDim[2]*
            referenceImage->nt*referenceImage->nu*
            referenceImage->nv*referenceImage->nw;
      referenceImage->dim[1]=referenceImage->nx=refNewDim[0];
      referenceImage->dim[2]=referenceImage->ny=refNewDim[1];
      referenceImage->dim[3]=referenceImage->nz=refNewDim[2];
      referenceImage->pixdim[1]=referenceImage->dx=floatingImage->dx;
      referenceImage->pixdim[2]=referenceImage->dy=floatingImage->dx;
      referenceImage->pixdim[3]=referenceImage->dz=floatingImage->dx;
      if(referenceImage->qform_code>0)
      {
         // Update the qform matrices
         referenceImage->qto_xyz = nifti_quatern_to_mat44(referenceImage->quatern_b,
         referenceImage->quatern_c,
         referenceImage->quatern_d,
         referenceImage->qoffset_x,
         referenceImage->qoffset_y,
         referenceImage->qoffset_z,
         referenceImage->dx,
         referenceImage->dy,
         referenceImage->dz,
         referenceImage->qfac);
         referenceImage->qto_ijk = nifti_mat44_inverse(referenceImage->qto_xyz);
      }
      if(referenceImage->sform_code>0)
      {
         // Update the sform matrices
         referenceImage->sto_xyz.m[0][0]=referenceImage->sto_xyz.m[0][0] / resolutionRatio[0];
         referenceImage->sto_xyz.m[1][0]=referenceImage->sto_xyz.m[1][0] / resolutionRatio[0];
         referenceImage->sto_xyz.m[2][0]=referenceImage->sto_xyz.m[2][0] / resolutionRatio[0];
         referenceImage->sto_xyz.m[0][1]=referenceImage->sto_xyz.m[0][1] / resolutionRatio[1];
         referenceImage->sto_xyz.m[1][1]=referenceImage->sto_xyz.m[1][1] / resolutionRatio[1];
         referenceImage->sto_xyz.m[2][1]=referenceImage->sto_xyz.m[2][1] / resolutionRatio[1];
         referenceImage->sto_xyz.m[0][2]=referenceImage->sto_xyz.m[0][2] / resolutionRatio[2];
         referenceImage->sto_xyz.m[1][2]=referenceImage->sto_xyz.m[1][2] / resolutionRatio[2];
         referenceImage->sto_xyz.m[2][2]=referenceImage->sto_xyz.m[2][2] / resolutionRatio[2];
         referenceImage->sto_ijk = nifti_mat44_inverse(referenceImage->sto_xyz);
      }
   }

//   // Tell the CLI that the process has started
//   startProgress("reg_resample");
//   // Set up progress indicators
//   float iProgressStep=1, nProgressSteps;

   /* *********************** */
   /* READ THE TRANSFORMATION */
   /* *********************** */
   nifti_image *inputTransformationImage = NULL;
   mat44 inputAffineTransformation;
   // Check if a transformation has been specified
   if(flag->inputTransFlag)
   {
      // First check if the input filename is an image
      if(reg_isAnImageFileName(param->inputTransName))
      {
         inputTransformationImage=reg_io_ReadImageFile(param->inputTransName);
         if(inputTransformationImage==NULL)
         {
            fprintf(stderr, "[NiftyReg ERROR] Error when reading the provided transformation: %s\n",
                    param->inputTransName);
            return 1;
         }
         reg_checkAndCorrectDimension(inputTransformationImage);
      }
      else
      {
         // the transformation is assumed to be affine
         reg_tool_ReadAffineFile(&inputAffineTransformation,
                                 param->inputTransName);
      }
   }
   else
   {
      // No transformation is specified, an identity transformation is used
      reg_mat44_eye(&inputAffineTransformation);
   }

//   // Update progress via CLI
//   progressXML(1, "Transform loaded...");

   // Create a deformation field
   nifti_image *deformationFieldImage = nifti_copy_nim_info(referenceImage);
   deformationFieldImage->dim[0]=deformationFieldImage->ndim=5;
   deformationFieldImage->dim[1]=deformationFieldImage->nx=referenceImage->nx;
   deformationFieldImage->dim[2]=deformationFieldImage->ny=referenceImage->ny;
   deformationFieldImage->dim[3]=deformationFieldImage->nz=referenceImage->nz;
   deformationFieldImage->dim[4]=deformationFieldImage->nt=1;
   deformationFieldImage->pixdim[4]=deformationFieldImage->dt=1.0;
   deformationFieldImage->dim[5]=deformationFieldImage->nu=referenceImage->nz>1?3:2;
   deformationFieldImage->dim[6]=deformationFieldImage->nv=1;
   deformationFieldImage->dim[7]=deformationFieldImage->nw=1;
   deformationFieldImage->nvox =(size_t)deformationFieldImage->nx*
                                deformationFieldImage->ny*deformationFieldImage->nz*
                                deformationFieldImage->nt*deformationFieldImage->nu;
   deformationFieldImage->scl_slope=1.f;
   deformationFieldImage->scl_inter=0.f;
   if(inputTransformationImage!=NULL)
   {
      deformationFieldImage->datatype = inputTransformationImage->datatype;
      deformationFieldImage->nbyper = inputTransformationImage->nbyper;
   }
   else
   {
      deformationFieldImage->datatype = NIFTI_TYPE_FLOAT32;
      deformationFieldImage->nbyper = sizeof(float);
   }
   deformationFieldImage->data = (void *)calloc(deformationFieldImage->nvox, deformationFieldImage->nbyper);

   // Initialise the deformation field with an identity transformation
   reg_tools_multiplyValueToImage(deformationFieldImage,deformationFieldImage,0.f);
   reg_getDeformationFromDisplacement(deformationFieldImage);
   deformationFieldImage->intent_p1=DEF_FIELD;

   // Compute the transformation to apply
   if(inputTransformationImage!=NULL)
   {
      switch(static_cast<int>(inputTransformationImage->intent_p1))
      {
      case SPLINE_GRID:
         reg_spline_getDeformationField(inputTransformationImage,
                                        deformationFieldImage,
                                        NULL,
                                        false,
                                        true);
         break;
      case DISP_VEL_FIELD:
         if(flag->usePSF)
         {
            reg_print_msg_error("The -psf option is not supported yet with a displacement flow as an input transformation.");
            reg_exit(0);
         }
         reg_getDeformationFromDisplacement(inputTransformationImage);
      case DEF_VEL_FIELD:
      {
            if(flag->usePSF)
            {
               reg_print_msg_error("The -psf option is not supported yet with a deformation flow as an input transformation.");
               reg_exit(0);
            }
         nifti_image *tempFlowField = nifti_copy_nim_info(deformationFieldImage);
         tempFlowField->data = (void *)malloc(tempFlowField->nvox*tempFlowField->nbyper);
         memcpy(tempFlowField->data,deformationFieldImage->data,
                tempFlowField->nvox*tempFlowField->nbyper);
         reg_defField_compose(inputTransformationImage,
                              tempFlowField,
                              NULL);
         tempFlowField->intent_p1=inputTransformationImage->intent_p1;
         tempFlowField->intent_p2=inputTransformationImage->intent_p2;
         reg_defField_getDeformationFieldFromFlowField(tempFlowField,
                                                       deformationFieldImage,
                                                       false);
         nifti_image_free(tempFlowField);
      }
      break;
      case SPLINE_VEL_GRID:
         reg_spline_getDefFieldFromVelocityGrid(inputTransformationImage,
               deformationFieldImage,
               false);
         break;
      case DISP_FIELD:
         if(flag->usePSF)
         {
            reg_print_msg_error("The -psf option is not supported yet with a displacement field as an input transformation.");
            reg_exit(0);
         }
         reg_getDeformationFromDisplacement(inputTransformationImage);
      default: // deformation field
         if(flag->usePSF)
         {
            reg_print_msg_error("The -psf option is not supported yet with a deformation field as an input transformation.");
            reg_exit(0);
         }
         reg_defField_compose(inputTransformationImage,
                              deformationFieldImage,
                              NULL);
         break;
      }
      nifti_image_free(inputTransformationImage);
      inputTransformationImage=NULL;
   }
   else
   {
      reg_affine_getDeformationField(&inputAffineTransformation,
                                     deformationFieldImage,
                                     false,
                                     NULL);
   }

//   // Update progress via CLI
//   progressXML(2, "Deformation field ready...");

   /* ************************* */
   /* WARP THE FLOATING IMAGE */
   /* ************************* */
   if(flag->outputResultFlag)
   {
      switch(param->interpolation)
      {
      case 0:
         param->interpolation=0;
         break;
      case 1:
         param->interpolation=1;
         break;
      case 4:
         param->interpolation=4;
         break;
      default:
         param->interpolation=3;
         break;
      }
      nifti_image *warpedImage = nifti_copy_nim_info(referenceImage);
      warpedImage->dim[0]=warpedImage->ndim=floatingImage->dim[0];
      warpedImage->dim[4]=warpedImage->nt=floatingImage->dim[4];
      warpedImage->cal_min=floatingImage->cal_min;
      warpedImage->cal_max=floatingImage->cal_max;
      warpedImage->scl_slope=floatingImage->scl_slope;
      warpedImage->scl_inter=floatingImage->scl_inter;
      if(param->paddingValue==std::numeric_limits<float>::quiet_NaN() &&
         (floatingImage->datatype!=NIFTI_TYPE_FLOAT32 ||
          floatingImage->datatype!=NIFTI_TYPE_FLOAT64)){
         warpedImage->datatype = NIFTI_TYPE_FLOAT32;
         reg_tools_changeDatatype<float>(floatingImage);
      }
      else warpedImage->datatype = floatingImage->datatype;
      warpedImage->nbyper = floatingImage->nbyper;
      warpedImage->nvox = (size_t)warpedImage->dim[1] * (size_t)warpedImage->dim[2] *
                          (size_t)warpedImage->dim[3] * (size_t)warpedImage->dim[4];
      warpedImage->data = (void *)calloc(warpedImage->nvox, warpedImage->nbyper);

      if((floatingImage->dim[4]==6 || floatingImage->dim[4]==7) && flag->isTensor==true)
      {
#ifndef NDEBUG
         printf("[NiftyReg DEBUG] DTI-based resampling\n");
#endif
         // Compute first the Jacobian matrices
         mat33 *jacobian = (mat33 *)malloc(deformationFieldImage->nx *
                                           deformationFieldImage->ny *
                                           deformationFieldImage->nz *
                                           sizeof(mat33));
         reg_defField_getJacobianMatrix(deformationFieldImage,
                                        jacobian);
         // resample the DTI image
         bool timepoints[7];
         for(int i=0; i<7; ++i) timepoints[i]=true;
         if(floatingImage->dim[4]==7) timepoints[0]=false;
         reg_resampleImage(floatingImage,
                           warpedImage,
                           deformationFieldImage,
                           NULL,
                           param->interpolation,
                           std::numeric_limits<float>::quiet_NaN(),
                           timepoints,
                           jacobian
                          );
      }
      else
      {
         reg_resampleImage(floatingImage,
                           warpedImage,
                           deformationFieldImage,
                           NULL,
                           param->interpolation,
                           param->paddingValue);
      }

      // Resample the high resolution warped image to the native reference image resolution
      if(flag->usePSF)
      {
         // Read the original reference image
         nifti_image *origRefImage = reg_io_ReadImageHeader(param->referenceImageName);
         // The warped image is first convolved with a Gaussian kernel
         bool *timePoint = new bool[warpedImage->nt*warpedImage->nu];
         for(int i=0; i<warpedImage->nt*warpedImage->nu; ++i) timePoint[i]=true;

         if(param->interpolation>0){
             float *kernelSize = new float[warpedImage->nt*warpedImage->nu];
             bool boolX[3]= {1,0,0};
             for(int i=0; i<warpedImage->nt*warpedImage->nu; ++i)
                 kernelSize[i]=sqrt(reg_pow2(origRefImage->dx/(2.f*sqrt(2.f*log(2.f))))-reg_pow2(referenceImage->dx/(2.f*sqrt(2.f*log(2.f)))));
             reg_tools_kernelConvolution(warpedImage,kernelSize,0,NULL,timePoint,boolX);
             bool boolY[3]= {0,1,0};
             for(int i=0; i<warpedImage->nt*warpedImage->nu; ++i)
                 kernelSize[i]=sqrt(reg_pow2(origRefImage->dy/(2.f*sqrt(2.f*log(2.f))))-reg_pow2(referenceImage->dy/(2.f*sqrt(2.f*log(2.f)))));
             reg_tools_kernelConvolution(warpedImage,kernelSize,0,NULL,timePoint,boolY);
             bool boolZ[3]= {0,0,1};
             for(int i=0; i<warpedImage->nt*warpedImage->nu; ++i)
                 kernelSize[i]=sqrt(reg_pow2(origRefImage->dz/(2.f*sqrt(2.f*log(2.f))))-reg_pow2(referenceImage->dz/(2.f*sqrt(2.f*log(2.f)))));
             reg_tools_kernelConvolution(warpedImage,kernelSize,0,NULL,timePoint,boolZ);
             delete []kernelSize;
         }
         else{
             float kernelVarianceX=reg_pow2(origRefImage->dx/(2.f*sqrt(2.f*log(2.f))))-reg_pow2(referenceImage->dx/(2.f*sqrt(2.f*log(2.f))));
             float kernelVarianceY=reg_pow2(origRefImage->dy/(2.f*sqrt(2.f*log(2.f))))-reg_pow2(referenceImage->dy/(2.f*sqrt(2.f*log(2.f))));
             float kernelVarianceZ=reg_pow2(origRefImage->dz/(2.f*sqrt(2.f*log(2.f))))-reg_pow2(referenceImage->dz/(2.f*sqrt(2.f*log(2.f))));
             reg_tools_labelKernelConvolution(warpedImage,kernelVarianceX,kernelVarianceY,kernelVarianceZ,NULL,timePoint);
         }
         delete []timePoint;

         // A new warped image is created based on the origin reference image
         nifti_image *origWarpedImage = nifti_copy_nim_info(origRefImage);
         origWarpedImage->dim[0]=origWarpedImage->ndim=floatingImage->dim[0];
         origWarpedImage->dim[4]=origWarpedImage->nt=floatingImage->dim[4];
         origWarpedImage->cal_min=floatingImage->cal_min;
         origWarpedImage->cal_max=floatingImage->cal_max;
         origWarpedImage->scl_slope=floatingImage->scl_slope;
         origWarpedImage->scl_inter=floatingImage->scl_inter;
         origWarpedImage->datatype = floatingImage->datatype;
         origWarpedImage->nbyper = floatingImage->nbyper;
         origWarpedImage->nvox = (size_t)origWarpedImage->dim[1] * origWarpedImage->dim[2] *
                             origWarpedImage->dim[3] * origWarpedImage->dim[4];
         origWarpedImage->data = (void *)calloc(origWarpedImage->nvox, origWarpedImage->nbyper);
         // An identity deformation field is created
         nifti_image *origDefFieldImage = nifti_copy_nim_info(origRefImage);
         origDefFieldImage->dim[0]=origDefFieldImage->ndim=5;
         origDefFieldImage->dim[1]=origDefFieldImage->nx=origDefFieldImage->nx;
         origDefFieldImage->dim[2]=origDefFieldImage->ny=origDefFieldImage->ny;
         origDefFieldImage->dim[3]=origDefFieldImage->nz=origDefFieldImage->nz;
         origDefFieldImage->dim[4]=origDefFieldImage->nt=1;
         origDefFieldImage->pixdim[4]=origDefFieldImage->dt=1.0;
         origDefFieldImage->dim[5]=origDefFieldImage->nu=origDefFieldImage->nz>1?3:2;
         origDefFieldImage->dim[6]=origDefFieldImage->nv=1;
         origDefFieldImage->dim[7]=origDefFieldImage->nw=1;
         origDefFieldImage->nvox =(size_t)origDefFieldImage->nx*
                                      origDefFieldImage->ny*origDefFieldImage->nz*
                                      origDefFieldImage->nt*origDefFieldImage->nu;
         origDefFieldImage->scl_slope=1.f;
         origDefFieldImage->scl_inter=0.f;
         origDefFieldImage->datatype = NIFTI_TYPE_FLOAT32;
         origDefFieldImage->nbyper = sizeof(float);
         origDefFieldImage->data = (void *)calloc(origDefFieldImage->nvox, origDefFieldImage->nbyper);
         reg_getDeformationFromDisplacement(origDefFieldImage);
         // The high resolution warped image is resampled into the low resolution warped image
          if(param->interpolation>0){
         reg_resampleImage(warpedImage,
                           origWarpedImage,
                           origDefFieldImage,
                           NULL,
                           1, // linear interpolation
                           0 // padding value set to 0 since field of view are aligned
                           );
          }else{
              reg_resampleImage(warpedImage,
                                origWarpedImage,
                                origDefFieldImage,
                                NULL,
                                0, // Nearest Neighbour
                                0 // padding value set to 0 since field of view are aligned
                                );

          }
         memset(origWarpedImage->descrip, 0, 80);
         strcpy (origWarpedImage->descrip,"Warped image using NiftyReg (reg_resample)");
         reg_io_WriteImageFile(origWarpedImage,param->outputResultName);
         nifti_image_free(origWarpedImage);
         nifti_image_free(origRefImage);
         nifti_image_free(origDefFieldImage);
      }
      else{
         memset(warpedImage->descrip, 0, 80);
         strcpy (warpedImage->descrip,"Warped image using NiftyReg (reg_resample)");
         reg_io_WriteImageFile(warpedImage,param->outputResultName);
      }
      if(verbose)
         printf("[NiftyReg] Resampled image has been saved: %s\n", param->outputResultName);
      nifti_image_free(warpedImage);
   }

   /* *********************** */
   /* RESAMPLE A REGULAR GRID */
   /* *********************** */
   if(flag->outputBlankFlag ||
         flag->outputBlankXYFlag ||
         flag->outputBlankYZFlag ||
         flag->outputBlankXZFlag )
   {
      nifti_image *gridImage = nifti_copy_nim_info(floatingImage);
      gridImage->cal_min=0;
      gridImage->cal_max=255;
      gridImage->scl_slope=1.f;
      gridImage->scl_inter=0.f;
      gridImage->dim[0]=gridImage->ndim=floatingImage->nz>1?3:2;
      gridImage->dim[1]=gridImage->nx=floatingImage->nx;
      gridImage->dim[2]=gridImage->ny=floatingImage->ny;
      gridImage->dim[3]=gridImage->nz=floatingImage->nz;
      gridImage->dim[4]=gridImage->nt=1;
      gridImage->dim[5]=gridImage->nu=1;
      gridImage->nvox=(size_t)gridImage->nx*
            gridImage->ny*gridImage->nz;
      gridImage->datatype = NIFTI_TYPE_UINT8;
      gridImage->nbyper = sizeof(unsigned char);
      gridImage->data = (void *)calloc(gridImage->nvox, gridImage->nbyper);
      unsigned char *gridImageValuePtr = static_cast<unsigned char *>(gridImage->data);
      for(int z=0; z<gridImage->nz; z++)
      {
         for(int y=0; y<gridImage->ny; y++)
         {
            for(int x=0; x<gridImage->nx; x++)
            {
               if(referenceImage->nz>1)
               {
                  if(flag->outputBlankXYFlag)
                  {
                     if( x/10==(float)x/10.0 || y/10==(float)y/10.0)
                        *gridImageValuePtr = 255;
                  }
                  else if(flag->outputBlankYZFlag)
                  {
                     if( y/10==(float)y/10.0 || z/10==(float)z/10.0)
                        *gridImageValuePtr = 255;
                  }
                  else if(flag->outputBlankXZFlag)
                  {
                     if( x/10==(float)x/10.0 || z/10==(float)z/10.0)
                        *gridImageValuePtr = 255;
                  }
                  else
                  {
                     if( x/10==(float)x/10.0 || y/10==(float)y/10.0 || z/10==(float)z/10.0)
                        *gridImageValuePtr = 255;
                  }
               }
               else
               {
                  if( x/10==(float)x/10.0 || x==referenceImage->nx-1 || y/10==(float)y/10.0 || y==referenceImage->ny-1)
                     *gridImageValuePtr = 255;
               }
               gridImageValuePtr++;
            }
         }
      }

      nifti_image *warpedImage = nifti_copy_nim_info(referenceImage);
      warpedImage->cal_min=0;
      warpedImage->cal_max=255;
      warpedImage->scl_slope=1.f;
      warpedImage->scl_inter=0.f;
      warpedImage->dim[0]=warpedImage->ndim=referenceImage->nz>1?3:2;
      warpedImage->dim[1]=warpedImage->nx=referenceImage->nx;
      warpedImage->dim[2]=warpedImage->ny=referenceImage->ny;
      warpedImage->dim[3]=warpedImage->nz=referenceImage->nz;
      warpedImage->dim[4]=warpedImage->nt=1;
      warpedImage->dim[5]=warpedImage->nu=1;
      warpedImage->datatype =NIFTI_TYPE_UINT8;
      warpedImage->nbyper = sizeof(unsigned char);
      warpedImage->data = (void *)calloc(warpedImage->nvox,
                                         warpedImage->nbyper);
      reg_resampleImage(gridImage,
                        warpedImage,
                        deformationFieldImage,
                        NULL,
                        1, // linear interpolation
                        0);
      memset(warpedImage->descrip, 0, 80);
      strcpy (warpedImage->descrip,"Warped regular grid using NiftyReg (reg_resample)");
      reg_io_WriteImageFile(warpedImage,param->outputBlankName);
      nifti_image_free(warpedImage);
      nifti_image_free(gridImage);
      if(verbose)
         printf("[NiftyReg] Resampled grid has been saved: %s\n", param->outputBlankName);
   }

//   // Tell the CLI that we finished
//   closeProgress("reg_resample", "Normal exit");

   nifti_image_free(referenceImage);
   nifti_image_free(floatingImage);
   nifti_image_free(deformationFieldImage);

   free(flag);
   free(param);
   return 0;
}

#endif
