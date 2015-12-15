/*
 *  reg_transform.cpp
 *
 *
 *  Created by Marc Modat on 08/11/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _MM_TRANSFORM_CPP
#define _MM_TRANSFORM_CPP

#include "_reg_ReadWriteImage.h"
#include "_reg_resampling.h"
#include "_reg_globalTransformation.h"
#include "_reg_localTransformation.h"
#include "_reg_tools.h"
#include "_reg_thinPlateSpline.h"

#include "reg_transform.h"

#include <fstream>
#include <vector>
#include <iostream>

typedef struct
{
   char *referenceImageName;
   char *referenceImage2Name;
   char *inputTransName;
   char *input2TransName;
   char *outputTransName;
   float affTransParam[12];
} PARAM;
typedef struct
{
   bool referenceImageFlag;
   bool referenceImage2Flag;
   bool outputDefFlag;
   bool outputDispFlag;
   bool outputFlowFlag;
   bool outputCompFlag;
   bool updSFormFlag;
   bool halfTransFlag;
   bool invertAffFlag;
   bool invertNRRFlag;
   bool flirtAff2NRFlag;
   bool makeAffFlag;
   bool aff2rigFlag;
} FLAG;


void PetitUsage(char *exec)
{
   fprintf(stderr,"Usage:\t%s [OPTIONS].\n",exec);
   fprintf(stderr,"\tSee the help for more details (-h).\n");
   return;
}
void Usage(char *exec)
{
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   printf("Usage:\t%s [OPTIONS].\n",exec);
   printf("* * OPTIONS * *\n\n");

   printf("\t-ref <filename>\n");
   printf("\t\tFilename of the reference image\n");
   printf("\t\tThe Reference image has to be specified when a cubic B-Spline parametrised control point grid is used*.\n");
   printf("\t-ref2 <filename>\n");
   printf("\t\tFilename of the second reference image to be used when dealing with composition\n\n");

   printf("\t-def <filename1> <filename2>\n");
   printf("\t\tTake a transformation of any recognised type* and compute the corresponding deformation field\n");
   printf("\t\tfilename1 - Input transformation file name\n");
   printf("\t\tfilename2 - Output deformation field file name\n\n");

   printf("\t-disp <filename1> <filename2>\n");
   printf("\t\tTake a transformation of any recognised type* and compute the corresponding displacement field\n");
   printf("\t\tfilename1 - Input transformation file name\n");
   printf("\t\tfilename2 - Output displacement field file name\n\n");

   printf("\t-flow <filename1> <filename2>\n");
   printf("\t\tTake a spline parametrised SVF and compute the corresponding flow field\n");
   printf("\t\tfilename1 - Input transformation file name\n");
   printf("\t\tfilename2 - Output flow field file name\n\n");

   printf("\t-comp <filename1> <filename2> <filename3>\n");
   printf("\t\tCompose two transformations of any recognised type* and returns a deformation field.\n");
   printf("\t\tTrans3(x) = Trans2(Trans1(x)).\n");
   printf("\t\tfilename1 - Input transformation 1 file name (associated with -ref if required)\n");
   printf("\t\tfilename2 - Input transformation 2 file name (associated with -ref2 if required)\n");
   printf("\t\tfilename3 - Output deformation field file name\n\n");

   printf("\t-updSform <filename1> <filename2> <filename3>\n");
   printf("\t\tUpdate the sform of an image using an affine transformation.\n");
   printf("\t\tFilename1 - Image to be updated\n");
   printf("\t\tFilename2 - Affine transformation defined as Affine x Reference = Floating\n");
   printf("\t\tFilename3 - Updated image.\n\n");

   printf("\t-invAff <filename1> <filename2>\n");
   printf("\t\tInvert an affine matrix.\n");
   printf("\t\tfilename1 - Input affine transformation file name\n");
   printf("\t\tfilename2 - Output inverted affine transformation file name\n\n");

   printf("\t-invNrr <filename1> <filename2> <filename3>\n");
   printf("\t\tInvert a non-rigid transformation and save the result as a deformation field.\n");
   printf("\t\tfilename1 - Input transformation file name\n");
   printf("\t\tfilename2 - Input floating (source) image where the inverted transformation is defined\n");
   printf("\t\tfilename3 - Output inverted transformation file name\n");
   printf("\t\tNote that the cubic b-spline grid parametrisations can not be inverted without approximation,\n");
   printf("\t\tas a result, they are converted into deformation fields before inversion.\n\n");

   printf("\t-half <filename1> <filename2>\n");
   printf("\t\tThe input transformation is halfed and stored using the same transformation type.\n");
   printf("\t\tfilename1 - Input transformation file name\n");
   printf("\t\tfilename2 - Output transformation file name\n\n");

   printf("\t-makeAff <rx> <ry> <rz> <tx> <ty> <tz> <sx> <sy> <sz> <shx> <shy> <shz> <outputFilename>\n");
   printf("\t\tCreate an affine transformation matrix\n\n");

   printf("\t-aff2rig <filename1> <filename2>\n");
   printf("\t\tExtract the rigid component from an affine transformation matrix\n");
   printf("\t\tfilename1 - Input transformation file name\n");
   printf("\t\tfilename2 - Output transformation file name\n\n");

   printf("\t-flirtAff2NR <filename1> <filename2> <filename3> <filename4>\n");
   printf("\t\tConvert a flirt (FSL) affine transformation to a NiftyReg affine transformation\n");
   printf("\t\tfilename1 - Input FLIRT (FSL) affine transformation file name\n");
   printf("\t\tfilename2 - Image used as a reference (-ref arg in FLIRT)\n");
   printf("\t\tfilename3 - Image used as a floating (-in arg in FLIRT)\n");
   printf("\t\tfilename4 - Output affine transformation file name\n\n");

#ifdef _GIT_HASH
   printf("\n\t--version\t\tPrint current source code git hash key and exit\n\t\t\t\t(%s)\n",_GIT_HASH);
#endif

   printf("\t* The supported transformation types are:\n");
   printf("\t\t- cubic B-Spline parametrised grid (reference image is required)\n");
   printf("\t\t- a dense deformation field\n");
   printf("\t\t- a dense displacement field\n");
   printf("\t\t- a cubic B-Spline parametrised stationary velocity field (reference image is required)\n");
   printf("\t\t- a stationary velocity deformation field\n");
   printf("\t\t- a stationary velocity displacement field\n");
   printf("\t\t- an affine matrix\n\n");
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   return;
}

int main(int argc, char **argv)
{
   // Display the help if no arguments are provided
   if(argc==1)
   {
      PetitUsage(argv[0]);
      return 0;
   }

   // Set the variables used to store the parsed data
   PARAM *param = (PARAM *)calloc(1,sizeof(PARAM));
   FLAG *flag = (FLAG *)calloc(1,sizeof(FLAG));

   // Parse the input data
   for(int i=1; i<argc; ++i)
   {
      if(strcmp(argv[i],"-h")==0 || strcmp(argv[i],"--h")==0 ||
            strcmp(argv[i],"-H")==0 || strcmp(argv[i],"--H")==0 ||
            strcmp(argv[i],"-help")==0 || strcmp(argv[i],"--help")==0 ||
            strcmp(argv[i],"-Help")==0 || strcmp(argv[i],"--Help")==0 ||
            strcmp(argv[i],"-HELP")==0 || strcmp(argv[i],"--HELP")==0)
      {
         free(param);
         free(flag);
         Usage(argv[0]);
         return 0;
      }
#ifdef _GIT_HASH
      else if(strcmp(argv[i], "-version")==0 || strcmp(argv[i], "-Version")==0 ||
            strcmp(argv[i], "-V")==0 || strcmp(argv[i], "-v")==0 ||
            strcmp(argv[i], "--v")==0 || strcmp(argv[i], "--version")==0)
      {
         printf("%s\n",_GIT_HASH);
         return EXIT_SUCCESS;
      }
#endif
      else if(strcmp(argv[i],"-ref")==0 || strcmp(argv[i],"--ref")==0 || strcmp(argv[i],"-target")==0)
      {
         flag->referenceImageFlag=true;
         param->referenceImageName=argv[++i];
      }
      else if(strcmp(argv[i],"-ref2")==0 || strcmp(argv[i],"--ref2")==0 || strcmp(argv[i],"-target2")==0)
      {
         flag->referenceImage2Flag=true;
         param->referenceImage2Name=argv[++i];
      }
      else if(strcmp(argv[i],"-def")==0 || strcmp(argv[i],"--def")==0)
      {
         flag->outputDefFlag=true;
         param->inputTransName=argv[++i];
         param->outputTransName=argv[++i];
      }
      else if(strcmp(argv[i],"-disp")==0 || strcmp(argv[i],"--disp")==0)
      {
         flag->outputDispFlag=true;
         param->inputTransName=argv[++i];
         param->outputTransName=argv[++i];
      }
      else if(strcmp(argv[i],"-flow")==0 || strcmp(argv[i],"--flow")==0)
      {
         flag->outputFlowFlag=true;
         param->inputTransName=argv[++i];
         param->outputTransName=argv[++i];
      }
      else if(strcmp(argv[i],"-comp")==0 || strcmp(argv[i],"--comp")==0)
      {
         flag->outputCompFlag=true;
         param->inputTransName=argv[++i];
         param->input2TransName=argv[++i];
         param->outputTransName=argv[++i];
      }
      else if(strcmp(argv[i],"-updSform")==0 || strcmp(argv[i],"--comp")==0)
      {
         flag->updSFormFlag=true;
         param->inputTransName=argv[++i];
         param->input2TransName=argv[++i];
         param->outputTransName=argv[++i];
      }
      else if(strcmp(argv[i],"-half")==0 || strcmp(argv[i],"--half")==0)
      {
         flag->halfTransFlag=true;
         param->inputTransName=argv[++i];
         param->outputTransName=argv[++i];
      }
      else if(strcmp(argv[i],"-invAff")==0 || strcmp(argv[i],"--invAff")==0 ||
              strcmp(argv[i],"-invAffine")==0 || strcmp(argv[i],"--invAffine")==0)
      {
         flag->invertAffFlag=true;
         param->inputTransName=argv[++i];
         param->outputTransName=argv[++i];
      }
      else if(strcmp(argv[i],"-invNrr")==0 || strcmp(argv[i],"--invNrr")==0)
      {
         flag->invertNRRFlag=true;
         param->inputTransName=argv[++i];
         param->input2TransName=argv[++i];
         param->outputTransName=argv[++i];
      }
      else if(strcmp(argv[i],"-makeAff")==0 || strcmp(argv[i],"--makeAff")==0)
      {
         flag->makeAffFlag=true;
         for(int j=0; j<12; ++j)
            param->affTransParam[j]=static_cast<float>(atof(argv[++i]));
         param->outputTransName=argv[++i];
      }
      else if(strcmp(argv[i],"-aff2rig")==0 || strcmp(argv[i],"--aff2rig")==0)
      {
         flag->aff2rigFlag=true;
         param->inputTransName=argv[++i];
         param->outputTransName=argv[++i];
      }
      else if(strcmp(argv[i],"-flirtAff2NR")==0 || strcmp(argv[i],"--flirtAff2NR")==0)
      {
         flag->flirtAff2NRFlag=true;
         param->inputTransName=argv[++i];
         param->referenceImageName=argv[++i];
         param->referenceImage2Name=argv[++i];
         param->outputTransName=argv[++i];
      }
      else
      {
         fprintf(stderr, "[NiftyReg ERROR] Unrecognised argument: %s\n",
                 argv[i]);
         return 1;
      }
   }

   /* ********************************************** */
   // Generate the deformation or displacement field //
   /* ********************************************** */
   if(flag->outputDefFlag || flag->outputDispFlag || flag->outputFlowFlag)
   {
      // Create some variables
      mat44 *affineTransformation=NULL;
      nifti_image *referenceImage=NULL;
      nifti_image *inputTransformationImage=NULL;
      nifti_image *outputTransformationImage=NULL;
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
         // If the input transformation is a grid, check that the reference image has been specified
         if(inputTransformationImage->intent_p1==SPLINE_GRID ||
               inputTransformationImage->intent_p1==SPLINE_VEL_GRID)
         {
            if(!flag->referenceImageFlag)
            {
               fprintf(stderr, "[NiftyReg ERROR] When using a control point grid parametrisation (%s),",
                       param->inputTransName);
               fprintf(stderr, " a reference image shoud be specified (-ref flag).\n");
               return 1;
            }
            referenceImage=reg_io_ReadImageHeader(param->referenceImageName);
            if(referenceImage==NULL)
            {
               fprintf(stderr, "[NiftyReg ERROR] Error when reading the reference image: %s\n",
                       param->referenceImageName);
               return 1;
            }
            reg_checkAndCorrectDimension(referenceImage);
         }
      }
      else
      {
         // Read the affine transformation
         affineTransformation=(mat44 *)malloc(sizeof(mat44));
         reg_tool_ReadAffineFile(affineTransformation,param->inputTransName);
         if(!flag->referenceImageFlag)
         {
            fprintf(stderr, "[NiftyReg ERROR] When using an affine transformation (%s),",
                    param->inputTransName);
            fprintf(stderr, " a reference image shoud be specified (-ref flag).\n");
            return 1;
         }
         referenceImage=reg_io_ReadImageHeader(param->referenceImageName);
         if(referenceImage==NULL)
         {
            fprintf(stderr, "[NiftyReg ERROR] Error when reading the reference image: %s\n",
                    param->referenceImageName);
            return 1;
         }
         reg_checkAndCorrectDimension(referenceImage);
      }
      // Create a dense field
      if(affineTransformation!=NULL ||
            inputTransformationImage->intent_p1==SPLINE_GRID ||
            inputTransformationImage->intent_p1==SPLINE_VEL_GRID)
      {
         // Create a field image from the reference image
         outputTransformationImage=nifti_copy_nim_info(referenceImage);
         outputTransformationImage->ndim=outputTransformationImage->dim[0]=5;
         outputTransformationImage->nt=outputTransformationImage->dim[4]=1;
         outputTransformationImage->nu=outputTransformationImage->dim[5]=outputTransformationImage->nz>1?3:2;
         outputTransformationImage->nvox=(size_t)outputTransformationImage->nx *
                                         outputTransformationImage->ny * outputTransformationImage->nz *
                                         outputTransformationImage->nt * outputTransformationImage->nu;
         outputTransformationImage->nbyper=sizeof(float);
         outputTransformationImage->datatype=NIFTI_TYPE_FLOAT32;
         outputTransformationImage->intent_code=NIFTI_INTENT_VECTOR;
         memset(outputTransformationImage->intent_name, 0, 16);
         strcpy(outputTransformationImage->intent_name,"NREG_TRANS");
         outputTransformationImage->scl_slope=1.f;
         outputTransformationImage->scl_inter=0.f;
      }
      else
      {
         // Create a deformation field from in the input transformation
         outputTransformationImage=nifti_copy_nim_info(inputTransformationImage);
      }
      // Allocate the output field data array
      outputTransformationImage->data=(void *)malloc
                                      (outputTransformationImage->nvox*outputTransformationImage->nbyper);
      // Create a flow field image
      if(flag->outputFlowFlag)
      {
         if(affineTransformation!=NULL)
         {
            fprintf(stderr,"[NiftyReg ERROR] A flow field transformation can not be generated from an affine transformation\n");
            return 1;
         }
         if(inputTransformationImage->intent_p1==SPLINE_GRID)
         {
            fprintf(stderr,"[NiftyReg ERROR] A flow field transformation can not be generated from a cubic spline grid\n");
            return 1;
         }
         if(inputTransformationImage->intent_p1==DEF_FIELD)
         {
            fprintf(stderr,"[NiftyReg ERROR] A flow field transformation can not be generated from a deformation field\n");
            return 1;
         }
         if(inputTransformationImage->intent_p1==DISP_FIELD)
         {
            fprintf(stderr,"[NiftyReg ERROR] A flow field transformation can not be generated from a displacement field\n");
            return 1;
         }
         switch(static_cast<int>(inputTransformationImage->intent_p1))
         {
            break;
         case DEF_VEL_FIELD:
            printf("[NiftyReg] The specified transformation is a deformation velocity field:\n[NiftyReg] %s\n",
                   inputTransformationImage->fname);
            // The current input transformation is copied
            memcpy(outputTransformationImage->data,inputTransformationImage->data,
                   outputTransformationImage->nvox*outputTransformationImage->nbyper);
            break;
         case DISP_VEL_FIELD:
            printf("[NiftyReg] The specified transformation is a displacement velocity field:\n[NiftyReg] %s\n",
                   inputTransformationImage->fname);
            // The current input transformation is copied and converted
            memcpy(outputTransformationImage->data,inputTransformationImage->data,
                   outputTransformationImage->nvox*outputTransformationImage->nbyper);
            reg_getDisplacementFromDeformation(outputTransformationImage);
            break;
         case SPLINE_VEL_GRID:
            printf("[NiftyReg] The specified transformation is a spline velocity parametrisation:\n[NiftyReg] %s\n",
                   inputTransformationImage->fname);
            reg_spline_getFlowFieldFromVelocityGrid(inputTransformationImage,
                                                    outputTransformationImage);
            break;
         default:
            fprintf(stderr,"[NiftyReg ERROR] Unknown input transformation type\n");
            return 1;
         }
         outputTransformationImage->intent_p1=DEF_VEL_FIELD;
         outputTransformationImage->intent_p2=inputTransformationImage->intent_p2;
      }
      // Create a deformation or displacement field
      else if(flag->outputDefFlag || flag->outputDispFlag)
      {
         if(affineTransformation!=NULL)
         {
            reg_affine_getDeformationField(affineTransformation,outputTransformationImage);
         }
         else
         {
            switch(static_cast<int>(reg_round(inputTransformationImage->intent_p1)))
            {
            case DEF_FIELD:
               printf("[NiftyReg] The specified transformation is a deformation field:\n[NiftyReg] %s\n",
                      inputTransformationImage->fname);
               // the current in transformation is copied
               memcpy(outputTransformationImage->data,inputTransformationImage->data,
                      outputTransformationImage->nvox*outputTransformationImage->nbyper);
               break;
            case DISP_FIELD:
               printf("[NiftyReg] The specified transformation is a displacement field:\n[NiftyReg] %s\n",
                      inputTransformationImage->fname);
               // the current in transformation is copied and converted
               memcpy(outputTransformationImage->data,inputTransformationImage->data,
                      outputTransformationImage->nvox*outputTransformationImage->nbyper);
               reg_getDeformationFromDisplacement(outputTransformationImage);
               break;
            case SPLINE_GRID:
               printf("[NiftyReg] The specified transformation is a spline parametrisation:\n[NiftyReg] %s\n",
                      inputTransformationImage->fname);
               // The output field is filled with an identity deformation field
               memset(outputTransformationImage->data,
                      0,
                      outputTransformationImage->nvox*outputTransformationImage->nbyper);
               reg_getDeformationFromDisplacement(outputTransformationImage);
               // The spline transformation is composed with the identity field
               reg_spline_getDeformationField(inputTransformationImage,
                                              outputTransformationImage,
                                              NULL, // no mask
                                              true, // composition is used,
                                              true // b-spline are used
                                             );
               break;
            case DEF_VEL_FIELD:
               printf("[NiftyReg] The specified transformation is a deformation velocity field:\n[NiftyReg] %s\n",
                      inputTransformationImage->fname);
               // The flow field is exponentiated
               reg_defField_getDeformationFieldFromFlowField(inputTransformationImage,
                     outputTransformationImage,
                     false // step number is not updated
                                                            );
               break;
            case DISP_VEL_FIELD:
               printf("[NiftyReg] The specified transformation is a displacement velocity field:\n[NiftyReg] %s\n",
                      inputTransformationImage->fname);
               // The input transformation is converted into a def flow
               reg_getDeformationFromDisplacement(outputTransformationImage);
               // The flow field is exponentiated
               reg_defField_getDeformationFieldFromFlowField(inputTransformationImage,
                     outputTransformationImage,
                     false // step number is not updated
                                                            );
               break;
            case SPLINE_VEL_GRID:
               printf("[NiftyReg] The specified transformation is a spline velocity parametrisation:\n[NiftyReg] %s\n",
                      inputTransformationImage->fname);
               // The spline parametrisation is converted into a dense flow and exponentiated
               reg_spline_getDefFieldFromVelocityGrid(inputTransformationImage,
                     outputTransformationImage,
                     false // step number is not updated
                                                             );
               break;
            default:
               fprintf(stderr,"[NiftyReg ERROR] Unknown input transformation type\n");
               return 1;
            }
         }
         outputTransformationImage->intent_p1=DEF_FIELD;
         outputTransformationImage->intent_p2=0;
         if(flag->outputDispFlag)
            reg_getDisplacementFromDeformation(outputTransformationImage);
      }
      // Save the generated transformation
      reg_io_WriteImageFile(outputTransformationImage,param->outputTransName);
      switch(static_cast<int>(round(outputTransformationImage->intent_p1)))
      {
      case DEF_FIELD:
         printf("[NiftyReg] The deformation field has been saved as:\n[NiftyReg] %s\n",
                param->outputTransName);
         break;
      case DISP_FIELD:
         printf("[NiftyReg] The displacement field has been saved as:\n[NiftyReg] %s\n",
                param->outputTransName);
         break;
      case DEF_VEL_FIELD:
         printf("[NiftyReg] The flow field has been saved as:\n[NiftyReg] %s\n",
                param->outputTransName);
         break;
      }
      // Free the allocated images and arrays
      if(affineTransformation!=NULL) free(affineTransformation);
      if(referenceImage!=NULL) nifti_image_free(referenceImage);
      if(inputTransformationImage!=NULL) nifti_image_free(inputTransformationImage);
      if(outputTransformationImage!=NULL) nifti_image_free(outputTransformationImage);
   }

   /* ************************************ */
   // Start the transformation composition //
   /* ************************************ */
   if(flag->outputCompFlag)
   {
      printf("[NiftyReg] Starting the composition of two transformations\n");
      // Create some variables
      mat44 *affine1Trans=NULL;
      mat44 *affine2Trans=NULL;
      nifti_image *referenceImage=NULL;
      nifti_image *referenceImage2=NULL;
      nifti_image *input1TransImage=NULL;
      nifti_image *input2TransImage=NULL;
      nifti_image *output1TransImage=NULL;
      nifti_image *output2TransImage=NULL;
      // Read the first transformation
      if(!reg_isAnImageFileName(param->inputTransName))
      {
         affine1Trans=(mat44 *)malloc(sizeof(mat44));
         reg_tool_ReadAffineFile(affine1Trans,param->inputTransName);
         printf("[NiftyReg] Transformation 1 is an affine parametrisation:\n[NiftyReg] %s\n",
                param->inputTransName);
      }
      else
      {
         input1TransImage = reg_io_ReadImageFile(param->inputTransName);
         if(input1TransImage==NULL)
         {
            fprintf(stderr, "[NiftyReg ERROR] Error when reading the transformation image: %s\n",
                    param->inputTransName);
            return 1;
         }
         reg_checkAndCorrectDimension(input1TransImage);
      }
      // Read the second transformation
      if(!reg_isAnImageFileName(param->input2TransName))
      {
         affine2Trans=(mat44 *)malloc(sizeof(mat44));
         reg_tool_ReadAffineFile(affine2Trans,param->input2TransName);
      }
      else
      {
         input2TransImage = reg_io_ReadImageFile(param->input2TransName);
         if(input2TransImage==NULL)
         {
            fprintf(stderr, "[NiftyReg ERROR] Error when reading the transformation image: %s\n",
                    param->input2TransName);
            return 1;
         }
         reg_checkAndCorrectDimension(input2TransImage);
      }
      // Check if the two input transformations are affine transformation
      if(affine1Trans!=NULL && affine2Trans!=NULL)
      {
         printf("[NiftyReg] Transformation 2 is an affine parametrisation:\n[NiftyReg] %s\n",
                param->input2TransName);
         *affine1Trans=reg_mat44_mul(affine2Trans,affine1Trans);
         reg_tool_WriteAffineFile(affine1Trans,param->outputTransName);
      }
      else
      {
         // Check if the reference image is required
         if(affine1Trans!=NULL)
         {
            if(!flag->referenceImageFlag)
            {
               fprintf(stderr, "[NiftyReg ERROR] When using an affine transformation (%s),",
                       param->inputTransName);
               fprintf(stderr, " a reference image shoud be specified (-res flag).\n");
               return 1;
            }
            referenceImage=reg_io_ReadImageHeader(param->referenceImageName);
            if(referenceImage==NULL)
            {
               fprintf(stderr, "[NiftyReg ERROR] Error when reading the reference image: %s\n",
                       param->referenceImageName);
               return 1;
            }
            reg_checkAndCorrectDimension(referenceImage);
         }
         else if(input1TransImage->intent_p1==SPLINE_GRID || input1TransImage->intent_p1==SPLINE_VEL_GRID)
         {
            if(!flag->referenceImageFlag)
            {
               fprintf(stderr, "[NiftyReg ERROR] When using an cubic b-spline parametrisation (%s),",
                       param->inputTransName);
               fprintf(stderr, " a reference image shoud be specified (-ref flag).\n");
               return 1;
            }
            referenceImage=reg_io_ReadImageHeader(param->referenceImageName);
            if(referenceImage==NULL)
            {
               fprintf(stderr, "[NiftyReg ERROR] Error when reading the reference image: %s\n",
                       param->referenceImageName);
               return 1;
            }
            reg_checkAndCorrectDimension(referenceImage);
         }
         // Read the second reference image if specified
         if(flag->referenceImage2Flag==true)
         {
            referenceImage2=reg_io_ReadImageHeader(param->referenceImage2Name);
            if(referenceImage2==NULL)
            {
               fprintf(stderr, "[NiftyReg ERROR] Error when reading the second reference image: %s\n",
                       param->referenceImage2Name);
               return 1;
            }
         }
         // Generate the first deformation field
         if(referenceImage!=NULL)
         {
            // The field is created using the reference image space
            output1TransImage=nifti_copy_nim_info(referenceImage);
            output1TransImage->ndim=output1TransImage->dim[0]=5;
            output1TransImage->nt=output1TransImage->dim[4]=1;
            output1TransImage->nu=output1TransImage->dim[5]=output1TransImage->nz>1?3:2;
            output1TransImage->nvox=(size_t)output1TransImage->nx *
                                    output1TransImage->ny * output1TransImage->nz *
                                    output1TransImage->nt * output1TransImage->nu;
            output1TransImage->scl_slope=1.f;
            output1TransImage->scl_inter=0.f;
            if(referenceImage->datatype!=NIFTI_TYPE_FLOAT32)
            {
               output1TransImage->nbyper=sizeof(float);
               output1TransImage->datatype=NIFTI_TYPE_FLOAT32;
            }
            printf("[NiftyReg] Transformation 1 is defined in the space of image:\n[NiftyReg] %s\n",
                   referenceImage->fname);
         }
         else
         {
            // The field is created using the input transformation image space
            output1TransImage=nifti_copy_nim_info(input1TransImage);
         }
         output1TransImage->intent_code=NIFTI_INTENT_VECTOR;
         memset(output1TransImage->intent_name, 0, 16);
         strcpy(output1TransImage->intent_name,"NREG_TRANS");
         output1TransImage->intent_p1=DEF_FIELD;
         output1TransImage->data=(void *)calloc
                                 (output1TransImage->nvox,output1TransImage->nbyper);
         if(affine1Trans!=NULL)
         {
            reg_affine_getDeformationField(affine1Trans,output1TransImage);
         }
         else switch(reg_round(input1TransImage->intent_p1))
            {
            case SPLINE_GRID:
               printf("[NiftyReg] Transformation 1 is a spline parametrisation:\n[NiftyReg] %s\n",
                      input1TransImage->fname);
               reg_tools_multiplyValueToImage(output1TransImage,output1TransImage,0.f);
               output1TransImage->intent_p1=DISP_FIELD;
               reg_getDeformationFromDisplacement(output1TransImage);
               reg_spline_getDeformationField(input1TransImage,
                                              output1TransImage,
                                              NULL,
                                              true,
                                              true);
               break;
            case DEF_FIELD:
               printf("[NiftyReg] Transformation 1 is a deformation field:\n[NiftyReg] %s\n",
                      input1TransImage->fname);
               memcpy(output1TransImage->data,input1TransImage->data,
                      output1TransImage->nbyper*output1TransImage->nvox);
               break;
            case DISP_FIELD:
               printf("[NiftyReg] Transformation 1 is a displacement field:\n[NiftyReg] %s\n",
                      input1TransImage->fname);
               memcpy(output1TransImage->data,input1TransImage->data,
                      output1TransImage->nbyper*output1TransImage->nvox);
               reg_getDeformationFromDisplacement(output1TransImage);
               break;
            case SPLINE_VEL_GRID:
               printf("[NiftyReg] Transformation 1 is a spline velocity field parametrisation:\n[NiftyReg] %s\n",
                      input1TransImage->fname);
               reg_spline_getDefFieldFromVelocityGrid(input1TransImage,
                     output1TransImage,
                     false // the number of step is not automatically updated
                                                             );
               break;
            case DEF_VEL_FIELD:
               printf("[NiftyReg] Transformation 1 is a deformation field velocity:\n[NiftyReg] %s\n",
                      input1TransImage->fname);
               reg_defField_getDeformationFieldFromFlowField(input1TransImage,
                     output1TransImage,
                     false // the number of step is not automatically updated
                                                            );
               break;
            case DISP_VEL_FIELD:
               printf("[NiftyReg] Transformation 1 is a displacement field velocity:\n[NiftyReg] %s\n",
                      input1TransImage->fname);
               reg_getDeformationFromDisplacement(output1TransImage);
               reg_defField_getDeformationFieldFromFlowField(input1TransImage,
                     output1TransImage,
                     false // the number of step is not automatically updated
                                                            );
               break;
            default:
               fprintf(stderr,"[NiftyReg ERROR] The specified first input transformation type is not recognised: %s\n",
                       param->input2TransName);
               return 1;
            }
         if(affine2Trans!=NULL)
         {
            printf("[NiftyReg] Transformation 2 is an affine parametrisation:\n[NiftyReg] %s\n",
                   param->input2TransName);
            // The field is created using the previous image space
            output2TransImage=nifti_copy_nim_info(output1TransImage);
            output2TransImage->intent_code=NIFTI_INTENT_VECTOR;
            memset(output2TransImage->intent_name, 0, 16);
            strcpy(output2TransImage->intent_name,"NREG_TRANS");
            output2TransImage->intent_p1=DEF_FIELD;
            output2TransImage->data=(void *)calloc
                                    (output2TransImage->nvox,output2TransImage->nbyper);
            reg_affine_getDeformationField(affine2Trans,output2TransImage);
            reg_defField_compose(output2TransImage,output1TransImage,NULL);
         }
         else
         {
            switch(reg_round(input2TransImage->intent_p1))
            {
            case SPLINE_GRID:
               printf("[NiftyReg] Transformation 2 is a spline parametrisation:\n[NiftyReg] %s\n",
                      input2TransImage->fname);
               reg_spline_getDeformationField(input2TransImage,
                                              output1TransImage,
                                              NULL,
                                              true, // composition
                                              true // b-spline
                                             );
               break;
            case DEF_FIELD:
               printf("[NiftyReg] Transformation 2 is a deformation field:\n[NiftyReg] %s\n",
                      input2TransImage->fname);
               reg_defField_compose(input2TransImage,output1TransImage,NULL);
               break;
            case DISP_FIELD:
               printf("[NiftyReg] Transformation 2 is a displacement field:\n[NiftyReg] %s\n",
                      input2TransImage->fname);
               reg_getDeformationFromDisplacement(input2TransImage);
               reg_defField_compose(input2TransImage,output1TransImage,NULL);
               break;
            case SPLINE_VEL_GRID:
               // The field is created using the second reference image space
               if(referenceImage2!=NULL)
               {
                  output2TransImage=nifti_copy_nim_info(referenceImage2);
                  output2TransImage->scl_slope=1.f;
                  output2TransImage->scl_inter=0.f;
                  printf("[NiftyReg] Transformation 2 is defined in the space of image:\n[NiftyReg] %s\n",
                         referenceImage2->fname);
               }
               else
               {
                  output2TransImage=nifti_copy_nim_info(output1TransImage);
               }
               output2TransImage->ndim=output2TransImage->dim[0]=5;
               output2TransImage->nt=output2TransImage->dim[4]=1;
               output2TransImage->nu=output2TransImage->dim[5]=output2TransImage->nz>1?3:2;
               output2TransImage->nvox=(size_t)output2TransImage->nx *
                                       output2TransImage->ny * output2TransImage->nz *
                                       output2TransImage->nt * output2TransImage->nu;
               output2TransImage->nbyper=output1TransImage->nbyper;
               output2TransImage->datatype=output1TransImage->datatype;
               output2TransImage->data=(void *)calloc
                                       (output2TransImage->nvox,output2TransImage->nbyper);
               printf("[NiftyReg] Transformation 2 is a spline velocity field parametrisation:\n[NiftyReg] %s\n",
                      input2TransImage->fname);
               reg_spline_getDefFieldFromVelocityGrid(input2TransImage,
                     output2TransImage,
                     false // the number of step is not automatically updated
                                                             );
               reg_defField_compose(output2TransImage,output1TransImage,NULL);
               break;
            case DEF_VEL_FIELD:
               printf("[NiftyReg] Transformation 2 is a deformation field velocity:\n[NiftyReg] %s\n",
                      input2TransImage->fname);
               output2TransImage=nifti_copy_nim_info(input2TransImage);
               output2TransImage->intent_p1=DEF_FIELD;
               output2TransImage->data=(void *)calloc
                                       (output2TransImage->nvox,output2TransImage->nbyper);
               reg_defField_getDeformationFieldFromFlowField(input2TransImage,
                     output2TransImage,
                     false // the number of step is not automatically updated
                                                            );
               reg_defField_compose(output2TransImage,output1TransImage,NULL);
               break;
            case DISP_VEL_FIELD:
               printf("[NiftyReg] Transformation 2 is a displacement field velocity:\n[NiftyReg] %s\n",
                      input2TransImage->fname);
               output2TransImage=nifti_copy_nim_info(input2TransImage);
               output2TransImage->intent_p1=DEF_FIELD;
               output2TransImage->data=(void *)calloc
                                       (output2TransImage->nvox,output2TransImage->nbyper);
               reg_getDeformationFromDisplacement(input2TransImage);
               reg_defField_getDeformationFieldFromFlowField(input2TransImage,
                     output2TransImage,
                     false // the number of step is not automatically updated
                                                            );
               reg_defField_compose(output2TransImage,output1TransImage,NULL);
               break;
            default:
               fprintf(stderr,"[NiftyReg ERROR] The specified second input transformation type is not recognised: %s\n",
                       param->input2TransName);
               return 1;
            }
         }
         // Save the composed transformation
         reg_io_WriteImageFile(output1TransImage,param->outputTransName);
         printf("[NiftyReg] The final deformation field has been saved as:\n[NiftyReg] %s\n",
                param->outputTransName);
      }
      // Free allocated object
      if(affine1Trans!=NULL) free(affine1Trans);
      if(affine2Trans!=NULL) free(affine2Trans);
      if(referenceImage!=NULL) nifti_image_free(referenceImage);
      if(referenceImage2!=NULL) nifti_image_free(referenceImage2);
      if(input1TransImage!=NULL) nifti_image_free(input1TransImage);
      if(input2TransImage!=NULL) nifti_image_free(input2TransImage);
      if(output1TransImage!=NULL) nifti_image_free(output1TransImage);
      if(output2TransImage!=NULL) nifti_image_free(output2TransImage);
   }

   /* **************************************** */
   // Update the SForm matrix of a given image //
   /* **************************************** */
   if(flag->updSFormFlag)
   {
      // Read the input image
      nifti_image *image = reg_io_ReadImageFile(param->inputTransName);
      if(image==NULL)
      {
         fprintf(stderr,"[NiftyReg ERROR] Error when reading the input image: %s\n",
                 param->inputTransName);
         return 1;
      }
      reg_checkAndCorrectDimension(image);
      // Read the affine transformation
      mat44 *affineTransformation = (mat44 *)calloc(1,sizeof(mat44));
      reg_tool_ReadAffineFile(affineTransformation,
                              param->input2TransName);
      //Invert the affine transformation since the flaoting is updated
      *affineTransformation = nifti_mat44_inverse(*affineTransformation);

      // Update the sform
      if(image->sform_code>0)
      {
         image->sto_xyz = reg_mat44_mul(affineTransformation, &(image->sto_xyz));
      }
      else
      {
         image->sform_code = 1;
         image->sto_xyz = reg_mat44_mul(affineTransformation, &(image->qto_xyz));
      }
      image->sto_ijk = nifti_mat44_inverse(image->sto_xyz);

      // Write the output image
      reg_io_WriteImageFile(image,param->outputTransName);
      // Free the allocated image and array
      nifti_image_free(image);
      free(affineTransformation);
   }
   /* ******************************** */
   // Half the provided transformation //
   /* ******************************** */
   if(flag->halfTransFlag)
   {
      // Read the input transformation
      mat44 *affineTrans=NULL;
      nifti_image *inputTransImage=NULL;
      if(!reg_isAnImageFileName(param->inputTransName))
      {
         // An affine transformation is considered
         affineTrans=(mat44 *)malloc(sizeof(mat44));
         reg_tool_ReadAffineFile(affineTrans,param->inputTransName);
         // The affine transformation is halfed
         *affineTrans=reg_mat44_logm(affineTrans);
         *affineTrans=reg_mat44_mul(affineTrans,0.5);
         *affineTrans=reg_mat44_expm(affineTrans);
         // The affine transformation is saved
         reg_tool_WriteAffineFile(affineTrans,param->outputTransName);
      }
      else
      {
         // A non-rigid parametrisation is considered
         inputTransImage = reg_io_ReadImageFile(param->inputTransName);
         if(inputTransImage==NULL)
         {
            fprintf(stderr,"[NiftyReg ERROR] Error when reading the input image: %s\n",
                    param->inputTransName);
            return 1;
         }
         reg_checkAndCorrectDimension(inputTransImage);
         switch(reg_round(inputTransImage->intent_p1))
         {
         case SPLINE_GRID:
            reg_getDisplacementFromDeformation(inputTransImage);
            reg_tools_multiplyValueToImage(inputTransImage,inputTransImage,0.5f);
            reg_getDeformationFromDisplacement(inputTransImage);
            break;
         case DEF_FIELD:
            reg_getDisplacementFromDeformation(inputTransImage);
            reg_tools_multiplyValueToImage(inputTransImage,inputTransImage,0.5f);
            reg_getDeformationFromDisplacement(inputTransImage);
            break;
         case DISP_FIELD:
            reg_tools_multiplyValueToImage(inputTransImage,inputTransImage,0.5f);
            break;
         case SPLINE_VEL_GRID:
            reg_getDisplacementFromDeformation(inputTransImage);
            reg_tools_multiplyValueToImage(inputTransImage,inputTransImage,0.5f);
            reg_getDeformationFromDisplacement(inputTransImage);
            --inputTransImage->intent_p2;
            if(inputTransImage->num_ext>1)
               --inputTransImage->num_ext;
            break;
         case DEF_VEL_FIELD:
            reg_getDisplacementFromDeformation(inputTransImage);
            reg_tools_multiplyValueToImage(inputTransImage,inputTransImage,0.5f);
            reg_getDeformationFromDisplacement(inputTransImage);
            --inputTransImage->intent_p2;
            break;
         case DISP_VEL_FIELD:
            reg_tools_multiplyValueToImage(inputTransImage,inputTransImage,0.5f);
            --inputTransImage->intent_p2;
            break;
         default:
            fprintf(stderr,"[NiftyReg ERROR] The specified input transformation type is not recognised: %s\n",
                    param->inputTransName);
            return 1;
         }
         // Save the image
         reg_io_WriteImageFile(inputTransImage,param->outputTransName);
      }
      // Clear the allocated arrays
      if(affineTrans!=NULL) free(affineTrans);
   }
   /* ******************************************** */
   // Invert the provided non-rigid transformation //
   /* ******************************************** */
   if(flag->invertNRRFlag)
   {
      // Read the provided transformation
      nifti_image *inputTransImage = reg_io_ReadImageFile(param->inputTransName);
      if(inputTransImage==NULL)
      {
         fprintf(stderr,"[NiftyReg ERROR] Error when reading the input image: %s\n",
                 param->inputTransName);
         return 1;
      }
      reg_checkAndCorrectDimension(inputTransImage);
      // Read the provided floating space image
      nifti_image *floatingImage = reg_io_ReadImageFile(param->input2TransName);
      if(floatingImage==NULL)
      {
         fprintf(stderr,"[NiftyReg ERROR] Error when reading the input image: %s\n",
                 param->input2TransName);
         return 1;
      }
      reg_checkAndCorrectDimension(floatingImage);
      // Create a field to store the transformation
      nifti_image *outputTransImage=nifti_copy_nim_info(floatingImage);
      outputTransImage->ndim=outputTransImage->dim[0]=5;
      outputTransImage->nt=outputTransImage->dim[4]=1;
      outputTransImage->nu=outputTransImage->dim[5]=outputTransImage->nz>1?3:2;
      outputTransImage->nvox=(size_t)outputTransImage->nx *
                             outputTransImage->ny * outputTransImage->nz *
                             outputTransImage->nt * outputTransImage->nu;
      outputTransImage->nbyper=inputTransImage->nbyper;
      outputTransImage->datatype=inputTransImage->datatype;
      outputTransImage->intent_code=NIFTI_INTENT_VECTOR;
      memset(outputTransImage->intent_name, 0, 16);
      strcpy(outputTransImage->intent_name,"NREG_TRANS");
      outputTransImage->intent_p1=inputTransImage->intent_p1;
      outputTransImage->intent_p2=inputTransImage->intent_p2;
      outputTransImage->scl_slope=1.f;
      outputTransImage->scl_inter=0.f;
      outputTransImage->data=(void *)malloc
                             (outputTransImage->nvox*outputTransImage->nbyper);
      // Convert the spline parametrisation into a dense deformation parametrisation
      if(inputTransImage->intent_p1==SPLINE_GRID || inputTransImage->intent_p1==SPLINE_VEL_GRID)
      {
         // Read the reference image
         if(!flag->referenceImageFlag)
         {
            fprintf(stderr, "[NiftyReg ERROR] When using an spline parametrisation transformation (%s),",
                    param->inputTransName);
            fprintf(stderr, " a reference image shoud be specified (-res flag).\n");
            return 1;
         }
         nifti_image *referenceImage=reg_io_ReadImageHeader(param->referenceImageName);
         if(referenceImage==NULL)
         {
            fprintf(stderr, "[NiftyReg ERROR] Error when reading the reference image: %s\n",
                    param->referenceImageName);
            return 1;
         }
         reg_checkAndCorrectDimension(referenceImage);
         // Create a deformation field or a flow field
         nifti_image *tempField=nifti_copy_nim_info(referenceImage);
         tempField->ndim=tempField->dim[0]=5;
         tempField->nt=tempField->dim[4]=1;
         tempField->nu=tempField->dim[5]=tempField->nz>1?3:2;
         tempField->nvox=(size_t)tempField->nx * tempField->ny * tempField->nz *
                         tempField->nt * tempField->nu;
         tempField->nbyper=inputTransImage->nbyper;
         tempField->datatype=inputTransImage->datatype;
         tempField->intent_code=NIFTI_INTENT_VECTOR;
         memset(tempField->intent_name, 0, 16);
         strcpy(tempField->intent_name,"NREG_TRANS");
         tempField->intent_p1=DEF_FIELD;
         if(inputTransImage->intent_p1==SPLINE_VEL_GRID)
         {
            tempField->intent_p1=DEF_VEL_FIELD;
            tempField->intent_p2=inputTransImage->intent_p2;
         }
         tempField->scl_slope=1.f;
         tempField->scl_inter=0.f;
         tempField->data=(void *)calloc(tempField->nvox,tempField->nbyper);
         // Compute the dense field
         if(inputTransImage->intent_p1==SPLINE_GRID)
            reg_spline_getDeformationField(inputTransImage,
                                           tempField,
                                           NULL,
                                           false,
                                           true);
         else
            reg_spline_getFlowFieldFromVelocityGrid(inputTransImage,
                                                    tempField);
         // The provided transformation file is replaced by the compute dense field
         nifti_image_free(referenceImage);
         nifti_image_free(inputTransImage);
         inputTransImage=tempField;
         tempField=NULL;
      }
      // Invert the provided
      switch(reg_round(inputTransImage->intent_p1))
      {
      case DEF_FIELD:
         reg_defFieldInvert(inputTransImage,outputTransImage,1.0e-6f);
         break;
      case DISP_FIELD:
         reg_getDeformationFromDisplacement(inputTransImage);
         reg_defFieldInvert(inputTransImage,outputTransImage,1.0e-6f);
         reg_getDisplacementFromDeformation(outputTransImage);
         break;
      case DEF_VEL_FIELD:
      {
         // create a temp deformation field containing an identity transformation
         nifti_image *tempField=nifti_copy_nim_info(outputTransImage);
         tempField->intent_p1=DEF_FIELD;
         tempField->data=(void *)calloc(tempField->nvox,tempField->nbyper);
         reg_getDeformationFromDisplacement(tempField);
         reg_getDisplacementFromDeformation(inputTransImage);
         reg_resampleGradient(inputTransImage,
                              outputTransImage,
                              tempField,
                              1,
                              0);
         nifti_image_free(tempField);
         reg_getDeformationFromDisplacement(outputTransImage);
         outputTransImage->intent_p2 *= -1.f;
         break;
      }
      case DISP_VEL_FIELD:
      {
         // create a temp deformation field containing an identity transformation
         nifti_image *tempField=nifti_copy_nim_info(outputTransImage);
         tempField->intent_p1=DEF_FIELD;
         tempField->data=(void *)calloc(tempField->nvox,tempField->nbyper);
         reg_getDeformationFromDisplacement(tempField);
         reg_resampleGradient(inputTransImage,
                              outputTransImage,
                              tempField,
                              1,
                              0);
         nifti_image_free(tempField);
         outputTransImage->intent_p2 *= -1.f;
         break;
      }
      default:
         fprintf(stderr,"[NiftyReg ERROR] The specified input transformation type is not recognised: %s\n",
                 param->inputTransName);
         return 1;
      }
      // Save the inverted transformation
      reg_io_WriteImageFile(outputTransImage,param->outputTransName);
      // Free the allocated images
      nifti_image_free(inputTransImage);
      nifti_image_free(outputTransImage);
   }
   /* ***************************************** */
   // Invert the provided affine transformation //
   /* ***************************************** */
   if(flag->invertAffFlag)
   {
      // Read the affine transformation
      mat44 affineTrans;
      reg_tool_ReadAffineFile(&affineTrans,param->inputTransName);
      // Invert the transformation
      affineTrans = nifti_mat44_inverse(affineTrans);
      // Save the inverted transformation
      reg_tool_WriteAffineFile(&affineTrans,param->outputTransName);
   }
   /* ******************************* */
   // Create an affine transformation //
   /* ******************************* */
   if(flag->makeAffFlag)
   {
      // Create all the required matrices
      mat44 rotationX;
      reg_mat44_eye(&rotationX);
      mat44 translation;
      reg_mat44_eye(&translation);
      mat44 rotationY;
      reg_mat44_eye(&rotationY);
      mat44 rotationZ;
      reg_mat44_eye(&rotationZ);
      mat44 scaling;
      reg_mat44_eye(&scaling);
      mat44 shearing;
      reg_mat44_eye(&shearing);
      // Set up the rotation matrix along the YZ plane
      rotationX.m[1][1]=cosf(param->affTransParam[0]);
      rotationX.m[1][2]=-sinf(param->affTransParam[0]);
      rotationX.m[2][1]=sinf(param->affTransParam[0]);
      rotationX.m[2][2]=cosf(param->affTransParam[0]);
      // Set up the rotation matrix along the XZ plane
      rotationY.m[0][0]=cosf(param->affTransParam[1]);
      rotationY.m[0][2]=-sinf(param->affTransParam[1]);
      rotationY.m[2][0]=sinf(param->affTransParam[1]);
      rotationY.m[2][2]=cosf(param->affTransParam[1]);
      // Set up the rotation matrix along the XY plane
      rotationZ.m[0][0]=cosf(param->affTransParam[2]);
      rotationZ.m[0][1]=-sinf(param->affTransParam[2]);
      rotationZ.m[1][0]=sinf(param->affTransParam[2]);
      rotationZ.m[1][1]=cosf(param->affTransParam[2]);
      // Set up the translation matrix
      translation.m[0][3]=param->affTransParam[3];
      translation.m[1][3]=param->affTransParam[4];
      translation.m[2][3]=param->affTransParam[5];
      // Set up the scaling matrix
      scaling.m[0][0]=param->affTransParam[6];
      scaling.m[1][1]=param->affTransParam[7];
      scaling.m[2][2]=param->affTransParam[8];
      // Set up the shearing matrix
      shearing.m[1][0]=param->affTransParam[9];
      shearing.m[2][0]=param->affTransParam[10];
      shearing.m[2][1]=param->affTransParam[11];
      // Combine all the transformations
      mat44 affine=reg_mat44_mul(&rotationY,&rotationZ);
      affine=reg_mat44_mul(&rotationX,&affine);
      affine=reg_mat44_mul(&scaling,&affine);
      affine=reg_mat44_mul(&shearing,&affine);
      affine=reg_mat44_mul(&translation,&affine);
      // Save the new matrix
      reg_tool_WriteAffineFile(&affine,param->outputTransName);
   }
   /* ************************************************* */
   // Extract the rigid component from an affine matrix //
   /* ************************************************* */
   if(flag->aff2rigFlag)
   {
      mat44 affine;
      reg_tool_ReadAffineFile(&affine,param->inputTransName);
      // Compute the orthonormal matrix
      float qb,qc,qd,qx,qy,qz,dx,dy,dz,qfac;
      nifti_mat44_to_quatern(affine,&qb,&qc,&qd,&qx,&qy,&qz,&dx,&dy,&dz,&qfac);
      affine = nifti_quatern_to_mat44(qb,qc,qd,qx,qy,qz,1.f,1.f,1.f,qfac);
      reg_tool_WriteAffineFile(&affine, param->outputTransName);
   }
   /* ********************************************************** */
   // Convert a flirt affine transformation to a NiftyReg affine //
   /* ********************************************************** */
   if(flag->flirtAff2NRFlag)
   {
      mat44 affine;
      nifti_image *referenceImage=reg_io_ReadImageHeader(param->referenceImageName);
      nifti_image *floatingImage=reg_io_ReadImageHeader(param->referenceImage2Name);
      reg_tool_ReadAffineFile(&affine,referenceImage,floatingImage,param->inputTransName,true);
      reg_tool_WriteAffineFile(&affine, param->outputTransName);
      nifti_image_free(referenceImage);
      nifti_image_free(floatingImage);
   }
   // Free allocated object
   free(param);
   free(flag);

   return 0;
}

#endif
