/**
 * @file reg_measure.cpp
 * @author Marc Modat
 * @date 28/02/2014
 *
 *  Copyright (c) 2014, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_MEASURE_CPP
#define _REG_MEASURE_CPP

#include <limits>

#include "_reg_ReadWriteImage.h"
#include "_reg_resampling.h"
#include "_reg_tools.h"
#include "_reg_nmi.h"
#include "_reg_dti.h"
#include "_reg_ssd.h"
#include "_reg_KLdivergence.h"
#include "_reg_lncc.h"

typedef struct
{
   char *refImageName;
   char *floImageName;
   char *refMaskImageName;
   char *floMaskImageName;
   int interpolation;
   float paddingValue;
   char *outFileName;
} PARAM;
typedef struct
{
   bool refImageFlag;
   bool floImageFlag;
   bool refMaskImageFlag;
   bool floMaskImageFlag;
   bool returnNMIFlag;
   bool returnSSDFlag;
   bool returnLNCCFlag;
   bool returnNCCFlag;
   bool outFileFlag;
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
   printf("\t-ref <filename>\tFilename of the reference image (mandatory)\n");
   printf("\t-flo <filename>\tFilename of the floating image (mandatory)\n");
   printf("\t\tNote that the floating image is resampled into the reference\n");
   printf("\t\timage space using the header informations.\n");

   printf("* * OPTIONS * *\n");
   printf("\t-ncc\t\tReturns the NCC value\n");
   printf("\t-lncc\t\tReturns the LNCC value\n");
   printf("\t-nmi\t\tReturns the NMI value (64 bins are used)\n");
   printf("\t-ssd\t\tReturns the SSD value\n");
   printf("\n\t-out\t\tText file output where to store the value(s).\n\t\t\tThe stdout is used by default\n");
#ifdef _GIT_HASH
   printf("\n\t--version\tPrint current source code git hash key and exit\n\t\t\t(%s)\n",_GIT_HASH);
#endif
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   return;
}

int main(int argc, char **argv)
{
   PARAM *param = (PARAM *)calloc(1,sizeof(PARAM));
   FLAG *flag = (FLAG *)calloc(1,sizeof(FLAG));

   param->interpolation=3; // Cubic spline interpolation used by default
   param->paddingValue=std::numeric_limits<float>::quiet_NaN();

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
//      else if(strcmp(argv[i], "--xml")==0)
//      {
//         printf("%s",xml_measure);
//         return 0;
//      }
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
      else if((strcmp(argv[i],"-rmask")==0) ||
              (strcmp(argv[i],"--rmask")==0))
      {
         param->refMaskImageName=argv[++i];
         flag->refMaskImageFlag=1;
      }
      else if((strcmp(argv[i],"-flo")==0) || (strcmp(argv[i],"-source")==0) ||
              (strcmp(argv[i],"--flo")==0))
      {
         param->floImageName=argv[++i];
         flag->floImageFlag=1;
      }
      else if((strcmp(argv[i],"-fmask")==0) ||
              (strcmp(argv[i],"--fmask")==0))
      {
         param->floMaskImageName=argv[++i];
         flag->floMaskImageFlag=1;
      }
      else if(strcmp(argv[i], "-inter") == 0 ||
              (strcmp(argv[i],"--inter")==0))
      {
         param->interpolation=atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-pad") == 0 ||
              (strcmp(argv[i],"--pad")==0))
      {
         param->paddingValue=(float)atof(argv[++i]);
      }
      else if(strcmp(argv[i], "-ncc") == 0 ||
              (strcmp(argv[i],"--ncc")==0))
      {
         flag->returnNCCFlag=true;
      }
      else if(strcmp(argv[i], "-lncc") == 0 ||
              (strcmp(argv[i],"--lncc")==0))
      {
         flag->returnLNCCFlag=true;
      }
      else if(strcmp(argv[i], "-nmi") == 0 ||
              (strcmp(argv[i],"--nmi")==0))
      {
         flag->returnNMIFlag=true;
      }
      else if(strcmp(argv[i], "-ssd") == 0 ||
              (strcmp(argv[i],"--sdd")==0))
      {
         flag->returnSSDFlag=true;
      }
      else if(strcmp(argv[i], "-out") == 0 ||
              (strcmp(argv[i],"--out")==0))
      {
         flag->outFileFlag=true;
         param->outFileName=argv[++i];
      }
      else
      {
         fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
         PetitUsage(argv[0]);
         return 1;
      }
   }

   if(!flag->refImageFlag || !flag->floImageFlag)
   {
      fprintf(stderr,"[NiftyReg ERROR] The reference and the floating image have both to be defined.\n");
      PetitUsage(argv[0]);
      return 1;
   }

   /* Read the reference image */
   nifti_image *refImage = reg_io_ReadImageFile(param->refImageName);
   if(refImage == NULL)
   {
      fprintf(stderr,"[NiftyReg ERROR] Error when reading the reference image: %s\n",
              param->refImageName);
      return 1;
   }
   reg_checkAndCorrectDimension(refImage);
   reg_tools_changeDatatype<float>(refImage);

   /* Read the floating image */
   nifti_image *floImage = reg_io_ReadImageFile(param->floImageName);
   if(floImage == NULL)
   {
      fprintf(stderr,"[NiftyReg ERROR] Error when reading the floating image: %s\n",
              param->floImageName);
      return 1;
   }
   reg_checkAndCorrectDimension(floImage);
   reg_tools_changeDatatype<float>(floImage);

   /* Read and create the mask array */
   int *refMask=NULL;
   int refMaskVoxNumber=refImage->nx*refImage->ny*refImage->nz;
   if(flag->refMaskImageFlag){
      nifti_image *refMaskImage = reg_io_ReadImageFile(param->refMaskImageName);
      if(refMaskImage == NULL)
      {
         fprintf(stderr,"[NiftyReg ERROR] Error when reading the reference mask image: %s\n",
                 param->refMaskImageName);
         return 1;
      }
      reg_checkAndCorrectDimension(refMaskImage);
      reg_createMaskPyramid<float>(refMaskImage, &refMask, 1, 1, &refMaskVoxNumber);
   }
   else{
      refMask = (int *)calloc(refMaskVoxNumber,sizeof(int));
      for(int i=0;i<refMaskVoxNumber;++i) refMask[i]=i;
   }

   /* Create the warped floating image */
   nifti_image *warpedFloImage = nifti_copy_nim_info(refImage);
   warpedFloImage->ndim=warpedFloImage->dim[0]=floImage->ndim;
   warpedFloImage->nt=warpedFloImage->dim[4]=floImage->nt;
   warpedFloImage->nu=warpedFloImage->dim[5]=floImage->nu;
   warpedFloImage->nvox=(size_t)warpedFloImage->nx * warpedFloImage->ny *
         warpedFloImage->nz * warpedFloImage->nt * warpedFloImage->nu;
   warpedFloImage->cal_min=floImage->cal_min;
   warpedFloImage->cal_max=floImage->cal_max;
   warpedFloImage->scl_inter=floImage->scl_inter;
   warpedFloImage->scl_slope=floImage->scl_slope;
   warpedFloImage->datatype=floImage->datatype;
   warpedFloImage->nbyper=floImage->nbyper;
   warpedFloImage->data=(void *)malloc(warpedFloImage->nvox*warpedFloImage->nbyper);

   /* Create the deformation field */
   nifti_image *defField = nifti_copy_nim_info(refImage);
   defField->ndim=defField->dim[0]=5;
   defField->nt=defField->dim[4]=1;
   defField->nu=defField->dim[5]=refImage->nz>1?3:2;
   defField->nvox=(size_t)defField->nx * defField->ny *
         defField->nz * defField->nt * defField->nu;
   defField->datatype=NIFTI_TYPE_FLOAT32;
   defField->nbyper=sizeof(float);
   defField->data=(void *)calloc(defField->nvox,defField->nbyper);
   defField->scl_slope=1.f;
   defField->scl_inter=0.f;
   reg_tools_multiplyValueToImage(defField,defField,0.f);
   defField->intent_p1=DISP_FIELD;
   reg_getDeformationFromDisplacement(defField);

   /* Warp the floating image */
   reg_resampleImage(floImage,
                     warpedFloImage,
                     defField,
                     refMask,
                     param->interpolation,
                     param->paddingValue);
   nifti_image_free(defField);

   FILE *outFile=NULL;
   if(flag->outFileFlag)
      outFile=fopen(param->outFileName, "w");

   /* Compute the NCC if required */
   if(flag->returnNCCFlag){
      float *refPtr = static_cast<float *>(refImage->data);
      float *warPtr = static_cast<float *>(warpedFloImage->data);
      double refMeanValue =0.;
      double warMeanValue =0.;
      refMaskVoxNumber=0;
      for(size_t i=0; i<refImage->nvox; ++i){
         if(refMask[i]>-1 && refPtr[i]==refPtr[i] && warPtr[i]==warPtr[i]){
            refMeanValue += refPtr[i];
            warMeanValue += warPtr[i];
            ++refMaskVoxNumber;
         }
      }
      if(refMaskVoxNumber==0)
         fprintf(stderr, "No active voxel\n");
      refMeanValue /= (double)refMaskVoxNumber;
      warMeanValue /= (double)refMaskVoxNumber;
      double refSTDValue =0.;
      double warSTDValue =0.;
      double measure=0.;
      for(size_t i=0; i<refImage->nvox; ++i){
         if(refMask[i]>-1 && refPtr[i]==refPtr[i] && warPtr[i]==warPtr[i]){
            refSTDValue += reg_pow2((double)refPtr[i] - refMeanValue);
            warSTDValue += reg_pow2((double)warPtr[i] - warMeanValue);
            measure += ((double)refPtr[i] - refMeanValue) *
                  ((double)warPtr[i] - warMeanValue);
         }
      }
      refSTDValue /= (double)refMaskVoxNumber;
      warSTDValue /= (double)refMaskVoxNumber;
      measure /= sqrt(refSTDValue)*sqrt(warSTDValue)*
            (double)refMaskVoxNumber;
      if(outFile!=NULL)
         fprintf(outFile, "%g\n", measure);
      else printf("NCC: %g\n", measure);
   }
   /* Compute the LNCC if required */
   if(flag->returnLNCCFlag){
      reg_lncc *lncc_object=new reg_lncc();
      for(int i=0;i<(refImage->nt<warpedFloImage->nt?refImage->nt:warpedFloImage->nt);++i)
         lncc_object->SetActiveTimepoint(i);
      lncc_object->InitialiseMeasure(refImage,
                                    warpedFloImage,
                                    refMask,
                                    warpedFloImage,
                                    NULL,
                                    NULL);
      double measure=lncc_object->GetSimilarityMeasureValue();
      if(outFile!=NULL)
         fprintf(outFile, "%g\n", measure);
      else printf("LNCC: %g\n", measure);
      delete lncc_object;
   }
   /* Compute the NMI if required */
   if(flag->returnNMIFlag){
      reg_nmi *nmi_object=new reg_nmi();
      for(int i=0;i<(refImage->nt<warpedFloImage->nt?refImage->nt:warpedFloImage->nt);++i)
         nmi_object->SetActiveTimepoint(i);
      nmi_object->InitialiseMeasure(refImage,
                                    warpedFloImage,
                                    refMask,
                                    warpedFloImage,
                                    NULL,
                                    NULL);
      double measure=nmi_object->GetSimilarityMeasureValue();
      if(outFile!=NULL)
         fprintf(outFile, "%g\n", measure);
      else printf("NMI: %g\n", measure);
      delete nmi_object;
   }
   /* Compute the SSD if required */
   if(flag->returnSSDFlag){
      reg_ssd *ssd_object=new reg_ssd();
      for(int i=0;i<(refImage->nt<warpedFloImage->nt?refImage->nt:warpedFloImage->nt);++i)
         ssd_object->SetActiveTimepoint(i);
      ssd_object->InitialiseMeasure(refImage,
                                    warpedFloImage,
                                    refMask,
                                    warpedFloImage,
                                    NULL,
                                    NULL);
      double measure=ssd_object->GetSimilarityMeasureValue();
      if(outFile!=NULL)
         fprintf(outFile, "%g\n", measure);
      else printf("SSD: %g\n", measure);
      delete ssd_object;
   }

   // Close the output file if required
   if(outFile!=NULL)
      fclose(outFile);

   // Free the allocated images
   nifti_image_free(refImage);
   nifti_image_free(floImage);
   free(refMask);

   free(flag);
   free(param);
   return 0;
}

#endif
