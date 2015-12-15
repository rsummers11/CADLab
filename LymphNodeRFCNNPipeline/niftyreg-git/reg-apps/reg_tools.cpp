/*
 *  reg_tools.cpp
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */
#ifndef MM_TOOLS_CPP
#define MM_TOOLS_CPP

#include "_reg_ReadWriteImage.h"
#include "_reg_resampling.h"
#include "_reg_globalTransformation.h"
#include "_reg_localTransformation.h"
#include "_reg_tools.h"

#include "reg_tools.h"

int isNumeric (const char *s)
{
   if(s==NULL || *s=='\0' || isspace(*s))
      return 0;
   char * p;
   strtod (s, &p);
   return *p == '\0';
}

typedef struct
{
   char *inputImageName;
   char *outputImageName;
   char *operationImageName;
   char *rmsImageName;
   float operationValue;
   float smoothValueX;
   float smoothValueY;
   float smoothValueZ;
   float thresholdImageValue;
   float removeNanInfValue;
} PARAM;
typedef struct
{
   bool inputImageFlag;
   bool outputImageFlag;
   bool floatFlag;
   bool downsampleFlag;
   bool rmsImageFlag;
   bool smoothSplineFlag;
   bool smoothGaussianFlag;
   bool smoothLabFlag;
   bool smoothMeanFlag;
   bool binarisedImageFlag;
   bool thresholdImageFlag;
   bool nanMaskFlag;
   bool normFlag;
   int operationTypeFlag;
   bool iso;
   bool nosclFlag;
   bool removeNanInf;
} FLAG;


void PetitUsage(char *exec)
{
   fprintf(stderr,"Usage:\t%s -in  <targetImageName> [OPTIONS].\n",exec);
   fprintf(stderr,"\tSee the help for more details (-h).\n");
   return;
}
void Usage(char *exec)
{
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   printf("Usage:\t%s -in <filename> -out <filename> [OPTIONS].\n",exec);
   printf("\t-in <filename>\tFilename of the input image image (mandatory)\n");
   printf("* * OPTIONS * *\n");
   printf("\t-out <filename>\t\tFilename out the output image [output.nii]\n");
   printf("\t-float\t\t\tThe input image is converted to float\n");
   printf("\t-down\t\t\tThe input image is downsampled 2 times\n");
   printf("\t-smoS <float> <float> <float>\tThe input image is smoothed using a cubic b-spline kernel\n");
   printf("\t-smoG <float> <float> <float>\tThe input image is smoothed using Gaussian kernel\n");
   printf("\t-smoL <float> <float> <float>\tThe input label image is smoothed using Gaussian kernel\n");
   printf("\t-add <filename/float>\tThis image (or value) is added to the input\n");
   printf("\t-sub <filename/float>\tThis image (or value) is subtracted to the input\n");
   printf("\t-mul <filename/float>\tThis image (or value) is multiplied to the input\n");
   printf("\t-div <filename/float>\tThis image (or value) is divided to the input\n");
   printf("\t-rms <filename>\t\tCompute the mean rms between both image\n");
   printf("\t-bin \t\t\tBinarise the input image (val!=0?val=1:val=0)\n");
   printf("\t-thr <float>\t\tThreshold the input image (val<thr?val=0:val=1)\n");
   printf("\t-nan <filename>\t\tThis image is used to mask the input image.\n\t\t\t\tVoxels outside of the mask are set to nan\n");
   printf("\t-iso\t\t\tThe resulting image is made isotropic\n");
   printf("\t-noscl\t\t\tThe scl_slope and scl_inter are set to 1 and 0 respectively\n");
   printf("\t-rmNanInf <float>\t\tRemove the nan and inf from the input image and replace them by the specified value\n");
#ifdef _GIT_HASH
   printf("\n\t--version\t\tPrint current source code git hash key and exit\n\t\t\t\t(%s)\n",_GIT_HASH);
#endif
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   return;
}

int main(int argc, char **argv)
{
   PARAM *param = (PARAM *)calloc(1,sizeof(PARAM));
   FLAG *flag = (FLAG *)calloc(1,sizeof(FLAG));
   flag->operationTypeFlag=-1;

   if (argc < 2)
   {
      PetitUsage(argv[0]);
      return 1;
   }

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
         printf("%s",xml_tools);
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
      else if(strcmp(argv[i], "-in") == 0)
      {
         param->inputImageName=argv[++i];
         flag->inputImageFlag=1;
      }
      else if(strcmp(argv[i], "-out") == 0)
      {
         param->outputImageName=argv[++i];
         flag->outputImageFlag=1;
      }

      else if(strcmp(argv[i], "-add") == 0)
      {
         param->operationImageName=argv[++i];
         if(isNumeric(param->operationImageName)==true)
         {
            param->operationValue=(float)atof(param->operationImageName);
            param->operationImageName=NULL;
         }
         flag->operationTypeFlag=0;
      }
      else if(strcmp(argv[i], "-sub") == 0)
      {
         param->operationImageName=argv[++i];
         if(isNumeric(param->operationImageName))
         {
            param->operationValue=(float)atof(param->operationImageName);
            param->operationImageName=NULL;
         }
         flag->operationTypeFlag=1;
      }
      else if(strcmp(argv[i], "-mul") == 0)
      {
         param->operationImageName=argv[++i];
         if(isNumeric(param->operationImageName))
         {
            param->operationValue=(float)atof(param->operationImageName);
            param->operationImageName=NULL;
         }
         flag->operationTypeFlag=2;
      }
      else if(strcmp(argv[i], "-iso") == 0)
      {
         flag->iso=true;
      }
      else if(strcmp(argv[i], "-div") == 0)
      {
         param->operationImageName=argv[++i];
         if(isNumeric(param->operationImageName))
         {
            param->operationValue=(float)atof(param->operationImageName);
            param->operationImageName=NULL;
         }
         flag->operationTypeFlag=3;
      }
      else if(strcmp(argv[i], "-rms") == 0)
      {
         param->rmsImageName=argv[++i];
         flag->rmsImageFlag=1;
      }
      else if(strcmp(argv[i], "-down") == 0)
      {
         flag->downsampleFlag=1;
      }
      else if(strcmp(argv[i], "-float") == 0)
      {
         flag->floatFlag=1;
      }
      else if(strcmp(argv[i], "-smoS") == 0)
      {
         param->smoothValueX=atof(argv[++i]);
         param->smoothValueY=atof(argv[++i]);
         param->smoothValueZ=atof(argv[++i]);
         flag->smoothSplineFlag=1;
      }
      else if(strcmp(argv[i], "-smoG") == 0)
      {
         param->smoothValueX=atof(argv[++i]);
         param->smoothValueY=atof(argv[++i]);
         param->smoothValueZ=atof(argv[++i]);
         flag->smoothGaussianFlag=1;
      }
      else if(strcmp(argv[i], "-smoL") == 0)
      {
         param->smoothValueX=atof(argv[++i]);
         param->smoothValueY=atof(argv[++i]);
         param->smoothValueZ=atof(argv[++i]);
         flag->smoothLabFlag=1;
      }
      else if(strcmp(argv[i], "-smoM") == 0)
      {
         param->smoothValueX=atof(argv[++i]);
         param->smoothValueY=atof(argv[++i]);
         param->smoothValueZ=atof(argv[++i]);
         flag->smoothMeanFlag=1;
      }
      else if(strcmp(argv[i], "-bin") == 0)
      {
         flag->binarisedImageFlag=1;
      }
      else if(strcmp(argv[i], "-thr") == 0)
      {
         param->thresholdImageValue=atof(argv[++i]);
         flag->thresholdImageFlag=1;
      }
      else if(strcmp(argv[i], "-nan") == 0)
      {
         param->operationImageName=argv[++i];
         flag->nanMaskFlag=1;
      }
      else if(strcmp(argv[i], "-norm") == 0)
      {
         flag->normFlag=1;
      }
      else if(strcmp(argv[i], "-noscl") == 0)
      {
         flag->nosclFlag=1;
      }
      else if(strcmp(argv[i], "-rmNanInf") == 0)
      {
         flag->removeNanInf=1;
         param->removeNanInfValue=atof(argv[++i]);
      }
      else
      {
         fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
         PetitUsage(argv[0]);
         return 1;
      }
   }

   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

   /* Read the image */
   nifti_image *image = reg_io_ReadImageFile(param->inputImageName);
   if(image == NULL)
   {
      fprintf(stderr,"** ERROR Error when reading the target image: %s\n",param->inputImageName);
      return 1;
   }
   reg_checkAndCorrectDimension(image);

   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

   if(flag->floatFlag)
   {
      reg_tools_changeDatatype<float>(image);
      if(flag->outputImageFlag)
         reg_io_WriteImageFile(image, param->outputImageName);
      else reg_io_WriteImageFile(image, "output.nii");
   }

   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

   if(flag->downsampleFlag)
   {
      bool dim[8]= {true,true,true,true,true,true,true,true};
      reg_downsampleImage<float>(image, true, dim);
      if(flag->outputImageFlag)
         reg_io_WriteImageFile(image, param->outputImageName);
      else reg_io_WriteImageFile(image, "output.nii");
   }

   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

   // The image intensity are normalised between the 3 and 97%ile values
   if(flag->normFlag)
   {
      reg_tools_changeDatatype<float>(image);
      nifti_image *normImage = nifti_copy_nim_info(image);
      normImage->data = (void *)malloc(normImage->nvox * normImage->nbyper);
      memcpy(normImage->data, image->data, normImage->nvox*normImage->nbyper);
      reg_heapSort(static_cast<float *>(normImage->data), normImage->nvox);
      float minValue = static_cast<float *>(normImage->data)[static_cast<int>(reg_floor(03*(int)normImage->nvox/100))];
      float maxValue = static_cast<float *>(normImage->data)[static_cast<int>(reg_floor(97*(int)normImage->nvox/100))];
      reg_tools_substractValueToImage(image,normImage,minValue);
      reg_tools_divideValueToImage(normImage,normImage,maxValue-minValue);
      if(flag->outputImageFlag)
         reg_io_WriteImageFile(normImage, param->outputImageName);
      else reg_io_WriteImageFile(normImage, "output.nii");
      nifti_image_free(normImage);
   }

   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

   if(flag->smoothGaussianFlag || flag->smoothSplineFlag || flag->smoothMeanFlag)
   {
      nifti_image *smoothImg = nifti_copy_nim_info(image);
      smoothImg->data = (void *)malloc(smoothImg->nvox * smoothImg->nbyper);
      memcpy(smoothImg->data, image->data, smoothImg->nvox*smoothImg->nbyper);
      float *kernelSize = new float[smoothImg->nt*smoothImg->nu];
      bool *timePoint = new bool[smoothImg->nt*smoothImg->nu];
      for(int i=0; i<smoothImg->nt*smoothImg->nu; ++i) timePoint[i]=true;
      bool boolX[3]= {1,0,0};
      for(int i=0; i<smoothImg->nt*smoothImg->nu; ++i) kernelSize[i]=param->smoothValueX;
      if(flag->smoothMeanFlag)
         reg_tools_kernelConvolution(smoothImg,kernelSize,2,NULL,timePoint,boolX);
      else if(flag->smoothSplineFlag)
         reg_tools_kernelConvolution(smoothImg,kernelSize,1,NULL,timePoint,boolX);
      else reg_tools_kernelConvolution(smoothImg,kernelSize,0,NULL,timePoint,boolX);
      bool boolY[3]= {0,1,0};
      for(int i=0; i<smoothImg->nt*smoothImg->nu; ++i) kernelSize[i]=param->smoothValueY;
      if(flag->smoothMeanFlag)
         reg_tools_kernelConvolution(smoothImg,kernelSize,2,NULL,timePoint,boolY);
      else if(flag->smoothSplineFlag)
         reg_tools_kernelConvolution(smoothImg,kernelSize,1,NULL,timePoint,boolY);
      else reg_tools_kernelConvolution(smoothImg,kernelSize,0,NULL,timePoint,boolY);
      bool boolZ[3]= {0,0,1};
      for(int i=0; i<smoothImg->nt*smoothImg->nu; ++i) kernelSize[i]=param->smoothValueZ;
      if(flag->smoothMeanFlag)
         reg_tools_kernelConvolution(smoothImg,kernelSize,2,NULL,timePoint,boolZ);
      else if(flag->smoothSplineFlag)
         reg_tools_kernelConvolution(smoothImg,kernelSize,1,NULL,timePoint,boolZ);
      else reg_tools_kernelConvolution(smoothImg,kernelSize,0,NULL,timePoint,boolZ);
      delete []kernelSize;
      delete []timePoint;
      if(flag->outputImageFlag)
         reg_io_WriteImageFile(smoothImg, param->outputImageName);
      else reg_io_WriteImageFile(smoothImg, "output.nii");
      nifti_image_free(smoothImg);
   }


   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

   if(flag->smoothLabFlag)
   {
      nifti_image *smoothImg = nifti_copy_nim_info(image);
      smoothImg->data = (void *)malloc(smoothImg->nvox * smoothImg->nbyper);
      memcpy(smoothImg->data, image->data, smoothImg->nvox*smoothImg->nbyper);

      bool *timePoint = new bool[smoothImg->nt*smoothImg->nu];
      for(int i=0; i<smoothImg->nt*smoothImg->nu; ++i) timePoint[i]=true;

      float varX=param->smoothValueX;
      float varY=param->smoothValueY;
      float varZ=param->smoothValueZ;

      reg_tools_labelKernelConvolution(smoothImg,varX,varY,varZ,NULL,timePoint);

      delete []timePoint;
      if(flag->outputImageFlag)
         reg_io_WriteImageFile(smoothImg, param->outputImageName);
      else reg_io_WriteImageFile(smoothImg, "output.nii");
      nifti_image_free(smoothImg);
   }

   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

   if(flag->operationTypeFlag>-1)
   {
      nifti_image *image2=NULL;
      if(param->operationImageName!=NULL)
      {
         image2 = reg_io_ReadImageFile(param->operationImageName);
         if(image2 == NULL)
         {
            fprintf(stderr,"** ERROR Error when reading the image: %s\n",param->operationImageName);
            return 1;
         }
         reg_checkAndCorrectDimension(image2);
      }
      // Images are converted to the higher datatype
      if(image2!=NULL){
         switch(image->datatype>image2->datatype?image->datatype:image2->datatype)
         {
         case NIFTI_TYPE_UINT8:
            reg_tools_changeDatatype<unsigned char>(image,NIFTI_TYPE_UINT8);
            reg_tools_changeDatatype<unsigned char>(image2,NIFTI_TYPE_UINT8);
            break;
         case NIFTI_TYPE_INT8:
            reg_tools_changeDatatype<char>(image,NIFTI_TYPE_INT8);
            reg_tools_changeDatatype<char>(image2,NIFTI_TYPE_INT8);
            break;
         case NIFTI_TYPE_UINT16:
            reg_tools_changeDatatype<unsigned short>(image,NIFTI_TYPE_UINT16);
            reg_tools_changeDatatype<unsigned short>(image2,NIFTI_TYPE_UINT16);
            break;
         case NIFTI_TYPE_INT16:
            reg_tools_changeDatatype<short>(image,NIFTI_TYPE_INT16);
            reg_tools_changeDatatype<short>(image2,NIFTI_TYPE_INT16);
            break;
         case NIFTI_TYPE_UINT32:
            reg_tools_changeDatatype<unsigned int>(image,NIFTI_TYPE_UINT32);
            reg_tools_changeDatatype<unsigned int>(image2,NIFTI_TYPE_UINT32);
            break;
         case NIFTI_TYPE_INT32:
            reg_tools_changeDatatype<int>(image,NIFTI_TYPE_INT32);
            reg_tools_changeDatatype<int>(image2,NIFTI_TYPE_INT32);
            break;
         case NIFTI_TYPE_FLOAT32:
            reg_tools_changeDatatype<float>(image,NIFTI_TYPE_FLOAT32);
            reg_tools_changeDatatype<float>(image2,NIFTI_TYPE_FLOAT32);
            break;
         case NIFTI_TYPE_FLOAT64:
            reg_tools_changeDatatype<double>(image,NIFTI_TYPE_FLOAT64);
            reg_tools_changeDatatype<double>(image2,NIFTI_TYPE_FLOAT64);
            break;
         default:
            reg_print_msg_error("Unsurported data type.");
            reg_exit(1);
         }
      }

      nifti_image *resultImage = nifti_copy_nim_info(image);
      resultImage->data = (void *)malloc(resultImage->nvox * resultImage->nbyper);

      if(image2!=NULL)
      {
         switch(flag->operationTypeFlag)
         {
         case 0:
            reg_tools_addImageToImage(image, image2, resultImage);
            break;
         case 1:
            reg_tools_substractImageToImage(image, image2, resultImage);
            break;
         case 2:
            reg_tools_multiplyImageToImage(image, image2, resultImage);
            break;
         case 3:
            reg_tools_divideImageToImage(image, image2, resultImage);
            break;
         }
      }
      else
      {
         switch(flag->operationTypeFlag)
         {
         case 0:
            reg_tools_addValueToImage(image, resultImage, param->operationValue);
            break;
         case 1:
            reg_tools_substractValueToImage(image, resultImage, param->operationValue);
            break;
         case 2:
            reg_tools_multiplyValueToImage(image, resultImage, param->operationValue);
            break;
         case 3:
            reg_tools_divideValueToImage(image, resultImage, param->operationValue);
            break;
         }
      }
      if(flag->outputImageFlag)
         reg_io_WriteImageFile(resultImage,param->outputImageName);
      else reg_io_WriteImageFile(resultImage,"output.nii");

      nifti_image_free(resultImage);
      if(image2!=NULL) nifti_image_free(image2);
   }

   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

   if(flag->rmsImageFlag)
   {
      nifti_image *image2 = reg_io_ReadImageFile(param->rmsImageName);
      if(image2 == NULL)
      {
         fprintf(stderr,"** ERROR Error when reading the image: %s\n",param->rmsImageName);
         return 1;
      }
      reg_checkAndCorrectDimension(image2);
      // Check image dimension
      if(image->dim[0]!=image2->dim[0] ||
            image->dim[1]!=image2->dim[1] ||
            image->dim[2]!=image2->dim[2] ||
            image->dim[3]!=image2->dim[3] ||
            image->dim[4]!=image2->dim[4] ||
            image->dim[5]!=image2->dim[5] ||
            image->dim[6]!=image2->dim[6] ||
            image->dim[7]!=image2->dim[7])
      {
         fprintf(stderr,"Both images do not have the same dimension\n");
         return 1;
      }

      double meanRMSerror = reg_tools_getMeanRMS(image, image2);
      printf("%g\n", meanRMSerror);
      nifti_image_free(image2);
   }
   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
   if(flag->binarisedImageFlag)
   {
      reg_tools_binarise_image(image);
      reg_tools_changeDatatype<unsigned char>(image);
      if(flag->outputImageFlag)
         reg_io_WriteImageFile(image,param->outputImageName);
      else reg_io_WriteImageFile(image,"output.nii");
   }
   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
   if(flag->thresholdImageFlag)
   {
      reg_tools_binarise_image(image, param->thresholdImageValue);
      reg_tools_changeDatatype<unsigned char>(image);
      if(flag->outputImageFlag)
         reg_io_WriteImageFile(image,param->outputImageName);
      else reg_io_WriteImageFile(image,"output.nii");
   }
   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
   if(flag->nanMaskFlag)
   {
      nifti_image *maskImage = reg_io_ReadImageFile(param->operationImageName);
      if(maskImage == NULL)
      {
         fprintf(stderr,"** ERROR Error when reading the image: %s\n",param->operationImageName);
         return 1;
      }
      reg_checkAndCorrectDimension(maskImage);

      nifti_image *resultImage = nifti_copy_nim_info(image);
      resultImage->data = (void *)malloc(resultImage->nvox * resultImage->nbyper);

      reg_tools_nanMask_image(image,maskImage,resultImage);

      if(flag->outputImageFlag)
         reg_io_WriteImageFile(resultImage,param->outputImageName);
      else reg_io_WriteImageFile(resultImage,"output.nii");

      nifti_image_free(resultImage);
      nifti_image_free(maskImage);
   }
   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
   if(flag->iso)
   {
      nifti_image *resultImage = reg_makeIsotropic(image,3);
      if(flag->outputImageFlag)
         reg_io_WriteImageFile(resultImage,param->outputImageName);
      else reg_io_WriteImageFile(resultImage,"output.nii");
      nifti_image_free(resultImage);
   }
   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
   if(flag->nosclFlag)
   {
      reg_tools_removeSCLInfo(image);
      if(flag->outputImageFlag)
         reg_io_WriteImageFile(image,param->outputImageName);
      else reg_io_WriteImageFile(image,"output.nii");
   }
   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
   if(flag->removeNanInf)
   {
      size_t nanNumber=0, infNumber=0,finNumber=0;
      if(image->datatype==NIFTI_TYPE_FLOAT32)
      {
         float *imgDataPtr=static_cast<float *>(image->data);
         for(size_t i=0;i<image->nvox;++i){
            float value=imgDataPtr[i];
            if(value==std::numeric_limits<float>::quiet_NaN()){
               nanNumber++;
               imgDataPtr[i]=param->removeNanInfValue;
            }
            else if(value==std::numeric_limits<float>::infinity()){
               infNumber++;
               imgDataPtr[i]=param->removeNanInfValue;
            }
            else finNumber++;
         }
      }
      else if(image->datatype==NIFTI_TYPE_FLOAT64)
      {
         double *imgDataPtr=static_cast<double *>(image->data);
         for(size_t i=0;i<image->nvox;++i){
            double value=imgDataPtr[i];
            if(value==std::numeric_limits<double>::quiet_NaN()){
               nanNumber++;
               imgDataPtr[i]=param->removeNanInfValue;
            }
            else if(value==std::numeric_limits<double>::infinity()){
               infNumber++;
               imgDataPtr[i]=param->removeNanInfValue;
            }
            else finNumber++;
         }
      }
      else{
         reg_print_msg_error("Nan and Inf value can only be removed when the input image is of float or double datatype");
         return 1;
      }
      printf("The input image contained %lu NaN, %lu Inf and %lu finite values\n",
             nanNumber, infNumber, finNumber);
   }
   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

   //    reg_tools_changeDatatype<float>(image);
   //    nifti_image *temp=nifti_copy_nim_info(image);
   //    temp->scl_inter=0.f;
   //    temp->scl_slope=1.f;
   //    temp->data=(void *)calloc(temp->nbyper,temp->nvox);
   //    float *imgPtr = static_cast<float *>(image->data);
   //    float *tmpPtr = static_cast<float *>(temp->data);

   //    size_t blockNumber=(image->nx/4)*(image->ny/4)*(image->nz/4);

   //    float *block_values=new float[blockNumber];

   //    size_t blockIndex=0;
   //    int znum=0;
   //    for(size_t z=0; z<image->nz; z+=4){
   //        int blockZ[2]={z,z+4};


   //        if(blockZ[1]<=image->nz){
   //            znum++;

   //            for(size_t y=0; y<image->ny; y+=4){
   //                int blockY[2]={y,y+4};

   //                if(blockY[1]<=image->ny){

   //                    for(size_t x=0; x<image->nx; x+=4){
   //                        int blockX[2]={x,x+4};

   //                        if(blockX[1]<=image->nx){

   //                            float mean=0;
   //                            for(int zz=blockZ[0];zz<blockZ[1];++zz){
   //                                for(int yy=blockY[0];yy<blockY[1];++yy){
   //                                    for(int xx=blockX[0];xx<blockX[1];++xx){
   //                                        mean+=imgPtr[(zz*image->ny+yy)*image->nx+xx];
   //                                    }
   //                                }
   //                            }
   //                            mean/=64.f;
   //                            float stddev=0;
   //                            for(int zz=blockZ[0];zz<blockZ[1];++zz){
   //                                for(int yy=blockY[0];yy<blockY[1];++yy){
   //                                    for(int xx=blockX[0];xx<blockX[1];++xx){
   //                                        stddev+=(mean-imgPtr[(zz*image->ny+yy)*image->nx+xx])*(mean-imgPtr[(zz*image->ny+yy)*image->nx+xx]);
   //                                    }
   //                                }
   //                            }
   //                            stddev/=64.f;
   //                            block_values[blockIndex]=stddev;
   //                            blockIndex++;
   //                        } //ifx
   //                    } //x
   //                } //ify
   //            } //y
   //        } //ifz
   //    } //z
   //    int *block_index=new int[blockNumber];
   //    for(int i=0;i<blockNumber;++i){
   //        block_index[i]=i;
   //    }

   //    reg_heapSort(block_values,block_index,blockNumber);

   //    for(int i=blockNumber-1;i>blockNumber/2;--i){

   //        int z=block_index[i]/((int)floor(image->nx/4)*(int)floor(image->ny/4));
   //        int temporary=block_index[i]-z*(int)floor(image->nx/4)*(int)floor(image->ny/4);
   //        int y=temporary/(int)floor(image->nx/4);
   //        int x=temporary-y*(int)floor(image->nx/4);
   //        for(int zz=z*4;zz<z*4+4;++zz){
   //            for(int yy=y*4;yy<y*4+4;++yy){
   //                for(int xx=x*4;xx<x*4+4;++xx){
   //                    tmpPtr[(zz*image->ny+yy)*image->nx+xx]=1.f;
   //                }
   //            }
   //        }
   //    }
   //    delete []block_index;
   //    delete []block_values;
   //    reg_io_WriteImageFile(temp,param->outputImageName);
   //    nifti_image_free(temp);

   nifti_image_free(image);
   return 0;
}

#endif
