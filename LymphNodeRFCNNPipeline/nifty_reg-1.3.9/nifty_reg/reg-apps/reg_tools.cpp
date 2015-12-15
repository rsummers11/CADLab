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

#ifdef _USE_NR_DOUBLE
#define PrecisionTYPE double
#else
#define PrecisionTYPE float
#endif

int isNumeric (const char *s)
{
    if(s==NULL || *s=='\0' || isspace(*s))
      return 0;
    char * p;
    double a=0; //useless - here to avoid a warning
    a=strtod (s, &p);
    return *p == '\0';
}

typedef struct{
    char *inputImageName;
    char *outputImageName;
    char *operationImageName;
    char *rmsImageName;
    float operationValue;
    float smoothValue;
    float smoothValueX;
    float smoothValueY;
    float smoothValueZ;
    float thresholdImageValue;
}PARAM;
typedef struct{
    bool inputImageFlag;
    bool outputImageFlag;
    bool rmsImageFlag;
    bool smoothValueFlag;
    bool smoothGaussianFlag;
    bool binarisedImageFlag;
    bool thresholdImageFlag;
    bool nanMaskFlag;
    int operationTypeFlag;
}FLAG;


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
    printf("\t-add <filename/float>\tThis image (or value) is added to the input\n");
    printf("\t-sub <filename/float>\tThis image (or value) is subtracted to the input\n");
    printf("\t-mul <filename/float>\tThis image (or value) is multiplied to the input\n");
    printf("\t-div <filename/float>\tThis image (or value) is divided to the input\n");
    printf("\t-smo <float>\t\tThe input image is smoothed using a b-spline curve\n");
    printf("\t-smoG <float> <float> <float>\tThe input image is smoothed using Gaussian kernel\n");
    printf("\t-rms <filename>\t\tCompute the mean rms between both image\n");
    printf("\t-bin \t\t\tBinarise the input image (val!=0?val=1:val=0)\n");
    printf("\t-thr <float>\t\tThresold the input image (val<thr?val=0:val=1)\n");
    printf("\t-nan <filename>\t\tThis image is used to mask the input image.\n\t\t\t\tVoxels outside of the mask are set to nan\n");
#ifdef _SVN_REV
    printf("\t-v Print the subversion revision number\n");
#endif
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    return;
}

int main(int argc, char **argv)
{
    PARAM *param = (PARAM *)calloc(1,sizeof(PARAM));
    FLAG *flag = (FLAG *)calloc(1,sizeof(FLAG));
    flag->operationTypeFlag=-1;

    /* read the input parameter */
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
           strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
           strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0){
            Usage(argv[0]);
            return 0;
        }
        else if(strcmp(argv[i], "--xml")==0){
            printf("%s",xml_tools);
            return 0;
        }
#ifdef _SVN_REV
        if(strcmp(argv[i], "-version")==0 || strcmp(argv[i], "-Version")==0 ||
           strcmp(argv[i], "-V")==0 || strcmp(argv[i], "-v")==0 ||
           strcmp(argv[i], "--v")==0 || strcmp(argv[i], "--version")==0){
            printf("NiftyReg revision number: %i\n",_SVN_REV);
            return 0;
        }
#endif
        else if(strcmp(argv[i], "-in") == 0){
            param->inputImageName=argv[++i];
            flag->inputImageFlag=1;
        }
        else if(strcmp(argv[i], "-out") == 0){
            param->outputImageName=argv[++i];
            flag->outputImageFlag=1;
        }

        else if(strcmp(argv[i], "-add") == 0){
            param->operationImageName=argv[++i];
            if(isNumeric(param->operationImageName)==true){
                param->operationValue=(float)atof(param->operationImageName);
                param->operationImageName=NULL;
            }
            flag->operationTypeFlag=0;
        }
        else if(strcmp(argv[i], "-sub") == 0){
            param->operationImageName=argv[++i];
            if(isNumeric(param->operationImageName)){
                param->operationValue=(float)atof(param->operationImageName);
                param->operationImageName=NULL;
            }
            flag->operationTypeFlag=1;
        }
        else if(strcmp(argv[i], "-mul") == 0){
            param->operationImageName=argv[++i];
            if(isNumeric(param->operationImageName)){
                param->operationValue=(float)atof(param->operationImageName);
                param->operationImageName=NULL;
            }
            flag->operationTypeFlag=2;
        }
        else if(strcmp(argv[i], "-div") == 0){
            param->operationImageName=argv[++i];
            if(isNumeric(param->operationImageName)){
                param->operationValue=(float)atof(param->operationImageName);
                param->operationImageName=NULL;
            }
            flag->operationTypeFlag=3;
        }

        else if(strcmp(argv[i], "-rms") == 0){
            param->rmsImageName=argv[++i];
            flag->rmsImageFlag=1;
        }
        else if(strcmp(argv[i], "-smo") == 0){
            param->smoothValue=atof(argv[++i]);
            flag->smoothValueFlag=1;
        }
        else if(strcmp(argv[i], "-smoG") == 0){
            param->smoothValueX=atof(argv[++i]);
            param->smoothValueY=atof(argv[++i]);
            param->smoothValueZ=atof(argv[++i]);
            flag->smoothGaussianFlag=1;
        }
        else if(strcmp(argv[i], "-bin") == 0){
            flag->binarisedImageFlag=1;
        }
        else if(strcmp(argv[i], "-thr") == 0){
            param->thresholdImageValue=atof(argv[++i]);
            flag->thresholdImageFlag=1;
        }
        else if(strcmp(argv[i], "-nan") == 0){
            param->operationImageName=argv[++i];
            flag->nanMaskFlag=1;
        }
        else{
            fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
            PetitUsage(argv[0]);
            return 1;
        }
    }

    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    /* Read the image */
    nifti_image *image = reg_io_ReadImageFile(param->inputImageName);
    if(image == NULL){
        fprintf(stderr,"** ERROR Error when reading the target image: %s\n",param->inputImageName);
        return 1;
    }
    reg_checkAndCorrectDimension(image);

    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    if(flag->smoothValueFlag){
        nifti_image *smoothImg = nifti_copy_nim_info(image);
        smoothImg->data = (void *)malloc(smoothImg->nvox * smoothImg->nbyper);
        memcpy(smoothImg->data, image->data, smoothImg->nvox*smoothImg->nbyper);
        int radius[3];radius[0]=radius[1]=radius[2]=param->smoothValue;
        reg_smoothNormImageForCubicSpline<PrecisionTYPE>(smoothImg, radius);
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(smoothImg, param->outputImageName);
        else reg_io_WriteImageFile(smoothImg, "output.nii");
        nifti_image_free(smoothImg);
    }

    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    if(flag->smoothGaussianFlag){
        nifti_image *smoothImg = nifti_copy_nim_info(image);
        smoothImg->data = (void *)malloc(smoothImg->nvox * smoothImg->nbyper);
        memcpy(smoothImg->data, image->data, smoothImg->nvox*smoothImg->nbyper);
        bool boolX[8]={3,1,0,0,0,0,0,0};
        reg_gaussianSmoothing(smoothImg,param->smoothValueX,boolX);
        bool boolY[8]={3,0,1,0,0,0,0,0};
        reg_gaussianSmoothing(smoothImg,param->smoothValueY,boolY);
        bool boolZ[8]={3,0,0,1,0,0,0,0};
        reg_gaussianSmoothing(smoothImg,param->smoothValueZ,boolZ);
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(smoothImg, param->outputImageName);
        else reg_io_WriteImageFile(smoothImg, "output.nii");
        nifti_image_free(smoothImg);
    }

    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    if(flag->operationTypeFlag>-1){
        nifti_image *image2=NULL;
        if(param->operationImageName!=NULL){
            image2 = reg_io_ReadImageFile(param->operationImageName);
            if(image2 == NULL){
                fprintf(stderr,"** ERROR Error when reading the image: %s\n",param->operationImageName);
                return 1;
            }
            reg_checkAndCorrectDimension(image2);
        }

        nifti_image *resultImage = nifti_copy_nim_info(image);
        resultImage->data = (void *)malloc(resultImage->nvox * resultImage->nbyper);

        if(image2!=NULL)
            reg_tools_addSubMulDivImages(image, image2, resultImage, flag->operationTypeFlag);
        else reg_tools_addSubMulDivValue(image, resultImage, param->operationValue, flag->operationTypeFlag);

        if(flag->outputImageFlag)
            reg_io_WriteImageFile(resultImage,param->outputImageName);
        else reg_io_WriteImageFile(resultImage,"output.nii");

        nifti_image_free(resultImage);
        if(image2!=NULL) nifti_image_free(image2);
    }

    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//

    if(flag->rmsImageFlag){
        nifti_image *image2 = reg_io_ReadImageFile(param->rmsImageName);
        if(image2 == NULL){
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
           image->dim[7]!=image2->dim[7]){
            fprintf(stderr,"Both images do not have the same dimension\n");
            return 1;
        }

        double meanRMSerror = reg_tools_getMeanRMS(image, image2);
        printf("%g\n", meanRMSerror);
        nifti_image_free(image2);
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    if(flag->binarisedImageFlag){
        reg_tools_binarise_image(image);
        reg_tools_changeDatatype<unsigned char>(image);
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(image,param->outputImageName);
        else reg_io_WriteImageFile(image,"output.nii");
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    if(flag->thresholdImageFlag){
        reg_tools_binarise_image(image, param->thresholdImageValue);
        reg_tools_changeDatatype<unsigned char>(image);
        if(flag->outputImageFlag)
            reg_io_WriteImageFile(image,param->outputImageName);
        else reg_io_WriteImageFile(image,"output.nii");
    }
    //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//
    if(flag->nanMaskFlag){
        nifti_image *maskImage = reg_io_ReadImageFile(param->operationImageName);
        if(maskImage == NULL){
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

    nifti_image_free(image);
    return 0;
}

#endif
