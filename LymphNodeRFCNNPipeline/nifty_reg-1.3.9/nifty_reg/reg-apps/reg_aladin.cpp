/*
 *  reg_aladin.cpp
 *
 *
 *  Created by Marc Modat and Pankaj Daga on 12/08/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */


#ifndef _MM_ALADIN_CPP
#define _MM_ALADIN_CPP


#include "_reg_ReadWriteImage.h"
#include "_reg_aladin_sym.h"
#include "_reg_tools.h"
#include "reg_aladin.h"

#ifdef _WINDOWS
#include <time.h>
#endif

#ifdef _USE_NR_DOUBLE
#define PrecisionTYPE double
#else
#define PrecisionTYPE float
#endif

void PetitUsage(char *exec)
{
    fprintf(stderr,"\n");
    fprintf(stderr,"reg_aladin\n");
    fprintf(stderr,"Usage:\t%s -ref <referenceImageName> -flo <floatingImageName> [OPTIONS].\n",exec);
    fprintf(stderr,"\tSee the help for more details (-h).\n");
    fprintf(stderr,"\n");
    return;
}
void Usage(char *exec)
{
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    printf("Block Matching algorithm for global registration.\n");
    printf("Based on Ourselin et al., \"Reconstructing a 3D structure from serial histological sections\",\n");
    printf("Image and Vision Computing, 2001\n");
    printf("This code has been written by Marc Modat (m.modat@ucl.ac.uk) and Pankaj Daga,\n");
    printf("for any comment, please contact them.\n");
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    printf("Usage:\t%s -ref <filename> -flo <filename> [OPTIONS].\n",exec);
    printf("\t-ref <filename>\tFilename of the reference (target) image (mandatory)\n");
    printf("\t-flo <filename>\tFilename of the floating (source) image (mandatory)\n");
    printf("* * OPTIONS * *\n");
#ifdef _BUILD_NR_DEV
    printf("\t-sym \t\t\tUses symmetric version of the algorithm.\n");
#endif
    printf("\t-aff <filename>\t\tFilename which contains the output affine transformation [outputAffine.txt]\n");
    printf("\t-rigOnly\t\tTo perform a rigid registration only (rigid+affine by default)\n");
    printf("\t-affDirect\t\tDirectly optimize 12 DoF affine [default is rigid initially then affine]\n");
    printf("\t-inaff <filename>\tFilename which contains an input affine transformation (Affine*Reference=Floating) [none]\n");
    printf("\t-affFlirt <filename>\tFilename which contains an input affine transformation from Flirt [none]\n");
    printf("\t-rmask <filename>\tFilename of a mask image in the reference space\n");
    printf("\t-fmask <filename>\tFilename of a mask image in the floating space. Only used when symmetric turned on\n");
    printf("\t-res <filename>\tFilename of the resampled image [outputResult.nii]\n");
    printf("\t-maxit <int>\t\tNumber of iteration per level [5]\n");
    printf("\t-smooR <float>\t\tSmooth the reference image using the specified sigma (mm) [0]\n");
    printf("\t-smooF <float>\t\tSmooth the floating image using the specified sigma (mm) [0]\n");
    printf("\t-ln <int>\t\tNumber of level to perform [3]\n");
    printf("\t-lp <int>\t\tOnly perform the first levels [ln]\n");

    printf("\t-nac\t\t\tUse the nifti header origins to initialise the translation\n");

    printf("\t-%%v <int>\t\tPercentage of block to use [50]\n");
    printf("\t-%%i <int>\t\tPercentage of inlier for the LTS [50]\n");
#ifdef _USE_CUDA	
    printf("\t-gpu \t\t\tTo use the GPU implementation [no]\n");
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

    time_t start; time(&start);

    int symFlag=0;

    char *referenceImageName=NULL;
    int referenceImageFlag=0;

    char *floatingImageName=NULL;
    int floatingImageFlag=0;

    char *outputAffineName=NULL;
    int outputAffineFlag=0;

    char *inputAffineName=NULL;
    int inputAffineFlag=0;
    int flirtAffineFlag=0;

    char *referenceMaskName=NULL;
    int referenceMaskFlag=0;

    char *floatingMaskName=NULL;
    int floatingMaskFlag=0;

    char *outputResultName=NULL;
    int outputResultFlag=0;

    int maxIter=5;
    int nLevels=3;
    int levelsToPerform=3;
    int affineFlag=1;
    int rigidFlag=1;
    float blockPercentage=50.0f;
    float inlierLts=50.0f;
    int alignCentre=1;
    int interpolation=1;
    float floatingSigma=0.0;
    float referenceSigma=0.0;

    /* read the input parameter */
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
           strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
           strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0){
            Usage(argv[0]);
            return 0;
        }
        else if(strcmp(argv[i], "--xml")==0){
            printf("%s",xml_aladin);
            return 0;
        }
#ifdef _SVN_REV
        if( strcmp(argv[i], "-version")==0 ||
            strcmp(argv[i], "-Version")==0 ||
            strcmp(argv[i], "-V")==0 ||
            strcmp(argv[i], "-v")==0 ||
            strcmp(argv[i], "--v")==0 ||
            strcmp(argv[i], "--version")==0){
            printf("NiftyReg revision number: %i\n",_SVN_REV);
            return 0;
        }
#endif
        else if(strcmp(argv[i], "-ref")==0 || strcmp(argv[i], "-target")==0 || strcmp(argv[i], "--ref")==0){
            referenceImageName=argv[++i];
            referenceImageFlag=1;
        }
        else if(strcmp(argv[i], "-flo")==0 || strcmp(argv[i], "-source")==0 || strcmp(argv[i], "--flo")==0){
            floatingImageName=argv[++i];
            floatingImageFlag=1;
        }
        else if(strcmp(argv[i], "-sym")==0 || strcmp(argv[i], "--sym")==0){
            symFlag=1;
        }
        else if(strcmp(argv[i], "-aff")==0 || strcmp(argv[i], "--aff")==0){
            outputAffineName=argv[++i];
            outputAffineFlag=1;
        }
        else if(strcmp(argv[i], "-inaff")==0 || strcmp(argv[i], "--inaff")==0){
            inputAffineName=argv[++i];
            inputAffineFlag=1;
        }
        else if(strcmp(argv[i], "-affFlirt")==0 || strcmp(argv[i], "--affFlirt")==0){
            inputAffineName=argv[++i];
            inputAffineFlag=1;
            flirtAffineFlag=1;
        }
        else if(strcmp(argv[i], "-rmask")==0 || strcmp(argv[i], "-tmask")==0 || strcmp(argv[i], "--rmask")==0){
            referenceMaskName=argv[++i];
            referenceMaskFlag=1;
        }
        else if(strcmp(argv[i], "-fmask")==0 || strcmp(argv[i], "-smask")==0 || strcmp(argv[i], "--fmask")==0){
            floatingMaskName=argv[++i];
            floatingMaskFlag=1;
        }
        else if(strcmp(argv[i], "-res")==0 || strcmp(argv[i], "-result")==0 || strcmp(argv[i], "--res")==0){
            outputResultName=argv[++i];
            outputResultFlag=1;
        }
        else if(strcmp(argv[i], "-maxit")==0 || strcmp(argv[i], "--maxit")==0){
            maxIter = atoi(argv[++i]);
        }
        else if(strcmp(argv[i], "-ln")==0 || strcmp(argv[i], "--ln")==0){
            nLevels=atoi(argv[++i]);
        }
        else if(strcmp(argv[i], "-lp")==0 || strcmp(argv[i], "--lp")==0){
            levelsToPerform=atoi(argv[++i]);
        }
        else if(strcmp(argv[i], "-smooR")==0 || strcmp(argv[i], "-smooT")==0 || strcmp(argv[i], "--smooR")==0){
            referenceSigma = (float)(atof(argv[++i]));
        }
        else if(strcmp(argv[i], "-smooF")==0 || strcmp(argv[i], "-smooS")==0 || strcmp(argv[i], "--smooF")==0){
            floatingSigma=(float)(atof(argv[++i]));
        }
        else if(strcmp(argv[i], "-rigOnly")==0 || strcmp(argv[i], "--rigOnly")==0){
            rigidFlag=1;
            affineFlag=0;
            }
        else if(strcmp(argv[i], "-affDirect")==0 || strcmp(argv[i], "--affDirect")==0){
          rigidFlag=0;
          affineFlag=1;
        }
        else if(strcmp(argv[i], "-nac")==0 || strcmp(argv[i], "--nac")==0){
            alignCentre=0;
        }
        else if(strcmp(argv[i], "-%v")==0 || strcmp(argv[i], "--vv")==0){
            blockPercentage=atof(argv[++i]);
        }
        else if(strcmp(argv[i], "-%i")==0 || strcmp(argv[i], "--ii")==0){
            inlierLts=atof(argv[++i]);
        }
        else if(strcmp(argv[i], "-interp")==0 || strcmp(argv[i], "--interp")==0){
            interpolation=atoi(argv[++i]);
        }
        else{
            fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
            PetitUsage(argv[0]);
            return 1;
        }
    }

    if(!referenceImageFlag || !floatingImageFlag){
        fprintf(stderr,"Err:\tThe reference and the floating image have to be defined.\n");
        PetitUsage(argv[0]);
        return 1;
    }

    // Output the command line
    printf("\n[NiftyReg ALADIN] Command line:\n\t");
    for(int i=0;i<argc;i++)
        printf(" %s", argv[i]);
    printf("\n\n");

    reg_aladin<PrecisionTYPE> *REG;
#ifdef _BUILD_NR_DEV
    if(symFlag)
    {
        REG = new reg_aladin_sym<PrecisionTYPE>;
        if ( (referenceMaskFlag && !floatingMaskName) || (!referenceMaskFlag && floatingMaskName) )
        {
            fprintf(stderr,"[NiftyReg Warning] You have one image mask option turned on but not the other.\n");
            fprintf(stderr,"[NiftyReg Warning] This will affect the degree of symmetry achieved.\n");
        }
    }
    else
    {
#endif
        REG = new reg_aladin<PrecisionTYPE>;
#ifdef _BUILD_NR_DEV
        if (floatingMaskFlag)
        {
            fprintf(stderr,"Note: Floating mask flag only used in symmetric method. Ignoring this option\n");
        }
    }
#endif
    REG->SetMaxIterations(maxIter);
    REG->SetNumberOfLevels(nLevels);
    REG->SetLevelsToPerform(levelsToPerform);
    REG->SetReferenceSigma(referenceSigma);
    REG->SetFloatingSigma(floatingSigma);
    REG->SetAlignCentre(alignCentre);
    REG->SetPerformAffine(affineFlag);
    REG->SetPerformRigid(rigidFlag);
    REG->SetBlockPercentage(blockPercentage);
    REG->SetInlierLts(inlierLts);
    REG->SetInterpolation(interpolation);

    if(REG->GetLevelsToPerform() > REG->GetNumberOfLevels())
        REG->SetLevelsToPerform(REG->GetNumberOfLevels());

    /* Read the reference image and check its dimension */
    nifti_image *referenceHeader = reg_io_ReadImageFile(referenceImageName);
    if(referenceHeader == NULL){
        fprintf(stderr,"* ERROR Error when reading the reference  image: %s\n",referenceImageName);
        return 1;
    }

    /* Read teh floating image and check its dimension */
    nifti_image *floatingHeader = reg_io_ReadImageFile(floatingImageName);
    if(floatingHeader == NULL){
        fprintf(stderr,"* ERROR Error when reading the floating image: %s\n",floatingImageName);
        return 1;
    }

    // Set the reference and floating image
    REG->SetInputReference(referenceHeader);
    REG->SetInputFloating(floatingHeader);

    // Set the input affine transformation if defined
    if(inputAffineFlag==1)
        REG->SetInputTransform(inputAffineName,flirtAffineFlag);

    /* read the reference mask image */
    nifti_image *referenceMaskImage=NULL;
    if(referenceMaskFlag){
        referenceMaskImage = reg_io_ReadImageFile(referenceMaskName);
        if(referenceMaskImage == NULL){
            fprintf(stderr,"* ERROR Error when reading the reference mask image: %s\n",referenceMaskName);
            return 1;
        }
        /* check the dimension */
        for(int i=1; i<=referenceHeader->dim[0]; i++){
            if(referenceHeader->dim[i]!=referenceMaskImage->dim[i]){
                fprintf(stderr,"* ERROR The reference image and its mask do not have the same dimension\n");
                return 1;
            }
        }
        REG->SetInputMask(referenceMaskImage);
    }
#ifdef _BUILD_NR_DEV
    nifti_image *floatingMaskImage=NULL;
    if(floatingMaskFlag && symFlag){
        floatingMaskImage = reg_io_ReadImageFile(floatingMaskName);
        if(floatingMaskImage == NULL){
            fprintf(stderr,"* ERROR Error when reading the floating mask image: %s\n",referenceMaskName);
            return 1;
        }
        /* check the dimension */
        for(int i=1; i<=floatingHeader->dim[0]; i++){
            if(floatingHeader->dim[i]!=floatingMaskImage->dim[i]){
                fprintf(stderr,"* ERROR The floating image and its mask do not have the same dimension\n");
                return 1;
            }
        }
        REG->SetInputFloatingMask(floatingMaskImage);
    }
#endif
    REG->Run();

    // The warped image is saved
    nifti_image *outputResultImage=REG->GetFinalWarpedImage();
    if(!outputResultFlag) outputResultName=(char *)"outputResult.nii";
    reg_io_WriteImageFile(outputResultImage,outputResultName);
    nifti_image_free(outputResultImage);

    /* The affine transformation is saved */
    if(outputAffineFlag)
        reg_tool_WriteAffineFile(REG->GetTransformationMatrix(), outputAffineName);
    else reg_tool_WriteAffineFile(REG->GetTransformationMatrix(), (char *)"outputAffine.txt");

    nifti_image_free(referenceHeader);
    nifti_image_free(floatingHeader);

    delete REG;
    time_t end; time(&end);
    int minutes=(int)floorf((end-start)/60.0f);
    int seconds=(int)(end-start - 60*minutes);
    printf("Registration Performed in %i min %i sec\n", minutes, seconds);
    printf("Have a good day !\n");

    return 0;
}

#endif
