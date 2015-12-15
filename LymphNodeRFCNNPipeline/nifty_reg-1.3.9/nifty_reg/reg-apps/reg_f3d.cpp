/*
 *  reg_f3d.cpp
 *
 *
 *  Created by Marc Modat on 26/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_ReadWriteImage.h"
#include "_reg_f3d2.h"
#include "reg_f3d.h"

#ifdef _USE_CUDA
#include "_reg_f3d_gpu.h"
#endif
#include "float.h"
#include <limits>

#ifdef _WINDOWS
    #include <time.h>
#endif

#ifdef _USE_NR_DOUBLE
    #define PrecisionTYPE double
#else
    #define PrecisionTYPE float
#endif

void HelpPenaltyTerm()
{
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    printf("Additional help on the penalty term that have been implemented in F3D\n");
    printf("\t-be\t Bending Energy, sum of the second derivatives of the transformation T\n");
    printf("\t\t\t (d2T/dxx)^2 + (d2T/dyy)^2 + (d2T/dzz)^2 + 2*((d2T/dxy)^2 + (d2T/dyz)^2 + (d2T/dxz)^2)\n");
    printf("\t-le\t Linear Elasticity, 2 parameters weighted differently:\n");
    printf("\t\t\t 1: Squared member of the symmetric part of the Jacobian matrix\n");
    printf("\t\t\t 1: (dTx/dx)^2 + (dTy/dy)^2 + (dTz/dz)^2 + 1/2 * ( (dTx/dy+dTy/dx)^2 +  (dTx/dz+dTz/dx)^2 +  (dTy/dz+dTz/dy)^2 ) \n");
    printf("\t\t\t 2: Divergence\n");
    printf("\t\t\t 2: (dTx/dx)^2 + (dTy/dy)^2 + (dTz/dz)^2\n");
    printf("\t-l2\t Squared Eucliean distance of the displacement field D\n");
    printf("\t\t\t (Dx)^2 + (Dy)^2 + (Dz)^2\n");
    printf("\t-jl\t Penalty term based on the Jacobian determiant |J(T)|. Squared log of the Jacobian determinant\n");
    printf("\t\t\t log^2(|J(T)|)\n");
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    return;
}
void PetitUsage(char *exec)
{
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    fprintf(stderr,"Fast Free-Form Deformation algorithm for non-rigid registration.\n");
    fprintf(stderr,"Usage:\t%s -target <targetImageName> -source <sourceImageName> [OPTIONS].\n",exec);
    fprintf(stderr,"\tSee the help for more details (-h).\n");
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    return;
}
void Usage(char *exec)
{
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    printf("Fast Free-Form Deformation algorithm for non-rigid registration.\n");
    printf("This implementation is a re-factoring of Daniel Rueckert' 99 TMI work.\n");
    printf("The code is presented in Modat et al., \"Fast Free-Form Deformation using\n");
    printf("graphics processing units\", CMPB, 2010\n");
    printf("Cubic B-Spline are used to deform a source image in order to optimise a objective function\n");
    printf("based on the Normalised Mutual Information and a penalty term. The penalty term could\n");
    printf("be either the bending energy or the squared Jacobian determinant log.\n");
    printf("This code has been written by Marc Modat (m.modat@ucl.ac.uk), for any comment,\n");
    printf("please contact him.\n");
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    printf("Usage:\t%s -ref <filename> -flo <filename> [OPTIONS].\n",exec);
    printf("\t-ref <filename>\tFilename of the reference image (mandatory)\n");
    printf("\t-flo <filename>\tFilename of the floating image (mandatory)\n");
    printf("\n***************\n*** OPTIONS ***\n***************\n");
#ifdef _SVN_REV
    fprintf(stderr,"\n-v Print the subversion revision number\n");
#endif

    printf("\n*** Initial transformation options (One option will be considered):\n");
    printf("\t-aff <filename>\t\tFilename which contains an affine transformation (Affine*Reference=Floating)\n");
    printf("\t-affFlirt <filename>\tFilename which contains a flirt affine transformation (Flirt from the FSL package)\n");
    printf("\t-incpp <filename>\tFilename ofloatf control point grid input\n\t\t\t\tThe coarse spacing is defined by this file.\n");

    printf("\n*** Output options:\n");
    printf("\t-cpp <filename>\t\tFilename of control point grid [outputCPP.nii]\n");
    printf("\t-res <filename> \tFilename of the resampled image [outputResult.nii]\n");

    printf("\n*** Input image options:\n");
    printf("\t-rmask <filename>\t\tFilename of a mask image in the reference space\n");
    printf("\t-smooR <float>\t\t\tSmooth the reference image using the specified sigma (mm) [0]\n");
    printf("\t-smooF <float>\t\t\tSmooth the floating image using the specified sigma (mm) [0]\n");
    printf("\t--rbn <int>\t\t\tNumber of bin to use for the joint histogram [64]. Identical value for every timepoint.\n");
    printf("\t--fbn <int>\t\t\tNumber of bin to use for the joint histogram [64]. Identical value for every timepoint.\n");
    printf("\t-rbn <timepoint> <int>\t\tNumber of bin to use for the joint histogram [64]\n");
    printf("\t-fbn <timepoint> <int>\t\tNumber of bin to use for the joint histogram [64]\n");
    printf("\t--rLwTh <float>\t\t\tLower threshold to apply to the reference image intensities [none]. Identical value for every timepoint.*\n");
    printf("\t--rUpTh <float>\t\t\tUpper threshold to apply to the reference image intensities [none]. Identical value for every timepoint.*\n");
    printf("\t--fLwTh <float>\t\t\tLower threshold to apply to the floating image intensities [none]. Identical value for every timepoint.*\n");
    printf("\t--fUpTh <float>\t\t\tUpper threshold to apply to the floating image intensities [none]. Identical value for every timepoint.*\n");
    printf("\t-rLwTh <timepoint> <float>\tLower threshold to apply to the reference image intensities [none]*\n");
    printf("\t-rUpTh <timepoint> <float>\tUpper threshold to apply to the reference image intensities [none]*\n");
    printf("\t-fLwTh <timepoint> <float>\tLower threshold to apply to the floating image intensities [none]*\n");
    printf("\t-fUpTh <timepoint> <float>\tUpper threshold to apply to the floating image intensities [none]*\n");
    printf("\t* The scl_slope and scl_inter from the nifti header are taken into account for the thresholds\n");

    printf("\n*** Spline options:\n");
    printf("\t-sx <float>\t\tFinal grid spacing along the x axis in mm (in voxel if negative value) [5 voxels]\n");
    printf("\t-sy <float>\t\tFinal grid spacing along the y axis in mm (in voxel if negative value) [sx value]\n");
    printf("\t-sz <float>\t\tFinal grid spacing along the z axis in mm (in voxel if negative value) [sx value]\n");

    printf("\n*** Objective function options:\n");
    printf("\t-be <float>\t\tWeight of the bending energy penalty term [0.005]\n");
    printf("\t-le <float> <float>\tWeights of linear elasticity penalty term [0.0 0.0]\n");
    printf("\t-l2 <float>\t\tWeights of L2 norm displacement penalty term [0.0]\n");
    printf("\t-jl <float>\t\tWeight of log of the Jacobian determinant penalty term [0.0]\n");
    printf("\t-noAppJL\t\tTo not approximate the JL value only at the control point position\n");
    printf("\t-noConj\t\t\tTo not use the conjuage gradient optimisation but a simple gradient ascent\n");
    printf("\t-ssd\t\t\tTo use the SSD as the similiarity measure (NMI by default)\n");
    printf("\t-kld\t\t\tTo use the KL divergence as the similiarity measure (NMI by default)*\n");
    printf("\t-amc\t\t\tTo use the additive NMI for multichannel data (bivariate NMI by default)*\n");
    printf("\t* For the Kullback–Leibler divergence, reference and floating are expected to be probabilities\n");

    printf("\n*** Optimisation options:\n");
    printf("\t-maxit <int>\t\tMaximal number of iteration per level [300]\n");
    printf("\t-ln <int>\t\tNumber of level to perform [3]\n");
    printf("\t-lp <int>\t\tOnly perform the first levels [ln]\n");
    printf("\t-nopy\t\t\tDo not use a pyramidal approach\n");

    printf("\n*** F3D_SYM options:\n");
    printf("\t-sym \t\t\tUse symmetric approach\n");
    printf("\t-fmask <filename>\tFilename of a mask image in the floating space\n");
    printf("\t-ic <float>\t\tWeight of the inverse consistency penalty term [0.01]\n");

#ifdef _BUILD_NR_DEV
    printf("\n*** F3D2 options:\n");
    printf("\t-vel \t\t\tUse a velocity field integrationto generate the deformation\n");
    printf("\t-step <int>\t\tNumber of composition step [6].\n");
#endif // _BUILD_NR_DEV

    printf("\n*** Other options:\n");
    printf("\t-smoothGrad <float>\tTo smooth the metric derivative (in mm) [0]\n");
    printf("\t-pad <float>\t\tPadding value [nan]\n");
    printf("\t-voff\t\t\tTo turn verbose off\n");

#ifdef _USE_CUDA
    printf("\n*** GPU-related options:\n");
    printf("\t-mem\t\t\tDisplay an approximate memory requierment and exit\n");
    printf("\t-gpu \t\t\tTo use the GPU implementation [no]\n");
#endif
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    printf("For further description of the penalty term, use: %s -helpPenalty\n", exec);
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

    char *referenceName=NULL;
    char *floatingName=NULL;
    char *referenceMaskName=NULL;
    char *inputControlPointGridName=NULL;
    char *outputControlPointGridName=NULL;
    char *outputWarpedName=NULL;
    char *affineTransformationName=NULL;
    bool flirtAffine=false;
    PrecisionTYPE bendingEnergyWeight=std::numeric_limits<PrecisionTYPE>::quiet_NaN();
    PrecisionTYPE linearEnergyWeight0=std::numeric_limits<PrecisionTYPE>::quiet_NaN();
    PrecisionTYPE linearEnergyWeight1=std::numeric_limits<PrecisionTYPE>::quiet_NaN();
    PrecisionTYPE L2NormWeight=std::numeric_limits<PrecisionTYPE>::quiet_NaN();
    PrecisionTYPE jacobianLogWeight=std::numeric_limits<PrecisionTYPE>::quiet_NaN();
    bool jacobianLogApproximation=true;
    int maxiterationNumber=-1;
    PrecisionTYPE referenceSmoothingSigma=std::numeric_limits<PrecisionTYPE>::quiet_NaN();
    PrecisionTYPE floatingSmoothingSigma=std::numeric_limits<PrecisionTYPE>::quiet_NaN();
    PrecisionTYPE referenceThresholdUp[10];
    PrecisionTYPE referenceThresholdLow[10];
    PrecisionTYPE floatingThresholdUp[10];
    PrecisionTYPE floatingThresholdLow[10];
    unsigned int referenceBinNumber[10];
    unsigned int floatingBinNumber[10];
    for(int i=0;i<10;i++){
        referenceThresholdUp[i]=std::numeric_limits<PrecisionTYPE>::quiet_NaN();
        referenceThresholdLow[i]=std::numeric_limits<PrecisionTYPE>::quiet_NaN();
        floatingThresholdUp[i]=std::numeric_limits<PrecisionTYPE>::quiet_NaN();
        floatingThresholdLow[i]=std::numeric_limits<PrecisionTYPE>::quiet_NaN();
        referenceBinNumber[i]=0;
        floatingBinNumber[i]=0;
    }
    bool parzenWindowApproximation=true;
    PrecisionTYPE warpedPaddingValue=std::numeric_limits<PrecisionTYPE>::quiet_NaN();
    PrecisionTYPE spacing[3];
    spacing[0]=std::numeric_limits<PrecisionTYPE>::quiet_NaN();
    spacing[1]=std::numeric_limits<PrecisionTYPE>::quiet_NaN();
    spacing[2]=std::numeric_limits<PrecisionTYPE>::quiet_NaN();
    unsigned int levelNumber=0;
    unsigned int levelToPerform=0;
    PrecisionTYPE gradientSmoothingSigma=std::numeric_limits<PrecisionTYPE>::quiet_NaN();
    bool verbose=true;
    bool useConjugate=true;
    bool useSSD=false;
    bool useKLD=false;
    bool noPyramid=0;
    int interpolation=1;
    bool xOptimisation=true;
    bool yOptimisation=true;
    bool zOptimisation=true;
    bool gridRefinement=true;

    bool useSym=false;
    bool additiveNMI = false;
    char *floatingMaskName=NULL;
    float inverseConsistencyWeight=std::numeric_limits<PrecisionTYPE>::quiet_NaN();

#ifdef _BUILD_NR_DEV
    int stepNumber=-1;
    bool useVel=false;
#endif

#ifdef _USE_CUDA
    bool useGPU=false;
    bool checkMem=false;
    int cardNumber=-1;
#endif

    /* read the input parameter */
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
           strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
           strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0){
            Usage(argv[0]);
            return 0;
        }
        else if(strcmp(argv[i], "--xml")==0){
            printf("%s",xml_f3d);
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
        if(strcmp(argv[i], "-helpPenalty")==0){
            HelpPenaltyTerm();
            return 0;
        }
        else if((strcmp(argv[i],"-ref")==0) || (strcmp(argv[i],"-target")==0) || (strcmp(argv[i],"--ref")==0)){
            referenceName=argv[++i];
        }
        else if((strcmp(argv[i],"-flo")==0) || (strcmp(argv[i],"-source")==0) || (strcmp(argv[i],"--flo")==0)){
            floatingName=argv[++i];
        }
        else if(strcmp(argv[i], "-voff")==0){
            verbose=false;
        }
        else if(strcmp(argv[i], "-aff")==0 || (strcmp(argv[i],"--aff")==0)){
            affineTransformationName=argv[++i];
        }
        else if(strcmp(argv[i], "-affFlirt")==0 || (strcmp(argv[i],"--affFlirt")==0)){
            affineTransformationName=argv[++i];
            flirtAffine=1;
        }
        else if(strcmp(argv[i], "-incpp")==0 || (strcmp(argv[i],"--incpp")==0)){
            inputControlPointGridName=argv[++i];
        }
        else if((strcmp(argv[i],"-rmask")==0) || (strcmp(argv[i],"-tmask")==0) || (strcmp(argv[i],"--rmask")==0)){
            referenceMaskName=argv[++i];
        }
        else if((strcmp(argv[i],"-res")==0) || (strcmp(argv[i],"-result")==0) || (strcmp(argv[i],"--res")==0)){
            outputWarpedName=argv[++i];
        }
        else if(strcmp(argv[i], "-cpp")==0 || (strcmp(argv[i],"--cpp")==0)){
            outputControlPointGridName=argv[++i];
        }
        else if(strcmp(argv[i], "-maxit")==0 || strcmp(argv[i], "--maxit")==0){
            maxiterationNumber=atoi(argv[++i]);
        }
        else if(strcmp(argv[i], "-sx")==0 || strcmp(argv[i], "--sx")==0){
            spacing[0]=(float)atof(argv[++i]);
        }
        else if(strcmp(argv[i], "-sy")==0 || strcmp(argv[i], "--sy")==0){
            spacing[1]=(float)atof(argv[++i]);
        }
        else if(strcmp(argv[i], "-sz")==0 || strcmp(argv[i], "--sz")==0){
            spacing[2]=(float)atof(argv[++i]);
        }
        else if((strcmp(argv[i],"-rbn")==0) || (strcmp(argv[i],"-tbn")==0)){
            referenceBinNumber[atoi(argv[i+1])]=atoi(argv[i+2]);
            i+=2;
        }
        else if((strcmp(argv[i],"--rbn")==0) ){
            referenceBinNumber[0]=atoi(argv[++i]);
            for(unsigned j=1;j<10;++j) referenceBinNumber[j]=referenceBinNumber[0];
        }
        else if((strcmp(argv[i],"-fbn")==0) || (strcmp(argv[i],"-sbn")==0)){
            floatingBinNumber[atoi(argv[i+1])]=atoi(argv[i+2]);
            i+=2;
        }
        else if((strcmp(argv[i],"--fbn")==0) ){
            floatingBinNumber[0]=atoi(argv[++i]);
            for(unsigned j=1;j<10;++j) floatingBinNumber[j]=floatingBinNumber[0];
        }
        else if(strcmp(argv[i], "-ln")==0 || strcmp(argv[i], "--ln")==0){
            levelNumber=atoi(argv[++i]);
        }
        else if(strcmp(argv[i], "-lp")==0 || strcmp(argv[i], "--lp")==0){
           levelToPerform=atoi(argv[++i]);
        }
        else if(strcmp(argv[i], "-be")==0 || strcmp(argv[i], "--be")==0){
            bendingEnergyWeight=(PrecisionTYPE)(atof(argv[++i]));
        }
        else if(strcmp(argv[i], "-le")==0){
            linearEnergyWeight0=(PrecisionTYPE)(atof(argv[++i]));
            linearEnergyWeight1=(PrecisionTYPE)(atof(argv[++i]));
        }
        else if(strcmp(argv[i], "-l2")==0 || strcmp(argv[i], "--l2")==0){
            L2NormWeight=(PrecisionTYPE)(atof(argv[++i]));
        }
        else if(strcmp(argv[i], "-jl")==0 || strcmp(argv[i], "--jl")==0){
            jacobianLogWeight=(PrecisionTYPE)(atof(argv[++i]));
        }
        else if(strcmp(argv[i], "-noAppJL")==0 || strcmp(argv[i], "--noAppJL")==0){
            jacobianLogApproximation=false;
        }
        else if((strcmp(argv[i],"-smooR")==0) || (strcmp(argv[i],"-smooT")==0) || strcmp(argv[i], "--smooR")==0){
            referenceSmoothingSigma=(PrecisionTYPE)(atof(argv[++i]));
        }
        else if((strcmp(argv[i],"-smooF")==0) || (strcmp(argv[i],"-smooS")==0) || strcmp(argv[i], "--smooF")==0){
            floatingSmoothingSigma=(PrecisionTYPE)(atof(argv[++i]));
        }
        else if((strcmp(argv[i],"-rLwTh")==0) || (strcmp(argv[i],"-tLwTh")==0)){
           referenceThresholdLow[atoi(argv[i+1])]=(PrecisionTYPE)(atof(argv[i+2]));
           i+=2;
        }
        else if((strcmp(argv[i],"-rUpTh")==0) || strcmp(argv[i],"-tUpTh")==0){
            referenceThresholdUp[atoi(argv[i+1])]=(PrecisionTYPE)(atof(argv[i+2]));
            i+=2;
        }
        else if((strcmp(argv[i],"-fLwTh")==0) || (strcmp(argv[i],"-sLwTh")==0)){
            floatingThresholdLow[atoi(argv[i+1])]=(PrecisionTYPE)(atof(argv[i+2]));
            i+=2;
        }
        else if((strcmp(argv[i],"-fUpTh")==0) || (strcmp(argv[i],"-sUpTh")==0)){
            floatingThresholdUp[atoi(argv[i+1])]=(PrecisionTYPE)(atof(argv[i+2]));
            i+=2;
        }
        else if((strcmp(argv[i],"--rLwTh")==0) ){
            referenceThresholdLow[0]=atof(argv[++i]);
            for(unsigned j=1;j<10;++j) referenceThresholdLow[j]=referenceThresholdLow[0];
        }
        else if((strcmp(argv[i],"--rUpTh")==0) ){
            referenceThresholdUp[0]=atof(argv[++i]);
            for(unsigned j=1;j<10;++j) referenceThresholdUp[j]=referenceThresholdUp[0];
        }
        else if((strcmp(argv[i],"--fLwTh")==0) ){
            floatingThresholdLow[0]=atof(argv[++i]);
            for(unsigned j=1;j<10;++j) floatingThresholdLow[j]=floatingThresholdLow[0];
        }
        else if((strcmp(argv[i],"--fUpTh")==0) ){
            floatingThresholdUp[0]=atof(argv[++i]);
            for(unsigned j=1;j<10;++j) floatingThresholdUp[j]=floatingThresholdUp[0];
        }
        else if(strcmp(argv[i], "-smoothGrad")==0){
            gradientSmoothingSigma=(PrecisionTYPE)(atof(argv[++i]));
        }
        else if(strcmp(argv[i], "-ssd")==0 || strcmp(argv[i], "--ssd")==0){
            useSSD=true;
            useKLD=false;
        }
        else if(strcmp(argv[i], "-kld")==0 || strcmp(argv[i], "--kld")==0){
            useSSD=false;
            useKLD=true;
        }
        else if(strcmp(argv[i], "-amc")==0){
            additiveNMI = true;
        }

        else if(strcmp(argv[i], "-pad")==0){
            warpedPaddingValue=(PrecisionTYPE)(atof(argv[++i]));
        }
        else if(strcmp(argv[i], "-nopy")==0 || strcmp(argv[i], "--nopy")==0){
            noPyramid=1;
        }
        else if(strcmp(argv[i], "-noConj")==0 || strcmp(argv[i], "--noConj")==0){
           useConjugate=false;
        }
        else if(strcmp(argv[i], "-interp")==0 || strcmp(argv[i], "--interp")==0){
            interpolation=atoi(argv[++i]);
        }
        else if(strcmp(argv[i], "-noAppPW")==0){
            parzenWindowApproximation=false;
        }
        else if(strcmp(argv[i], "-sym") ==0 || strcmp(argv[i], "--sym") ==0){
            useSym=true;
        }
        else if((strcmp(argv[i],"-fmask")==0) || (strcmp(argv[i],"-smask")==0) ||
                (strcmp(argv[i],"--fmask")==0) || (strcmp(argv[i],"--smask")==0)){
            floatingMaskName=argv[++i];
        }
        else if(strcmp(argv[i], "-ic")==0 || strcmp(argv[i], "--ic")==0){
            inverseConsistencyWeight=atof(argv[++i]);
        }
        else if(strcmp(argv[i], "-nox") ==0){
            xOptimisation=false;
        }
        else if(strcmp(argv[i], "-noy") ==0){
            yOptimisation=false;
        }
        else if(strcmp(argv[i], "-noz") ==0){
            zOptimisation=false;
        }
        else if(strcmp(argv[i], "-nogr") ==0){
            gridRefinement=false;
        }
#ifdef _BUILD_NR_DEV
        else if(strcmp(argv[i], "-vel")==0 || strcmp(argv[i], "--vel")==0){
            useVel=true;
        }
        else if(strcmp(argv[i], "-step")==0 || strcmp(argv[i], "--step")==0){
           stepNumber=atoi(argv[++i]);
        }
#endif
#ifdef _USE_CUDA
        else if(strcmp(argv[i], "-gpu")==0){
            useGPU=true;
        }
        else if(strcmp(argv[i], "-card")==0){
            cardNumber=atoi(argv[++i]);
        }
        else if(strcmp(argv[i], "-mem")==0){
            checkMem=true;
        }
#endif
        else{
            fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
            PetitUsage(argv[0]);
            return 1;
        }
    }

    if(referenceName==NULL || floatingName==NULL){
        fprintf(stderr,"Err:\tThe reference and the floating image have to be defined.\n");
        PetitUsage(argv[0]);
        return 1;
    }

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] *******************************************\n");
    printf("[NiftyReg DEBUG] *******************************************\n");
    printf("[NiftyReg DEBUG] NiftyReg has been compiled in DEBUG mode\n");
    printf("[NiftyReg DEBUG] Please rerun cmake to set the variable\n");
    printf("[NiftyReg DEBUG] CMAKE_BUILD_TYPE to \"Release\" if required\n");
    printf("[NiftyReg DEBUG] *******************************************\n");
    printf("[NiftyReg DEBUG] *******************************************\n");
#endif

    // Output the command line
#ifdef NDEBUG
    if(verbose==true){
#endif
        printf("\n[NiftyReg F3D] Command line:\n\t");
        for(int i=0;i<argc;i++)
            printf(" %s", argv[i]);
        printf("\n\n");
#ifdef NDEBUG
    }
#endif

    // Read the reference image
    if(referenceName==NULL){
        fprintf(stderr, "Error. No reference image has been defined\n");
        PetitUsage(argv[0]);
        return 1;
    }
    nifti_image *referenceImage = reg_io_ReadImageFile(referenceName);
    if(referenceImage==NULL){
        fprintf(stderr, "Error when reading the reference image %s\n",referenceName);
        return 1;
    }

    // Read the floating image
    if(floatingName==NULL){
        fprintf(stderr, "Error. No floating image has been defined\n");
        PetitUsage(argv[0]);
        return 1;
    }
    nifti_image *floatingImage = reg_io_ReadImageFile(floatingName);
    if(floatingImage==NULL){
        fprintf(stderr, "Error when reading the floating image %s\n",floatingName);
        return 1;
    }

    // Read the mask images
    nifti_image *referenceMaskImage=NULL;
    if(referenceMaskName!=NULL){
        referenceMaskImage=reg_io_ReadImageFile(referenceMaskName);
        if(referenceMaskImage==NULL){
            fprintf(stderr, "Error when reading the reference mask image %s\n",referenceMaskName);
            return 1;
        }
    }
    nifti_image *floatingMaskImage=NULL;
    if(floatingMaskName!=NULL){
        floatingMaskImage=reg_io_ReadImageFile(floatingMaskName);
        if(floatingMaskImage==NULL){
            fprintf(stderr, "Error when reading the reference mask image %s\n",floatingMaskName);
            return 1;
        }
    }

    // Read the input control point grid image
    nifti_image *controlPointGridImage=NULL;
    if(inputControlPointGridName!=NULL){
        controlPointGridImage=reg_io_ReadImageFile(inputControlPointGridName);
        if(controlPointGridImage==NULL){
            fprintf(stderr, "Error when reading the input control point grid image %s\n",inputControlPointGridName);
            return 1;
        }
#ifdef _BUILD_NR_DEV
        if( controlPointGridImage->intent_code==NIFTI_INTENT_VECTOR &&
            strcmp(controlPointGridImage->intent_name,"NREG_VEL_STEP")==0 &&
            fabs(controlPointGridImage->intent_p1)>1 )
            useVel=true;
#endif
    }

    // Read the affine transformation
    mat44 *affineTransformation=NULL;
    if(affineTransformationName!=NULL){
        affineTransformation = (mat44 *)malloc(sizeof(mat44));
        // Check first if the specified affine file exist
        if(FILE *aff=fopen(affineTransformationName, "r")){
            fclose(aff);
        }
        else{
            fprintf(stderr,"The specified input affine file (%s) can not be read\n",affineTransformationName);
            return 1;
        }
        reg_tool_ReadAffineFile(affineTransformation,
                                referenceImage,
                                floatingImage,
                                affineTransformationName,
                                flirtAffine);
    }

    // Create the reg_f3d object
    reg_f3d<PrecisionTYPE> *REG=NULL;
#ifdef _USE_CUDA
    CUdevice dev;
    CUcontext ctx;
    if(useGPU){

        if(linearEnergyWeight0==linearEnergyWeight0 ||
           linearEnergyWeight1==linearEnergyWeight1 ||
           L2NormWeight==L2NormWeight){
            fprintf(stderr,"NiftyReg ERROR CUDA] The linear elasticity has not been implemented with CUDA yet. Exit.\n");
            exit(0);
        }

        if(useSym){
            fprintf(stderr,"\n[NiftyReg ERROR CUDA] GPU implementation of the symmetric registration is not available yet. Exit\n");
            exit(0);
        }
#ifdef _BUILD_NR_DEV
        if(useVel){
            fprintf(stderr,"\n[NiftyReg ERROR CUDA] GPU implementation of velocity field parametrisartion is not available yet. Exit\n");
            exit(0);
        }
#endif

        if((referenceImage->dim[4]==1&&floatingImage->dim[4]==1) || (referenceImage->dim[4]==2&&floatingImage->dim[4]==2)){

            // The CUDA card is setup
            cuInit(0);
            struct cudaDeviceProp deviceProp;
            int device_count = 0;
            cudaGetDeviceCount( &device_count );
            int device=cardNumber;
            if(cardNumber==-1){
                // following code is from cutGetMaxGflopsDeviceId()
                int max_gflops_device = 0;
                int max_gflops = 0;
                int current_device = 0;
                while( current_device < device_count ){
                    cudaGetDeviceProperties( &deviceProp, current_device );
                    int gflops = deviceProp.multiProcessorCount * deviceProp.clockRate;
                    if( gflops > max_gflops ){
                        max_gflops = gflops;
                        max_gflops_device = current_device;
                    }
                    ++current_device;
                }
                device = max_gflops_device;
            }
            NR_CUDA_SAFE_CALL(cudaSetDevice( device ));
            NR_CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device ));
            cuDeviceGet(&dev,device);
            cuCtxCreate(&ctx, 0, dev);
            if (deviceProp.major < 1){
                printf("[NiftyReg ERROR CUDA] The specified graphical card does not exist.\n");
                return 1;
            }
            REG = new reg_f3d_gpu<PrecisionTYPE>(referenceImage->nt, floatingImage->nt);
#ifdef NDEBUG
            if(verbose==true){
  #endif
                printf("\n[NiftyReg F3D] GPU implementation is used\n");
#ifdef NDEBUG
            }
#endif
        }
        else{
            fprintf(stderr,"[NiftyReg ERROR] The GPU implementation only handle 1 to 1 or 2 to 2 image(s) registration\n");
            exit(1);
        }
    }
    else
#endif
    {
        if(useSym){
            REG = new reg_f3d_sym<PrecisionTYPE>(referenceImage->nt, floatingImage->nt);
#ifdef NDEBUG
            if(verbose==true){
#endif // NDEBUG
                printf("\n[NiftyReg F3D SYM] CPU implementation is used\n");
#ifdef NDEBUG
            }
#endif // NDEBUG
        }
#ifdef _BUILD_NR_DEV
        else if(useVel){
            REG = new reg_f3d2<PrecisionTYPE>(referenceImage->nt, floatingImage->nt);
#ifdef NDEBUG
            if(verbose==true){
  #endif
                printf("\n[NiftyReg F3D2] CPU implementation is used\n");
#ifdef NDEBUG
            }
#endif
        }
#endif // _BUILD_NR_DEV
        else{
            REG = new reg_f3d<PrecisionTYPE>(referenceImage->nt, floatingImage->nt);
#ifdef NDEBUG
            if(verbose==true){
#endif // NDEBUG
                printf("\n[NiftyReg F3D] CPU implementation is used\n");
#ifdef NDEBUG
            }
#endif // NDEBUG
        }
    }
#ifdef _OPENMP
    int maxThreadNumber = omp_get_max_threads();
#ifdef NDEBUG
    if(verbose==true){
#endif // NDEBUG
        printf("[NiftyReg F3D] OpenMP is used with %i thread(s)\n", maxThreadNumber);
#ifdef NDEBUG
    }
#endif // NDEBUG
#endif // _OPENMP

    // Set the reg_f3d parameters

    REG->SetReferenceImage(referenceImage);

    REG->SetFloatingImage(floatingImage);

    if(verbose==false) REG->DoNotPrintOutInformation();
    else REG->PrintOutInformation();

    if(referenceMaskImage!=NULL)
       REG->SetReferenceMask(referenceMaskImage);

    if(controlPointGridImage!=NULL)
       REG->SetControlPointGridImage(controlPointGridImage);

    if(affineTransformation!=NULL)
       REG->SetAffineTransformation(affineTransformation);

    if(bendingEnergyWeight==bendingEnergyWeight)
        REG->SetBendingEnergyWeight(bendingEnergyWeight);

    if(linearEnergyWeight0==linearEnergyWeight0 || linearEnergyWeight1==linearEnergyWeight1){
        if(linearEnergyWeight0!=linearEnergyWeight0) linearEnergyWeight0=0.0;
        if(linearEnergyWeight1!=linearEnergyWeight1) linearEnergyWeight1=0.0;
        REG->SetLinearEnergyWeights(linearEnergyWeight0,linearEnergyWeight1);
    }
    if(L2NormWeight==L2NormWeight)
        REG->SetL2NormDisplacementWeight(L2NormWeight);

    if(jacobianLogWeight==jacobianLogWeight)
        REG->SetJacobianLogWeight(jacobianLogWeight);

    if(jacobianLogApproximation)
        REG->ApproximateJacobianLog();
    else REG->DoNotApproximateJacobianLog();

    if(parzenWindowApproximation)
        REG->ApproximateParzenWindow();
    else REG->DoNotApproximateParzenWindow();

    if(maxiterationNumber>-1)
        REG->SetMaximalIterationNumber(maxiterationNumber);

    if(referenceSmoothingSigma==referenceSmoothingSigma)
        REG->SetReferenceSmoothingSigma(referenceSmoothingSigma);

    if(floatingSmoothingSigma==floatingSmoothingSigma)
        REG->SetFloatingSmoothingSigma(floatingSmoothingSigma);

    for(unsigned int t=0;t<(unsigned int)referenceImage->nt;t++)
        if(referenceThresholdUp[t]==referenceThresholdUp[t])
            REG->SetReferenceThresholdUp(t,referenceThresholdUp[t]);

    for(unsigned int t=0;t<(unsigned int)referenceImage->nt;t++)
        if(referenceThresholdLow[t]==referenceThresholdLow[t])
            REG->SetReferenceThresholdLow(t,referenceThresholdLow[t]);

    for(unsigned int t=0;t<(unsigned int)floatingImage->nt;t++)
        if(floatingThresholdUp[t]==floatingThresholdUp[t])
            REG->SetFloatingThresholdUp(t,floatingThresholdUp[t]);

    for(unsigned int t=0;t<(unsigned int)floatingImage->nt;t++)
        if(floatingThresholdLow[t]==floatingThresholdLow[t])
            REG->SetFloatingThresholdLow(t,floatingThresholdLow[t]);

    for(unsigned int t=0;t<(unsigned int)referenceImage->nt;t++)
        if(referenceBinNumber[t]>0)
            REG->SetReferenceBinNumber(t,referenceBinNumber[t]);

    for(unsigned int t=0;t<(unsigned int)floatingImage->nt;t++)
        if(floatingBinNumber[t]>0)
            REG->SetFloatingBinNumber(t,floatingBinNumber[t]);

    if(warpedPaddingValue==warpedPaddingValue)
        REG->SetWarpedPaddingValue(warpedPaddingValue);

    for(unsigned int s=0;s<3;s++)
        if(spacing[s]==spacing[s])
            REG->SetSpacing(s,spacing[s]);

    if(levelNumber>0)
        REG->SetLevelNumber(levelNumber);

    if(levelToPerform>0)
        REG->SetLevelToPerform(levelToPerform);

    if(gradientSmoothingSigma==gradientSmoothingSigma)
        REG->SetGradientSmoothingSigma(gradientSmoothingSigma);

    if(useSSD)
        REG->UseSSD();
    else REG->DoNotUseSSD();

    if(useKLD)
        REG->UseKLDivergence();
    else REG->DoNotUseKLDivergence();

    if(useConjugate==true)
        REG->UseConjugateGradient();
    else REG->DoNotUseConjugateGradient();

    if(noPyramid==1)
        REG->DoNotUsePyramidalApproach();

    switch(interpolation){
    case 0:REG->UseNeareatNeighborInterpolation();
        break;
    case 1:REG->UseLinearInterpolation();
        break;
    case 3:REG->UseCubicSplineInterpolation();
        break;
    default:REG->UseLinearInterpolation();
        break;
    }

    if(xOptimisation==false)
        REG->NoOptimisationAlongX();

    if(yOptimisation==false)
        REG->NoOptimisationAlongY();

    if(zOptimisation==false)
        REG->NoOptimisationAlongZ();

    if(gridRefinement==false)
        REG->NoGridRefinement();

    if (additiveNMI) REG->SetAdditiveMC();

    // F3D SYM arguments
    if(floatingMaskImage!=NULL){
        if(useSym)
            REG->SetFloatingMask(floatingMaskImage);
#ifdef _BUILD_NR_DEV
        else if(useVel)
            REG->SetFloatingMask(floatingMaskImage);
#endif
        else{
            fprintf(stderr, "[NiftyReg WARNING] The floating mask image is only used for symmetric or velocity field parametrisation\n");
            fprintf(stderr, "[NiftyReg WARNING] The floating mask image is ignored\n");
        }
    }

    if(inverseConsistencyWeight==inverseConsistencyWeight)
       REG->SetInverseConsistencyWeight(inverseConsistencyWeight);

#ifdef _BUILD_NR_DEV
    // F3D2 arguments
    if(stepNumber>-1)
        REG->SetCompositionStepNumber(stepNumber);
#endif

    // Run the registration
#ifdef _USE_CUDA
    if(useGPU && checkMem){
        size_t free, total, requiredMemory = REG->CheckMemoryMB_f3d();
        cuMemGetInfo(&free, &total);
        printf("[NiftyReg CUDA] The required memory to run the registration is %lu Mb\n",
               (unsigned long int)requiredMemory);
        printf("[NiftyReg CUDA] The GPU card has %lu Mb from which %lu Mb are currenlty free\n",
               (unsigned long int)total/(1024*1024), (unsigned long int)free/(1024*1024));
    }
    else{
#endif
        REG->Run_f3d();

        // Save the control point result
        nifti_image *outputControlPointGridImage = REG->GetControlPointPositionImage();
        if(outputControlPointGridName==NULL) outputControlPointGridName=(char *)"outputCPP.nii";
        memset(outputControlPointGridImage->descrip, 0, 80);
        strcpy (outputControlPointGridImage->descrip,"Control point position from NiftyReg (reg_f3d)");
#ifdef _BUILD_NR_DEV
        if(useVel)
            strcpy (outputControlPointGridImage->descrip,"Velocity field grid from NiftyReg (reg_f3d2)");
#endif
        reg_io_WriteImageFile(outputControlPointGridImage,outputControlPointGridName);
        nifti_image_free(outputControlPointGridImage);outputControlPointGridImage=NULL;

        // Save the backward control point result
#ifdef _BUILD_NR_DEV
        if(useSym || useVel){
#else
        if(useSym){
#endif
            // _backward is added to the forward control point grid image name
            std::string b(outputControlPointGridName);
            if(b.find( ".nii.gz") != std::string::npos)
                b.replace(b.find( ".nii.gz"),7,"_backward.nii.gz");
            else if(b.find( ".nii") != std::string::npos)
                b.replace(b.find( ".nii"),4,"_backward.nii");
            else if(b.find( ".hdr") != std::string::npos)
                b.replace(b.find( ".hdr"),4,"_backward.hdr");
            else if(b.find( ".img.gz") != std::string::npos)
                b.replace(b.find( ".img.gz"),7,"_backward.img.gz");
            else if(b.find( ".img") != std::string::npos)
                b.replace(b.find( ".img"),4,"_backward.img");
#ifdef _USE_NR_PNG
            else if(b.find( ".png") != std::string::npos)
                b.replace(b.find( ".png"),4,"_backward.png");
#endif
#ifdef _USE_NR_NRRD
            else if(b.find( ".nrrd") != std::string::npos)
                b.replace(b.find( ".nrrd"),5,"_backward.nrrd");
#endif
            else b.append("_backward.nii");
            nifti_image *outputBackwardControlPointGridImage = REG->GetBackwardControlPointPositionImage();
            memset(outputBackwardControlPointGridImage->descrip, 0, 80);
            strcpy (outputBackwardControlPointGridImage->descrip,"Backward Control point position from NiftyReg (reg_f3d)");
#ifdef _BUILD_NR_DEV
            if(useVel)
                strcpy (outputBackwardControlPointGridImage->descrip,"Backward velocity field grid from NiftyReg (reg_f3d2)");
#endif
            reg_io_WriteImageFile(outputBackwardControlPointGridImage,b.c_str());
            nifti_image_free(outputBackwardControlPointGridImage);outputBackwardControlPointGridImage=NULL;
        }

        // Save the warped image result(s)
        nifti_image **outputWarpedImage=(nifti_image **)malloc(2*sizeof(nifti_image *));
        outputWarpedImage[0]=outputWarpedImage[1]=NULL;
        outputWarpedImage = REG->GetWarpedImage();
        if(outputWarpedName==NULL) outputWarpedName=(char *)"outputResult.nii";
        memset(outputWarpedImage[0]->descrip, 0, 80);
        strcpy (outputWarpedImage[0]->descrip,"Warped image using NiftyReg (reg_f3d)");
        if(useSym){
            strcpy (outputWarpedImage[0]->descrip,"Warped image using NiftyReg (reg_f3d_sym)");
            strcpy (outputWarpedImage[1]->descrip,"Warped image using NiftyReg (reg_f3d_sym)");
#ifdef _BUILD_NR_DEV
        }
        if(useVel){
            strcpy (outputWarpedImage[0]->descrip,"Warped image using NiftyReg (reg_f3d2)");
            strcpy (outputWarpedImage[1]->descrip,"Warped image using NiftyReg (reg_f3d2)");
        }
        if(useSym || useVel){
#endif
            if(outputWarpedImage[1]!=NULL){
                std::string b(outputWarpedName);
                if(b.find( ".nii.gz") != std::string::npos)
                    b.replace(b.find( ".nii.gz"),7,"_backward.nii.gz");
                else if(b.find( ".nii") != std::string::npos)
                    b.replace(b.find( ".nii"),4,"_backward.nii");
                else if(b.find( ".hdr") != std::string::npos)
                    b.replace(b.find( ".hdr"),4,"_backward.hdr");
                else if(b.find( ".img.gz") != std::string::npos)
                    b.replace(b.find( ".img.gz"),7,"_backward.img.gz");
                else if(b.find( ".img") != std::string::npos)
                    b.replace(b.find( ".img"),4,"_backward.img");
#ifdef _USE_NR_PNG
                else if(b.find( ".png") != std::string::npos)
                    b.replace(b.find( ".png"),4,"_backward.png");
#endif
#ifdef _USE_NR_NRRD
                else if(b.find( ".nrrd") != std::string::npos)
                    b.replace(b.find( ".nrrd"),5,"_backward.nrrd");
#endif
                else b.append("_backward.nii");
                if(useSym)
                    strcpy (outputWarpedImage[1]->descrip,"Warped image using NiftyReg (reg_f3d_sym)");
#ifdef _BUILD_NR_DEV
                if(useVel)
                    strcpy (outputWarpedImage[1]->descrip,"Warped image using NiftyReg (reg_f3d2)");
#endif
                reg_io_WriteImageFile(outputWarpedImage[1],b.c_str());
                nifti_image_free(outputWarpedImage[1]);outputWarpedImage[1]=NULL;
            }
        }
        reg_io_WriteImageFile(outputWarpedImage[0],outputWarpedName);
        nifti_image_free(outputWarpedImage[0]);outputWarpedImage[0]=NULL;
        free(outputWarpedImage);
#ifdef _USE_CUDA
    }
    cuCtxDetach(ctx);
#endif
    // Erase the registration object
    delete REG;

    // Clean the allocated images
    if(referenceImage!=NULL) nifti_image_free(referenceImage);
    if(floatingImage!=NULL) nifti_image_free(floatingImage);
    if(controlPointGridImage!=NULL) nifti_image_free(controlPointGridImage);
    if(affineTransformation!=NULL) free(affineTransformation);
    if(referenceMaskImage!=NULL) nifti_image_free(referenceMaskImage);
    if(floatingMaskImage!=NULL) nifti_image_free(floatingMaskImage);

#ifdef NDEBUG
    if(verbose){
#endif
        time_t end; time( &end );
        int minutes = (int)floorf(float(end-start)/60.0f);
        int seconds = (int)(end-start - 60*minutes);

#ifdef _USE_CUDA
        if(!checkMem){
#endif
            printf("[NiftyReg F3D] Registration Performed in %i min %i sec\n", minutes, seconds);
            printf("[NiftyReg F3D] Have a good day !\n");
#ifdef _USE_CUDA
        }
#endif
#ifdef NDEBUG
    }
#endif
    return 0;
}
