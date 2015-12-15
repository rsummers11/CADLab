/*
 *  _reg_f3_symd.cpp
 *
 *
 *  Created by Marc Modat on 10/11/2011.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_F3D_SYM_CPP
#define _REG_F3D_SYM_CPP

#include "_reg_f3d_sym.h"

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d_sym<T>::reg_f3d_sym(int refTimePoint,int floTimePoint)
    :reg_f3d<T>::reg_f3d(refTimePoint,floTimePoint)
{
    this->executableName=(char *)"NiftyReg F3D SYM";

    this->backwardControlPointGrid=NULL;
    this->backwardWarped=NULL;
    this->backwardWarpedGradientImage=NULL;
    this->backwardDeformationFieldImage=NULL;
    this->backwardVoxelBasedMeasureGradientImage=NULL;
    this->backwardNodeBasedGradientImage=NULL;

    this->backwardBestControlPointPosition=NULL;
    this->backwardConjugateG=NULL;
    this->backwardConjugateH=NULL;

    this->backwardProbaJointHistogram=NULL;
    this->backwardLogJointHistogram=NULL;

    this->floatingMaskImage=NULL;
    this->currentFloatingMask=NULL;
    this->floatingMaskPyramid=NULL;
    this->backwardActiveVoxelNumber=NULL;

    this->inverseConsistencyWeight=0.1;

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_sym constructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_f3d_sym<T>::~reg_f3d_sym()
{
    if(this->backwardControlPointGrid!=NULL){
        nifti_image_free(this->backwardControlPointGrid);
        this->backwardControlPointGrid=NULL;
    }

    if(this->floatingMaskPyramid!=NULL){
        if(this->usePyramid){
            for(unsigned int i=0;i<this->levelToPerform;i++){
                if(this->floatingMaskPyramid[i]!=NULL){
                    free(this->floatingMaskPyramid[i]);
                    this->floatingMaskPyramid[i]=NULL;
                }
            }
        }
        else{
            if(this->floatingMaskPyramid[0]!=NULL){
                free(this->floatingMaskPyramid[0]);
                this->floatingMaskPyramid[0]=NULL;
            }
        }
        free(this->floatingMaskPyramid);
        floatingMaskPyramid=NULL;
    }

    if(this->backwardActiveVoxelNumber!=NULL){
        free(this->backwardActiveVoxelNumber);
        this->backwardActiveVoxelNumber=NULL;
    }

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_sym destructor called\n");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::SetFloatingMask(nifti_image *m)
{
    this->floatingMaskImage = m;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::SetInverseConsistencyWeight(T w)
{
    this->inverseConsistencyWeight = w;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateCurrentInputImage()
{
    reg_f3d<T>::AllocateCurrentInputImage();
    if(this->currentLevel!=0)
        reg_bspline_refineControlPointGrid(this->currentFloating, this->backwardControlPointGrid);

    if(this->usePyramid){
        this->currentFloatingMask = this->floatingMaskPyramid[this->currentLevel];
    }
    else{
        this->currentFloatingMask = this->floatingMaskPyramid[0];
    }

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearCurrentInputImage()
{
    reg_f3d<T>::ClearCurrentInputImage();
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateWarped()
{
    this->ClearWarped();

    reg_f3d<T>::AllocateWarped();
    if(this->currentFloating==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The floating image is not defined\n");
        exit(1);
    }
    this->backwardWarped = nifti_copy_nim_info(this->currentFloating);
    this->backwardWarped->dim[0]=this->backwardWarped->ndim=this->currentReference->ndim;
    this->backwardWarped->dim[4]=this->backwardWarped->nt=this->currentReference->nt;
    this->backwardWarped->pixdim[4]=this->backwardWarped->dt=1.0;
    this->backwardWarped->nvox = this->backwardWarped->nx *
            this->backwardWarped->ny *
            this->backwardWarped->nz *
            this->backwardWarped->nt;
    this->backwardWarped->datatype = this->currentReference->datatype;
    this->backwardWarped->nbyper = this->currentReference->nbyper;
    this->backwardWarped->data = (void *)calloc(this->backwardWarped->nvox, this->backwardWarped->nbyper);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearWarped()
{
    reg_f3d<T>::ClearWarped();
    if(this->backwardWarped!=NULL){
        nifti_image_free(this->backwardWarped);
        this->backwardWarped=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateDeformationField()
{
    this->ClearDeformationField();

    reg_f3d<T>::AllocateDeformationField();
    if(this->currentFloating==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The floating image is not defined\n");
        exit(1);
    }
    if(this->backwardControlPointGrid==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The backward control point image is not defined\n");
        exit(1);
    }
    this->backwardDeformationFieldImage = nifti_copy_nim_info(this->currentFloating);
    this->backwardDeformationFieldImage->dim[0]=this->backwardDeformationFieldImage->ndim=5;
    this->backwardDeformationFieldImage->dim[1]=this->backwardDeformationFieldImage->nx=this->currentFloating->nx;
    this->backwardDeformationFieldImage->dim[2]=this->backwardDeformationFieldImage->ny=this->currentFloating->ny;
    this->backwardDeformationFieldImage->dim[3]=this->backwardDeformationFieldImage->nz=this->currentFloating->nz;
    this->backwardDeformationFieldImage->dim[4]=this->backwardDeformationFieldImage->nt=1;
    this->backwardDeformationFieldImage->pixdim[4]=this->backwardDeformationFieldImage->dt=1.0;
    if(this->currentFloating->nz==1)
        this->backwardDeformationFieldImage->dim[5]=this->backwardDeformationFieldImage->nu=2;
    else this->backwardDeformationFieldImage->dim[5]=this->backwardDeformationFieldImage->nu=3;
    this->backwardDeformationFieldImage->pixdim[5]=this->backwardDeformationFieldImage->du=1.0;
    this->backwardDeformationFieldImage->dim[6]=this->backwardDeformationFieldImage->nv=1;
    this->backwardDeformationFieldImage->pixdim[6]=this->backwardDeformationFieldImage->dv=1.0;
    this->backwardDeformationFieldImage->dim[7]=this->backwardDeformationFieldImage->nw=1;
    this->backwardDeformationFieldImage->pixdim[7]=this->backwardDeformationFieldImage->dw=1.0;
    this->backwardDeformationFieldImage->nvox=	this->backwardDeformationFieldImage->nx *
            this->backwardDeformationFieldImage->ny *
            this->backwardDeformationFieldImage->nz *
            this->backwardDeformationFieldImage->nt *
            this->backwardDeformationFieldImage->nu;
    this->backwardDeformationFieldImage->nbyper = this->backwardControlPointGrid->nbyper;
    this->backwardDeformationFieldImage->datatype = this->backwardControlPointGrid->datatype;
    this->backwardDeformationFieldImage->data = (void *)calloc(this->backwardDeformationFieldImage->nvox, this->backwardDeformationFieldImage->nbyper);

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearDeformationField()
{
    reg_f3d<T>::ClearDeformationField();
    if(this->backwardDeformationFieldImage!=NULL){
        nifti_image_free(this->backwardDeformationFieldImage);
        this->backwardDeformationFieldImage=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateWarpedGradient()
{
    this->ClearWarpedGradient();

    reg_f3d<T>::AllocateWarpedGradient();
    if(this->backwardDeformationFieldImage==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The backward control point image is not defined\n");
        exit(1);
    }
    this->backwardWarpedGradientImage = nifti_copy_nim_info(this->backwardDeformationFieldImage);
    this->backwardWarpedGradientImage->dim[0]=this->backwardWarpedGradientImage->ndim=5;
    this->backwardWarpedGradientImage->nt = this->backwardWarpedGradientImage->dim[4] = this->currentReference->nt;
    this->backwardWarpedGradientImage->nvox =	this->backwardWarpedGradientImage->nx *
            this->backwardWarpedGradientImage->ny *
            this->backwardWarpedGradientImage->nz *
            this->backwardWarpedGradientImage->nt *
            this->backwardWarpedGradientImage->nu;
    this->backwardWarpedGradientImage->data = (void *)calloc(this->backwardWarpedGradientImage->nvox, this->backwardWarpedGradientImage->nbyper);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearWarpedGradient()
{
    reg_f3d<T>::ClearWarpedGradient();
    if(this->backwardWarpedGradientImage!=NULL){
        nifti_image_free(this->backwardWarpedGradientImage);
        this->backwardWarpedGradientImage=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateVoxelBasedMeasureGradient()
{
    this->ClearVoxelBasedMeasureGradient();

    reg_f3d<T>::AllocateVoxelBasedMeasureGradient();
    if(this->backwardDeformationFieldImage==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The backward control point image is not defined\n");
        exit(1);
    }
    this->backwardVoxelBasedMeasureGradientImage = nifti_copy_nim_info(this->backwardDeformationFieldImage);
    this->backwardVoxelBasedMeasureGradientImage->data =
            (void *)calloc(this->backwardVoxelBasedMeasureGradientImage->nvox,
                           this->backwardVoxelBasedMeasureGradientImage->nbyper);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearVoxelBasedMeasureGradient()
{
    reg_f3d<T>::ClearVoxelBasedMeasureGradient();
    if(this->backwardVoxelBasedMeasureGradientImage!=NULL){
        nifti_image_free(this->backwardVoxelBasedMeasureGradientImage);
        this->backwardVoxelBasedMeasureGradientImage=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateNodeBasedGradient()
{
    this->ClearNodeBasedGradient();

    reg_f3d<T>::AllocateNodeBasedGradient();
    if(this->backwardControlPointGrid==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The backward control point image is not defined\n");
        exit(1);
    }
    this->backwardNodeBasedGradientImage = nifti_copy_nim_info(this->backwardControlPointGrid);
    this->backwardNodeBasedGradientImage->data =
            (void *)calloc(this->backwardNodeBasedGradientImage->nvox,
                           this->backwardNodeBasedGradientImage->nbyper);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearNodeBasedGradient()
{
    reg_f3d<T>::ClearNodeBasedGradient();
    if(this->backwardNodeBasedGradientImage!=NULL){
        nifti_image_free(this->backwardNodeBasedGradientImage);
        this->backwardNodeBasedGradientImage=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateConjugateGradientVariables()
{
    this->ClearConjugateGradientVariables();

    reg_f3d<T>::AllocateConjugateGradientVariables();
    if(this->backwardControlPointGrid==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The backward control point image is not defined\n");
        exit(1);
    }
    this->backwardConjugateG = (T *)calloc(this->backwardControlPointGrid->nvox, sizeof(T));
    this->backwardConjugateH = (T *)calloc(this->backwardControlPointGrid->nvox, sizeof(T));
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearConjugateGradientVariables()
{
    reg_f3d<T>::ClearConjugateGradientVariables();
    if(this->backwardConjugateG!=NULL){
        free(this->backwardConjugateG);
        this->backwardConjugateG=NULL;
    }
    if(this->backwardConjugateH!=NULL){
        free(this->backwardConjugateH);
        this->backwardConjugateH=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateBestControlPointArray()
{
    this->ClearBestControlPointArray();

    reg_f3d<T>::AllocateBestControlPointArray();
    if(this->backwardControlPointGrid==NULL){
        fprintf(stderr, "[NiftyReg ERROR] The backward control point image is not defined\n");
        exit(1);
    }
    this->backwardBestControlPointPosition =
            (T *)malloc(this->backwardControlPointGrid->nvox *
                        this->backwardControlPointGrid->nbyper);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearBestControlPointArray()
{
    reg_f3d<T>::ClearBestControlPointArray();
    if(this->backwardBestControlPointPosition!=NULL){
        free(this->backwardBestControlPointPosition);
        this->backwardBestControlPointPosition=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::AllocateJointHistogram()
{
    this->ClearJointHistogram();

    reg_f3d<T>::AllocateJointHistogram();
    this->backwardProbaJointHistogram = (double *)malloc(this->totalBinNumber*sizeof(double));
    this->backwardLogJointHistogram = (double *)malloc(this->totalBinNumber*sizeof(double));
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::ClearJointHistogram()
{
    reg_f3d<T>::ClearJointHistogram();
    if(this->backwardProbaJointHistogram!=NULL){
        free(this->backwardProbaJointHistogram);
        this->backwardProbaJointHistogram=NULL;
    }
    if(this->backwardLogJointHistogram!=NULL){
        free(this->backwardLogJointHistogram);
        this->backwardLogJointHistogram=NULL;
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::SaveCurrentControlPoint()
{
    reg_f3d<T>::SaveCurrentControlPoint();
    memcpy(this->backwardBestControlPointPosition, this->backwardControlPointGrid->data,
           this->backwardControlPointGrid->nvox*this->backwardControlPointGrid->nbyper);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::RestoreCurrentControlPoint()
{
    reg_f3d<T>::RestoreCurrentControlPoint();
    memcpy(this->backwardControlPointGrid->data, this->backwardBestControlPointPosition,
           this->backwardControlPointGrid->nvox*this->controlPointGrid->nbyper);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::CheckParameters_f3d()
{

    reg_f3d<T>::CheckParameters_f3d();

    if(this->affineTransformation!=NULL){
        fprintf(stderr, "[NiftyReg F3D_SYM ERROR] The inverse consistency parametrisation does not handle affine input\n");
        fprintf(stderr, "[NiftyReg F3D_SYM ERROR] Please update your source image sform using reg_transform\n");
        fprintf(stderr, "[NiftyReg F3D_SYM ERROR] and use the updated source image as an input\n.");
        exit(1);
    }

    // CHECK THE FLOATING MASK DIMENSION IF IT IS DEFINED
    if(this->floatingMaskImage!=NULL){
        if(this->inputFloating->nx != this->floatingMaskImage->nx ||
                this->inputFloating->ny != this->floatingMaskImage->ny ||
                this->inputFloating->nz != this->floatingMaskImage->nz){
            fprintf(stderr,"* The floating mask image has different x, y or z dimension than the floating image.\n");
            exit(1);
        }
    }

    // NORMALISE THE OBJECTIVE FUNCTION WEIGHTS
    T penaltySum=
            this->bendingEnergyWeight
            +this->linearEnergyWeight0
            +this->linearEnergyWeight1
            +this->L2NormWeight
            +this->jacobianLogWeight
            +this->inverseConsistencyWeight;
    if(penaltySum>=1){
        this->similarityWeight=0;
        this->bendingEnergyWeight /= penaltySum;
        this->linearEnergyWeight0 /= penaltySum;
        this->linearEnergyWeight1 /= penaltySum;
        this->L2NormWeight /= penaltySum;
        this->jacobianLogWeight /= penaltySum;
        this->inverseConsistencyWeight /= penaltySum;
    }
    else this->similarityWeight=1.0 - penaltySum;

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::Initisalise_f3d()
{
    reg_f3d<T>::Initisalise_f3d();

    /* allocate the backward control point image */

    /* Convert the spacing from voxel to mm if necessary */
    float spacingInMillimeter[3]={this->spacing[0],this->spacing[1],this->spacing[2]};
    if(this->usePyramid){
        if(spacingInMillimeter[0]<0) spacingInMillimeter[0] *= -1.0f * this->floatingPyramid[this->levelToPerform-1]->dx;
        if(spacingInMillimeter[1]<0) spacingInMillimeter[1] *= -1.0f * this->floatingPyramid[this->levelToPerform-1]->dy;
        if(spacingInMillimeter[2]<0) spacingInMillimeter[2] *= -1.0f * this->floatingPyramid[this->levelToPerform-1]->dz;
    }
    else{
        if(spacingInMillimeter[0]<0) spacingInMillimeter[0] *= -1.0f * this->floatingPyramid[0]->dx;
        if(spacingInMillimeter[1]<0) spacingInMillimeter[1] *= -1.0f * this->floatingPyramid[0]->dy;
        if(spacingInMillimeter[2]<0) spacingInMillimeter[2] *= -1.0f * this->floatingPyramid[0]->dz;
    }

    // Define the spacing for the first level
    float gridSpacing[3];
    gridSpacing[0] = spacingInMillimeter[0] * powf(2.0f, (float)(this->levelToPerform-1));
    gridSpacing[1] = spacingInMillimeter[1] * powf(2.0f, (float)(this->levelToPerform-1));
    gridSpacing[2] = 1.0f;
    if(this->floatingPyramid[0]->nz>1)
        gridSpacing[2] = spacingInMillimeter[2] * powf(2.0f, (float)(this->levelToPerform-1));

    // Create and allocate the control point image
    reg_createControlPointGrid<T>(&this->backwardControlPointGrid,
                                  this->floatingPyramid[0],
                                  gridSpacing);

    // the backward control point is initialised using an affine transformation
    mat44 matrixAffine;
    matrixAffine.m[0][0]=1.f;
    matrixAffine.m[0][1]=0.f;
    matrixAffine.m[0][2]=0.f;
    matrixAffine.m[0][3]=0.f;
    matrixAffine.m[1][0]=0.f;
    matrixAffine.m[1][1]=1.f;
    matrixAffine.m[1][2]=0.f;
    matrixAffine.m[1][3]=0.f;
    matrixAffine.m[2][0]=0.f;
    matrixAffine.m[2][1]=0.f;
    matrixAffine.m[2][2]=1.f;
    matrixAffine.m[2][3]=0.f;
    matrixAffine.m[3][0]=0.f;
    matrixAffine.m[3][1]=0.f;
    matrixAffine.m[3][2]=0.f;
    matrixAffine.m[3][3]=1.f;
    if(reg_bspline_initialiseControlPointGridWithAffine(&matrixAffine, this->controlPointGrid))
        exit(1);
    if(reg_bspline_initialiseControlPointGridWithAffine(&matrixAffine, this->backwardControlPointGrid))
        exit(1);

    // Set the floating mask image pyramid
    if(this->usePyramid){
        this->floatingMaskPyramid = (int **)malloc(this->levelToPerform*sizeof(int *));
        this->backwardActiveVoxelNumber= (int *)malloc(this->levelToPerform*sizeof(int));
    }
    else{
        this->floatingMaskPyramid = (int **)malloc(sizeof(int *));
        this->backwardActiveVoxelNumber= (int *)malloc(sizeof(int));
    }

    if(this->usePyramid){
        if (this->floatingMaskImage!=NULL)
            reg_createMaskPyramid<T>(this->floatingMaskImage,
                                     this->floatingMaskPyramid,
                                     this->levelNumber,
                                     this->levelToPerform,
                                     this->backwardActiveVoxelNumber);
        else{
            for(unsigned int l=0;l<this->levelToPerform;++l){
                this->backwardActiveVoxelNumber[l]=this->floatingPyramid[l]->nx*this->floatingPyramid[l]->ny*this->floatingPyramid[l]->nz;
                this->floatingMaskPyramid[l]=(int *)calloc(backwardActiveVoxelNumber[l],sizeof(int));
            }
        }
    }
    else{ // no pyramid
        if (this->floatingMaskImage!=NULL)
            reg_createMaskPyramid<T>(this->floatingMaskImage, this->floatingMaskPyramid, 1, 1, this->backwardActiveVoxelNumber);
        else{
            this->backwardActiveVoxelNumber[0]=this->floatingPyramid[0]->nx*this->floatingPyramid[0]->ny*this->floatingPyramid[0]->nz;
            this->floatingMaskPyramid[0]=(int *)calloc(backwardActiveVoxelNumber[0],sizeof(int));
        }
    }

#ifdef NDEBUG
    if(this->verbose){
#endif
    printf("[%s]\n", this->executableName);
    printf("[%s] Inverse consistency error penalty term weight: %g\n",
           this->executableName, this->inverseConsistencyWeight);
#ifdef NDEBUG
    }
#endif

#ifndef NDEBUG
    printf("[NiftyReg DEBUG] reg_f3d_sym::Initialise_f3d() done\n");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetDeformationField()
{
    reg_f3d<T>::GetDeformationField();
    if(this->backwardDeformationFieldImage!=NULL)
        reg_spline_getDeformationField(this->backwardControlPointGrid,
                                       this->currentFloating,
                                       this->backwardDeformationFieldImage,
                                       this->currentFloatingMask,
                                       false, //composition
                                       true // bspline
                                       );
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::WarpFloatingImage(int inter)
{
    // Compute the deformation fields
    this->GetDeformationField();

    // Resample the floating image
    reg_resampleSourceImage(this->currentReference,
                            this->currentFloating,
                            this->warped,
                            this->deformationFieldImage,
                            this->currentMask,
                            inter,
                            this->warpedPaddingValue);
    // Resample the reference image
    reg_resampleSourceImage(this->currentFloating,
                            this->currentReference,
                            this->backwardWarped,
                            this->backwardDeformationFieldImage,
                            this->currentFloatingMask,
                            inter,
                            this->warpedPaddingValue);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_sym<T>::ComputeSimilarityMeasure()
{

    double measure=0.;
    if(this->useSSD){
        // forward
        measure = -reg_getSSD(this->currentReference,
                              this->warped,
                              NULL,
                              this->currentMask);
        // backward
        measure+= -reg_getSSD(this->currentFloating,
                              this->backwardWarped,
                              NULL,
                              this->currentFloatingMask);
        if(this->usePyramid)
            measure /= this->maxSSD[this->currentLevel];
        else measure /= this->maxSSD[0];
    }
    else if(this->useKLD){
        measure = -reg_getKLDivergence(this->currentReference,
                                       this->warped,
                                       NULL,
                                       this->currentMask);
        measure += -reg_getKLDivergence(this->currentFloating,
                                        this->backwardWarped,
                                        NULL,
                                        this->currentFloatingMask);
    }
    else{
        reg_getEntropies(this->currentReference,
                         this->warped,
                         this->referenceBinNumber,
                         this->floatingBinNumber,
                         this->probaJointHistogram,
                         this->logJointHistogram,
                         this->entropies,
                         this->currentMask,
                         this->approxParzenWindow);
        reg_getEntropies(this->currentFloating,
                         this->backwardWarped,
                         this->floatingBinNumber,
                         this->referenceBinNumber,
                         this->backwardProbaJointHistogram,
                         this->backwardLogJointHistogram,
                         this->backwardEntropies,
                         this->currentFloatingMask,
                         this->approxParzenWindow);
        measure = (this->entropies[0]+this->entropies[1])/this->entropies[2] +
                  (this->backwardEntropies[0]+this->backwardEntropies[1])/this->backwardEntropies[2];

//        fprintf(stderr, "[%i] for %g - bck %g | for entropies : [%g %g %g] | bck entropies : [%g %g %g]\n",
//                this->currentIteration,
//                (this->entropies[0]+this->entropies[1])/this->entropies[2],
//                (this->backwardEntropies[0]+this->backwardEntropies[1])/this->backwardEntropies[2],
//                this->entropies[0], this->entropies[1], this->entropies[2],
//                this->backwardEntropies[0], this->backwardEntropies[1], this->backwardEntropies[2]);
    }
    return double(this->similarityWeight) * measure;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_sym<T>::ComputeJacobianBasedPenaltyTerm(int type)
{
    if (this->jacobianLogWeight<=0) return 0.;

    double forwardPenaltyTerm=reg_f3d<T>::ComputeJacobianBasedPenaltyTerm(type);

    double backwardPenaltyTerm=0.;

    if(type==2){
        backwardPenaltyTerm = reg_bspline_jacobian(this->backwardControlPointGrid,
                                     this->currentFloating,
                                     false);
    }
    else{
        backwardPenaltyTerm = reg_bspline_jacobian(this->backwardControlPointGrid,
                                     this->currentFloating,
                                     this->jacobianLogApproximation);
    }
    unsigned int maxit=5;
    if(type>0) maxit=20;
    unsigned int it=0;
    while(backwardPenaltyTerm!=backwardPenaltyTerm && it<maxit){
        if(type==2){
            backwardPenaltyTerm = reg_bspline_correctFolding(this->backwardControlPointGrid,
                                               this->currentFloating,
                                               false);
        }
        else{
            backwardPenaltyTerm = reg_bspline_correctFolding(this->backwardControlPointGrid,
                                               this->currentFloating,
                                               this->jacobianLogApproximation);
        }
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] Folding correction - Backward transformation\n");
#endif
        it++;
    }
    if(type>0){
        if(backwardPenaltyTerm!=backwardPenaltyTerm){
            this->RestoreCurrentControlPoint();
            fprintf(stderr, "[NiftyReg ERROR] The backward transformation folding correction scheme failed\n");
        }
        else{
#ifdef NDEBUG
            if(this->verbose){
#endif
                printf("[%s] Backward transformation folding correction, %i step(s)\n", this->executableName, it);
#ifdef NDEBUG
            }
#endif
        }
    }
    backwardPenaltyTerm *= (double)this->jacobianLogWeight;

    return forwardPenaltyTerm+backwardPenaltyTerm;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_sym<T>::ComputeBendingEnergyPenaltyTerm()
{
    if (this->bendingEnergyWeight<=0) return 0.;

    double forwardPenaltyTerm=reg_f3d<T>::ComputeBendingEnergyPenaltyTerm();

    double value = reg_bspline_bendingEnergy(this->backwardControlPointGrid);
    return forwardPenaltyTerm + this->bendingEnergyWeight * value;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_sym<T>::ComputeLinearEnergyPenaltyTerm()
{
    if(this->linearEnergyWeight0<=0 && this->linearEnergyWeight1<=0) return 0.;

    double forwardPenaltyTerm=reg_f3d<T>::ComputeLinearEnergyPenaltyTerm();

    double values_le[2]={0.,0.};
    reg_bspline_linearEnergy(this->backwardControlPointGrid, values_le);

    double backwardPenaltyTerm = this->linearEnergyWeight0*values_le[0] +
                                 this->linearEnergyWeight1*values_le[1];

    return forwardPenaltyTerm+backwardPenaltyTerm;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
double reg_f3d_sym<T>::ComputeL2NormDispPenaltyTerm()
{
    if(this->L2NormWeight<=0) return 0.;

    // Compute the L2 norm penalty term along the forward direction
    double forwardPenaltyTerm=reg_f3d<T>::ComputeL2NormDispPenaltyTerm();

    // Compute the L2 norm penalty term along the backward direction
    double backwardPenaltyTerm= (double)this->L2NormWeight *
            reg_bspline_L2norm_displacement(this->backwardControlPointGrid);

    // Return the sum of the forward and backward squared L2 norm of the displacement
    return forwardPenaltyTerm+backwardPenaltyTerm;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetVoxelBasedGradient()
{
    // The intensity gradient is first computed - floating warped into reference
    reg_getSourceImageGradient(this->currentReference,
                               this->currentFloating,
                               this->warpedGradientImage,
                               this->deformationFieldImage,
                               this->currentMask,
                               this->interpolation);

    // The intensity gradient is first computed - reference warped into floating
    reg_getSourceImageGradient(this->currentFloating,
                               this->currentReference,
                               this->backwardWarpedGradientImage,
                               this->backwardDeformationFieldImage,
                               this->currentFloatingMask,
                               this->interpolation);

    if(this->useSSD){
        T localMaxSSD=this->maxSSD[0];
        if(this->usePyramid)
            localMaxSSD=this->maxSSD[this->currentLevel];
        // Compute the voxel based SSD gradient - forward
        reg_getVoxelBasedSSDGradient(this->currentReference,
                                     this->warped,
                                     this->warpedGradientImage,
                                     this->voxelBasedMeasureGradientImage,
                                     NULL,
                                     localMaxSSD,
                                     this->currentMask
                                     );
        // Compute the voxel based SSD gradient - backward
        reg_getVoxelBasedSSDGradient(this->currentFloating,
                                     this->backwardWarped,
                                     this->backwardWarpedGradientImage,
                                     this->backwardVoxelBasedMeasureGradientImage,
                                     NULL,
                                     localMaxSSD,
                                     this->currentFloatingMask
                                     );
    }
    else if(this->useKLD){
        // Compute the voxel based KL divergence gradient - forward
        reg_getKLDivergenceVoxelBasedGradient(this->currentReference,
                                              this->warped,
                                              this->warpedGradientImage,
                                              this->voxelBasedMeasureGradientImage,
                                              NULL,
                                              this->currentMask
                                              );
        // Compute the voxel based KL divergence gradient - backward
        reg_getKLDivergenceVoxelBasedGradient(this->currentFloating,
                                              this->backwardWarped,
                                              this->backwardWarpedGradientImage,
                                              this->backwardVoxelBasedMeasureGradientImage,
                                              NULL,
                                              this->currentFloatingMask
                                              );
    }
    else{
        // Compute the voxel based NMI gradient - forward
        reg_getVoxelBasedNMIGradientUsingPW(this->currentReference,
                                            this->warped,
                                            this->warpedGradientImage,
                                            this->referenceBinNumber,
                                            this->floatingBinNumber,
                                            this->logJointHistogram,
                                            this->entropies,
                                            this->voxelBasedMeasureGradientImage,
                                            this->currentMask,
                                            this->approxParzenWindow);
        // Compute the voxel based NMI gradient - backward
        reg_getVoxelBasedNMIGradientUsingPW(this->currentFloating,
                                            this->backwardWarped,
                                            this->backwardWarpedGradientImage,
                                            this->floatingBinNumber,
                                            this->referenceBinNumber,
                                            this->backwardLogJointHistogram,
                                            this->backwardEntropies,
                                            this->backwardVoxelBasedMeasureGradientImage,
                                            this->currentFloatingMask,
                                            this->approxParzenWindow);
    }

    return;
}

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetSimilarityMeasureGradient()
{
    this->GetVoxelBasedGradient();

    reg_f3d<T>::GetSimilarityMeasureGradient();

    // The voxel based NMI gradient is convolved with a spline kernel
    int smoothingRadius[3];
    smoothingRadius[0] = (int)( 2.0*this->backwardControlPointGrid->dx/this->currentFloating->dx );
    smoothingRadius[1] = (int)( 2.0*this->backwardControlPointGrid->dy/this->currentFloating->dy );
    smoothingRadius[2] = (int)( 2.0*this->backwardControlPointGrid->dz/this->currentFloating->dz );
    reg_tools_CubicSplineKernelConvolution<T>(this->backwardVoxelBasedMeasureGradientImage,
                                              smoothingRadius);

    // The node based NMI gradient is extracted
    reg_voxelCentric2NodeCentric(this->backwardNodeBasedGradientImage,
                                 this->backwardVoxelBasedMeasureGradientImage,
                                 this->similarityWeight,
                                 false);

    /* The gradient is converted from voxel space to real space */
    mat44 *referenceMatrix_xyz=NULL;
    int controlPointNumber=this->backwardControlPointGrid->nx *
            this->backwardControlPointGrid->ny *
            this->backwardControlPointGrid->nz;
    int i;
    if(this->currentReference->sform_code>0)
        referenceMatrix_xyz = &(this->currentReference->sto_xyz);
    else referenceMatrix_xyz = &(this->currentReference->qto_xyz);
    if(this->currentFloating->nz==1){
        T *gradientValuesX = static_cast<T *>(this->backwardNodeBasedGradientImage->data);
        T *gradientValuesY = &gradientValuesX[controlPointNumber];
        T newGradientValueX, newGradientValueY;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(gradientValuesX, gradientValuesY, referenceMatrix_xyz, controlPointNumber) \
    private(newGradientValueX, newGradientValueY, i)
#endif
        for(i=0; i<controlPointNumber; i++){
            newGradientValueX = gradientValuesX[i] * referenceMatrix_xyz->m[0][0] +
                    gradientValuesY[i] * referenceMatrix_xyz->m[0][1];
            newGradientValueY = gradientValuesX[i] * referenceMatrix_xyz->m[1][0] +
                    gradientValuesY[i] * referenceMatrix_xyz->m[1][1];
            gradientValuesX[i] = newGradientValueX;
            gradientValuesY[i] = newGradientValueY;
        }
    }
    else{
        T *gradientValuesX = static_cast<T *>(this->backwardNodeBasedGradientImage->data);
        T *gradientValuesY = &gradientValuesX[controlPointNumber];
        T *gradientValuesZ = &gradientValuesY[controlPointNumber];
        T newGradientValueX, newGradientValueY, newGradientValueZ;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(gradientValuesX, gradientValuesY, gradientValuesZ, referenceMatrix_xyz, controlPointNumber) \
    private(newGradientValueX, newGradientValueY, newGradientValueZ, i)
#endif
        for(i=0; i<controlPointNumber; i++){

            newGradientValueX = gradientValuesX[i] * referenceMatrix_xyz->m[0][0] +
                    gradientValuesY[i] * referenceMatrix_xyz->m[0][1] +
                    gradientValuesZ[i] * referenceMatrix_xyz->m[0][2];
            newGradientValueY = gradientValuesX[i] * referenceMatrix_xyz->m[1][0] +
                    gradientValuesY[i] * referenceMatrix_xyz->m[1][1] +
                    gradientValuesZ[i] * referenceMatrix_xyz->m[1][2];
            newGradientValueZ = gradientValuesX[i] * referenceMatrix_xyz->m[2][0] +
                    gradientValuesY[i] * referenceMatrix_xyz->m[2][1] +
                    gradientValuesZ[i] * referenceMatrix_xyz->m[2][2];
            gradientValuesX[i] = newGradientValueX;
            gradientValuesY[i] = newGradientValueY;
            gradientValuesZ[i] = newGradientValueZ;
        }
    }

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetJacobianBasedGradient()
{
    if(this->jacobianLogWeight<=0) return;

    reg_f3d<T>::GetJacobianBasedGradient();

    reg_bspline_jacobianDeterminantGradient(this->backwardControlPointGrid,
                                            this->currentFloating,
                                            this->backwardNodeBasedGradientImage,
                                            this->jacobianLogWeight,
                                            this->jacobianLogApproximation);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetBendingEnergyGradient()
{
    if(this->bendingEnergyWeight<=0) return;

    reg_f3d<T>::GetBendingEnergyGradient();
    reg_bspline_bendingEnergyGradient(this->backwardControlPointGrid,
                                      this->currentFloating,
                                      this->backwardNodeBasedGradientImage,
                                      this->bendingEnergyWeight);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetLinearEnergyGradient()
{
    if(this->linearEnergyWeight0<=0 && this->linearEnergyWeight1<=0 && this->L2NormWeight<=0) return;

    reg_f3d<T>::GetLinearEnergyGradient();

    reg_bspline_linearEnergyGradient(this->backwardControlPointGrid,
                                     this->currentFloating,
                                     this->nodeBasedGradientImage,
                                     this->linearEnergyWeight0,
                                     this->linearEnergyWeight1);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::GetL2NormDispGradient()
{
    if(this->L2NormWeight<=0) return;

    reg_f3d<T>::GetL2NormDispGradient();

    reg_bspline_L2norm_dispGradient(this->backwardControlPointGrid,
                                    this->currentFloating,
                                    this->backwardNodeBasedGradientImage,
                                    this->L2NormWeight);
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

template <class T>
void reg_f3d_sym<T>::ComputeConjugateGradient()
{
    reg_f3d<T>::ComputeConjugateGradient();

    int nodeNumber = this->backwardNodeBasedGradientImage->nx *
            this->backwardNodeBasedGradientImage->ny *
            this->backwardNodeBasedGradientImage->nz;
    int i;
    if(this->currentIteration==1){
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] Backward conjugate gradient initialisation\n");
#endif
        // first conjugate gradient iteration
        if(this->currentFloating->nz==1){
            T *conjGPtrX = &this->backwardConjugateG[0];
            T *conjGPtrY = &conjGPtrX[nodeNumber];
            T *conjHPtrX = &this->backwardConjugateH[0];
            T *conjHPtrY = &conjHPtrX[nodeNumber];
            T *gradientValuesX = static_cast<T *>(this->backwardNodeBasedGradientImage->data);
            T *gradientValuesY = &gradientValuesX[nodeNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(conjHPtrX, conjHPtrY, conjGPtrX, conjGPtrY, \
    gradientValuesX, gradientValuesY, nodeNumber) \
    private(i)
#endif
            for(i=0; i<nodeNumber;i++){
                conjHPtrX[i] = conjGPtrX[i] = - gradientValuesX[i];
                conjHPtrY[i] = conjGPtrY[i] = - gradientValuesY[i];
            }
        }else{
            T *conjGPtrX = &this->backwardConjugateG[0];
            T *conjGPtrY = &conjGPtrX[nodeNumber];
            T *conjGPtrZ = &conjGPtrY[nodeNumber];
            T *conjHPtrX = &this->backwardConjugateH[0];
            T *conjHPtrY = &conjHPtrX[nodeNumber];
            T *conjHPtrZ = &conjHPtrY[nodeNumber];
            T *gradientValuesX = static_cast<T *>(this->backwardNodeBasedGradientImage->data);
            T *gradientValuesY = &gradientValuesX[nodeNumber];
            T *gradientValuesZ = &gradientValuesY[nodeNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(conjHPtrX, conjHPtrY, conjHPtrZ, conjGPtrX, conjGPtrY, conjGPtrZ, \
    gradientValuesX, gradientValuesY, gradientValuesZ, nodeNumber) \
    private(i)
#endif
            for(i=0; i<nodeNumber;i++){
                conjHPtrX[i] = conjGPtrX[i] = - gradientValuesX[i];
                conjHPtrY[i] = conjGPtrY[i] = - gradientValuesY[i];
                conjHPtrZ[i] = conjGPtrZ[i] = - gradientValuesZ[i];
            }
        }
    }
    else{
#ifndef NDEBUG
        printf("[NiftyReg DEBUG] Backward conjugate gradient update\n");
#endif
        double dgg=0.0, gg=0.0;
        if(this->currentFloating->nz==1){
            T *conjGPtrX = &this->backwardConjugateG[0];
            T *conjGPtrY = &conjGPtrX[nodeNumber];
            T *conjHPtrX = &this->backwardConjugateH[0];
            T *conjHPtrY = &conjHPtrX[nodeNumber];
            T *gradientValuesX = static_cast<T *>(this->backwardNodeBasedGradientImage->data);
            T *gradientValuesY = &gradientValuesX[nodeNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(conjHPtrX, conjHPtrY, conjGPtrX, conjGPtrY, \
    gradientValuesX, gradientValuesY, nodeNumber) \
    private(i) \
    reduction(+:gg) \
    reduction(+:dgg)
#endif
            for(i=0; i<nodeNumber;i++){
                gg += conjHPtrX[i] * conjGPtrX[i];
                gg += conjHPtrY[i] * conjGPtrY[i];
                dgg += (gradientValuesX[i] + conjGPtrX[i]) * gradientValuesX[i];
                dgg += (gradientValuesY[i] + conjGPtrY[i]) * gradientValuesY[i];
            }
            double gam = dgg/gg;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(conjHPtrX, conjHPtrY, conjGPtrX, conjGPtrY, \
    gradientValuesX, gradientValuesY, nodeNumber, gam) \
    private(i)
#endif
            for(i=0; i<nodeNumber;i++){
                conjGPtrX[i] = - gradientValuesX[i];
                conjGPtrY[i] = - gradientValuesY[i];
                conjHPtrX[i] = (float)(conjGPtrX[i] + gam * conjHPtrX[i]);
                conjHPtrY[i] = (float)(conjGPtrY[i] + gam * conjHPtrY[i]);
                gradientValuesX[i] = - conjHPtrX[i];
                gradientValuesY[i] = - conjHPtrY[i];
            }
        }
        else{
            T *conjGPtrX = &this->backwardConjugateG[0];
            T *conjGPtrY = &conjGPtrX[nodeNumber];
            T *conjGPtrZ = &conjGPtrY[nodeNumber];
            T *conjHPtrX = &this->backwardConjugateH[0];
            T *conjHPtrY = &conjHPtrX[nodeNumber];
            T *conjHPtrZ = &conjHPtrY[nodeNumber];
            T *gradientValuesX = static_cast<T *>(this->backwardNodeBasedGradientImage->data);
            T *gradientValuesY = &gradientValuesX[nodeNumber];
            T *gradientValuesZ = &gradientValuesY[nodeNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(conjHPtrX, conjHPtrY, conjHPtrZ, conjGPtrX, conjGPtrY, conjGPtrZ, \
    gradientValuesX, gradientValuesY, gradientValuesZ, nodeNumber) \
    private(i) \
    reduction(+:gg) \
    reduction(+:dgg)
#endif
            for(i=0; i<nodeNumber;i++){
                gg += conjHPtrX[i] * conjGPtrX[i];
                gg += conjHPtrY[i] * conjGPtrY[i];
                gg += conjHPtrZ[i] * conjGPtrZ[i];
                dgg += (gradientValuesX[i] + conjGPtrX[i]) * gradientValuesX[i];
                dgg += (gradientValuesY[i] + conjGPtrY[i]) * gradientValuesY[i];
                dgg += (gradientValuesZ[i] + conjGPtrZ[i]) * gradientValuesZ[i];
            }
            double gam = dgg/gg;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(conjHPtrX, conjHPtrY, conjHPtrZ, conjGPtrX, conjGPtrY, conjGPtrZ, \
    gradientValuesX, gradientValuesY, gradientValuesZ, nodeNumber, gam) \
    private(i)
#endif
            for(i=0; i<nodeNumber;i++){
                conjGPtrX[i] = - gradientValuesX[i];
                conjGPtrY[i] = - gradientValuesY[i];
                conjGPtrZ[i] = - gradientValuesZ[i];
                conjHPtrX[i] = (float)(conjGPtrX[i] + gam * conjHPtrX[i]);
                conjHPtrY[i] = (float)(conjGPtrY[i] + gam * conjHPtrY[i]);
                conjHPtrZ[i] = (float)(conjGPtrZ[i] + gam * conjHPtrZ[i]);
                gradientValuesX[i] = - conjHPtrX[i];
                gradientValuesY[i] = - conjHPtrY[i];
                gradientValuesZ[i] = - conjHPtrZ[i];
            }
        }
    }
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::SetGradientImageToZero()
{
    reg_f3d<T>::SetGradientImageToZero();

    T* nodeGradPtr = static_cast<T *>(this->backwardNodeBasedGradientImage->data);
    for(unsigned int i=0; i<this->backwardNodeBasedGradientImage->nvox; ++i)
        *nodeGradPtr++=0;
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
T reg_f3d_sym<T>::GetMaximalGradientLength()
{
    T forwardLength=reg_f3d<T>::GetMaximalGradientLength();
    T backwardLength= reg_getMaximalLength<T>(this->backwardNodeBasedGradientImage);
    return forwardLength>backwardLength?forwardLength:backwardLength;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::UpdateControlPointPosition(T scale)
{
    reg_f3d<T>::UpdateControlPointPosition(scale);

    int nodeNumber = this->backwardControlPointGrid->nx *
            this->backwardControlPointGrid->ny *
            this->backwardControlPointGrid->nz;
    int i;
    if(this->currentFloating->nz==1){
        T *controlPointValuesX = static_cast<T *>(this->backwardControlPointGrid->data);
        T *controlPointValuesY = &controlPointValuesX[nodeNumber];
        T *bestControlPointValuesX = &this->backwardBestControlPointPosition[0];
        T *bestControlPointValuesY = &bestControlPointValuesX[nodeNumber];
        T *gradientValuesX = static_cast<T *>(this->backwardNodeBasedGradientImage->data);
        T *gradientValuesY = &gradientValuesX[nodeNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(controlPointValuesX, controlPointValuesY, bestControlPointValuesX, \
    bestControlPointValuesY, gradientValuesX, gradientValuesY, nodeNumber, scale) \
    private(i)
#endif
        for(i=0; i<nodeNumber;i++){
            controlPointValuesX[i] = bestControlPointValuesX[i] + scale * gradientValuesX[i];
            controlPointValuesY[i] = bestControlPointValuesY[i] + scale * gradientValuesY[i];
        }
    }
    else{
        T *controlPointValuesX = static_cast<T *>(this->backwardControlPointGrid->data);
        T *controlPointValuesY = &controlPointValuesX[nodeNumber];
        T *controlPointValuesZ = &controlPointValuesY[nodeNumber];
        T *bestControlPointValuesX = &this->backwardBestControlPointPosition[0];
        T *bestControlPointValuesY = &bestControlPointValuesX[nodeNumber];
        T *bestControlPointValuesZ = &bestControlPointValuesY[nodeNumber];
        T *gradientValuesX = static_cast<T *>(this->backwardNodeBasedGradientImage->data);
        T *gradientValuesY = &gradientValuesX[nodeNumber];
        T *gradientValuesZ = &gradientValuesY[nodeNumber];
#ifdef _OPENMP
#pragma omp parallel for default(none) \
    shared(controlPointValuesX, controlPointValuesY, controlPointValuesZ, \
    bestControlPointValuesX, bestControlPointValuesY, bestControlPointValuesZ, \
    gradientValuesX, gradientValuesY, gradientValuesZ, nodeNumber, scale) \
    private(i)
#endif
        for(i=0; i<nodeNumber;i++){
            controlPointValuesX[i] = bestControlPointValuesX[i] + scale * gradientValuesX[i];
            controlPointValuesY[i] = bestControlPointValuesY[i] + scale * gradientValuesY[i];
            controlPointValuesZ[i] = bestControlPointValuesZ[i] + scale * gradientValuesZ[i];
        }
    }

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_f3d_sym<T>::DisplayCurrentLevelParameters()
{
    reg_f3d<T>::DisplayCurrentLevelParameters();
#ifdef NDEBUG
        if(this->verbose){
#endif
            printf("[%s] Current backward control point image\n", this->executableName);
            printf("[%s] \t* image dimension: %i x %i x %i\n", this->executableName,
                   this->backwardControlPointGrid->nx, this->backwardControlPointGrid->ny,
                   this->backwardControlPointGrid->nz);
            printf("[%s] \t* image spacing: %g x %g x %g mm\n", this->executableName,
                   this->backwardControlPointGrid->dx, this->backwardControlPointGrid->dy,
                   this->backwardControlPointGrid->dz);
#ifdef NDEBUG
        }
#endif

#ifndef NDEBUG

        if(this->backwardControlPointGrid->sform_code>0)
            reg_mat44_disp(&(this->backwardControlPointGrid->sto_xyz), (char *)"[NiftyReg DEBUG] Backward CPP sform");
        else reg_mat44_disp(&(this->backwardControlPointGrid->qto_xyz), (char *)"[NiftyReg DEBUG] Backward CPP qform");
#endif
    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::GetInverseConsistencyErrorField()
{
    if (this->inverseConsistencyWeight<=0) return;

    if(this->similarityWeight<=0){
        reg_spline_getDeformationField(this->controlPointGrid,
                                       this->currentReference,
                                       this->deformationFieldImage,
                                       this->currentMask,
                                       false, // composition
                                       true // use B-Spline
                                       );
        reg_spline_getDeformationField(this->backwardControlPointGrid,
                                       this->currentFloating,
                                       this->backwardDeformationFieldImage,
                                       this->currentFloatingMask,
                                       false, // composition
                                       true // use B-Spline
                                       );
    }

    reg_spline_getDeformationField(this->backwardControlPointGrid,
                                   this->currentReference,
                                   this->deformationFieldImage,
                                   this->currentMask,
                                   true, // composition
                                   true // use B-Spline
                                   );
    reg_getDisplacementFromDeformation(this->deformationFieldImage);
    reg_spline_getDeformationField(this->controlPointGrid,
                                   this->currentFloating,
                                   this->backwardDeformationFieldImage,
                                   this->currentFloatingMask,
                                   true, // composition
                                   true // use B-Spline
                                   );
    reg_getDisplacementFromDeformation(this->backwardDeformationFieldImage);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
double reg_f3d_sym<T>::GetInverseConsistencyPenaltyTerm()
{
    if (this->inverseConsistencyWeight<=0) return 0.;

    this->GetInverseConsistencyErrorField();

    double ferror=0.;
    unsigned int voxelNumber=this->deformationFieldImage->nx *
            this->deformationFieldImage->ny *
            this->deformationFieldImage->nz;
    T *dispPtrX=static_cast<T *>(this->deformationFieldImage->data);
    T *dispPtrY=&dispPtrX[voxelNumber];
    if(this->deformationFieldImage->nz>1){
        T *dispPtrZ=&dispPtrY[voxelNumber];
        for(unsigned int i=0; i<voxelNumber; ++i){
            if(this->currentMask[i]>-1){
                double dist=POW2(dispPtrX[i]) + POW2(dispPtrY[i]) + POW2(dispPtrZ[i]);
                ferror += dist;
            }
        }
    }
    else{
        for(unsigned int i=0; i<voxelNumber; ++i){
            if(this->currentMask[i]>-1){
                double dist=POW2(dispPtrX[i]) + POW2(dispPtrY[i]);
                ferror += dist;
            }
        }
    }

    double berror=0.;
    voxelNumber=this->backwardDeformationFieldImage->nx *
            this->backwardDeformationFieldImage->ny *
            this->backwardDeformationFieldImage->nz;
    dispPtrX=static_cast<T *>(this->backwardDeformationFieldImage->data);
    dispPtrY=&dispPtrX[voxelNumber];
    if(this->backwardDeformationFieldImage->nz>1){
        T *dispPtrZ=&dispPtrY[voxelNumber];
        for(unsigned int i=0; i<voxelNumber; ++i){
            if(this->currentFloatingMask[i]>-1){
                double dist=POW2(dispPtrX[i]) + POW2(dispPtrY[i]) + POW2(dispPtrZ[i]);
                berror += dist;
            }
        }
    }
    else{
        for(unsigned int i=0; i<voxelNumber; ++i){
            if(this->currentFloatingMask[i]>-1){
                double dist=POW2(dispPtrX[i]) + POW2(dispPtrY[i]);
                berror += dist;
            }
        }
    }
    double error = (ferror + berror) / double(this->activeVoxelNumber[this->currentLevel]+this->backwardActiveVoxelNumber[this->currentLevel]);
    return double(this->inverseConsistencyWeight) * error;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
void reg_f3d_sym<T>::GetInverseConsistencyGradient()
{
    if(this->inverseConsistencyWeight<=0) return;

    /* FORWARD CONTROL POINT GRADIENT */

    // Derivative of || Backward(Forward(x)) ||^2 against the forward control point position
    reg_spline_getDeformationField(this->controlPointGrid,
                                   this->currentReference,
                                   this->deformationFieldImage,
                                   this->currentMask,
                                   false, // composition
                                   true // use B-Spline
                                   );
    reg_spline_getDeformationField(this->backwardControlPointGrid,
                                   this->currentReference,
                                   this->deformationFieldImage,
                                   this->currentMask,
                                   true, // composition
                                   true // use B-Spline
                                   );
    reg_getDisplacementFromDeformation(this->deformationFieldImage);
    unsigned int voxelNumber=this->deformationFieldImage->nx *this->deformationFieldImage->ny *this->deformationFieldImage->nz;
    T *defPtrX=static_cast<T* >(this->deformationFieldImage->data);
    T *defPtrY=&defPtrX[voxelNumber];
    T *defPtrZ=&defPtrY[voxelNumber];
    for(unsigned int i=0; i<voxelNumber; ++i){
        if(this->currentMask[i]<0){
            defPtrX[i]=0;
            defPtrY[i]=0;
            if(this->deformationFieldImage->nz>1) defPtrZ[i]=0;
        }
    }
    int smoothingRadius[3];
    smoothingRadius[0] = (int)( 2.0*this->controlPointGrid->dx/this->currentReference->dx );
    smoothingRadius[1] = (int)( 2.0*this->controlPointGrid->dy/this->currentReference->dy );
    smoothingRadius[2] = (int)( 2.0*this->controlPointGrid->dz/this->currentReference->dz );
    reg_tools_CubicSplineKernelConvolution<T>(this->deformationFieldImage, smoothingRadius);
    reg_voxelCentric2NodeCentric(this->nodeBasedGradientImage,
                                 this->deformationFieldImage,
                                 2.f * this->inverseConsistencyWeight,
                                 true); // update?

    // Derivative of || Forward(Backward(x)) ||^2 against the forward control point position
    reg_tools_addSubMulDivValue(this->deformationFieldImage,this->deformationFieldImage, 0.f, 2); // multiplication by 0
    reg_getDeformationFromDisplacement(this->deformationFieldImage);
    reg_spline_getDeformationField(this->backwardControlPointGrid,
                                   this->currentReference,
                                   this->deformationFieldImage,
                                   this->currentMask,
                                   true, // composition
                                   true // use B-Spline
                                   );
    reg_spline_getDeformationField(this->controlPointGrid,
                                   this->currentReference,
                                   this->deformationFieldImage,
                                   this->currentMask,
                                   true, // composition
                                   true // use B-Spline
                                   );
    reg_getDisplacementFromDeformation(this->deformationFieldImage);
    voxelNumber=this->deformationFieldImage->nx *this->deformationFieldImage->ny *this->deformationFieldImage->nz;
    defPtrX=static_cast<T* >(this->deformationFieldImage->data);
    defPtrY=&defPtrX[voxelNumber];
    defPtrZ=&defPtrY[voxelNumber];
    for(unsigned int i=0; i<voxelNumber; ++i){
        if(this->currentMask[i]<0){
            defPtrX[i]=0;
            defPtrY[i]=0;
            if(this->deformationFieldImage->nz>1) defPtrZ[i]=0;
        }
    }
    smoothingRadius[0] = (int)( 2.0*this->controlPointGrid->dx/this->currentReference->dx );
    smoothingRadius[1] = (int)( 2.0*this->controlPointGrid->dy/this->currentReference->dy );
    smoothingRadius[2] = (int)( 2.0*this->controlPointGrid->dz/this->currentReference->dz );
    reg_tools_CubicSplineKernelConvolution<T>(this->deformationFieldImage, smoothingRadius);
    reg_voxelCentric2NodeCentric(this->nodeBasedGradientImage,
                                 this->deformationFieldImage,
                                 2.f * this->inverseConsistencyWeight,
                                 true); // update?


    /* BACKWARD CONTROL POINT GRADIENT */

    // Derivative of || Forward(Backward(x)) ||^2 against the backward control point position
    reg_spline_getDeformationField(this->backwardControlPointGrid,
                                   this->currentFloating,
                                   this->backwardDeformationFieldImage,
                                   this->currentFloatingMask,
                                   false, // composition
                                   true // use B-Spline
                                   );
    reg_spline_getDeformationField(this->controlPointGrid,
                                   this->currentFloating,
                                   this->backwardDeformationFieldImage,
                                   this->currentFloatingMask,
                                   true, // composition
                                   true // use B-Spline
                                   );
    reg_getDisplacementFromDeformation(this->backwardDeformationFieldImage);
    voxelNumber=this->backwardDeformationFieldImage->nx *this->backwardDeformationFieldImage->ny *this->backwardDeformationFieldImage->nz;
    defPtrX=static_cast<T* >(this->backwardDeformationFieldImage->data);
    defPtrY=&defPtrX[voxelNumber];
    defPtrZ=&defPtrY[voxelNumber];
    for(unsigned int i=0; i<voxelNumber; ++i){
        if(this->currentFloatingMask[i]<0){
            defPtrX[i]=0;
            defPtrY[i]=0;
            if(this->backwardDeformationFieldImage->nz>1) defPtrZ[i]=0;
        }
    }
    smoothingRadius[0] = (int)( 2.0*this->backwardControlPointGrid->dx/this->currentFloating->dx );
    smoothingRadius[1] = (int)( 2.0*this->backwardControlPointGrid->dy/this->currentFloating->dy );
    smoothingRadius[2] = (int)( 2.0*this->backwardControlPointGrid->dz/this->currentFloating->dz );
    reg_tools_CubicSplineKernelConvolution<T>(this->backwardDeformationFieldImage, smoothingRadius);
    reg_voxelCentric2NodeCentric(this->backwardNodeBasedGradientImage,
                                 this->backwardDeformationFieldImage,
                                 2.f * this->inverseConsistencyWeight,
                                 true); // update?


    // Derivative of || Backward(Forward(x)) ||^2 against the backward control point position
    reg_tools_addSubMulDivValue(this->backwardDeformationFieldImage,this->backwardDeformationFieldImage, 0.f, 2); // multiplication by 0
    reg_getDeformationFromDisplacement(this->backwardDeformationFieldImage);
    reg_spline_getDeformationField(this->controlPointGrid,
                                   this->currentFloating,
                                   this->backwardDeformationFieldImage,
                                   this->currentFloatingMask,
                                   true, // composition
                                   true // use B-Spline
                                   );
    reg_spline_getDeformationField(this->backwardControlPointGrid,
                                   this->currentFloating,
                                   this->backwardDeformationFieldImage,
                                   this->currentFloatingMask,
                                   true, // composition
                                   true // use B-Spline
                                   );
    reg_getDisplacementFromDeformation(this->backwardDeformationFieldImage);
    voxelNumber=this->backwardDeformationFieldImage->nx *this->backwardDeformationFieldImage->ny *this->backwardDeformationFieldImage->nz;
    defPtrX=static_cast<T* >(this->backwardDeformationFieldImage->data);
    defPtrY=&defPtrX[voxelNumber];
    defPtrZ=&defPtrY[voxelNumber];
    for(unsigned int i=0; i<voxelNumber; ++i){
        if(this->currentFloatingMask[i]<0){
            defPtrX[i]=0;
            defPtrY[i]=0;
            if(this->backwardDeformationFieldImage->nz>1) defPtrZ[i]=0;
        }
    }
    smoothingRadius[0] = (int)( 2.0*this->backwardControlPointGrid->dx/this->currentFloating->dx );
    smoothingRadius[1] = (int)( 2.0*this->backwardControlPointGrid->dy/this->currentFloating->dy );
    smoothingRadius[2] = (int)( 2.0*this->backwardControlPointGrid->dz/this->currentFloating->dz );
    reg_tools_CubicSplineKernelConvolution<T>(this->backwardDeformationFieldImage, smoothingRadius);
    reg_voxelCentric2NodeCentric(this->backwardNodeBasedGradientImage,
                                 this->backwardDeformationFieldImage,
                                 2.f * this->inverseConsistencyWeight,
                                 true); // update?

    return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
nifti_image **reg_f3d_sym<T>::GetWarpedImage()
{
    // The initial images are used
    if(this->inputReference==NULL ||
       this->inputFloating==NULL ||
       this->controlPointGrid==NULL ||
       this->backwardControlPointGrid==NULL){
        fprintf(stderr,"[NiftyReg ERROR] reg_f3d_sym::GetWarpedImage()\n");
        fprintf(stderr," * The reference, floating and both control point grid images have to be defined\n");
    }

    reg_f3d_sym<T>::currentReference = this->inputReference;
    reg_f3d_sym<T>::currentFloating = this->inputFloating;
    reg_f3d_sym<T>::currentMask = NULL;
    reg_f3d_sym<T>::currentFloatingMask = NULL;

    reg_f3d_sym<T>::AllocateWarped();
    reg_f3d_sym<T>::AllocateDeformationField();

    reg_f3d_sym<T>::WarpFloatingImage(3); // cubic spline interpolation

    reg_f3d_sym<T>::ClearDeformationField();

    nifti_image **resultImage=(nifti_image **)malloc(2*sizeof(nifti_image *));
    resultImage[0] = nifti_copy_nim_info(this->warped);
    resultImage[0]->cal_min=this->inputFloating->cal_min;
    resultImage[0]->cal_max=this->inputFloating->cal_max;
    resultImage[0]->scl_slope=this->inputFloating->scl_slope;
    resultImage[0]->scl_inter=this->inputFloating->scl_inter;
    resultImage[0]->data=(void *)malloc(resultImage[0]->nvox*resultImage[0]->nbyper);
    memcpy(resultImage[0]->data, this->warped->data, resultImage[0]->nvox*resultImage[0]->nbyper);

    resultImage[1] = nifti_copy_nim_info(this->backwardWarped);
    resultImage[1]->cal_min=this->inputReference->cal_min;
    resultImage[1]->cal_max=this->inputReference->cal_max;
    resultImage[1]->scl_slope=this->inputReference->scl_slope;
    resultImage[1]->scl_inter=this->inputReference->scl_inter;
    resultImage[1]->data=(void *)malloc(resultImage[1]->nvox*resultImage[1]->nbyper);
    memcpy(resultImage[1]->data, this->backwardWarped->data, resultImage[1]->nvox*resultImage[1]->nbyper);

    reg_f3d_sym<T>::ClearWarped();
    return resultImage;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template<class T>
nifti_image * reg_f3d_sym<T>::GetBackwardControlPointPositionImage()
{
    // Create a control point grid nifti image
    nifti_image *returnedControlPointGrid = nifti_copy_nim_info(this->backwardControlPointGrid);
    // Allocate the new image data array
    returnedControlPointGrid->data=(void *)malloc(returnedControlPointGrid->nvox*returnedControlPointGrid->nbyper);
    // Copy the final backward control point grid image
    memcpy(returnedControlPointGrid->data, this->backwardControlPointGrid->data,
           returnedControlPointGrid->nvox*returnedControlPointGrid->nbyper);
    // Return the new control point grid
    return returnedControlPointGrid;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */

#endif
