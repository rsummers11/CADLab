/*
 *  _reg_f3d_sym.h
 *
 *
 *  Created by Marc Modat on 10/11/2011.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */


#ifndef _REG_F3D_SYM_H
#define _REG_F3D_SYM_H

#include "_reg_f3d.h"

template <class T>
class reg_f3d_sym : public reg_f3d<T>
{
  protected:
    nifti_image *floatingMaskImage;
    int **floatingMaskPyramid;
    int *currentFloatingMask;
    int *backwardActiveVoxelNumber;

    nifti_image *backwardControlPointGrid;
    nifti_image *backwardDeformationFieldImage;
    nifti_image *backwardWarped;
    nifti_image *backwardWarpedGradientImage;
    nifti_image *backwardVoxelBasedMeasureGradientImage;
    nifti_image *backwardNodeBasedGradientImage;

    T *backwardBestControlPointPosition;
    T *backwardConjugateG;
    T *backwardConjugateH;

    double *backwardProbaJointHistogram;
    double *backwardLogJointHistogram;
    double backwardEntropies[4];

    T inverseConsistencyWeight;

    virtual void AllocateWarped();
    virtual void ClearWarped();
    virtual void AllocateDeformationField();
    virtual void ClearDeformationField();
    virtual void AllocateWarpedGradient();
    virtual void ClearWarpedGradient();
    virtual void AllocateVoxelBasedMeasureGradient();
    virtual void ClearVoxelBasedMeasureGradient();
    virtual void AllocateNodeBasedGradient();
    virtual void ClearNodeBasedGradient();
    virtual void AllocateConjugateGradientVariables();
    virtual void ClearConjugateGradientVariables();
    virtual void AllocateBestControlPointArray();
    virtual void ClearBestControlPointArray();
    virtual void AllocateJointHistogram();
    virtual void ClearJointHistogram();
    virtual void AllocateCurrentInputImage();
    virtual void ClearCurrentInputImage();

    virtual void SaveCurrentControlPoint();
    virtual void RestoreCurrentControlPoint();
    virtual double ComputeJacobianBasedPenaltyTerm(int);
    virtual double ComputeBendingEnergyPenaltyTerm();
    virtual double ComputeLinearEnergyPenaltyTerm();
    virtual double ComputeL2NormDispPenaltyTerm();
    virtual void GetDeformationField();
    virtual void WarpFloatingImage(int);
    virtual double ComputeSimilarityMeasure();
    virtual void GetVoxelBasedGradient();
    virtual void GetSimilarityMeasureGradient();
    virtual void GetBendingEnergyGradient();
    virtual void GetLinearEnergyGradient();
    virtual void GetL2NormDispGradient();
    virtual void GetJacobianBasedGradient();
    virtual void ComputeConjugateGradient();
    virtual T GetMaximalGradientLength();
    virtual void SetGradientImageToZero();
    virtual void UpdateControlPointPosition(T);
    virtual void DisplayCurrentLevelParameters();

    virtual void GetInverseConsistencyErrorField();
    virtual double GetInverseConsistencyPenaltyTerm();
    virtual void GetInverseConsistencyGradient();

public:
    virtual void SetFloatingMask(nifti_image *);
    virtual void SetInverseConsistencyWeight(T);

    reg_f3d_sym(int refTimePoint,int floTimePoint);
    ~reg_f3d_sym();
    virtual void CheckParameters_f3d();
    virtual void Initisalise_f3d();
    virtual nifti_image *GetBackwardControlPointPositionImage();
    virtual nifti_image **GetWarpedImage();
};

#include "_reg_f3d_sym.cpp"

#endif
