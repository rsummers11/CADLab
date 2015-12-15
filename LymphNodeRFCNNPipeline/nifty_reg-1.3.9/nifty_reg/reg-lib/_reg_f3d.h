/*
 *  _reg_f3d.h
 *
 *
 *  Created by Marc Modat on 19/11/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_F3D_H
#define _REG_F3D_H

#include "_reg_resampling.h"
#include "_reg_globalTransformation.h"
#include "_reg_localTransformation.h"
#include "_reg_mutualinformation.h"
#include "_reg_ssd.h"
#include "_reg_KLdivergence.h"
#include "_reg_tools.h"
#include "_reg_ReadWriteImage.h"
#include "float.h"
#include <limits>

template <class T>
class reg_f3d
{
  protected:
    char *executableName;
    int referenceTimePoint;
    int floatingTimePoint;
    nifti_image *inputReference; // pointer to external
    nifti_image *inputFloating; // pointer to external
    nifti_image *inputControlPointGrid; // pointer to external
    nifti_image *maskImage; // pointer to external
    mat44 *affineTransformation; // pointer to external
    int *referenceMask;
    nifti_image *controlPointGrid;
    T similarityWeight;
    T bendingEnergyWeight;
    T linearEnergyWeight0;
    T linearEnergyWeight1;
    T L2NormWeight;
    T jacobianLogWeight;
    bool jacobianLogApproximation;
    unsigned int maxiterationNumber;
    T referenceSmoothingSigma;
    T floatingSmoothingSigma;
    float *referenceThresholdUp;
    float *referenceThresholdLow;
    float *floatingThresholdUp;
    float *floatingThresholdLow;
    unsigned int *referenceBinNumber;
    unsigned int *floatingBinNumber;
    T warpedPaddingValue;
    T spacing[3];
    unsigned int levelNumber;
    unsigned int levelToPerform;
    T gradientSmoothingSigma;
    bool useSSD;
    bool useKLD;
    bool useConjGradient;
    bool verbose;
    bool usePyramid;
    int interpolation;
//    int threadNumber;

    bool initialised;
    nifti_image **referencePyramid;
    nifti_image **floatingPyramid;
    int **maskPyramid;
    int *activeVoxelNumber;
    nifti_image *currentReference;
    nifti_image *currentFloating;
    int *currentMask;
    nifti_image *warped;
    nifti_image *deformationFieldImage;
    nifti_image *warpedGradientImage;
    nifti_image *voxelBasedMeasureGradientImage;
    nifti_image *nodeBasedGradientImage;
    T *conjugateG;
    T *conjugateH;
    T *bestControlPointPosition;
    double *probaJointHistogram;
    double *logJointHistogram;
    double entropies[4];
    bool approxParzenWindow;
    T *maxSSD;
    unsigned int currentLevel;
    unsigned totalBinNumber;
    bool xOptimisation;
    bool yOptimisation;
    bool zOptimisation;
    bool gridRefinement;

    bool additive_mc_nmi; // Additive multi channel NMI

    unsigned int currentIteration;

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

    void (*funcProgressCallback)(float pcntProgress, void *params);
    void *paramsProgressCallback;

public:
    reg_f3d(int refTimePoint,int floTimePoint);
    virtual ~reg_f3d();

    void SetReferenceImage(nifti_image *);
    void SetFloatingImage(nifti_image *);
    void SetControlPointGridImage(nifti_image *);
    void SetReferenceMask(nifti_image *);
    void SetAffineTransformation(mat44 *);
    void SetBendingEnergyWeight(T);
    void SetLinearEnergyWeights(T,T);
    void SetL2NormDisplacementWeight(T);
    void SetJacobianLogWeight(T);
    void ApproximateJacobianLog();
    void DoNotApproximateJacobianLog();
    void ApproximateParzenWindow();
    void DoNotApproximateParzenWindow();
    void SetReferenceSmoothingSigma(T);
    void SetFloatingSmoothingSigma(T);
    void SetReferenceThresholdUp(unsigned int,T);
    void SetReferenceThresholdLow(unsigned int,T);
    void SetFloatingThresholdUp(unsigned int, T);
    void SetFloatingThresholdLow(unsigned int,T);
    void SetWarpedPaddingValue(T);
    void SetSpacing(unsigned int ,T);
    void SetLevelNumber(unsigned int);
    void SetLevelToPerform(unsigned int);
    void SetGradientSmoothingSigma(T);

    // Set the multi channel implementation to additive.
    void SetAdditiveMC() { this->additive_mc_nmi = true; }

    void UseComposition();
    void DoNotUseComposition();
    void UseSSD();
    void DoNotUseSSD();
    void UseKLDivergence();
    void DoNotUseKLDivergence();
    void UseConjugateGradient();
    void DoNotUseConjugateGradient();
    void PrintOutInformation();
    void DoNotPrintOutInformation();
    void SetMaximalIterationNumber(unsigned int);
    void SetReferenceBinNumber(int, unsigned int);
    void SetFloatingBinNumber(int, unsigned int);
    void DoNotUsePyramidalApproach();
    void UseNeareatNeighborInterpolation();
    void UseLinearInterpolation();
    void UseCubicSplineInterpolation();
    void NoOptimisationAlongX(){this->xOptimisation=false;}
    void NoOptimisationAlongY(){this->yOptimisation=false;}
    void NoOptimisationAlongZ(){this->zOptimisation=false;}
    void NoGridRefinement(){this->gridRefinement=false;}
//    int SetThreadNumber(int t);

    // F3D2 specific options
    virtual void SetCompositionStepNumber(int){return;}
    virtual void ApproximateComposition(){return;}
    virtual void UseSimilaritySymmetry(){return;}

    // F3D_SYM specific options
    virtual void SetFloatingMask(nifti_image *){return;}
    virtual void SetInverseConsistencyWeight(T){return;}
    virtual nifti_image *GetBackwardControlPointPositionImage(){return NULL;}
    virtual double GetInverseConsistencyPenaltyTerm(){return 0.;}
    virtual void GetInverseConsistencyGradient(){return;}

    // F3D_gpu specific option
    virtual int CheckMemoryMB_f3d(){return 0;}

    virtual void CheckParameters_f3d();
    void Run_f3d();
    virtual void Initisalise_f3d();
    nifti_image *GetControlPointPositionImage();
    virtual nifti_image **GetWarpedImage();

    void SetProgressCallbackFunction( void (*funcProgCallback)(float pcntProgress,
							       void *params), 
				      void *paramsProgCallback ) {
      funcProgressCallback = funcProgCallback;
      paramsProgressCallback = paramsProgCallback;
    }
};

#include "_reg_f3d.cpp"

#endif
