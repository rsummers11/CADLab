/**
 * @file _reg_base.h
 * @author Marc Modat
 * @date 15/11/2012
 *
 * Copyright (c) 2012, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_BASE_H
#define _REG_BASE_H

#include "_reg_resampling.h"
#include "_reg_globalTransformation.h"
#include "_reg_localTransformation.h"
#include "_reg_nmi.h"
#include "_reg_dti.h"
#include "_reg_ssd.h"
#include "_reg_KLdivergence.h"
#include "_reg_lncc.h"
#include "_reg_tools.h"
#include "_reg_ReadWriteImage.h"
#include "_reg_optimiser.h"
#include "float.h"
#include <limits>

template <class T>
class reg_base : public InterfaceOptimiser
{
protected:
   // Optimiser related variables
   reg_optimiser<T> *optimiser;
   size_t maxiterationNumber;
   size_t perturbationNumber;
   bool optimiseX;
   bool optimiseY;
   bool optimiseZ;

   // Optimiser related function
   virtual void SetOptimiser();

   // Measure related variables
   reg_ssd *measure_ssd;
   reg_kld *measure_kld;
   reg_dti *measure_dti;
   reg_lncc *measure_lncc;
   reg_nmi *measure_nmi;
   reg_multichannel_nmi *measure_multichannel_nmi;

   char *executableName;
   int referenceTimePoint;
   int floatingTimePoint;
   nifti_image *inputReference; // pointer to external
   nifti_image *inputFloating; // pointer to external
   nifti_image *maskImage; // pointer to external
   mat44 *affineTransformation; // pointer to external
   int *referenceMask;
   T referenceSmoothingSigma;
   T floatingSmoothingSigma;
   float *referenceThresholdUp;
   float *referenceThresholdLow;
   float *floatingThresholdUp;
   float *floatingThresholdLow;
   T warpedPaddingValue;
   unsigned int levelNumber;
   unsigned int levelToPerform;
   T gradientSmoothingSigma;
   T similarityWeight;
   bool additive_mc_nmi;
   bool useConjGradient;
   bool useApproxGradient;
   bool verbose;
   bool usePyramid;
   int interpolation;

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
   unsigned int currentLevel;

   mat33 *forwardJacobianMatrix;

   double bestWMeasure;
   double currentWMeasure;

   virtual void AllocateWarped();
   virtual void ClearWarped();
   virtual void AllocateDeformationField();
   virtual void ClearDeformationField();
   virtual void AllocateWarpedGradient();
   virtual void ClearWarpedGradient();
   virtual void AllocateVoxelBasedMeasureGradient();
   virtual void ClearVoxelBasedMeasureGradient();
   virtual T InitialiseCurrentLevel()
   {
      return 0.;
   }
   virtual void ClearCurrentInputImage();

   virtual void WarpFloatingImage(int);
   virtual double ComputeSimilarityMeasure();
   virtual void GetVoxelBasedGradient();
   virtual void SmoothGradient()
   {
      return;
   }
   virtual void InitialiseSimilarity();

   // Virtual empty functions that have to be filled
   virtual void GetDeformationField()
   {
      return;  // Need to be filled
   }
   virtual void SetGradientImageToZero()
   {
      return;  // Need to be filled
   }
   virtual void GetApproximatedGradient()
   {
      return;  // Need to be filled
   }
   virtual double GetObjectiveFunctionValue()
   {
      return std::numeric_limits<double>::quiet_NaN();  // Need to be filled
   }
   virtual void UpdateParameters(float)
   {
      return;  // Need to be filled
   }
   virtual T NormaliseGradient()
   {
      return std::numeric_limits<float>::quiet_NaN();  // Need to be filled
   }
   virtual void GetSimilarityMeasureGradient()
   {
      return;  // Need to be filled
   }
   virtual void GetObjectiveFunctionGradient()
   {
      return;  // Need to be filled
   }
   virtual void DisplayCurrentLevelParameters()
   {
      return;  // Need to be filled
   }
   virtual void UpdateBestObjFunctionValue()
   {
      return;  // Need to be filled
   }
   virtual void PrintCurrentObjFunctionValue(T)
   {
      return;  // Need to be filled
   }
   virtual void PrintInitialObjFunctionValue()
   {
      return;  // Need to be filled
   }
   virtual void AllocateTransformationGradient()
   {
      return;  // Need to be filled
   }
   virtual void ClearTransformationGradient()
   {
      return;  // Need to be filled
   }
   virtual void CorrectTransformation()
   {
      return;  // Need to be filled
   }

   void (*funcProgressCallback)(float pcntProgress, void *params);
   void *paramsProgressCallback;

public:
   reg_base(int refTimePoint,int floTimePoint);
   virtual ~reg_base();

   // Optimisation related functions
   void SetMaximalIterationNumber(unsigned int);
   void NoOptimisationAlongX()
   {
      this->optimiseX=false;
   }
   void NoOptimisationAlongY()
   {
      this->optimiseY=false;
   }
   void NoOptimisationAlongZ()
   {
      this->optimiseZ=false;
   }
   void SetPerturbationNumber(size_t v)
   {
      this->perturbationNumber=v;
   }
   void UseConjugateGradient();
   void DoNotUseConjugateGradient();
   void UseApproximatedGradient();
   void DoNotUseApproximatedGradient();
   // Measure of similarity related functions
//    void ApproximateParzenWindow();
//    void DoNotApproximateParzenWindow();
   virtual void UseNMISetReferenceBinNumber(int,int);
   virtual void UseNMISetFloatingBinNumber(int,int);
   virtual void UseMultiChannelNMI(int timepointNumber);
   virtual void UseSSD(int timepoint);
   virtual void UseKLDivergence(int timepoint);
   virtual void UseDTI(bool *timepoint);
   virtual void UseLNCC(int timepoint, float stdDevKernel);
   virtual void SetLNCCKernelType(int type);

   void SetReferenceImage(nifti_image *);
   void SetFloatingImage(nifti_image *);
   void SetReferenceMask(nifti_image *);
   void SetAffineTransformation(mat44 *);
   void SetReferenceSmoothingSigma(T);
   void SetFloatingSmoothingSigma(T);
   void SetGradientSmoothingSigma(T);
   void SetReferenceThresholdUp(unsigned int,T);
   void SetReferenceThresholdLow(unsigned int,T);
   void SetFloatingThresholdUp(unsigned int, T);
   void SetFloatingThresholdLow(unsigned int,T);
   void SetWarpedPaddingValue(T);
   void SetLevelNumber(unsigned int);
   void SetLevelToPerform(unsigned int);
   void PrintOutInformation();
   void DoNotPrintOutInformation();
   void DoNotUsePyramidalApproach();
   void UseNeareatNeighborInterpolation();
   void UseLinearInterpolation();
   void UseCubicSplineInterpolation();

   virtual void CheckParameters();
   void Run();
   virtual void Initialise();
   nifti_image **GetWarpedImage()
   {
      return NULL;  // Need to be filled
   }
   virtual char * GetExecutableName()
   {
      return this->executableName;
   }
   virtual bool GetSymmetricStatus()
   {
      return false;
   }

   // Function required for the NiftyReg pluggin in NiftyView
   void SetProgressCallbackFunction(void (*funcProgCallback)(float pcntProgress,
                                    void *params),
                                    void *paramsProgCallback)
   {
      funcProgressCallback = funcProgCallback;
      paramsProgressCallback = paramsProgCallback;
   }

   // Function used for testing
   virtual void reg_test_setOptimiser(reg_optimiser<T> *opt)
   {
      this->optimiser=opt;
   }
};

#include "_reg_base.cpp"

#endif // _REG_BASE_H
