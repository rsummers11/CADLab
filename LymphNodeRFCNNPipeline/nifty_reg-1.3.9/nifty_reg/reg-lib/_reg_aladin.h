/*
 *  _reg_aladin.h
 *
 *
 *  Created by Marc Modat on 08/12/2011.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_ALADIN_H
#define _REG_ALADIN_H
#define CONVERGENCE_EPS 0.00001
#define RIGID 0
#define AFFINE 1
#include "_reg_macros.h"
#include "_reg_resampling.h"
#include "_reg_blockMatching.h"
#include "_reg_globalTransformation.h"
#include "_reg_mutualinformation.h"
#include "_reg_ssd.h"
#include "_reg_tools.h"
#include "float.h"
#include <limits>

template <class T>
class reg_aladin
{
    protected:
      char *ExecutableName;
      nifti_image *InputReference;
      nifti_image *InputFloating;
      nifti_image *InputReferenceMask;
      nifti_image **ReferencePyramid;
      nifti_image **FloatingPyramid;
      int **ReferenceMaskPyramid;
      nifti_image *CurrentReference;
      nifti_image *CurrentFloating;
      nifti_image *CurrentWarped;
      nifti_image *deformationFieldImage;
      int *CurrentReferenceMask;
      int *activeVoxelNumber;

      char *InputTransformName;
      bool InputTransformFromFlirt;
      mat44 *TransformationMatrix;

      bool Verbose;

      unsigned int MaxIterations;

      unsigned int CurrentLevel;
      unsigned int NumberOfLevels;
      unsigned int LevelsToPerform;

      bool PerformRigid;
      bool PerformAffine;

      int BlockPercentage;
      int InlierLts;
      _reg_blockMatchingParam blockMatchingParams;

      bool AlignCentre;

      int Interpolation;

      float FloatingSigma;

      float ReferenceSigma;

      bool TestMatrixConvergence(mat44 *mat);

      virtual void InitialiseRegistration();
      virtual void SetCurrentImages();
      virtual void ClearCurrentInputImage();
      virtual void AllocateWarpedImage();
      virtual void ClearWarpedImage();
      virtual void AllocateDeformationField();
      virtual void ClearDeformationField();

      virtual void InitialiseBlockMatching(int);
      virtual void GetDeformationField();
      virtual void GetWarpedImage(int);
      virtual mat44 GetUpdateTransformationMatrix(int);
      virtual void UpdateTransformationMatrix(mat44);

      void (*funcProgressCallback)(float pcntProgress, void *params);
      void *paramsProgressCallback;

    public:
      reg_aladin();
      virtual ~reg_aladin();
      GetStringMacro(ExecutableName);

      //No allocating of the images here...
      void SetInputReference(nifti_image *input) {this->InputReference = input;}
      nifti_image* GetInputReference() {return this->InputReference;}

      void SetInputFloating(nifti_image *input) {this->InputFloating=input;}
      nifti_image *GetInputFloating() {return this->InputFloating;}

      void SetInputMask(nifti_image *input) {this->InputReferenceMask=input;}
      nifti_image *GetInputMask() {return this->InputReferenceMask;}

      void SetInputTransform(const char *filename, bool IsFlirt);
      mat44* GetInputTransform() {return this->InputTransform;}

      mat44* GetTransformationMatrix() {return this->TransformationMatrix;}
      nifti_image *GetFinalWarpedImage();

      SetMacro(MaxIterations,unsigned int);
      GetMacro(MaxIterations,unsigned int);

      SetMacro(NumberOfLevels,unsigned int);
      GetMacro(NumberOfLevels,unsigned int);

      SetMacro(LevelsToPerform,unsigned int);
      GetMacro(LevelsToPerform,unsigned int);

      SetMacro(BlockPercentage,float);
      GetMacro(BlockPercentage,float);

      SetMacro(InlierLts,float);
      GetMacro(InlierLts,float);

      SetMacro(ReferenceSigma,float);
      GetMacro(ReferenceSigma,float);

      SetMacro(FloatingSigma,float);
      GetMacro(FloatingSigma,float);

      SetMacro(PerformRigid,int);
      GetMacro(PerformRigid,int);
      BooleanMacro(PerformRigid, int);

      SetMacro(PerformAffine,int);
      GetMacro(PerformAffine,int);
      BooleanMacro(PerformAffine, int);

      GetMacro(AlignCentre,int);
      SetMacro(AlignCentre,int);
      BooleanMacro(AlignCentre, int);

      SetClampMacro(Interpolation,int,0,3);
      GetMacro(Interpolation, int);

      virtual void SetInputFloatingMask (nifti_image*) {fprintf(stderr, "Floating mask not used in one way affine\n"); }
      void SetInterpolationToNearestNeighbor() {this->SetInterpolation(0);}
      void SetInterpolationToTrilinear() {this->SetInterpolation(1);}
      void SetInterpolationToCubic() {this->SetInterpolation(3);}

      virtual int Check();
      virtual int Print();
      virtual void Run();

      virtual void DebugPrintLevelInfo(int);

      void SetProgressCallbackFunction( void (*funcProgCallback)(float pcntProgress,
								 void *params), 
					void *paramsProgCallback ) {
	funcProgressCallback = funcProgCallback;
	paramsProgressCallback = paramsProgCallback;
      }

};

#include "_reg_aladin.cpp"
#endif // _REG_ALADIN_H
