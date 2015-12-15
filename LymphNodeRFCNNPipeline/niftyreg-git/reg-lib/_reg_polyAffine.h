/**
 * @file _reg_polyAffine.h
 * @author Marc Modat
 * @date 16/11/2012
 *
 * Copyright (c) 2012, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_POLYAFFINE_H
#define _REG_POLYAFFINE_H

#include "_reg_base.h"

template <class T>
class reg_polyAffine : public reg_base<T>
{
protected:
   void GetDeformationField();
   void SetGradientImageToZero();
   void GetApproximatedGradient();
   double GetObjectiveFunctionValue();
   void UpdateParameters(float);
   T NormaliseGradient();
   void GetSimilarityMeasureGradient();
   void GetObjectiveFunctionGradient();
   void DisplayCurrentLevelParameters();
   void UpdateBestObjFunctionValue();
   void PrintCurrentObjFunctionValue(T);
   void PrintInitialObjFunctionValue();
   void AllocateTransformationGradient();
   void ClearTransformationGradient();

public:
   reg_polyAffine(int refTimePoint,int floTimePoint);
   ~reg_polyAffine();
};

#include "_reg_polyAffine.cpp"

#endif // _REG_POLYAFFINE_H
