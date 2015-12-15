/**
 * @file _reg_f3d2.h
 * @author Marc Modat
 * @date 19/11/2011
 *
 *  Copyright (c) 2011, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_f3d_sym.h"

#ifndef _REG_F3D2_H
#define _REG_F3D2_H

template <class T>
class reg_f3d2 : public reg_f3d_sym<T>
{
protected:
   bool BCHUpdate;
   bool useGradientCumulativeExp;
   int BCHUpdateValue;

   virtual void GetDeformationField();
   virtual void GetInverseConsistencyErrorField(bool forceAll);
   virtual void GetInverseConsistencyGradient();
   virtual void GetVoxelBasedGradient();
   virtual void UpdateParameters(float);
   virtual void ExponentiateGradient();
   virtual void UseBCHUpdate(int);
   virtual void UseGradientCumulativeExp();

public:
   reg_f3d2(int refTimePoint,int floTimePoint);
   ~reg_f3d2();
   virtual void Initialise();
   virtual nifti_image **GetWarpedImage();
};

#include "_reg_f3d2.cpp"

#endif
