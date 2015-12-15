/**
 * @file _reg_f3d2.h
 * @author Marc Modat
 * @date 19/11/2010
 *
 *  Created by Marc Modat on 19/11/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_f3d_sym.h"

#ifdef _BUILD_NR_DEV

#ifndef _REG_F3D2_H
#define _REG_F3D2_H

template <class T>
class reg_f3d2 : public reg_f3d_sym<T>
{
  protected:
    int stepNumber;
    mat33 *forward2backward_reorient;
    mat33 *backward2forward_reorient;

    virtual void DefineReorientationMatrices();
    virtual void GetDeformationField();
    virtual void GetInverseConsistencyErrorField();
    virtual void GetInverseConsistencyGradient();
    virtual void GetSimilarityMeasureGradient();
    virtual void UpdateControlPointPosition(T);

public:
    virtual void SetCompositionStepNumber(int);
    reg_f3d2(int refTimePoint,int floTimePoint);
    ~reg_f3d2();
    virtual void Initisalise_f3d();
    virtual nifti_image **GetWarpedImage();
};

#include "_reg_f3d2.cpp"

#endif

#endif
