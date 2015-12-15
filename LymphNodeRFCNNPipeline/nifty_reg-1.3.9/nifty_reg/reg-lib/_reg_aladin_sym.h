/*
 *  _reg_aladin_sym.h
 *
 *
 *  Created by David Cash on 28/02/2012.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_ALADIN_SYM_H
#define _REG_ALADIN_SYM_H

#include "_reg_aladin.h"
#ifdef _BUILD_NR_DEV

template <class T>
class reg_aladin_sym : public reg_aladin<T>
{
protected:
    nifti_image *InputFloatingMask;
    nifti_image *CurrentBackwardWarped;
    int ** FloatingMaskPyramid;
    nifti_image *BackwardDeformationFieldImage;
    int *CurrentFloatingMask;
    int *BackwardActiveVoxelNumber;

    _reg_blockMatchingParam BackwardBlockMatchingParams;

    mat44 *BackwardTransformationMatrix;

    virtual void ClearCurrentInputImage();
    virtual void AllocateBackwardWarpedImage();
    virtual void ClearBackwardWarpedImage();
    virtual void AllocateBackwardDeformationField();
    virtual void ClearBackwardDeformationField();
    virtual void GetBackwardDeformationField();
    virtual void UpdateTransformationMatrix(mat44);

    virtual void InitialiseRegistration();
    virtual void InitialiseBlockMatching(int);
    virtual void SetCurrentImages();
    virtual void GetWarpedImage(int);

public:
    reg_aladin_sym();
    ~reg_aladin_sym();
    //int Check();
    //int Print();
    //void Run();

    virtual void SetInputFloatingMask(nifti_image *);
    virtual mat44 GetUpdateTransformationMatrix(int);
    virtual void DebugPrintLevelInfo(int);

};

#include "_reg_aladin_sym.cpp"

#endif // _BUILD_NR_DEV
#endif // _REG_ALADIN_SYM_H
