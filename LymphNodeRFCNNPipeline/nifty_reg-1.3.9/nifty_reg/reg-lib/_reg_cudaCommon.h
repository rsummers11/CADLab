/*
 *  _reg_cudaCommon.h
 *  
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_CUDACOMMON_H
#define _REG_CUDACOMMON_H

#include "_reg_blocksize_gpu.h"

/* ******************************** */
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(cudaArray **, int *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(cudaArray **, cudaArray **, int *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE **, int *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_allocateArrayToDevice(DTYPE **, DTYPE **, int *);
/* ******************************** */
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray **, nifti_image *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(cudaArray **, cudaArray **, nifti_image *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(DTYPE **, nifti_image *);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferNiftiToArrayOnDevice(DTYPE **, DTYPE **, nifti_image *);
/* ******************************** */
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferFromDeviceToNifti(nifti_image *, DTYPE **);
/* ******************************** */
extern "C++"
template <class DTYPE>
int cudaCommon_transferFromDeviceToNifti(nifti_image *, DTYPE **, DTYPE **);
/* ******************************** */
/* ******************************** */
extern "C++"
void cudaCommon_free(cudaArray **);
/* ******************************** */
extern "C++" template <class DTYPE>
void cudaCommon_free(DTYPE **);
/* ******************************** */
/* ******************************** */
#endif
