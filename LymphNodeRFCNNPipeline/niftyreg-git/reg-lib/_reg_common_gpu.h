/** @file _reg_common_gpu.h
 * @author Marc Modat
 * @date 25/03/2009.
 * Copyright (c) 2009, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 */

#ifndef _REG_COMMON_GPU_H
#define _REG_COMMON_GPU_H

#include "_reg_blocksize_gpu.h"

/* ******************************** */
/* ******************************** */
int cudaCommon_setCUDACard(CUcontext *ctx,
                           bool verbose);
/* ******************************** */
void cudaCommon_unsetCUDACard(CUcontext *ctx);
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
int cudaCommon_allocateArrayToDevice(DTYPE **, int);
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
