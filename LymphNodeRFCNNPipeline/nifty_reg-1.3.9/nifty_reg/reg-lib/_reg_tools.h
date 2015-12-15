/**
 * @file _reg_tools.h
 * @author Marc Modat
 * @date 25/03/2009
 * @brief Set of useful functions
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TOOLS_H
#define _REG_TOOLS_H

#include "nifti1_io.h"
#include <fstream>
#include <limits>

#include "_reg_maths.h"


#if defined(_WIN32) && !defined(__CYGWIN__)

#include <float.h>
#include <time.h>

#ifndef M_PI
    #define M_PI (3.14159265358979323846)
#endif
#ifndef isnan(_X)
    #define isnan(_X) _isnan(_X)
#endif
#ifndef strtof(_s, _t)
    #define strtof(_s, _t) (float) strtod(_s, _t)
#endif

template<class PrecisionType> inline int round(PrecisionType x) { return int(x > 0.0 ? (x + 0.5) : (x - 0.5)); }
template<typename T> inline bool isinf(T value) { return std::numeric_limits<T>::has_infinity && value == std::numeric_limits<T>::infinity(); }
inline int fabs(int _x) { return (int)fabs((float)(_x)); }

#endif //If on windows...

/** @brief This function check some header parameters and correct them in
 * case of error. For example no dimension is lower than one. The scl_sclope
 * can not be equal to zero. The qto_xyz and qto_ijk are populated if
 * both qform_code and sform_code are set to zero.
 * @param image Input image to check and correct if necessary
 */
extern "C++"
void reg_checkAndCorrectDimension(nifti_image *image);

/** @brief Rescale an input image between two user-defined values.
 * Some threshold can also be applied concurrenlty
 * @param image Image to be rescaled
 * @param newMin Intensity lower bound after rescaling
 * @param newMax Intensity higher bound after rescaling
 * @param lowThr Intensity to use as lower threshold
 * @param upThr Intensity to use as higher threshold
 */
extern "C++"
void reg_intensityRescale(nifti_image *image,
                          float *newMin,
                          float *newMax,
                          float *lowThr,
                          float *upThr
                          );

/** @brief Convolve a cubic spline kernel with the provided image
 * @param image Image to be convolved with the kernel
 * @param radius Radius of the cubic spline kernel. The array
 * contains the radius along each axis
 */
extern "C++" template <class PrecisionTYPE>
void reg_tools_CubicSplineKernelConvolution(nifti_image *image,
                                            int radius[3]
                                            );

/** @brief Smooth an image using a
 * @param
 */
extern "C++" template <class PrecisionTYPE>
void reg_smoothNormImageForCubicSpline(nifti_image *image,
                                       int radius[3]
                                       );

/** @brief
 * @param
 */
extern "C++" template <class PrecisionTYPE>
void reg_smoothImageForTrilinear(nifti_image *image,
                                 int radius[3]
                                 );

/** @brief
 * @param
 */
extern "C++" template <class PrecisionTYPE>
void reg_gaussianSmoothing(nifti_image *image,
                           PrecisionTYPE sigma,
                           bool[8]);

/** @brief
 * @param
 */
extern "C++" template <class PrecisionTYPE>
void reg_downsampleImage(nifti_image *image, int, bool[8]);

/** @brief
 * @param
 */
extern "C++" template <class PrecisionTYPE>
PrecisionTYPE reg_getMaximalLength(nifti_image *image);

/** @brief
 * @param
 */
extern "C++" template <class NewTYPE>
void reg_tools_changeDatatype(nifti_image *image);

/** @brief
 * @param
 */
extern "C++"
void reg_tools_addSubMulDivImages(nifti_image *,
                                  nifti_image *,
                                  nifti_image *,
                                  int);

/** @brief
 * @param
 */
extern "C++"
void reg_tools_addSubMulDivValue(nifti_image *,
                                 nifti_image *,
                                 float,
                                 int);

/** @brief
 * @param
 */
extern "C++"
void reg_tools_binarise_image(nifti_image *);

/** @brief
 * @param
 */
extern "C++"
void reg_tools_binarise_image(nifti_image *, float);

/** @brief
 * @param
 */
extern "C++"
void reg_tools_binaryImage2int(nifti_image *, int *, int &);

/** @brief
 * @param
 */
extern "C++"
double reg_tools_getMeanRMS(nifti_image *, nifti_image *);

/** @brief
 * @param
 */
extern "C++"
int reg_tools_nanMask_image(nifti_image *, nifti_image *, nifti_image *);

/** @brief
 * @param
 */
extern "C++"
float reg_tools_getMinValue(nifti_image *);

/** @brief
 * @param
 */
extern "C++"
float reg_tools_getMaxValue(nifti_image *);

/** @brief
 * @param
 */
extern "C++" template<class DTYPE>
int reg_createImagePyramid(nifti_image *, nifti_image **, unsigned int, unsigned int);

/** @brief
 * @param
 */
extern "C++" template<class DTYPE>
int reg_createMaskPyramid(nifti_image *, int **, unsigned int , unsigned int , int *);

/** @brief
 * @param
 */
/** this function will threshold an image to the values provided,
 * set the scl_slope and sct_inter of the image to 1 and 0 (SSD uses actual image data values),
 * and sets cal_min and cal_max to have the min/max image data values
 */
extern "C++" template<class T>
void reg_thresholdImage(nifti_image *image,
                        T lowThr,
                        T upThr
                        );

/** @brief This function flipp the specified axis
 * @param
 */
extern "C++"
void reg_flippAxis(nifti_image *image,
                   void *array,
                   std::string cmd
                   );
#endif
