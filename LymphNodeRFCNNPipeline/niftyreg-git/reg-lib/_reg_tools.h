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

#include <fstream>
#include <limits>
#include <map>
#include "_reg_maths.h"


/* *************************************************************** */
/** @brief This function check some header parameters and correct them in
 * case of error. For example no dimension is lower than one. The scl_sclope
 * can not be equal to zero. The qto_xyz and qto_ijk are populated if
 * both qform_code and sform_code are set to zero.
 * @param image Input image to check and correct if necessary
 */
extern "C++"
void reg_checkAndCorrectDimension(nifti_image *image);

/* *************************************************************** */
/** @brief Check if the specified filename corresponds to an image.
 * @param name Input filename
 * @return True is the specified filename corresponds to an image,
 * false otherwise.
 */
extern "C++"
bool reg_isAnImageFileName(char *name);

/* *************************************************************** */
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
                          int timepoint,
                          float newMin,
                          float newMax
                         );


/* *************************************************************** */
/** @brief Set the scl_slope to 1 and the scl_inter to 0 and rescale
 * the intensity values
 * @param image Image to be updated
 */
extern "C++"
void reg_tools_removeSCLInfo(nifti_image *img);

/* *************************************************************** */
/** @brief reg_getRealImageSpacing
 * @param image image
 * @param spacingValues spacingValues
 */
extern "C++"
void reg_getRealImageSpacing(nifti_image *image,
                             float *spacingValues);
/* *************************************************************** */
/** @brief Smooth an image using a Gaussian kernel
 * @param image Image to be smoothed
 * @param sigma Standard deviation of the Gaussian kernel
 * to use. The kernel is bounded between +/- 3 sigma.
 * @param axis Boolean array to specify which axis have to be
 * smoothed. The array follow the dim array of the nifti header.
 */
extern "C++"
void reg_tools_kernelConvolution(nifti_image *image,
                                 float *sigma,
                                 int kernelType,
                                 int *mask = NULL,
                                 bool *timePoints = NULL,
                                 bool *axis = NULL);

/* *************************************************************** */
/** @brief Smooth a label image using a Gaussian kernel
 * @param image Image to be smoothed
 * @param varianceX The variance of the Gaussian kernel in X
 * @param varianceY The variance of the Gaussian kernel in Y
 * @param varianceZ The variance of the Gaussian kernel in Z
 * @param mask An integer mask over which the Gaussian smoothing should occour
 * @param timePoint Boolean array to specify which timepoints have to be
 * smoothed.
 */
extern "C++"
void reg_tools_labelKernelConvolution(nifti_image *image,
                                      float varianceX,
                                      float varianceY,
                                      float varianceZ,
                                      int *mask=NULL,
                                      bool *timePoint=NULL);


/* *************************************************************** */
/** @brief Downsample an image by a ratio of two
 * @param image Image to be downsampled
 * @param type The image is first smoothed  using a Gaussian
 * kernel of 0.7 voxel standard deviation before being downsample
 * if type is set to true.
 * @param axis Boolean array to specify which axis have to be
 * downsampled. The array follow the dim array of the nifti header.
 */
extern "C++" template <class PrecisionTYPE>
void reg_downsampleImage(nifti_image *image,
                         int type,
                         bool *axis
                        );
/* *************************************************************** */
/** @brief Returns the maximal euclidean distance from a
 * deformation field image
 * @param image Vector image to be considered
 * @return Scalar value that corresponds to the longest
 * euclidean distance
 */
extern "C++" template <class PrecisionTYPE>
PrecisionTYPE reg_getMaximalLength(nifti_image *image);
/* *************************************************************** */
/** @brief Change the datatype of a nifti image
 * @param image Image to be updated.
 */
extern "C++" template <class NewTYPE>
void reg_tools_changeDatatype(nifti_image *image,
                              int type=-1);
/* *************************************************************** */
/** @brief Add two images.
 * @param img1 First image to consider
 * @param img2 Second image to consider
 * @param out Result image that contains the result of the operation
 * between the first and second image.
 */
extern "C++"
void reg_tools_addImageToImage(nifti_image *img1,
                               nifti_image *img2,
                               nifti_image *out);
/* *************************************************************** */
/** @brief Substract two images.
 * @param img1 First image to consider
 * @param img2 Second image to consider
 * @param out Result image that contains the result of the operation
 * between the first and second image.
 */
extern "C++"
void reg_tools_substractImageToImage(nifti_image *img1,
                                     nifti_image *img2,
                                     nifti_image *out);
/* *************************************************************** */
/** @brief Multiply two images.
 * @param img1 First image to consider
 * @param img2 Second image to consider
 * @param out Result image that contains the result of the operation
 * between the first and second image.
 */
extern "C++"
void reg_tools_multiplyImageToImage(nifti_image *img1,
                                    nifti_image *img2,
                                    nifti_image *out);
/* *************************************************************** */
/** @brief Divide two images.
 * @param img1 First image to consider
 * @param img2 Second image to consider
 * @param out Result image that contains the result of the operation
 * between the first and second image.
 */
extern "C++"
void reg_tools_divideImageToImage(nifti_image *img1,
                                  nifti_image *img2,
                                  nifti_image *out);

/* *************************************************************** */
/** @brief Add a scalar to all image intensity
 * @param img1 Input image
 * @param out Result image that contains the result of the operation.
 * @param val Value to be added to input image
 */
extern "C++"
void reg_tools_addValueToImage(nifti_image *img1,
                               nifti_image *out,
                               float val);
/* *************************************************************** */
/** @brief Substract a scalar to all image intensity
 * @param img1 Input image
 * @param out Result image that contains the result of the operation.
 * @param val Value to be substracted to input image
 */
extern "C++"
void reg_tools_substractValueToImage(nifti_image *img1,
                                     nifti_image *out,
                                     float val);
/* *************************************************************** */
/** @brief Multiply a scalar to all image intensity
 * @param img1 Input image
 * @param out Result image that contains the result of the operation.
 * @param val Value to be multiplied to input image
 */
extern "C++"
void reg_tools_multiplyValueToImage(nifti_image *img1,
                                    nifti_image *out,
                                    float val);
/* *************************************************************** */
/** @brief Mivide a scalar to all image intensity
 * @param img1 Input image
 * @param out Result image that contains the result of the operation.
 * @param val Value to be divided to input image
 */
extern "C++"
void reg_tools_divideValueToImage(nifti_image *img1,
                                  nifti_image *out,
                                  float val);

/* *************************************************************** */
/** @brief Binarise an input image. All values different
 * from 0 are set to 1, 0 otherwise.
 * @param img Image that will be binarise inline
 */
extern "C++"
void reg_tools_binarise_image(nifti_image *img);

/* *************************************************************** */
/** @brief Binarise an input image. The binarisation is
 * performed according to a threshold value that is
 * user-defined.
 * @param img Image that will be binarise inline
 * @param thr Threshold value used for binarisation.
 * All values bellow thr are set to 0. All values equal
 * or bellow thr are set to 1
 */
extern "C++"
void reg_tools_binarise_image(nifti_image *img,
                              float thr);

/* *************************************************************** */
/** @brief Convert a binary image into an array of int.
 * This is used to define a mask within the registration
 * function.
 * @param img Input image
 * @param array The data array from the input nifti image
 * is binarised and stored in this array.
 * @param activeVoxelNumber This reference is updated
 * with the number of voxel that are included into the
 * mask
 */
extern "C++"
void reg_tools_binaryImage2int(nifti_image *img,
                               int *array,
                               int &activeVoxelNumber);

/* *************************************************************** */
/** @brief Compute the mean root mean squared error between
 * two vector images
 * @param imgA Input vector image
 * @param imgB Input vector image
 * @return Mean rsoot mean squared error valueis returned
 */
extern "C++"
double reg_tools_getMeanRMS(nifti_image *imgA,
                            nifti_image *imgB);
/* *************************************************************** */
/** @brief Set all voxels from an image to NaN if the voxel
 * bellong to the mask
 * @param img Input image to be masked with NaN value
 * @param mask Input mask that defines which voxels
 * have to be set to NaN
 * @param res Output image
 */
extern "C++"
int reg_tools_nanMask_image(nifti_image *img,
                            nifti_image *mask,
                            nifti_image *res);
/* *************************************************************** */
/** @brief Get the minimal value of an image
 * @param img Input image
 * @return min value
 */
extern "C++"
float reg_tools_getMinValue(nifti_image *img);
/* *************************************************************** */
/** @brief Get the maximal value of an image
 * @param img Input image
 * @return max value
 */
extern "C++"
float reg_tools_getMaxValue(nifti_image *img);
/* *************************************************************** */
/** @brief Generate a pyramid from an input image.
 * @param input Input image to be downsampled to create the pyramid
 * @param pyramid Output array of images that will contains the
 * different resolution images of the pyramid
 * @param levelNumber Number of level to use to create the pyramid.
 * 1 level corresponds to the original image resolution.
 * @param levelToPerform Number to level that will be perform during
 * the registration.
 */
extern "C++" template<class DTYPE>
int reg_createImagePyramid(nifti_image * input,
                           nifti_image **pyramid,
                           unsigned int levelNumber,
                           unsigned int levelToPerform);
/* *************************************************************** */
/** @brief Generate a pyramid from an input mask image.
 * @param input Input image to be downsampled to create the pyramid
 * @param pyramid Output array of mask images that will contains the
 * different resolution images of the pyramid
 * @param levelNumber Number of level to use to create the pyramid.
 * 1 level corresponds to the original image resolution.
 * @param levelToPerform Number to level that will be perform during
 * the registration.
 * @param activeVoxelNumber Array that contains the number of active
 * voxel for each level of the pyramid
 */
extern "C++" template<class DTYPE>
int reg_createMaskPyramid(nifti_image *input,
                          int **pyramid,
                          unsigned int levelNumber,
                          unsigned int levelToPerform,
                          int *activeVoxelNumber);
/* *************************************************************** */
/** @brief this function will threshold an image to the values provided,
 * set the scl_slope and sct_inter of the image to 1 and 0
 * (SSD uses actual image data values),
 * and sets cal_min and cal_max to have the min/max image data values.
 * @param image Input image to be thresholded.
 * @param lowThr Lower threshold value. All Value bellow the threshold
 * are set to the threshold value.
 * @param upThr Upper threshold value. All Value above the threshold
 * are set to the threshold value.
 */
extern "C++" template<class T>
void reg_thresholdImage(nifti_image *image,
                        T lowThr,
                        T upThr
                       );
/* *************************************************************** */
/** @brief This function flipp the specified axis
 * @param image Input image to be flipped
 * @param array Array that will contain the flipped
 * input image->data array
 * @param cmd String that contains the letter(s) of the axis
 * to flip (xyztuvw)
 */
extern "C++"
void reg_flippAxis(nifti_image *image,
                   void *array,
                   std::string cmd
                  );
/* *************************************************************** */
/** @brief This function converts an image containing deformation
 * field into a displacement field
 * The conversion is done using the appropriate qform/sform
 * @param image Image that contains a deformation field and will be
 * converted into a displacement field
 */
extern "C++"
int reg_getDisplacementFromDeformation(nifti_image *image);
/* *************************************************************** */
/** @brief This function converts an image containing a displacement field
 * into a displacement field.
 * The conversion is done using the appropriate qform/sform
 * @param image Image that contains a deformation field and will be
 * converted into a displacement field
 */
extern "C++"
int reg_getDeformationFromDisplacement(nifti_image *image);
/* *************************************************************** */
/** @brief The functions returns the largest ratio between two arrays
 * The returned value is the largest value computed as ((A/B)-1)
 * If A or B are zeros then the (A-B) value is returned.
 */
extern "C++" template<class DTYPE>
float reg_test_compare_arrays(DTYPE *ptrA,
                              DTYPE *ptrB,
                              size_t nvox);
/* *************************************************************** */
/** @brief The functions returns the largest ratio between input image intensities
 * The returned value is the largest value computed as ((A/B)-1)
 * If A or B are zeros then the (A-B) value is returned.
 */
extern "C++"
float reg_test_compare_images(nifti_image *imgA,
                              nifti_image *imgB);
/* *************************************************************** */
/** @brief The absolute operator is applied to the input image
 */
extern "C++"
void reg_tools_abs_image(nifti_image *img);
/* *************************************************************** */
/** @brief This function tells the progress to the CLI */
//extern "C++"
//void progressXML(unsigned long p, std::string text);
/* *************************************************************** */
/** @brief This function initiates progress updates through the CLI */
//extern "C++"
//void startProgress(std::string name);
/* *************************************************************** */
/** @brief This function closes progress updates through the CLI */
//extern "C++"
//void closeProgress(std::string name, std::string status);
/* *************************************************************** */
#endif
