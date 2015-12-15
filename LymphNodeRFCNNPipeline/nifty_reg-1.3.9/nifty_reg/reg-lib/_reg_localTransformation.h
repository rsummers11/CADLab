/**
 * @file _reg_localTransformation.h
 * @brief Library that contains local deformation related functions
 * @author Marc Modat
 * @date 25/03/2009
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_TRANSFORMATION_H
#define _REG_TRANSFORMATION_H

#include "nifti1_io.h"
#include "_reg_globalTransformation.h"
#include "float.h"
#include <limits>
#include "_reg_maths.h"
#include "_reg_tools.h"

#if _USE_SSE
	#include <emmintrin.h>
#endif


/* *********************************************** */
/* ****      CUBIC SPLINE BASED FUNCTIONS     **** */
/* *********************************************** */

/* *************************************************************** */
/** @brief Generate a control point grid image based on the dimension of a
 * reference image and on a spacing.
 * The function set the qform and sform code to overlay the reference
 * image.
 * @param controlPointGridImage The resulting control point grid will be
 * store in this pointer
 * @param referenceImage Reference image which dimension will be used to
 * define the control point grid image space
 * @param spacingMillimeter Control point spacing along each axis
 */
extern "C++" template <class DTYPE>
void reg_createControlPointGrid(nifti_image **controlPointGridImage,
                                nifti_image *referenceImage,
                                float *spacingMillimeter);

/* *************************************************************** */
/** @brief Compute a dense deformation field in the space of a reference
 * image from a grid of control point.
 * @param controlPointGridImage Control point grid that contains the deformation
 * parametrisation
 * @param referenceImage Reference image that defined the space of the deformation field
 * @param deformationField Output image that will be populated with the deformation field
 * @param mask Array that contains the a mask. Any voxel with a positive value is included
 * into the mask
 * @param composition A composition scheme is used if this value is set to true,
 * the deformation is starting from a blank grid otherwise.
 * @param bspline A cubic B-Spline scheme is used if the value is set to true,
 * a cubic spline scheme is used otherwise (interpolant spline).
 */
extern "C++"
void reg_spline_getDeformationField(nifti_image *controlPointGridImage,
                                    nifti_image *referenceImage,
                                    nifti_image *deformationField,
                                    int *mask,
                                    bool composition,
                                    bool bspline
                                    );
/* *************************************************************** */
/** @brief Compute and return the average bending energy computed using cubic b-spline.
 * The value is approximated as the bending energy is computated at
 * the control point position only.
 * @param controlPointGridImage Control point grid that contains the deformation
 * parametrisation
 * @return The normalised bending energy. Normalised by the number of voxel
 */
extern "C++"
double reg_bspline_bendingEnergy(nifti_image *controlPointGridImage);
/* *************************************************************** */
/** @brief Compute and return the approximated (at the control point position)
 * bending energy gradient for each control point
 * @param controlPointGridImage Image that contains the control point
 * grid used to parametrise the transformation
 * @param referenceImage Image that defines the space of the transformation
 * @param gradientImage Image of identical size that the control
 * point grid image. The gradient of the bending-energy will be added
 * at every control point position.
 * @param weight Scalar which will be multiplied by the bending-energy gradient
 */
extern "C++"
void reg_bspline_bendingEnergyGradient(nifti_image *controlPointGridImage,
                                       nifti_image *referenceImage,
                                       nifti_image *gradientImage,
                                       float weight
                                       );
/* *************************************************************** */
/** @brief Compute and return the linear elastic energy terms approximated
 * at the control point positions only.
 * @param controlPointGridImage Image that contains the transformation
 * parametrisation
 * @param values Array[2] that contains (0) the penalty term based on
 * the anti-symmetric part of the Jacobian matrices and (1) the squared
 * trace of the Jacobian matrices
 */
extern "C++"
void reg_bspline_linearEnergy(nifti_image *controlPointGridImage,
                              double *values
                              );
/* *************************************************************** */
/** @brief Compute the gradient of the linear elastic energy terms
 * approximated at the control point positions only.
 * @param controlPointGridImage Image that contains the transformation
 * parametrisation
 * @param referenceImage Reference image to define the deformation
 * field space
 * @param gradientImage Image of similar size than the control point
 * grid and that contains the gradient of the objective function.
 * The gradient of the linear elasticily terms are added to the
 * current values
 * @param weight0 Weight to apply to the first term of the penalty
 * @param weight1 Weight to apply to the second term of the penalty
 */
extern "C++"
void reg_bspline_linearEnergyGradient(nifti_image *controlPointGridImage,
                                      nifti_image *referenceImage,
                                      nifti_image *gradientImage,
                                      float weight0,
                                      float weight1
                                      );
/* *************************************************************** */
/** @brief Compute and return the L2 norm of the displacement approximated
  * at the control point positions only.
  * @param controlPointGridImage Image that contains the transformation parameters
  * @return The sum of squared Euclidean displacement at every control
  * point position
  */
extern "C++"
double reg_bspline_L2norm_displacement(nifti_image *controlPointGridImage);
/* *************************************************************** */
/** @brief Compute the gradient of the L2 norm of the displacement approximated
 * at the control point positions only.
 * @param controlPointGridImage Image that contains the transformation parameters
 * @param referenceImage Image that defines the space of the deformation field
 * @param gradientImage Image of similar size than the control point
 * grid and that contains the gradient of the objective function.
 * The gradient of the L2 norm of the displacement terms are added to the
 * current values
 * @param weight The gradient of the Euclidean displacement of the control
 * point position is weighted by this value
 */
extern "C++"
void reg_bspline_L2norm_dispGradient(nifti_image *controlPointGridImage,
                                     nifti_image *referenceImage,
                                     nifti_image *gradientImage,
                                     float weight);
/* *************************************************************** */
/** @brief Compute the Jacobian determinant map using a cubic b-spline
 * @param controlPointGridImage Image that contains the transformation
 * parametrisation.
 * @param jacobianImage Image that will be populated with the determinant
 * of the Jacobian matrix of the transformation at every voxel posision.
 */
extern "C++"
void reg_bspline_GetJacobianMap(nifti_image *controlPointGridImage,
                                nifti_image *jacobianImage
                                );
/* *************************************************************** */
/** @brief Compute the average Jacobian determinant
 * @param controlPointGridImage Image that contains the transformation
 * parametrisation.
 * @param referenceImage Image that defines the space of the deformation
 * field for the transformation
 * @param approx Approximate the average Jacobian determinant by using
 * only the information from the control point if the value is set to true;
 * all voxels are considered if the value is set to false.
 */
extern "C++"
double reg_bspline_jacobian(nifti_image *controlPointGridImage,
                            nifti_image *referenceImage,
                            bool approx
                            );
/* *************************************************************** */
/** @brief Compute the gradient at every control point position of the
 * Jacobian determinant based penalty term
 * @param controlPointGridImage Image that contains the transformation
 * parametrisation.
 * @param referenceImage Image that defines the space of the deformation
 * field for the transformation
 * @param gradientImage Image of similar size than the control point
 * grid and that contains the gradient of the objective function.
 * The gradient of the Jacobian determinant based penalty term is added
 * to the current values
 * @param weight The gradient of the Euclidean displacement of the control
 * point position is weighted by this value
 * @param approx Approximate the gradient by using only the information
 * from the control point if the value is set to true; all voxels are
 * considered if the value is set to false.
 */
extern "C++"
void reg_bspline_jacobianDeterminantGradient(nifti_image *controlPointGridImage,
                                             nifti_image *referenceImage,
                                             nifti_image *gradientImage,
                                             float weight,
                                             bool approx
                                             );
/* *************************************************************** */
/** @brief Compute the Jacobian matrix at every voxel position
 * using a cubic b-spline parametrisation. This function does not require
 * the control point grid to perfectly overlay the reference image.
 * @param referenceImage Image that defines the space of the deformation
 * field
 * @param controlPointGridImage Control point grid position that defines
 * the cubic B-Spline parametrisation
 * @param jacobianImage Array that is filled with the Jacobian matrices
 * for every voxel.
 */
extern "C++"
void reg_bspline_GetJacobianMatrixFull(nifti_image *referenceImage,
                                       nifti_image *controlPointGridImage,
                                       mat33 *jacobianImage
                                       );
/* *************************************************************** */
/** @brief Compute the Jacobian matrix at every voxel position
 * using a cubic b-spline parametrisation. This function does require
 * the control point grid to perfectly overlay the reference image.
 * @param referenceImage Image that defines the space of the deformation
 * field
 * @param controlPointGridImage Control point grid position that defines
 * the cubic B-Spline parametrisation
 * @param jacobianImage Array that is filled with the Jacobian matrices
 * for every voxel.
 */
extern "C++"
void reg_bspline_GetJacobianMatrix(nifti_image *referenceImage,
                                   nifti_image *controlPointGridImage,
                                   mat33 *jacobianImage
                                   );
/* *************************************************************** */
/** @brief Correct the folding in the transformation parametrised through
 * cubic B-Spline
 * @param controlPointGridImage Image that contains the cubic B-Spline
 * parametrisation
 * @param referenceImage Image that defines the space of the transformation
 * @param approx The function can be run be considering only the control
 * point position (approx==false) or every voxel (approx==true)
 */
extern "C++"
double reg_bspline_correctFolding(nifti_image *controlPointGridImage,
                                  nifti_image *referenceImage,
                                  bool approx
                                  );
/* *************************************************************** */
/** @brief Upsample an image from voxel space to node space using
 * millimiter correspendences.
 * @param nodeImage This image is a coarse representation of the
 * transformation (typically a grid of control point). This image
 * values are going to be updated
 * @param voxelImage This image contains a dense representation
 * if the transformation (typically a voxel-based gradient)
 * @param weight The values from used to update the node image
 * will be multiplied by the weight
 * @param update The values in node image will be incremented if
 * update is set to true; a blank node image is considered otherwise
 */
extern "C++"
void reg_voxelCentric2NodeCentric(nifti_image *nodeImage,
                                  nifti_image *voxelImage,
                                  float weight,
                                  bool update
                                  );
/* *************************************************************** */
/** @brief Refine a grid of control points
 * @param referenceImage Image that defined the space of the reference
 * image
 * @param controlPointGridImage This control point grid will be refined
 * by dividing the control point spacing by a ratio of 2
 */
extern "C++"
void reg_bspline_refineControlPointGrid(nifti_image *referenceImage,
                                        nifti_image *controlPointGridImage
                                        );
/* *************************************************************** */
/** @brief Initialise a lattice of control point to generate a global deformation
 * @param affineTransformation Matrix that contains an affine transformation
 * @param controlPointGridImage This grid of control point will be set to reproduce
 * the global transformation define by the matrix
 */
extern "C++"
int reg_bspline_initialiseControlPointGridWithAffine(mat44 *affineTransformation,
                                                     nifti_image *controlPointGridImage
                                                     );
/* *************************************************************** */
/** @brief This function compose the a first control point image with a second one:
 * Grid2(x) <= Grid1(Grid2(x)).
 * Grid1 and Grid2 have to contain either displacement or deformation.
 * The output will be a deformation field if grid1 is a deformation,
 * The output will be a displacement field if grid1 is a displacement.
 * @param grid1 Image that contains the first grid of control points
 * @param grid2 Image that contains the second grid of control points
 * @param displacement1 The first grid is a displacement field if this
 * value is set to true, a deformation field otherwise
 * @param displacement2 The second grid is a displacement field if this
 * value is set to true, a deformation field otherwise
 * @param Cubic B-Spline can be used (bspline==true)
 * or cubic Spline (bspline==false)
 */
extern "C++"
int reg_spline_cppComposition(nifti_image *grid1,
                              nifti_image *grid2,
                              bool displacement1,
                              bool displacement2,
                              bool bspline
                              );
/* *************************************************************** */


/* *********************************************** */
/* ****   DEFORMATION FIELD BASED FUNCTIONS   **** */
/* *********************************************** */

/* *************************************************************** */
/** @brief Compute the Jacobian determinant at every voxel position
 * from a deformation field. A linear interpolation is
 * assumed
 * @param deformationField Image that contains a deformation field
 * @param jacobianImage This image will be fill with the Jacobian
 * determinant of the transformation of every voxel.
 */
extern "C++"
void reg_defField_getJacobianMap(nifti_image *deformationField,
                                 nifti_image *jacobianImage);
/* *************************************************************** */
/** @brief Compute the Jacobian matrix at every voxel position
 * from a deformation field. A linear interpolation is
 * assumed
 * @param deformationField Image that contains a deformation field
 * @param jacobianMatrices This array will be fill with the Jacobian
 * matrices of the transformation of every voxel.
 */
extern "C++"
void reg_defField_getJacobianMatrix(nifti_image *deformationField,
                                    mat33 *jacobianMatrices);
/* *************************************************************** */
/** @brief Preforms the composition of two deformation fields
 * The deformation field image is applied to the second image:
 * dfToUpdate. Both images are expected to contain deformation
 * field.
 * @param deformationField Image that contains the deformation field
 * that will be applied
 * @param dfToUpdate Image that contains the deformation field that
 * is being updated
 * @param mask Mask overlaid on the dfToUpdate field where only voxel
 * within the mask will be updated. All positive values in the maks
 * are considered as belonging to the mask.
 */
extern "C++"
void reg_defField_compose(nifti_image *deformationField,
                          nifti_image *dfToUpdate,
                          int *mask);
/* *************************************************************** */

/* *********************************************** */
/* ****     VELOCITY FIELD BASED FUNCTIONS    **** */
/* *********************************************** */

/* *************************************************************** */
/** @brief This function computed Jacobian matrices by integrating
 * the velocity field
 * @param referenceImage Image that defines the space of the deformation
 * field
 * @param velocityFieldImage Image that contains a velocity field
 * parametrised using a grid of control points
 * @param jacobianMatrices Array of matrices that will be filled with
 * the Jacobian matrices of the transformation
 */
extern "C++"
int reg_bspline_GetJacobianMatricesFromVelocityField(nifti_image* referenceImage,
                                                     nifti_image* velocityFieldImage,
                                                     mat33* jacobianMatrices
                                                     );
/* *************************************************************** */
/** @brief This function computed a Jacobian determinant map by integrating
 * the velocity field
 * @param jacobianDetImage This image will be filled with the Jacobian
 * determinants of every voxel.
 * @param velocityFieldImage Image that contains a velocity field
 * parametrised using a grid of control points
 */
extern "C++"
int reg_bspline_GetJacobianDetFromVelocityField(nifti_image* jacobianDetImage,
                                                nifti_image* velocityFieldImage
                                                );
/* *************************************************************** */
/** @brief The deformation field (img2) is computed by integrating
 * a velocity field (img1)
 * @param velocityFieldImage Image that contains a velocity field
 * parametrised using a grid of control points
 * @param deformationFieldImage Deformation field image that will
 * be filled using the exponentiation of the velocity field.
 */
extern "C++"
void reg_bspline_getDeformationFieldFromVelocityGrid(nifti_image *velocityFieldGrid,
                                                     nifti_image *deformationFieldImage);
/* *************************************************************** */


/* *********************************************** */
/* ****            OTHER FUNCTIONS            **** */
/* *********************************************** */

/* *************************************************************** */
/** @brief This function converts an image containing deformation
 * field into a displacement field
 * The conversion is done using the appropriate qform/sform
 * @param image Image that contains a deformation field and will be
 * converted into a displacement field
 */
int reg_getDisplacementFromDeformation(nifti_image *image);
/* *************************************************************** */
/** @brief This function converts an image containing a displacement field
 * into a displacement field.
 * The conversion is done using the appropriate qform/sform
 * @param image Image that contains a deformation field and will be
 * converted into a displacement field
 */
int reg_getDeformationFromDisplacement(nifti_image *image);

/* *************************************************************** */
/** @brief This function compute the BCH update using an initial verlocity field
 * and its gradient.
 * @param img1 Image that contains the velocity field parametrisation
 * This image is updated
 * @param img2 This image contains the gradient to use
 * @param type The type encodes the number of component of the serie
 * to be considered:\n
 * 0 - w=u+v\n
 * 1 - w=u+v+0.5*[u,v]\n
 * 2 - w=u+v+0.5*[u,v]+[u,[u,v]]/12\n
 * 3 - w=u+v+0.5*[u,v]+[u,[u,v]]/12-[v,[u,v]]/12\n
 * 4 - w=u+v+0.5*[u,v]+[u,[u,v]]/12-[v,[u,v]]/12-[v,[u,[u,g]]]/24
 */
extern "C++"
void compute_BCH_update(nifti_image *img1,
                        nifti_image *img2,
                        int type);

/* *************************************************************** */
/** @brief This function deconvolve an image by a cubic B-Spline kernel
 * in order to get cubic B-Spline coefficient
 * @param img Image to be deconvolved
 */
extern "C++"
void reg_spline_GetDeconvolvedCoefficents(nifti_image *img);

/* *************************************************************** */
#endif
