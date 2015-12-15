/**
 * @file _reg_globalTransformation.h
 * @author Marc Modat
 * @date 25/03/2009
 * @brief library that contains the function related to global transformation
 *
 *  Created by Marc Modat on 25/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_AFFINETRANSFORMATION_H
#define _REG_AFFINETRANSFORMATION_H

#include "nifti1_io.h"
#include <fstream>
#include <limits>
#include "_reg_maths.h"


/** @brief This Function compute a deformation field based
 * on affine transformation matrix
 * @param affine This matrix contains the affine transformation
 * used to parametrise the transformation
 * @param referenceImage The image represents the reference space
 * of the transformation
 * @param deformationField Image that contains the deformation field
 * that is being updated
 */
extern "C++"
void reg_affine_positionField(mat44 *affine,
                nifti_image *referenceImage,
                nifti_image *deformationField);

/** @brief Read a text file that contains a affine transformation
 * and store it into a mat44 structure. This function can also read
 * affine parametrisation from Flirt (FSL package) and convert it
 * to a standard millimeter parametrisation
 * @param mat Structure that will be updated with the affine
 * transformation matrix
 * @param referenceImage Reference image of the current transformation
 * @param floatingImage Floating image of the current transformation.
 * Note that referenceImage and floating image have to be defined but
 * are only used when dealing with a Flirt affine matrix.
 * @param filename Filename for the text file that contains the matrix
 * to read
 * @param flirtFile If this flag is set to true the matrix is converted
 * from a Flirt (FSL) parametrisation to a standard parametrisation
 */
extern "C++"
void reg_tool_ReadAffineFile(mat44 *mat,
                             nifti_image* referenceImage,
                             nifti_image* floatingImage,
                             char *fileName,
                             bool flirtFile);
/** @brief Read a file that contains a 4-by-4 matrix and store it into
 * a mat44 structure
 * @param mat mat44 structure that will be updated with the affine matrix
 * @param filename Filename of the text file that contains the matrix to read
 */
extern "C++"
void reg_tool_ReadAffineFile(	mat44 *mat,
                                char *filename);

/** @brief This function save a 4-by-4 matrix to the disk as a text file
 * @param mat Matrix to be saved on the disk
 * @param filename Name of the text file to save on the disk
 */
extern "C++"
void reg_tool_WriteAffineFile(	mat44 *mat,
                                const char *fileName);

#endif
