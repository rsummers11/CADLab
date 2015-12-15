/**
 * @file _reg_femTransformation_gpu.h
 * @author Marc Modat
 * @date 02/11/2011
 * @brief Functions built to interface between NiftyReg and NiftySim
 * It basically allows to populate a dense deformation
 *
 *  Created by Marc Modat on 02/11/2011.
 *  Copyright (c) 2011, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_FEMTRANSFORMATION_H
#define _REG_FEMTRANSFORMATION_H

#include "nifti1_io.h"
#include <fstream>
#include <limits>
#include "_reg_maths.h"

/** @brief Initialise multiples arrays to populate a dense deformation
 * field from a FEM parametrisation
 * @param elementNodes Arrays that contains the nodes use to define each element.
 * @param nodePositions Arrays that contains the position in mm of
 * every node
 * @param deformationFieldImage Deformation field image, at this stage it
 * is only used to define the space of the transformation
 * @param closestNodes This array will contain for every voxel the closest
 * nodes to be used for interpolation
 * @param femInterpolationWeight This arrayt will contain for every voxel
 * the weight associated with the closest node.
 */
void reg_fem_InitialiseTransformation(int *elementNodes,
                                      unsigned int elementNumber,
                                      float *nodePositions,
                                      nifti_image *deformationFieldImage,
                                      unsigned int *closestNodes,
                                      float *femInterpolationWeight
                                     );

/** @brief A dense deformation field is filled using interpolation
 * from a coarse mesh
 * @param nodePositions Array that contains the position of every node
 * @param deformationFieldImage Deformation field image that will be
 * filled
 * @param closestNodes Array that contains for every voxel the closest
 * nodes from the mesh
 * @param femInterpolationWeight Array that contains for every voxel,
 * the weight associated with the closest nodes.
 */
void reg_fem_getDeformationField(float *nodePositions,
                                 nifti_image *deformationFieldImage,
                                 unsigned int *closestNodes,
                                 float *femInterpolationWeight
                                );

/** @brief Convert a dense gradient image into a mesh based gradient image
 * @param voxelBasedGradient Image that contains the gradient image
 * @param closestNodes Array that contains the closest nodes associated
 * with every voxel
 * @param femInterpolationWeight Array that contains for every voxel the
 * weight associated with the closest nodes
 * @param nodeNumber Scalar that contains the total number of node in the mesh
 * @param femBasedGradient Array that contains the gradient values at
 * every node.
 */
void reg_fem_voxelToNodeGradient(nifti_image *voxelBasedGradient,
                                 unsigned int *closestNodes,
                                 float *femInterpolationWeight,
                                 unsigned int nodeNumber,
                                 float *femBasedGradient);
#endif
