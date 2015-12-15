/**
 * @file _reg_blockMatching.h
 * @brief Functions related to the block matching approach
 * @author Marc Modat and Pankaj Daga
 * @date 24/03/2009
 *
 *  Created by Marc Modat and Pankaj Daga on 24/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef __REG_BLOCKMATCHING_H__
#define __REG_BLOCKMATCHING_H__

#include "_reg_maths.h"
#include <vector>

#define TOLERANCE 0.01
#define MAX_ITERATIONS 30

#define BLOCK_WIDTH 4
#define BLOCK_SIZE 64
#define BLOCK_2D_SIZE 16
#define OVERLAP_SIZE 3

#define NUM_BLOCKS_TO_COMPARE 343 // We compare in a 7x7x7 neighborhood.
#define NUM_BLOCKS_TO_COMPARE_2D 49
#define NUM_BLOCKS_TO_COMPARE_1D 7

/**
 *
 * Main algorithm of Ourselin et al.
 * The essence of the algorithm is as follows:
 * - Subdivide the target image into a number of blocks and find
 *   the block in the result image that is most similar.
 * - Get the point pair between the target and the result image block
 *   for the most similar block.
 *
 * target: Pointer to the nifti target image.
 * result: Pointer to the nifti result image.
 *
 *
 * block_size: Size of the block.
 * block_half_width: Half-width of the search neighborhood.
 * delta_1: Spacing between two consecutive blocks
 * delta_2: Sub-sampling value for a block
 *
 * Possible improvement: Take care of anisotropic data. Right now, we specify
 * the block size, neighborhood and the step sizes in voxels and it would be
 * better to specify it in millimeters and take the voxel size into account.
 * However, it would be more efficient to calculate this once (outside this
 * module) and pass these values for each axes. For the time being, we do this
 * simple implementation.
 *
 */

/**
 * @brief Structure which contains the block matching parameters
 */
struct _reg_blockMatchingParam
{
   int blockNumber[3];
   int percent_to_keep;

   float *targetPosition;
   float *resultPosition;

   int activeBlockNumber;
   int *activeBlock;

   int definedActiveBlock;

   int stepSize;

   _reg_blockMatchingParam()
      : targetPosition(0),
        resultPosition(0),
        activeBlock(0)
   {}

   ~_reg_blockMatchingParam()
   {
      if(targetPosition) free(targetPosition);
      if(resultPosition) free(resultPosition);
      if(activeBlock) free(activeBlock);
   }
};

/** @brief This function initialise a _reg_blockMatchingParam structure
 * according to the the provided arguments
 * @param referenceImage Reference image where the blocks are defined
 * @param params Block matching parameter structure that will be populated
 * @param percentToKeep_block Amount of block to block to keep for the
 * optimisation process
 * @param percentToKeep_opt Hmmmm ... I actually don't remember.
 * Need to check the source :)
 * @param mask Array than contains a mask of the voxel form the reference
 * image to consider for the registration
 * @param runningOnGPU Has to be set to true if the registration is
 * registration has to be performed on the GPU
 */
extern "C++"
void initialise_block_matching_method(nifti_image * referenceImage,
                                      _reg_blockMatchingParam *params,
                                      int percentToKeep_block,
                                      int percentToKeep_opt,
                                      int stepSize_block,
                                      int *mask,
                                      bool runningOnGPU = false);

/** @brief Interface for the block matching algorithm.
 * @param referenceImage Reference image in the currrent registration task
 * @param warpedImage Warped floating image in the currrent registration task
 * @param params Block matching parameter structure that contains all
 * relevant information
 * @param mask Maks array where only voxel defined as active are considered
 */
extern "C++"
void block_matching_method(nifti_image * referenceImage,
                           nifti_image * warpedImage,
                           _reg_blockMatchingParam *params,
                           int *mask);

/** @brief Apply the given affine transformation to a point
 * @todo I should remove this function as it is redondant
 * @param mat Transformation matrix
 * @param pt Input position
 * @param pr Output position
 */
void apply_affine(mat44 * mat,
                  float *pt,
                  float *pr);

/** @brief Find the optimal affine transformation that matches the points
 * in the target image to the point in the result image
 * @param params Block-matching structure that contains the relevant information
 * @param transformation_matrix Initial transformation matrix that is updated
 * @param affine Returns an affine transformation (12 DoFs) if set to true;
 * returns a rigid transformation (6 DoFs) otherwise
 */
void optimize(_reg_blockMatchingParam *params,
              mat44 * transformation_matrix,
              bool affine = true);



#endif
