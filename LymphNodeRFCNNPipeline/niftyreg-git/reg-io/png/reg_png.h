/**
 * @file reg_png.h
 * @author Marc Modat
 * @date 30/05/2012
 * @brief Interface between NiftyReg and pnglib
 *
 *  Created by Marc Modat on 30/05/2012.
 *  Copyright (c) 2012, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_PNG_H
#define _REG_PNG_H

#include "nifti1_io.h"
#include "_reg_tools.h"

/* *************************************************************** */
/** @brief This function read a png file from the hard-drive and convert
  * it into a nifti_structure. using this function, you can either
  * read the full image or only the header information
  * @param filename Filename of the png file to read
  * @param readData The actual data is read if the flag is set to true
  * @return Returns a pointer to the nifti_image that contains the PNG file
  */
nifti_image *reg_io_readPNGfile(const char *filename, bool readData);
/* *************************************************************** */
/** @brief This function first convert a nifti image into a png and then
  * save the png file.
  * @param image Nifti image that will first be converted to a png file
  * and then will be saved on the disk
  * @param filename Path where the png file will be saved on the disk
  */
void reg_io_writePNGfile(nifti_image *image, const char *filename);
/* *************************************************************** */

#endif
