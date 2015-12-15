/*
 *  _reg_ReadWriteImage.cpp
 *
 *
 *  Created by Marc Modat on 30/05/2012.
 *  Copyright (c) 2012, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_READWRITEIMAGE_CPP
#define _REG_READWRITEIMAGE_CPP

#include "_reg_ReadWriteImage.h"
#include "_reg_tools.h"

/* *************************************************************** */
void reg_hack_filename(nifti_image* image, const char *filename)
{
   std::string name(filename);
   name.append("\0");
   // Free the char arrays if already allocated
   if(image->fname) free(image->fname);
   if(image->iname) free(image->iname);
   // Allocate the char arrays
   image->fname = (char *)malloc((name.size()+1)*sizeof(char));
   image->iname = (char *)malloc((name.size()+1)*sizeof(char));
   // Copy the new name in the char arrays
   strcpy(image->fname,name.c_str());
   strcpy(image->iname,name.c_str());
   // Returns at the end of the function
   return;
}
/* *************************************************************** */
int reg_io_checkFileFormat(const char *filename)
{
   // Nifti format is used by default
   // Check the extention of the provided filename
   std::string b(filename);
   if(b.find( ".nii.gz") != std::string::npos)
      return NR_NII_FORMAT;
   else if(b.find( ".nii") != std::string::npos)
      return NR_NII_FORMAT;
   else if(b.find( ".hdr") != std::string::npos)
      return NR_NII_FORMAT;
   else if(b.find( ".img.gz") != std::string::npos)
      return NR_NII_FORMAT;
   else if(b.find( ".img") != std::string::npos)
      return NR_NII_FORMAT;
   else if(b.find( ".png") != std::string::npos)
      return NR_PNG_FORMAT;
   else if(b.find( ".nrrd") != std::string::npos)
      return NR_NRRD_FORMAT;
   else if(b.find( ".nhdr") != std::string::npos)
      return NR_NRRD_FORMAT;
   else fprintf(stderr, "[NiftyReg WARNING]: No filename extension provided - the Nifti library is used by default\n");

   return NR_NII_FORMAT;
}
/* *************************************************************** */
nifti_image *reg_io_ReadImageFile(const char *filename)
{
   // First read the fileformat in order to use the correct library
   int fileFormat=reg_io_checkFileFormat(filename);

   // Create the nifti image pointer
   nifti_image *image=NULL;

   // Read the image and convert it to nifti format if required
   switch(fileFormat)
   {
   case NR_NII_FORMAT:
      image=nifti_image_read(filename,true);
      reg_hack_filename(image,filename);
      break;
   case NR_PNG_FORMAT:
      image=reg_io_readPNGfile(filename,true);
      reg_hack_filename(image,filename);
      break;
   case NR_NRRD_FORMAT:
      Nrrd *nrrdImage = reg_io_readNRRDfile(filename);
      image = reg_io_nrdd2nifti(nrrdImage);
      nrrdNuke(nrrdImage);
      reg_hack_filename(image,filename);
      break;
   }
   reg_checkAndCorrectDimension(image);

   // Return the nifti image
   return image;
}
/* *************************************************************** */
nifti_image *reg_io_ReadImageHeader(const char *filename)
{
   // First read the fileformat in order to use the correct library
   int fileFormat=reg_io_checkFileFormat(filename);

   // Create the nifti image pointer
   nifti_image *image=NULL;

   // Read the image and convert it to nifti format if required
   switch(fileFormat)
   {
   case NR_NII_FORMAT:
      image=nifti_image_read(filename,false);
      break;
   case NR_PNG_FORMAT:
      image=reg_io_readPNGfile(filename,false);
      reg_hack_filename(image,filename);
      break;
   case NR_NRRD_FORMAT:
      Nrrd *nrrdImage = reg_io_readNRRDfile(filename);
      image = reg_io_nrdd2nifti(nrrdImage);
      nrrdNuke(nrrdImage);
      reg_hack_filename(image,filename);
      break;
   }
   reg_checkAndCorrectDimension(image);

   // Return the nifti image
   return image;
}
/* *************************************************************** */
void reg_io_WriteImageFile(nifti_image *image, const char *filename)
{
   // First read the fileformat in order to use the correct library
   int fileFormat=reg_io_checkFileFormat(filename);

   // Check if the images can be saved as a png file
   if( (image->nz>1 ||
         image->nt>1 ||
         image->nu>1 ||
         image->nv>1 ||
         image->nw>1 ) &&
         fileFormat==NR_PNG_FORMAT)
   {
      // If the image has more than two dimension,
      // the filename is converted to nifti
      std::string b(filename);
      b.replace(b.find( ".png"),4,".nii.gz");
      printf("[NiftyReg WARNING] The file can not be saved as png and is converted to nifti\n");
      printf("[NiftyReg WARNING] %s -> %s\n", filename, b.c_str());
      filename=b.c_str();
      fileFormat=NR_NII_FORMAT;
   }

   // Convert the image to the correct format if required, set the filename and save the file
   switch(fileFormat)
   {
   case NR_NII_FORMAT:
      nifti_set_filenames(image,filename,0,0);
      nifti_image_write(image);
      break;
   case NR_PNG_FORMAT:
      reg_io_writePNGfile(image,filename);
      break;
   case NR_NRRD_FORMAT:
      Nrrd *nrrdImage = reg_io_nifti2nrrd(image);
      reg_io_writeNRRDfile(nrrdImage,filename);
      nrrdNuke(nrrdImage);
   }

   // Return
   return;
}
/* *************************************************************** */

#endif
