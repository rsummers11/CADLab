/*
 *  reg_png.cpp
 *
 *
 *  Created by Marc Modat on 30/05/2012.
 *  Copyright (c) 2012, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _REG_PNG_CPP
#define _REG_PNG_CPP

#include "reg_png.h"

/* *************************************************************** */
nifti_image *reg_io_readPNGfile(const char *pngFileName, bool readData)
{
    // We first read the png file
    FILE *pngFile=NULL;
    pngFile = fopen (pngFileName, "r");
    if(pngFile==NULL){
        fprintf(stderr,"[NiftyReg ERROR]: Can not open the png file %s\n", pngFileName);
        exit(1);
    }

    uch sig[8];
    size_t a=0; // useless - here to avoid a warning
    a=fread(sig, 1, 8, fopen (pngFileName, "r"));
    if(!png_check_sig(sig, 8)) exit(1);

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr){
        fprintf(stderr,"[NiftyReg ERROR]: Error when reading the png file - out of memory\n");
        exit(1);
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fprintf(stderr,"[NiftyReg ERROR]: Error when reading the png file - out of memory\n");
        exit(1);
    }

    png_init_io(png_ptr, pngFile);
    png_read_info(png_ptr, info_ptr);

    png_uint_32 Width, Height;
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, &Width, &Height, &bit_depth,
                 &color_type, NULL, NULL, NULL);

    int Channels;
    ulg rowbytes;

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_expand(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_expand(png_ptr);

    if (bit_depth == 16)
        png_set_strip_16(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY ||
            color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    png_bytep *row_pointers= new png_bytep[Height];

    png_read_update_info(png_ptr, info_ptr);

    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    Channels = (int)png_get_channels(png_ptr, info_ptr);

    if(Channels > 3){
        printf("[NiftyReg WARNING]: the PNG file has %i channels. Only the first three are considered for RGB to gray conversion.\n", Channels);
    }

    if(Channels == 2){
        printf("[NiftyReg WARNING]: the PNG file has 2 channels. They will be average into one single channel.\n");
    }

    int dim[8]={2,Width,Height,1,1,1,1,1};
    nifti_image *niiImage=NULL;
    if(readData){

        uch *image_data;
        if ((image_data = (uch *)malloc(Width*Height*Channels*sizeof(uch))) == NULL)
            exit(1);

        for (png_uint_32 i=0;i<Height;++i){
            row_pointers[i] = image_data + i*rowbytes;
        }

        png_read_image(png_ptr, row_pointers);
        png_read_end(png_ptr, NULL);

        niiImage=nifti_make_new_nim(dim,NIFTI_TYPE_UINT8,true);
        uch *niiPtr=static_cast<uch *>(niiImage->data);
        for(size_t i=0;i<niiImage->nvox;++i) niiPtr[i]=0;
        // Define some weight to create a gray scale image
        float rgb2grayWeight[3];
        if(Channels==1){
            rgb2grayWeight[0]=1;
        }
        else if(Channels==2){
            rgb2grayWeight[0]=0.5;
            rgb2grayWeight[1]=0.5;
        }
        if(Channels>=3){ // rgb to y
            rgb2grayWeight[0]=0.299;
            rgb2grayWeight[1]=0.587;
            rgb2grayWeight[2]=0.114;
        }
        for(int c=0;c<(Channels<3?Channels:3);++c){
            for(png_uint_32 h=0;h<Height;++h){
                for(png_uint_32 w=0;w<Width;++w){
                    niiPtr[h*niiImage->nx+w] += (uch)((float)row_pointers[h][w*Channels+c]*rgb2grayWeight[c]);
                }
            }
        }
    }
    else{
        niiImage=nifti_make_new_nim(dim,NIFTI_TYPE_UINT8,false);
    }
    delete []row_pointers;
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose (pngFile);

    nifti_set_filenames(niiImage, pngFileName,0,0);
    return niiImage;
}

/* *************************************************************** */
void reg_io_writePNGfile(nifti_image *image, const char *filename)
{
    // We first check the nifti image dimension
    if(image->nz>1 || image->nt>1 || image->nu>1 || image->nv>1 || image->nw>1){
        fprintf(stderr,"[NiftyReg Error] reg_writePNGfile: Image with dimension larger than 2 can be saved as png\n");
        exit(1);
    }

    // Check the min and max values of the nifti image
    float minValue = reg_tools_getMinValue(image);
    float maxValue = reg_tools_getMaxValue(image);

    // Rescale the image intensites if  they are outside of the range
    if(minValue<0 || maxValue>255){
        float newMinValue[1]={0};
        float newMaxValue[1]={255};
        float lowThrValue[1]={-std::numeric_limits<float>::max()};
        float higThrValue[1]={std::numeric_limits<float>::max()};
        reg_intensityRescale(image,
                             newMinValue,
                             newMaxValue,
                             lowThrValue,
                             higThrValue);
        printf("[NiftyReg WARNING] reg_writePNGfile: the image intensities have been rescaled from [%g %g] to [0 255].\n",
               minValue, maxValue);
    }

    // The nifti image is converted as unsigned char if required
    if(image->datatype!=NIFTI_TYPE_UINT8)
        reg_tools_changeDatatype<uch>(image);

    // Create pointer the nifti image data
    uch *niiImgPtr = static_cast<uch *>(image->data);

    // Check first if the png file can be writen
    FILE *fp=fopen(filename, "wb");
    if(!fp){
        fprintf(stderr,"[NiftyReg Error] reg_writePNGfile: The png file can not be written: %s\n", filename);
        exit(1);
    }
    // The png file structures are created
    png_structp png_ptr = png_create_write_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr==NULL){
        fprintf(stderr,"[NiftyReg Error] reg_writePNGfile: The png pointer could not be created\n");
        exit(1);
    }
    png_infop info_ptr = png_create_info_struct (png_ptr);
    if(info_ptr==NULL){
        fprintf(stderr,"[NiftyReg Error] reg_writePNGfile: The png structure could not be created\n");
        exit(1);
    }
    // Set the png header information
    png_set_IHDR (png_ptr,
                  info_ptr,
                  image->nx, // width
                  image->ny, // height
                  8, // depth
                  PNG_COLOR_TYPE_GRAY,
                  PNG_INTERLACE_NONE,
                  PNG_COMPRESSION_TYPE_DEFAULT,
                  PNG_FILTER_TYPE_DEFAULT);
    // The rows of the png are intialised
    png_byte **row_pointers = (png_byte **)png_malloc(png_ptr, image->ny*sizeof(png_byte *));
    // The data are copied over from the nifti structure to the png structure
    size_t niiIndex=0;
    for (int y = 0; y < image->ny; ++y) {
        png_byte *row = (png_byte *)png_malloc(png_ptr, sizeof(uch)*image->nx);
        row_pointers[y] = row;
        for (int x = 0; x < image->nx;++x) {
            *row++ = niiImgPtr[niiIndex++];
        }
    }
    // Write the image data to the file
    png_init_io (png_ptr, fp);
    png_set_rows (png_ptr, info_ptr, row_pointers);
    png_write_png (png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    // Free the allocated png arrays
    for(int y=0; y<image->ny;++y)
        png_free(png_ptr, row_pointers[y]);
    png_free(png_ptr, row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    // Finally close the file on the hard-drive
    fclose (fp);
}
/* *************************************************************** */
#endif
