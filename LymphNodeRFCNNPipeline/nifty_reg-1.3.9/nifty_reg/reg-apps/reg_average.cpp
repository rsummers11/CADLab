/*
 *  reg_average.cpp
 *
 *
 *  Created by Marc Modat on 29/10/2011.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */
#ifndef MM_AVERAGE_CPP
#define MM_AVERAGE_CPP

#include "_reg_ReadWriteImage.h"
#include "_reg_tools.h"
#include "_reg_maths.h"
#include "_reg_globalTransformation.h"

#include "reg_average.h"

#ifdef _USE_NR_DOUBLE
#define PrecisionTYPE double
#else
#define PrecisionTYPE float
#endif

void usage(char *exec)
{
    printf("%s is a command line program to average either images or affine transformations\n", exec);
    printf("usage:\n\t%s <outputFileName> <inputFileName1> <inputFileName2> <inputFileName3> ...\n", exec);
    printf("\t If the input are images, the intensities are averaged\n");
    printf("\t If the input are affine matrices, out=expm(logm(M1)+logm(M2)+logm(M3)+...)\n");
}

int main(int argc, char **argv)
{
    if(strcmp(argv[1], "--xml")==0){
        printf("%s",xml_average);
        return 0;
    }

    if(argc<3){
        usage(argv[0]);
        return EXIT_SUCCESS;
    }

    //Check the name of the first file to verify if they are analyse or nifti image
    std::string n(argv[2]);
    if(     n.find( ".nii.gz") != std::string::npos ||
            n.find( ".nii") != std::string::npos ||
            n.find( ".hdr") != std::string::npos ||
            n.find( ".img") != std::string::npos ||
            n.find( ".img.gz") != std::string::npos)
    {
        // Input arguments are image filename
        // Read the first image to average
        nifti_image *tempImage=reg_io_ReadImageHeader(argv[2]);
        if(tempImage==NULL){
            fprintf(stderr, "The following image can not be read: %s\n", argv[2]);
            return EXIT_FAILURE;
        }
        reg_checkAndCorrectDimension(tempImage);

        // Create the average image
        nifti_image *average_image=nifti_copy_nim_info(tempImage);
        nifti_image_free(tempImage);tempImage=NULL;
        average_image->datatype=NIFTI_TYPE_FLOAT32;
        if(sizeof(PrecisionTYPE)==sizeof(double))
            average_image->datatype=NIFTI_TYPE_FLOAT64;
        average_image->nbyper=sizeof(PrecisionTYPE);
        average_image->data=(void *)malloc(average_image->nvox*average_image->nbyper);
        reg_tools_addSubMulDivValue(average_image,average_image,0.f,2);

        int imageTotalNumber=0;
        for(int i=2;i<argc;++i){
            nifti_image *tempImage=reg_io_ReadImageFile(argv[i]);
            if(tempImage==NULL){
                fprintf(stderr, "[!] The following image can not be read: %s\n", argv[i]);
                return EXIT_FAILURE;
            }
            reg_checkAndCorrectDimension(tempImage);
            if(average_image->nvox!=tempImage->nvox){
                fprintf(stderr, "[!] All images must have the same size. Error when processing: %s\n", argv[i]);
                return EXIT_FAILURE;
            }
            reg_tools_addSubMulDivImages(average_image,tempImage,average_image,0);
            imageTotalNumber++;
            nifti_image_free(tempImage);tempImage=NULL;
        }
        reg_tools_addSubMulDivValue(average_image,average_image,(float)imageTotalNumber,3);
        reg_io_WriteImageFile(average_image,argv[1]);
        nifti_image_free(average_image);
    }
    else{
        // input arguments are assumed to be text file name
        // Create an mat44 array to store all input matrices
        const size_t matrixNumber=argc-2;
        mat44 *inputMatrices=(mat44 *)malloc(matrixNumber * sizeof(mat44));
        // Read all the input matrices
        for(size_t m=0;m<matrixNumber;++m){
            if(FILE *aff=fopen(argv[m+2], "r")){
                fclose(aff);
            }
            else{
                fprintf(stderr,"The specified input affine file (%s) can not be read\n",argv[m+2]);
                exit(1);
            }
            // Read the current matrix file
            std::ifstream affineFile;
            affineFile.open(argv[m+2]);
            if(affineFile.is_open()){
                // Transfer the values into the mat44 array
                int i=0;
                float value1,value2,value3,value4;
                while(!affineFile.eof()){
                    affineFile >> value1 >> value2 >> value3 >> value4;
                    inputMatrices[m].m[i][0] = value1;
                    inputMatrices[m].m[i][1] = value2;
                    inputMatrices[m].m[i][2] = value3;
                    inputMatrices[m].m[i][3] = value4;
                    i++;
                    if(i>3) break;
                }
            }
            affineFile.close();
        }
        // All the input matrices are log-ed
        for(size_t m=0;m<matrixNumber;++m){
            inputMatrices[m] = reg_mat44_logm(&inputMatrices[m]);
        }
        // All the exponentiated matrices are summed up into one matrix
        //temporary double are used to avoid error accumulation
        double tempValue[16]={0,0,0,0,
                              0,0,0,0,
                              0,0,0,0,
                              0,0,0,0};
        for(size_t m=0;m<matrixNumber;++m){
            tempValue[0]+= (double)inputMatrices[m].m[0][0];
            tempValue[1]+= (double)inputMatrices[m].m[0][1];
            tempValue[2]+= (double)inputMatrices[m].m[0][2];
            tempValue[3]+= (double)inputMatrices[m].m[0][3];
            tempValue[4]+= (double)inputMatrices[m].m[1][0];
            tempValue[5]+= (double)inputMatrices[m].m[1][1];
            tempValue[6]+= (double)inputMatrices[m].m[1][2];
            tempValue[7]+= (double)inputMatrices[m].m[1][3];
            tempValue[8]+= (double)inputMatrices[m].m[2][0];
            tempValue[9]+= (double)inputMatrices[m].m[2][1];
            tempValue[10]+=(double)inputMatrices[m].m[2][2];
            tempValue[11]+=(double)inputMatrices[m].m[2][3];
            tempValue[12]+=(double)inputMatrices[m].m[3][0];
            tempValue[13]+=(double)inputMatrices[m].m[3][1];
            tempValue[14]+=(double)inputMatrices[m].m[3][2];
            tempValue[15]+=(double)inputMatrices[m].m[3][3];
        }
        // Average matrix is computed
        tempValue[0] /= (double)matrixNumber;
        tempValue[1] /= (double)matrixNumber;
        tempValue[2] /= (double)matrixNumber;
        tempValue[3] /= (double)matrixNumber;
        tempValue[4] /= (double)matrixNumber;
        tempValue[5] /= (double)matrixNumber;
        tempValue[6] /= (double)matrixNumber;
        tempValue[7] /= (double)matrixNumber;
        tempValue[8] /= (double)matrixNumber;
        tempValue[9] /= (double)matrixNumber;
        tempValue[10]/= (double)matrixNumber;
        tempValue[11]/= (double)matrixNumber;
        tempValue[12]/= (double)matrixNumber;
        tempValue[13]/= (double)matrixNumber;
        tempValue[14]/= (double)matrixNumber;
        tempValue[15]/= (double)matrixNumber;
        // The final matrix is exponentiated
        mat44 outputMatrix;
        outputMatrix.m[0][0]=(float)tempValue[0];
        outputMatrix.m[0][1]=(float)tempValue[1];
        outputMatrix.m[0][2]=(float)tempValue[2];
        outputMatrix.m[0][3]=(float)tempValue[3];
        outputMatrix.m[1][0]=(float)tempValue[4];
        outputMatrix.m[1][1]=(float)tempValue[5];
        outputMatrix.m[1][2]=(float)tempValue[6];
        outputMatrix.m[1][3]=(float)tempValue[7];
        outputMatrix.m[2][0]=(float)tempValue[8];
        outputMatrix.m[2][1]=(float)tempValue[9];
        outputMatrix.m[2][2]=(float)tempValue[10];
        outputMatrix.m[2][3]=(float)tempValue[11];
        outputMatrix.m[3][0]=(float)tempValue[12];
        outputMatrix.m[3][1]=(float)tempValue[13];
        outputMatrix.m[3][2]=(float)tempValue[14];
        outputMatrix.m[3][3]=(float)tempValue[15];
        outputMatrix = reg_mat44_expm(&outputMatrix);
        // Free the array containing the input matrices
        free(inputMatrices);
        // The final matrix is saved
        reg_tool_WriteAffineFile(&outputMatrix,argv[1]);
    }

    return EXIT_SUCCESS;
}

#endif
