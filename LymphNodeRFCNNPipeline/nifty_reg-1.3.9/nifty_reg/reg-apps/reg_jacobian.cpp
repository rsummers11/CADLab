/**
 * @file reg_jacobian.cpp
 * @author Marc Modat
 * @date 15/11/2010
 * @brief Executable use to generate Jacobian matrices and determinant
 * images.
 *
 *  Created by Marc Modat on 15/11/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _MM_JACOBIAN_CPP
#define _MM_JACOBIAN_CPP

#include "_reg_ReadWriteImage.h"
#include "_reg_globalTransformation.h"
#include "_reg_localTransformation.h"
#include "_reg_tools.h"
#include "_reg_resampling.h"
#include "reg_jacobian.h"

#ifdef _USE_NR_DOUBLE
    #define PrecisionTYPE double
#else
    #define PrecisionTYPE float
#endif

typedef struct{
    char *referenceImageName;
    char *inputDEFName;
    char *inputCPPName;
    char *inputAFFName;
    char *jacobianMapName;
    char *jacobianMatrixName;
    char *logJacobianMapName;
}PARAM;
typedef struct{
    bool referenceImageFlag;
    bool inputDEFFlag;
    bool inputCPPFlag;
    bool inputAFFFlag;
    bool jacobianMapFlag;
    bool jacobianMatrixFlag;
    bool logJacobianMapFlag;
}FLAG;


void PetitUsage(char *exec)
{
    fprintf(stderr,"Usage:\t%s -ref <referenceImage> [OPTIONS].\n",exec);
    fprintf(stderr,"\tSee the help for more details (-h).\n");
    return;
}
void Usage(char *exec)
{
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    printf("Usage:\t%s -ref <filename> [OPTIONS].\n",exec);
    printf("\t-target <filename>\tFilename of the reference image (mandatory)\n");
#ifdef _SVN_REV
    fprintf(stderr,"\n-v Print the subversion revision number\n");
#endif

    printf("\n* * INPUT (Only one will be used) * *\n");
    printf("\t-def <filename>\n");
        printf("\t\tFilename of the deformation field (from reg_transform).\n");
    printf("\t-cpp <filename>\n");
        printf("\t\tFilename of the control point position lattice (from reg_f3d).\n");
    printf("\n* * OUTPUT * *\n");
    printf("\t-jac <filename>\n");
        printf("\t\tFilename of the Jacobian determinant map.\n");
    printf("\t-jacM <filename>\n");
        printf("\t\tFilename of the Jacobian matrix map. (9 or 4 values are stored as a 5D nifti).\n");
        printf("\t-jacL <filename>\n");
        printf("\t\tFilename of the Log of the Jacobian determinant map.\n");
    printf("\n* * EXTRA * *\n");
    printf("\t-aff <filename>\n");
        printf("\t\tFilename of the affine matrix that will be used to modulate the Jacobian determinant map.\n");
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    return;
}

int main(int argc, char **argv)
{
    PARAM *param = (PARAM *)calloc(1,sizeof(PARAM));
    FLAG *flag = (FLAG *)calloc(1,sizeof(FLAG));

    /* read the input parameter */
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
            strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
            strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0){
            Usage(argv[0]);
            return 0;
        }
        else if(strcmp(argv[i], "--xml")==0){
            printf("%s",xml_jacobian);
            return 0;
        }
#ifdef _SVN_REV
        if( strcmp(argv[i], "-version")==0 ||
            strcmp(argv[i], "-Version")==0 ||
            strcmp(argv[i], "-V")==0 ||
            strcmp(argv[i], "-v")==0 ||
            strcmp(argv[i], "--v")==0 ||
            strcmp(argv[i], "--version")==0){
            printf("NiftyReg revision number: %i\n",_SVN_REV);
            return 0;
        }
#endif
        else if((strcmp(argv[i],"-ref")==0) || (strcmp(argv[i],"-target")==0) ||
                (strcmp(argv[i],"--ref")==0)){
            param->referenceImageName=argv[++i];
            flag->referenceImageFlag=1;
        }
        else if(strcmp(argv[i], "-def") == 0 ||
                (strcmp(argv[i],"--def")==0)){
            param->inputDEFName=argv[++i];
            flag->inputDEFFlag=1;
        }
        else if(strcmp(argv[i], "-cpp") == 0 ||
                (strcmp(argv[i],"--cpp")==0)){
            param->inputCPPName=argv[++i];
            flag->inputCPPFlag=1;
        }
        else if(strcmp(argv[i], "-jac") == 0 ||
                (strcmp(argv[i],"--jac")==0)){
            param->jacobianMapName=argv[++i];
            flag->jacobianMapFlag=1;
        }
        else if(strcmp(argv[i], "-jacM") == 0 ||
                (strcmp(argv[i],"--jacM")==0)){
            param->jacobianMatrixName=argv[++i];
            flag->jacobianMatrixFlag=1;
        }
        else if(strcmp(argv[i], "-jacL") == 0 ||
                (strcmp(argv[i],"--jacL")==0)){
            param->logJacobianMapName=argv[++i];
            flag->logJacobianMapFlag=1;
        }
        else if(strcmp(argv[i], "-aff") == 0 ||
                (strcmp(argv[i],"--aff")==0)){
            param->inputAFFName=argv[++i];
            flag->inputAFFFlag=1;
        }
         else{
             fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
             PetitUsage(argv[0]);
             return 1;
         }
    }

    /* ************** */
    /* READ REFERENCE */
    /* ************** */
    nifti_image *image = reg_io_ReadImageHeader(param->referenceImageName);
    if(image == NULL){
        fprintf(stderr,"** ERROR Error when reading the target image: %s\n",param->referenceImageName);
        return 1;
    }
    reg_checkAndCorrectDimension(image);

    /* ******************* */
    /* READ TRANSFORMATION */
    /* ******************* */
    nifti_image *controlPointImage=NULL;
    nifti_image *deformationFieldImage=NULL;
    if(flag->inputCPPFlag){
        controlPointImage = reg_io_ReadImageFile(param->inputCPPName);
        if(controlPointImage == NULL){
            fprintf(stderr,"** ERROR Error when reading the control point image: %s\n",param->inputCPPName);
            nifti_image_free(image);
            return 1;
        }
        reg_checkAndCorrectDimension(controlPointImage);
    }
    else if(flag->inputDEFFlag){
        deformationFieldImage = reg_io_ReadImageFile(param->inputDEFName);
        if(deformationFieldImage == NULL){
            fprintf(stderr,"** ERROR Error when reading the deformation field image: %s\n",param->inputDEFName);
            nifti_image_free(image);
            return 1;
        }
        reg_checkAndCorrectDimension(deformationFieldImage);
    }
    else{
        fprintf(stderr, "No transformation has been provided.\n");
        nifti_image_free(image);
        return 1;
    }

    /* ******************** */
    /* COMPUTE JACOBIAN MAP */
    /* ******************** */
    if(flag->jacobianMapFlag || flag->logJacobianMapFlag){
        // Create first the Jacobian map image
        nifti_image *jacobianImage = nifti_copy_nim_info(image);
        jacobianImage->cal_min=0;
        jacobianImage->cal_max=0;
        jacobianImage->scl_slope = 1.0f;
        jacobianImage->scl_inter = 0.0f;
        if(sizeof(PrecisionTYPE)==8)
            jacobianImage->datatype = NIFTI_TYPE_FLOAT64;
        else jacobianImage->datatype = NIFTI_TYPE_FLOAT32;
        jacobianImage->nbyper = sizeof(PrecisionTYPE);
        jacobianImage->data = (void *)calloc(jacobianImage->nvox, jacobianImage->nbyper);

        // Compute the determinant
        if(flag->inputCPPFlag){
            if( controlPointImage->intent_code==NIFTI_INTENT_VECTOR &&
                strcmp(controlPointImage->intent_name,"NREG_VEL_STEP")==0){
                reg_bspline_GetJacobianDetFromVelocityField(jacobianImage,
                                                            controlPointImage
                                                            );
            }
            else{
                reg_bspline_GetJacobianMap(controlPointImage,
                                           jacobianImage
                                           );
            }
        }
        else if(flag->inputDEFFlag){
            reg_defField_getJacobianMap(deformationFieldImage,
                                        jacobianImage);
        }
        else{
            fprintf(stderr, "No transformation has been provided.\n");
            nifti_image_free(image);
            return 1;
        }

        // Modulate the Jacobian map
        if(flag->inputAFFFlag){
            mat44 affineMatrix;
            mat33 affineMatrix2;
            reg_tool_ReadAffineFile(&affineMatrix, param->inputAFFName);
            affineMatrix2.m[0][0]=affineMatrix.m[0][0];
            affineMatrix2.m[0][1]=affineMatrix.m[0][1];
            affineMatrix2.m[0][2]=affineMatrix.m[0][2];
            affineMatrix2.m[1][0]=affineMatrix.m[1][0];
            affineMatrix2.m[1][1]=affineMatrix.m[1][1];
            affineMatrix2.m[1][2]=affineMatrix.m[1][2];
            affineMatrix2.m[2][0]=affineMatrix.m[2][0];
            affineMatrix2.m[2][1]=affineMatrix.m[2][1];
            affineMatrix2.m[2][2]=affineMatrix.m[2][2];
            float affineDet = nifti_mat33_determ(affineMatrix2);
            reg_tools_addSubMulDivValue(jacobianImage,jacobianImage,affineDet,2);
        }

        // Export the Jacobian determinant map
        if(flag->jacobianMapFlag){
            memset(jacobianImage->descrip, 0, 80);
            strcpy (jacobianImage->descrip,"Jacobian determinant map created using NiftyReg");
            reg_io_WriteImageFile(jacobianImage,param->jacobianMapName);
            printf("Jacobian map image has been saved: %s\n", param->jacobianMapName);
        }
        else if(flag->logJacobianMapFlag){
            PrecisionTYPE *jacPtr=static_cast<PrecisionTYPE *>(jacobianImage->data);
            for(unsigned int i=0;i<jacobianImage->nvox;i++){
                *jacPtr = log(*jacPtr);
                jacPtr++;
            }
            memset(jacobianImage->descrip, 0, 80);
            strcpy (jacobianImage->descrip,"Log Jacobian determinant map created using NiftyReg");
            reg_io_WriteImageFile(jacobianImage,param->logJacobianMapName);
            printf("Log Jacobian map image has been saved: %s\n", param->logJacobianMapName);
        }
        nifti_image_free(jacobianImage);
    }

    /* *********************** */
    /* COMPUTE JACOBIAN MATRIX */
    /* *********************** */
    if(flag->jacobianMatrixFlag){
        // Create first the Jacobian matrix image
        nifti_image *jacobianImage = nifti_copy_nim_info(image);
        jacobianImage->cal_min=0;
        jacobianImage->cal_max=0;
        jacobianImage->scl_slope = 1.0f;
        jacobianImage->scl_inter = 0.0f;
        jacobianImage->dim[0] = jacobianImage->ndim = 5;
        jacobianImage->dim[4] = jacobianImage->nt = 1;
        if(image->nz>1)
            jacobianImage->dim[5] = jacobianImage->nu = 9;
        else jacobianImage->dim[5] = jacobianImage->nu = 4;
        if(sizeof(PrecisionTYPE)==8)
            jacobianImage->datatype = NIFTI_TYPE_FLOAT64;
        else jacobianImage->datatype = NIFTI_TYPE_FLOAT32;
        jacobianImage->nbyper = sizeof(PrecisionTYPE);
        jacobianImage->nvox = jacobianImage->nx * jacobianImage->ny * jacobianImage->nz *
                jacobianImage->nt * jacobianImage->nu;
        jacobianImage->data = (void *)calloc(jacobianImage->nvox, jacobianImage->nbyper);

        size_t voxelNumber=image->nx*image->ny*image->nz;
        mat33* jacobianMatricesArray=(mat33 *)malloc(voxelNumber*sizeof(mat33));

        // Compute the matrices
        if(flag->inputCPPFlag){
            if( controlPointImage->intent_code==NIFTI_INTENT_VECTOR &&
                strcmp(controlPointImage->intent_name,"NREG_VEL_STEP")==0){
                reg_bspline_GetJacobianMatricesFromVelocityField(image,
                                                                 controlPointImage,
                                                                 jacobianMatricesArray
                                                                 );
            }
            else{
                reg_bspline_GetJacobianMatrix(image,
                                              controlPointImage,
                                              jacobianMatricesArray
                                              );
            }
        }
        else if(flag->inputDEFFlag){
            reg_defField_getJacobianMatrix(deformationFieldImage,
                                           jacobianMatricesArray);
        }
        else{
            fprintf(stderr, "No transformation has been provided.\n");
            nifti_image_free(image);
            return 1;
        }

        // Export the Jacobian matrix image
        PrecisionTYPE *jacMatXXPtr=static_cast<PrecisionTYPE *>(jacobianImage->data);
        if(image->nz>1){
            PrecisionTYPE *jacMatXYPtr=&jacMatXXPtr[voxelNumber];
            PrecisionTYPE *jacMatXZPtr=&jacMatXYPtr[voxelNumber];
            PrecisionTYPE *jacMatYXPtr=&jacMatXZPtr[voxelNumber];
            PrecisionTYPE *jacMatYYPtr=&jacMatYXPtr[voxelNumber];
            PrecisionTYPE *jacMatYZPtr=&jacMatYYPtr[voxelNumber];
            PrecisionTYPE *jacMatZXPtr=&jacMatYZPtr[voxelNumber];
            PrecisionTYPE *jacMatZYPtr=&jacMatZXPtr[voxelNumber];
            PrecisionTYPE *jacMatZZPtr=&jacMatZYPtr[voxelNumber];
            for(size_t voxel=0;voxel<voxelNumber;++voxel){
                mat33 jacobianMatrix=jacobianMatricesArray[voxel];
                jacMatXXPtr[voxel]=jacobianMatrix.m[0][0];
                jacMatXYPtr[voxel]=jacobianMatrix.m[0][1];
                jacMatXZPtr[voxel]=jacobianMatrix.m[0][2];
                jacMatYXPtr[voxel]=jacobianMatrix.m[1][0];
                jacMatYYPtr[voxel]=jacobianMatrix.m[1][1];
                jacMatYZPtr[voxel]=jacobianMatrix.m[1][2];
                jacMatZXPtr[voxel]=jacobianMatrix.m[2][0];
                jacMatZYPtr[voxel]=jacobianMatrix.m[2][1];
                jacMatZZPtr[voxel]=jacobianMatrix.m[2][2];
            }
        }
        else{
            PrecisionTYPE *jacMatXYPtr=&jacMatXXPtr[voxelNumber];
            PrecisionTYPE *jacMatYXPtr=&jacMatXYPtr[voxelNumber];
            PrecisionTYPE *jacMatYYPtr=&jacMatYXPtr[voxelNumber];
            for(size_t voxel=0;voxel<voxelNumber;++voxel){
                mat33 jacobianMatrix=jacobianMatricesArray[voxel];
                jacMatXXPtr[voxel]=jacobianMatrix.m[0][0];
                jacMatXYPtr[voxel]=jacobianMatrix.m[0][1];
                jacMatYXPtr[voxel]=jacobianMatrix.m[1][0];
                jacMatYYPtr[voxel]=jacobianMatrix.m[1][1];
            }

        }
        free(jacobianMatricesArray);

        strcpy (jacobianImage->descrip,"Jacobian matrices image created using NiftyReg");
        reg_io_WriteImageFile(jacobianImage,param->jacobianMatrixName);
        printf("Jacobian matrices image has been saved: %s\n", param->jacobianMatrixName);
        nifti_image_free(jacobianImage);
    }

    nifti_image_free(controlPointImage);
    nifti_image_free(deformationFieldImage);
    nifti_image_free(image);

    return EXIT_SUCCESS;
}

#endif
