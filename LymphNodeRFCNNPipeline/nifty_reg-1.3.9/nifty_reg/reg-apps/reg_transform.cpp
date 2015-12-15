/*
 *  reg_transform.cpp
 *
 *
 *  Created by Marc Modat on 08/11/2010.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _MM_TRANSFORM_CPP
#define _MM_TRANSFORM_CPP

#include "_reg_ReadWriteImage.h"
#include "_reg_resampling.h"
#include "_reg_globalTransformation.h"
#include "_reg_localTransformation.h"
#include "_reg_tools.h"
#include "_reg_thinPlateSpline.h"

#include "reg_transform.h"

#include <fstream>
#include <vector>
#include <iostream>

#ifdef _USE_NR_DOUBLE
#define PrecisionTYPE double
#else
#define PrecisionTYPE float
#endif

typedef struct{
    char *referenceImageName;
    char *sourceImageName;

    char *inputSourceImageName;
    char *inputFirstCPPName;
    char *inputSecondCPPName;
    char *inputDeformationName;
    char *inputSeconDefName;
    char *inputDisplacementName;
    char *outputSourceImageName;
    char *outputDeformationName;
    char *outputDisplacementName;
    char *cpp2defInputName;
    char *cpp2defOutputName;
    char *outputVelName;
    char *inputAffineName;
    char *inputAffineName2;
    char *outputAffineName;
    char *tpsTextFilename;
    float CPPSpacing;
    float ApproxTPSWeight;
    char *outputCPPName;
}PARAM;
typedef struct{
    bool referenceImageFlag;
    bool cpp2defFlag;
    bool composeTransformation1Flag;
    bool composeTransformation2Flag;
    bool composeTransformation3Flag;
    bool def2dispFlag;
    bool disp2defFlag;
    bool updateSformFlag;
    bool aff2defFlag;
    bool invertAffineFlag;
    bool composeAffineFlag;
    bool invertVelFlag;
    bool invertDefFlag;
    bool tps2cppFlag;
    bool tps2defFlag;
}FLAG;


void PetitUsage(char *exec)
{
    fprintf(stderr,"Usage:\t%s -ref <referenceImageName> [OPTIONS].\n",exec);
    fprintf(stderr,"\tSee the help for more details (-h).\n");
    return;
}
void Usage(char *exec)
{
    printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
    printf("Usage:\t%s -ref <filename> [OPTIONS].\n",exec);
    printf("\t-ref <filename>\tFilename of the reference image (mandatory)\n");

#ifdef _SVN_REV
    fprintf(stderr,"\n-v Print the subversion revision number\n");
#endif
    printf("\n* * OPTIONS * *\n");
    printf("\t-cpp2def <filename1>  <filename2>\n");
    printf("\t\tConversion from control point position to deformation field.\n");
    printf("\t\tFilename1 of input lattice of control point positions (CPP).\n");
    printf("\t\tFilename2 of the output deformation field image (DEF).\n");
    printf("\t-comp1 <filename1>  <filename2> <filename3>\n");
    printf("\t\tComposition of two lattices of control points. CPP2(CPP1(x)).\n");
    printf("\t\tFilename1 of lattice of control point that contains the second deformation (CPP2).\n");
    printf("\t\tFilename2 of lattice of control point that contains the initial deformation (CPP1).\n");
    printf("\t\tFilename3 of the output deformation field.\n");
    printf("\t-comp2 <filename1>  <filename2> <filename3>\n");
    printf("\t\tComposition of a deformation field with a lattice of control points. CPP(DEF(x)).\n");
    printf("\t\tFilename1 of lattice of control point that contains the second deformation (CPP).\n");
    printf("\t\tFilename2 of the deformation field to be used as initial deformation (DEF).\n");
    printf("\t\tFilename3 of the output deformation field.\n");
    printf("\t-comp3 <filename1>  <filename2> <filename3>\n");
    printf("\t\tComposition of two deformation fields. DEF2(DEF1(x)).\n");
    printf("\t\tFilename1 of the second deformation field (DEF2).\n");
    printf("\t\tFilename2 of the first deformation field (DEF1).\n");
    printf("\t\tFilename3 of the output deformation field.\n");
    printf("\t-def2disp <filename1>  <filename2>\n");
    printf("\t\tConvert a deformation field into a displacement field.\n");
    printf("\t\tFilename1: deformation field x'=T(x)\n");
    printf("\t\tFilename2: displacement field x'=x+T(x)\n");
    printf("\t-disp2def <filename1>  <filename2>\n");
    printf("\t\tConvert a displacement field into a deformation field.\n");
    printf("\t\tFilename1: displacement field x'=x+T(x)\n");
    printf("\t\tFilename2: deformation field x'=T(x)\n");
    printf("\t-updSform <filename1> <filename2> <filename3>\n");
    printf("\t\tUpdate the sform of a Floating (Source) image using an affine transformation.\n");
    printf("\t\tFilename1: Image to be updated\n");
    printf("\t\tFilename2: Affine transformation defined as Affine x Reference = Floating\n");
    printf("\t\tFilename3: Updated image.\n");
    printf("\t-aff2def <filename1> <filename2> <filename3> <filename3>\n");
    printf("\t\tCompose a non-rigid with an affine. Non-rigid between A and B\n");
    printf("\t\tAffine from B to C. Returns a deformation field to deform A in the space of C\n");
    printf("\t\tFilename1: Affine transformation defined as Affine x Reference = Floating\n");
    printf("\t\tFilename2: Image used as a target for the non-rigid step (B).\n");
    printf("\t\tFilename3: Control point position or deformation field filename\n");
    printf("\t\tFilename4: Output deformation field.\n");
    printf("\t-invAffine <filename1> <filename2>\n");
    printf("\t\tInvert an affine transformation matrix\n");
    printf("\t\tFilename1: Input affine matrix\n");
    printf("\t\tFilename2: Inverted affine matrix\n");
    //    printf("\t-invDef <filename1> <filename2> <filename3>\n");
    //        printf("\t\tInvert a deformation field\n");
    //        printf("\t\tFilename1: Input deformation field filename\n");
    //        printf("\t\tFilename2: Source image filename\n");
    //        printf("\t\tFilename3: output deformation field filename\n");
    printf("\t-compAff <filename1> <filename2> <filename3>\n");
    printf("\t\tCompose two affine transformation matrices\n");
    printf("\t\tFilename1: First affine matrix\n");
    printf("\t\tFilename2: Second affine matrix\n");
    printf("\t\tFilename3: Composed affine matrix result\n");
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
            printf("%s",xml_transform);
            return 0;
        }
#ifdef _SVN_REV
        if(strcmp(argv[i], "-version")==0 || strcmp(argv[i], "-Version")==0 ||
                strcmp(argv[i], "-V")==0 || strcmp(argv[i], "-v")==0 ||
                strcmp(argv[i], "--v")==0 || strcmp(argv[i], "--version")==0){
            printf("NiftyReg revision number: %i\n",_SVN_REV);
            return 0;
        }
#endif
        else if((strcmp(argv[i],"-ref")==0) || (strcmp(argv[i],"-target")==0)){
            param->referenceImageName=argv[++i];
            flag->referenceImageFlag=1;
        }
        else if(strcmp(argv[i], "-comp1") == 0){
            param->inputSecondCPPName=argv[++i];
            param->inputFirstCPPName=argv[++i];
            param->outputDeformationName=argv[++i];
            flag->composeTransformation1Flag=1;
        }
        else if(strcmp(argv[i], "-comp2") == 0){
            param->inputSecondCPPName=argv[++i];
            param->inputDeformationName=argv[++i];
            param->outputDeformationName=argv[++i];
            flag->composeTransformation2Flag=1;
        }
        else if(strcmp(argv[i], "-comp3") == 0){
            param->inputSeconDefName=argv[++i];
            param->inputDeformationName=argv[++i];
            param->outputDeformationName=argv[++i];
            flag->composeTransformation3Flag=1;
        }
        else if(strcmp(argv[i], "-def2disp") == 0){
            param->inputDeformationName=argv[++i];
            param->outputDisplacementName=argv[++i];
            flag->def2dispFlag=1;
        }
        else if(strcmp(argv[i], "-disp2def") == 0){
            param->inputDisplacementName=argv[++i];
            param->outputDeformationName=argv[++i];
            flag->disp2defFlag=1;
        }
        else if(strcmp(argv[i], "-updSform") == 0){
            param->inputSourceImageName=argv[++i];
            param->inputAffineName=argv[++i];
            param->outputSourceImageName=argv[++i];
            flag->updateSformFlag=1;
        }
        else if(strcmp(argv[i], "-aff2def") == 0){
            param->inputAffineName=argv[++i];
            param->sourceImageName=argv[++i];
            param->inputFirstCPPName=argv[++i];
            param->outputDeformationName=argv[++i];
            flag->aff2defFlag=1;
        }
        else if(strcmp(argv[i], "-invAffine") == 0){
            param->inputAffineName=argv[++i];
            param->outputAffineName=argv[++i];
            flag->invertAffineFlag=1;
        }
        else if(strcmp(argv[i], "-invDef") == 0){
            param->inputDeformationName=argv[++i];
            param->sourceImageName=argv[++i];
            param->outputDeformationName=argv[++i];
            flag->invertDefFlag=1;
        }
        else if(strcmp(argv[i], "-cpp2def") == 0){
            param->cpp2defInputName=argv[++i];
            param->cpp2defOutputName=argv[++i];
            flag->cpp2defFlag=1;
        }
        else if(strcmp(argv[i], "-compAff") == 0){
            param->inputAffineName=argv[++i];
            param->inputAffineName2=argv[++i];
            param->outputAffineName=argv[++i];
            flag->composeAffineFlag=1;
        }
        else if(strcmp(argv[i], "-tps2cpp") == 0){
            param->tpsTextFilename=argv[++i];
            param->CPPSpacing=(float)atof(argv[++i]);
            param->ApproxTPSWeight=(float)atof(argv[++i]);
            param->outputCPPName=argv[++i];
            flag->tps2cppFlag=1;
        }
        else if(strcmp(argv[i], "-tps2def") == 0){
            param->tpsTextFilename=argv[++i];
            param->ApproxTPSWeight=(float)atof(argv[++i]);
            param->outputDeformationName=argv[++i];
            flag->tps2defFlag=1;
        }
        else{
            fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
            PetitUsage(argv[0]);
            return 1;
        }
    }

    if(!flag->referenceImageFlag){
        fprintf(stderr,"Err:\tThe reference image has to be defined.\n");
        PetitUsage(argv[0]);
        return 1;
    }

    /* Read the reference image */
    nifti_image *referenceImage = reg_io_ReadImageHeader(param->referenceImageName);
    if(referenceImage == NULL){
        fprintf(stderr,"[NiftyReg ERROR] Error when reading the reference image: %s\n",param->referenceImageName);
        PetitUsage(argv[0]);
        return 1;
    }
    reg_checkAndCorrectDimension(referenceImage);

    /* ************************************************ */
    /* INITIALISE A CPP DEF DEF FROM LANDMARKS WITH TPS */
    /* ************************************************ */
    if(flag->tps2cppFlag || flag->tps2defFlag){
        // Read the text file that contains the landmark coordinates
        std::vector<PrecisionTYPE> values;
        std::ifstream infile(param->tpsTextFilename);
        if(!infile){
            fprintf(stderr,"[ERROR] Can not open the file %s\n", param->tpsTextFilename);
            exit(1);
        }
        PrecisionTYPE val;
        while(!infile.eof()){
            infile >> val;
            values.push_back(val);
        }
        infile.close();
        size_t landmarkNumber=0;
        if(referenceImage->ndim==2) landmarkNumber=values.size()/4;
        else if(referenceImage->ndim==3) landmarkNumber=values.size()/6;
        else{
            fprintf(stderr,"[ERROR] The reference image is expected to be of dimension 2 or 3\n");
            exit(1);
        }
        printf("[NiftyReg] Landmark number %i\n", (int)landmarkNumber);

        PrecisionTYPE *px=(PrecisionTYPE *)malloc(landmarkNumber*sizeof(PrecisionTYPE));
        PrecisionTYPE *py=(PrecisionTYPE *)malloc(landmarkNumber*sizeof(PrecisionTYPE));
        PrecisionTYPE *cx=(PrecisionTYPE *)malloc(landmarkNumber*sizeof(PrecisionTYPE));
        PrecisionTYPE *cy=(PrecisionTYPE *)malloc(landmarkNumber*sizeof(PrecisionTYPE));
        PrecisionTYPE *pz=NULL;
        PrecisionTYPE *cz=NULL;
        if(referenceImage->ndim==3){
            pz=(PrecisionTYPE *)malloc(landmarkNumber*sizeof(PrecisionTYPE));
            cz=(PrecisionTYPE *)malloc(landmarkNumber*sizeof(PrecisionTYPE));
        }
        size_t i=0, l=0;
        while(l<landmarkNumber){
            px[l]=values[i];++i;
            py[l]=values[i];++i;
            if(referenceImage->ndim==3){pz[l]=values[i];++i;}
            cx[l]=values[i]-px[l];++i;
            cy[l]=values[i]-py[l];++i;
            if(referenceImage->ndim==3){cz[l]=values[i]-pz[l];++i;}
            ++l;
        }

        // Create the TPS object
        reg_tps<PrecisionTYPE> *tps =
                new reg_tps<PrecisionTYPE>(referenceImage->dim[0],landmarkNumber);
        tps->SetAproxInter(param->ApproxTPSWeight);
        // Set the landmark initial and final positions
        if(referenceImage->ndim==3) tps->SetPosition(px,py,pz,cx,cy,cz);
        else if(referenceImage->ndim==2) tps->SetPosition(px,py,cx,cy);
        free(cx);free(cy);free(px);free(py);
        if(referenceImage->ndim==3){
            free(cz);free(pz);
        }
        // Compute the coefficients
        tps->InitialiseTPS();

        nifti_image *outputImage=NULL;
        if(flag->tps2cppFlag){
            // Compute the cpp dimension
            float gridSpacing[3]={param->CPPSpacing,param->CPPSpacing,param->CPPSpacing};
            if(gridSpacing[0]<0) gridSpacing[0] *= -1.0f * referenceImage->dx;
            if(gridSpacing[1]<0) gridSpacing[1] *= -1.0f * referenceImage->dy;
            if(referenceImage->nz>1)
                if(gridSpacing[2]<0)
                    gridSpacing[2] *= -1.0f * referenceImage->dz;

            // Create and allocate the cpp image
            reg_createControlPointGrid<PrecisionTYPE>(&outputImage,
                                          referenceImage,
                                          gridSpacing);

            nifti_set_filenames(outputImage,param->outputCPPName,0,0);
        }
        else{
            // Creation of the deformation field image
            outputImage=nifti_copy_nim_info(referenceImage);
            outputImage->nbyper=sizeof(PrecisionTYPE);
            if(sizeof(PrecisionTYPE)==sizeof(float))
                outputImage->datatype=NIFTI_TYPE_FLOAT32;
            else outputImage->datatype=NIFTI_TYPE_FLOAT64;
            outputImage->ndim=outputImage->dim[0]=5;
            outputImage->nt=outputImage->dim[4]=1;
            if(referenceImage->nz>1)
                outputImage->nu=outputImage->dim[5]=3;
            else outputImage->nu=outputImage->dim[5]=2;
            outputImage->nvox=outputImage->nx*
                    outputImage->ny *
                    outputImage->nz *
                    outputImage->nt *
                    outputImage->nu;
            // allocate the deformation field data array
            outputImage->data=(void *)malloc(outputImage->nvox*outputImage->nbyper);
        }

        tps->FillDeformationField(outputImage);

        reg_io_WriteImageFile(outputImage,param->outputDeformationName);
        nifti_image_free(outputImage);
        delete tps;
    }

    /* *************************************** */
    /* GENERATE THE DEFORMATION FIELD FROM CPP */
    /* *************************************** */
    if(flag->cpp2defFlag){
        // Read the control point image
        nifti_image *controlPointImage = reg_io_ReadImageFile(param->cpp2defInputName);
        if(controlPointImage == NULL){
            fprintf(stderr,"[NiftyReg ERROR] Error when reading the control point position image: %s\n",param->cpp2defInputName);
            PetitUsage(argv[0]);
            return 1;
        }
        reg_checkAndCorrectDimension(controlPointImage);
        // Allocate the deformation field
        nifti_image *deformationFieldImage = nifti_copy_nim_info(referenceImage);
        deformationFieldImage->dim[0]=deformationFieldImage->ndim=5;
        deformationFieldImage->dim[1]=deformationFieldImage->nx=referenceImage->nx;
        deformationFieldImage->dim[2]=deformationFieldImage->ny=referenceImage->ny;
        deformationFieldImage->dim[3]=deformationFieldImage->nz=referenceImage->nz;
        deformationFieldImage->dim[4]=deformationFieldImage->nt=1;deformationFieldImage->pixdim[4]=deformationFieldImage->dt=1.0;
        if(referenceImage->nz>1) deformationFieldImage->dim[5]=deformationFieldImage->nu=3;
        else deformationFieldImage->dim[5]=deformationFieldImage->nu=2;
        deformationFieldImage->pixdim[5]=deformationFieldImage->du=1.0;
        deformationFieldImage->dim[6]=deformationFieldImage->nv=1;deformationFieldImage->pixdim[6]=deformationFieldImage->dv=1.0;
        deformationFieldImage->dim[7]=deformationFieldImage->nw=1;deformationFieldImage->pixdim[7]=deformationFieldImage->dw=1.0;
        deformationFieldImage->nvox=deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz*deformationFieldImage->nt*deformationFieldImage->nu;
        deformationFieldImage->datatype = controlPointImage->datatype;
        deformationFieldImage->nbyper = controlPointImage->nbyper;
        deformationFieldImage->data = (void *)calloc(deformationFieldImage->nvox, deformationFieldImage->nbyper);
        //Computation of the deformation field
        if( controlPointImage->intent_code==NIFTI_INTENT_VECTOR &&
            strcmp(controlPointImage->intent_name,"NREG_VEL_STEP")==0 )
            reg_bspline_getDeformationFieldFromVelocityGrid(controlPointImage,
                                                            deformationFieldImage
                                                            );
        else
            reg_spline_getDeformationField(controlPointImage,
                                           referenceImage,
                                           deformationFieldImage,
                                           NULL,
                                           false, //composition
                                           true // bspline
                                           );
        nifti_image_free(controlPointImage);
        // Ouput the deformation field image
        reg_io_WriteImageFile(deformationFieldImage,param->cpp2defOutputName);
        printf("The deformation field has been saved: %s\n", param->cpp2defOutputName);
        nifti_image_free(deformationFieldImage);
    }

    /* ********************* */
    /* START THE COMPOSITION */
    /* ********************* */
    if(flag->composeTransformation1Flag || flag->composeTransformation2Flag){

        nifti_image *secondControlPointImage = reg_io_ReadImageFile(param->inputSecondCPPName);
        if(secondControlPointImage == NULL){
            fprintf(stderr,"[NiftyReg ERROR] Error when reading the control point image: %s\n",param->inputSecondCPPName);
            PetitUsage(argv[0]);
            return 1;
        }
        reg_checkAndCorrectDimension(secondControlPointImage);

        // Here should be a check for the control point image. Does it suit the target image space.
        //TODO

        // Check if the input deformation can be read
        nifti_image *deformationFieldImage = NULL;

        if(flag->composeTransformation1Flag){
            // Read the initial deformation control point grid
            nifti_image *firstControlPointImage = reg_io_ReadImageFile(param->inputFirstCPPName);
            if(firstControlPointImage == NULL){
                fprintf(stderr,"[NiftyReg ERROR] Error when reading the control point image: %s\n",param->inputFirstCPPName);
                PetitUsage(argv[0]);
                return 1;
            }
            reg_checkAndCorrectDimension(firstControlPointImage);

            // Create the deformation field image
            deformationFieldImage = nifti_copy_nim_info(referenceImage);
            deformationFieldImage->cal_min=0;
            deformationFieldImage->cal_max=0;
            deformationFieldImage->scl_slope = 1.0f;
            deformationFieldImage->scl_inter = 0.0f;
            deformationFieldImage->dim[0]=deformationFieldImage->ndim=5;
            deformationFieldImage->dim[1]=deformationFieldImage->nx=referenceImage->nx;
            deformationFieldImage->dim[2]=deformationFieldImage->ny=referenceImage->ny;
            deformationFieldImage->dim[3]=deformationFieldImage->nz=referenceImage->nz;
            deformationFieldImage->dim[4]=deformationFieldImage->nt=1;deformationFieldImage->pixdim[4]=deformationFieldImage->dt=1.0;
            if(referenceImage->nz>1)
                deformationFieldImage->dim[5]=deformationFieldImage->nu=3;
            else deformationFieldImage->dim[5]=deformationFieldImage->nu=2;
            deformationFieldImage->pixdim[5]=deformationFieldImage->du=1.0;
            deformationFieldImage->dim[6]=deformationFieldImage->nv=1;deformationFieldImage->pixdim[6]=deformationFieldImage->dv=1.0;
            deformationFieldImage->dim[7]=deformationFieldImage->nw=1;deformationFieldImage->pixdim[7]=deformationFieldImage->dw=1.0;
            deformationFieldImage->nvox=deformationFieldImage->nx*deformationFieldImage->ny*deformationFieldImage->nz*deformationFieldImage->nt*deformationFieldImage->nu;
            if(sizeof(PrecisionTYPE)==4) deformationFieldImage->datatype = NIFTI_TYPE_FLOAT32;
            else deformationFieldImage->datatype = NIFTI_TYPE_FLOAT64;
            deformationFieldImage->nbyper = sizeof(PrecisionTYPE);
            deformationFieldImage->data = (void *)calloc(deformationFieldImage->nvox, deformationFieldImage->nbyper);

            //Compute the initial deformation
            reg_spline_getDeformationField(firstControlPointImage,
                                           referenceImage,
                                           deformationFieldImage,
                                           NULL,
                                           false, //composition
                                           true // bspline
                                           );
            nifti_image_free(firstControlPointImage);
        }
        else{
            // Read the deformation field
            deformationFieldImage = reg_io_ReadImageFile(param->inputDeformationName);
            if(deformationFieldImage == NULL){
                fprintf(stderr,"[NiftyReg ERROR] Error when reading the input deformation field image: %s\n",param->inputDeformationName);
                PetitUsage(argv[0]);
                return 1;
            }
            reg_checkAndCorrectDimension(deformationFieldImage);
        }

        // Are the deformation field and the target image defined in the same space
        //TODO

        // The deformation field is updated through composition
        reg_spline_getDeformationField(secondControlPointImage,
                                       deformationFieldImage,
                                       deformationFieldImage,
                                       NULL,
                                       true, //composition
                                       true // bspline
                                       );
        nifti_image_free(secondControlPointImage);

        // Ouput the composed deformation field
        reg_io_WriteImageFile(deformationFieldImage,param->outputDeformationName);
        nifti_image_free(deformationFieldImage);
        printf("Composed deformation field has been saved: %s\n", param->outputDeformationName);
    }

    if(flag->composeTransformation3Flag){
        // Read both deformation field
        nifti_image *def1=reg_io_ReadImageFile(param->inputDeformationName);
        nifti_image *def2=reg_io_ReadImageFile(param->inputSeconDefName);

        reg_defField_compose(def2,def1,NULL);

        reg_io_WriteImageFile(def1,param->outputDeformationName);

        nifti_image_free(def1);
        nifti_image_free(def2);
    }

    /* ******************** */
    /* START THE CONVERSION */
    /* ******************** */
    if(flag->disp2defFlag){
        // Read the input displacement field
        nifti_image *image = reg_io_ReadImageFile(param->inputDisplacementName);
        if(image == NULL){
            fprintf(stderr,"[NiftyReg ERROR] Error when reading the input displacement field image: %s\n",param->inputDisplacementName);
            PetitUsage(argv[0]);
            return 1;
        }
        reg_checkAndCorrectDimension(image);
        // Conversion from displacement to deformation
        reg_getDeformationFromDisplacement(image);

        // Ouput the deformation field
        reg_io_WriteImageFile(image,param->outputDeformationName);
        nifti_image_free(image);
        printf("The deformation field has been saved: %s\n", param->outputDeformationName);
    }
    if(flag->def2dispFlag){
        // Read the input deformation field
        nifti_image *image = reg_io_ReadImageFile(param->inputDeformationName);
        if(image == NULL){
            fprintf(stderr,"[NiftyReg ERROR] Error when reading the input deformation field image: %s\n",param->inputDeformationName);
            PetitUsage(argv[0]);
            return 1;
        }
        reg_checkAndCorrectDimension(image);
        // Conversion from displacement to deformation
        reg_getDisplacementFromDeformation(image);

        // Ouput the deformation field
        reg_io_WriteImageFile(image,param->outputDisplacementName);
        nifti_image_free(image);
        printf("The displacement field has been saved: %s\n", param->outputDisplacementName);
    }

    /* ******************* */
    /* UPDATE IMAGE S_FORM */
    /* ******************* */
    if(flag->updateSformFlag){
        // Read the input image
        nifti_image *image = reg_io_ReadImageFile(param->inputSourceImageName);
        if(image == NULL){
            fprintf(stderr,"[NiftyReg ERROR] Error when reading the input image: %s\n",
                    param->inputSourceImageName);
            PetitUsage(argv[0]);
            return 1;
        }
        reg_checkAndCorrectDimension(image);
        // Read the affine transformation
        mat44 *affineTransformation = (mat44 *)calloc(1,sizeof(mat44));
        reg_tool_ReadAffineFile(affineTransformation,
                                referenceImage,
                                image,
                                param->inputAffineName,
                                0);
        //Invert the affine transformation since the flaoting is updated
        *affineTransformation = nifti_mat44_inverse(*affineTransformation);

        // Update the sform
        if(image->sform_code>0){
            image->sto_xyz = reg_mat44_mul(affineTransformation, &(image->sto_xyz));
        }
        else{
            image->sform_code = 1;
            image->sto_xyz = reg_mat44_mul(affineTransformation, &(image->qto_xyz));
        }
        image->sto_ijk = nifti_mat44_inverse(image->sto_xyz);
        free(affineTransformation);

        // Write the output image
        reg_io_WriteImageFile(image,param->outputSourceImageName);
        nifti_image_free(image);
        printf("The sform has been updated and the image saved: %s\n", param->outputSourceImageName);
    }


    /* **************************** */
    /* INVERT AFFINE TRANSFORMATION */
    /* **************************** */
    if(flag->invertAffineFlag){
        // First read the affine transformation from a file
        mat44 *affineTransformation = (mat44 *)calloc(1,sizeof(mat44));
        reg_tool_ReadAffineFile(affineTransformation,
                                referenceImage,
                                referenceImage,
                                param->inputAffineName,
                                0);

        // Invert the matrix
        *affineTransformation = nifti_mat44_inverse(*affineTransformation);

        // Output the new affine transformation
        reg_tool_WriteAffineFile(affineTransformation,
                                 param->outputAffineName);
        free(affineTransformation);
    }

    //    /* ************************ */
    //    /* INVERT DEFORMATION FIELD */
    //    /* ************************ */
    //    if(flag->invertDefFlag){
    //        // Read the input deformation field
    //        nifti_image *deformationField = nifti_image_read(param->inputDeformationName,true);
    //        if(deformationField == NULL){
    //            fprintf(stderr,"[NiftyReg ERROR] Error when reading the input deformation field image: %s\n",param->inputDeformationName);
    //            PetitUsage(argv[0]);
    //            return 1;
    //        }
    //        reg_checkAndCorrectDimension(deformationField);
    //
    //        nifti_image *sourceImage = nifti_image_read(param->sourceImageName,false);
    //        if(sourceImage == NULL){
    //            fprintf(stderr,"[NiftyReg ERROR] Error when reading the source image: %s\n",param->sourceImageName);
    //            PetitUsage(argv[0]);
    //            return 1;
    //        }
    //        reg_checkAndCorrectDimension(sourceImage);
    //
    //        thinPlateSpline *tps=NULL;
    //        reg_initThinPlateSplineFromDeformationField(deformationField,
    //                                                    tps,
    //                                                    false);
    //
    //        nifti_image *outputDeformationFieldImage = nifti_copy_nim_info(sourceImage);
    //        outputDeformationFieldImage->cal_min=0;
    //        outputDeformationFieldImage->cal_max=0;
    //        outputDeformationFieldImage->scl_slope = 1.0f;
    //        outputDeformationFieldImage->scl_inter = 0.0f;
    //        outputDeformationFieldImage->dim[0]=outputDeformationFieldImage->ndim=5;
    //        outputDeformationFieldImage->dim[1]=outputDeformationFieldImage->nx=sourceImage->nx;
    //        outputDeformationFieldImage->dim[2]=outputDeformationFieldImage->ny=sourceImage->ny;
    //        outputDeformationFieldImage->dim[3]=outputDeformationFieldImage->nz=sourceImage->nz;
    //        outputDeformationFieldImage->dim[4]=outputDeformationFieldImage->nt=1;outputDeformationFieldImage->pixdim[4]=outputDeformationFieldImage->dt=1.0;
    //        if(referenceImage->nz>1)
    //            outputDeformationFieldImage->dim[5]=outputDeformationFieldImage->nu=3;
    //        else outputDeformationFieldImage->dim[5]=outputDeformationFieldImage->nu=2;
    //        outputDeformationFieldImage->pixdim[5]=outputDeformationFieldImage->du=1.0;
    //        outputDeformationFieldImage->dim[6]=outputDeformationFieldImage->nv=1;
    //        outputDeformationFieldImage->pixdim[6]=outputDeformationFieldImage->dv=1.0;
    //        outputDeformationFieldImage->dim[7]=outputDeformationFieldImage->nw=1;
    //        outputDeformationFieldImage->pixdim[7]=outputDeformationFieldImage->dw=1.0;
    //        outputDeformationFieldImage->nvox=outputDeformationFieldImage->nx*outputDeformationFieldImage->ny*
    //                                          outputDeformationFieldImage->nz*outputDeformationFieldImage->nt*
    //                                          outputDeformationFieldImage->nu;
    //        if(sizeof(PrecisionTYPE)==4) outputDeformationFieldImage->datatype = NIFTI_TYPE_FLOAT32;
    //        else outputDeformationFieldImage->datatype = NIFTI_TYPE_FLOAT64;
    //        outputDeformationFieldImage->nbyper = sizeof(PrecisionTYPE);
    //        outputDeformationFieldImage->data = (void *)calloc(outputDeformationFieldImage->nvox, outputDeformationFieldImage->nbyper);
    //
    //        reg_TPSfillDeformationField(tps, outputDeformationFieldImage);
    //
    //        nifti_set_filenames(outputDeformationFieldImage,param->outputDeformationName,0,0);
    //        nifti_image_write(outputDeformationFieldImage);
    //        nifti_image_free(sourceImage);
    //        nifti_image_free(deformationField);
    //        nifti_image_free(outputDeformationFieldImage);
    //    }

    /* ***************************
    APPLY AFFINE TO DEFORMATION
    Considering A to B a NRR, B to C an affine.
    It returns the deformation field to resample A into C
    *************************** */
    if(flag->aff2defFlag){

        nifti_image *middleImage = reg_io_ReadImageHeader(param->sourceImageName);
        if(middleImage == NULL){
            fprintf(stderr,"[NiftyReg ERROR] Error when reading the nrr-step target image: %s\n", param->sourceImageName);
            PetitUsage(argv[0]);
            return 1;
        }
        reg_checkAndCorrectDimension(middleImage);

        // Read the affine transformation
        mat44 *affineTransformation = (mat44 *)calloc(1,sizeof(mat44));
        reg_tool_ReadAffineFile(affineTransformation,
                                referenceImage,
                                middleImage,
                                param->inputAffineName,
                                0);
        reg_mat44_disp(affineTransformation, (char *)"affineTransformation");

        // An initial deformation field is created
        nifti_image *deformationFieldImage = nifti_copy_nim_info(referenceImage);
        deformationFieldImage->cal_min=0;
        deformationFieldImage->cal_max=0;
        deformationFieldImage->scl_slope = 1.0f;
        deformationFieldImage->scl_inter = 0.0f;
        deformationFieldImage->dim[0]=deformationFieldImage->ndim=5;
        deformationFieldImage->dim[1]=deformationFieldImage->nx=referenceImage->nx;
        deformationFieldImage->dim[2]=deformationFieldImage->ny=referenceImage->ny;
        deformationFieldImage->dim[3]=deformationFieldImage->nz=referenceImage->nz;
        deformationFieldImage->dim[4]=deformationFieldImage->nt=1;
        deformationFieldImage->pixdim[4]=deformationFieldImage->dt=1.0;
        if(referenceImage->nz>1)
            deformationFieldImage->dim[5]=deformationFieldImage->nu=3;
        else deformationFieldImage->dim[5]=deformationFieldImage->nu=2;
        deformationFieldImage->pixdim[5]=deformationFieldImage->du=1.0;
        deformationFieldImage->dim[6]=deformationFieldImage->nv=1;
        deformationFieldImage->pixdim[6]=deformationFieldImage->dv=1.0;
        deformationFieldImage->dim[7]=deformationFieldImage->nw=1;
        deformationFieldImage->pixdim[7]=deformationFieldImage->dw=1.0;
        deformationFieldImage->nvox=deformationFieldImage->nx*deformationFieldImage->ny
                *deformationFieldImage->nz*deformationFieldImage->nt
                *deformationFieldImage->nu;
        if(sizeof(PrecisionTYPE)==4) deformationFieldImage->datatype = NIFTI_TYPE_FLOAT32;
        else deformationFieldImage->datatype = NIFTI_TYPE_FLOAT64;
        deformationFieldImage->nbyper = sizeof(PrecisionTYPE);
        deformationFieldImage->data = (void *)calloc(deformationFieldImage->nvox,
                                                     deformationFieldImage->nbyper);
        // The deformation field is filled with the affine transformation
        reg_affine_positionField(affineTransformation, referenceImage, deformationFieldImage);

        // Read the input control point position or deformation field
        nifti_image *nrr_transformation = reg_io_ReadImageFile(param->inputFirstCPPName);
        if(nrr_transformation == NULL){
            fprintf(stderr,"[NiftyReg ERROR] Error when reading the non-rigid transformation image: %s\n",param->inputFirstCPPName);
            PetitUsage(argv[0]);
            return 1;
        }
        reg_checkAndCorrectDimension(nrr_transformation);

        if(nrr_transformation->nx==middleImage->nx){ // Deformation field
            reg_defField_compose(nrr_transformation,
                                 deformationFieldImage,
                                 NULL);
        }
        else{ // control point position
            reg_spline_getDeformationField(nrr_transformation,
                                           middleImage,
                                           deformationFieldImage,
                                           NULL,
                                           true, //composition
                                           true // bspline
                                           );
        }
        nifti_image_free(nrr_transformation);

        reg_io_WriteImageFile(deformationFieldImage,param->outputDeformationName);
        nifti_image_free(deformationFieldImage);
        nifti_image_free(middleImage);
    }
    /* Compose two affine transformations */
    // It is basically a matrix multiplication
    if(flag->composeAffineFlag){
        // Read the first affine
        mat44 affine1;
        reg_tool_ReadAffineFile(&affine1,
                                param->inputAffineName);
        // Read the second affine
        mat44 affine2;
        reg_tool_ReadAffineFile(&affine2,
                                param->inputAffineName2);
        // Compose both affine and save the result
        mat44 affineResult = reg_mat44_mul(&affine1, &affine2);
        reg_tool_WriteAffineFile(&affineResult,
                                 param->outputAffineName);
    }

    nifti_image_free(referenceImage);

    free(flag);
    free(param);

    return 0;
}

#endif
