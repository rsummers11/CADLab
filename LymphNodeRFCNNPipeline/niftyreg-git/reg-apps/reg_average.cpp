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
#include "_reg_resampling.h"
#include "_reg_globalTransformation.h"
#include "_reg_localTransformation.h"

#include "reg_average.h"

#ifdef _USE_NR_DOUBLE
#define PrecisionTYPE double
#else
#define PrecisionTYPE float
#endif

void usage(char *exec)
{
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   printf("usage:\n\t%s <outputFileName> [OPTIONS]\n\n", exec);
   printf("\t-avg <inputAffineName1> <inputAffineName2> ... <inputAffineNameN> \n");
   printf("\t\tIf the input are images, the intensities are averaged\n");
   printf("\t\tIf the input are affine matrices, out=expm((logm(M1)+logm(M2)+...+logm(MN))/N)\n\n");
   printf("\t-avg_lts <AffineMat1> <AffineMat2> ... <AffineMatN> \n");
   printf("\t\tIt will estimate the robust average affine matrix by considering half of the matrices as ouliers.\n\n");
   printf("\t-avg_tran <referenceImage> <transformationFileName1> <floatingImage1> ... <transformationFileNameN> <floatingImageN> \n");
   printf("\t\tAll input images are resampled into the space of <reference image> and averaged\n");
   printf("\t\tA cubic spline interpolation scheme is used for resampling\n\n");
   printf("\t-demean1 <referenceImage> <AffineMat1> <floatingImage1> ...  <AffineMatN> <floatingImageN>\n");
   printf("\t\tThe demean1 option enforces the mean of all affine matrices to have\n");
   printf("\t\ta Jacobian determinant equal to one. This is done by computing the\n");
   printf("\t\taverage transformation by considering only the scaling and shearing\n");
   printf("\t\targuments.The inverse of this computed average matrix is then removed\n");
   printf("\t\tto all input affine matrix beforeresampling all floating images to the\n");
   printf("\t\tuser-defined reference space\n\n");
   printf("\t-demean2 <referenceImage> <NonRigidTrans1> <floatingImage1> ... <NonRigidTransN> <floatingImageN>\n");
   printf("\t-demean3 <referenceImage> <AffineMat1> <NonRigidTrans1> <floatingImage1> ...  <AffineMatN> <NonRigidTransN> <floatingImageN>\n\n");
#ifdef _GIT_HASH
   printf("\n\t--version\t\tPrint current source code git hash key and exit\n\t\t\t\t(%s)\n",_GIT_HASH);
#endif
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
}

template <class DTYPE>
void average_norm_intensity(nifti_image *image)
{
   DTYPE *rankedIntensities = (DTYPE *)malloc(image->nvox*sizeof(DTYPE));
   memcpy(rankedIntensities,image->data,image->nvox*sizeof(DTYPE));
   reg_heapSort(rankedIntensities,static_cast<int>(image->nvox));
   DTYPE lowerValue=rankedIntensities[static_cast<unsigned int>(static_cast<float>(image->nvox)*0.03f)];
   DTYPE higherValue=rankedIntensities[static_cast<unsigned int>(static_cast<float>(image->nvox)*0.97f)];
   reg_tools_substractValueToImage(image,image,lowerValue);
   reg_tools_multiplyValueToImage(image,image,255.f/(higherValue-lowerValue));
   free(rankedIntensities);
   return;
}

int main(int argc, char **argv)
{
   // Check that the number of argument is sufficient
   if(argc<2)
   {
      usage(argv[0]);
      return EXIT_FAILURE;
   }
   // Check if the --xml information is required
   if(strcmp(argv[1], "--xml")==0)
   {
      printf("%s",xml_average);
      return 0;
   }
   // Check if help is required
   for(int i=1; i<argc; ++i)
   {
      if(strcmp(argv[i],"-h")==0 ||
            strcmp(argv[i],"-H")==0 ||
            strcmp(argv[i],"-help")==0 ||
            strcmp(argv[i],"-HELP")==0 ||
            strcmp(argv[i],"-Help")==0
        )
      {
         usage(argv[0]);
         return EXIT_SUCCESS;
      }
      // Check if the --xml information is required
      else if(strcmp(argv[i], "--xml")==0)
      {
         printf("%s",xml_average);
         return 0;
      }
#ifdef _GIT_HASH
      else if(strcmp(argv[i], "-version")==0 || strcmp(argv[i], "-Version")==0 ||
            strcmp(argv[i], "-V")==0 || strcmp(argv[i], "-v")==0 ||
            strcmp(argv[i], "--v")==0 || strcmp(argv[i], "--version")==0)
      {
         printf("%s\n",_GIT_HASH);
         return EXIT_SUCCESS;
      }
#endif
   }
   // Command line
   printf("\nCommand line:\n\t");
   for(int i=0; i<argc; ++i)
      printf("%s ",argv[i]);
   printf("\n\n");

   // Set the name of the file to output
   char *outputName = argv[1];

   // Check what operation is required
   int operation;
   if(strcmp(argv[2],"-avg")==0)
      operation=0;
   else if(strcmp(argv[2],"-avg_lts")==0 || strcmp(argv[2],"-lts_avg")==0)
      operation=1;
   else if(strcmp(argv[2],"-avg_tran")==0)
      operation=2;
   else if(strcmp(argv[2],"-demean1")==0)
      operation=3;
   else if(strcmp(argv[2],"-demean2")==0)
      operation=4;
   else if(strcmp(argv[2],"-demean3")==0)
      operation=5;
   else
   {
      reg_print_msg_error("unknow operation. Options are \"-avg\", \"-avg_lts\", \"-avg_tran\", ");
      reg_print_msg_error("\"-demean1\", \"-demean2\" or \"-demean3\". Specified argument:");
      reg_print_msg_error(argv[2]);
      usage(argv[0]);
      return EXIT_FAILURE;
   }

   // Create the average image or average matrix
   if(operation==0)
   {
      //Check the name of the first file to verify if they are analyse or nifti image
      std::string n(argv[3]);
      if(     n.find( ".nii.gz") != std::string::npos ||
              n.find( ".nii") != std::string::npos ||
              n.find( ".hdr") != std::string::npos ||
              n.find( ".img") != std::string::npos ||
              n.find( ".img.gz") != std::string::npos)
      {
         // Input arguments are image filename
         // Read the first image to average
         nifti_image *tempImage=reg_io_ReadImageHeader(argv[3]);
         if(tempImage==NULL)
         {
            reg_print_msg_error("The following image can not be read:\n");
            reg_print_msg_error(argv[3]);
            return EXIT_FAILURE;
         }
         reg_checkAndCorrectDimension(tempImage);

         // Create the average image
         nifti_image *averageImage=nifti_copy_nim_info(tempImage);
         averageImage->scl_slope=1.f;
         averageImage->scl_inter=0.f;
         nifti_image_free(tempImage);
         tempImage=NULL;
         averageImage->datatype=NIFTI_TYPE_FLOAT32;
         if(sizeof(PrecisionTYPE)==sizeof(double))
            averageImage->datatype=NIFTI_TYPE_FLOAT64;
         averageImage->nbyper=sizeof(PrecisionTYPE);
         averageImage->data=(void *)calloc(averageImage->nvox,averageImage->nbyper);

         int imageTotalNumber=0;
         for(int i=3; i<argc; ++i)
         {
            nifti_image *tempImage=reg_io_ReadImageFile(argv[i]);
            if(tempImage==NULL)
            {
               reg_print_msg_error("The following image can not be read:\n");
               reg_print_msg_error(argv[i]);
               return EXIT_FAILURE;
            }
            reg_checkAndCorrectDimension(tempImage);
            if(sizeof(PrecisionTYPE)==sizeof(double))
               reg_tools_changeDatatype<double>(tempImage);
            else reg_tools_changeDatatype<float>(tempImage);
            if(averageImage->nvox!=tempImage->nvox)
            {
               reg_print_msg_error(" All images must have the same size. Error when processing:\n");
               reg_print_msg_error(argv[i]);
               return EXIT_FAILURE;
            }
//            if(sizeof(PrecisionTYPE)==sizeof(double))
//               average_norm_intensity<double>(tempImage);
//            else average_norm_intensity<float>(tempImage);
            reg_tools_addImageToImage(averageImage,tempImage,averageImage);
            imageTotalNumber++;
            nifti_image_free(tempImage);
            tempImage=NULL;
         }
         reg_tools_divideValueToImage(averageImage,averageImage,(float)imageTotalNumber);

         reg_io_WriteImageFile(averageImage,outputName);
         nifti_image_free(averageImage);
      }
      else
      {
         // input arguments are assumed to be text file name
         // Create an mat44 array to store all input matrices
         const size_t matrixNumber=argc-3;
         mat44 *inputMatrices=(mat44 *)malloc(matrixNumber * sizeof(mat44));
         // Read all the input matrices
         for(size_t m=0; m<matrixNumber; ++m)
         {
            if(FILE *aff=fopen(argv[m+3], "r"))
            {
               fclose(aff);
            }
            else
            {
               reg_print_msg_error("The specified input affine file can not be read\n");
               reg_print_msg_error(argv[m+3]);
               reg_exit(1);
            }
            // Read the current matrix file
            std::ifstream affineFile;
            affineFile.open(argv[m+3]);
            if(affineFile.is_open())
            {
               // Transfer the values into the mat44 array
               int i=0;
               float value1,value2,value3,value4;
               while(!affineFile.eof())
               {
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
         for(size_t m=0; m<matrixNumber; ++m)
         {
            inputMatrices[m] = reg_mat44_logm(&inputMatrices[m]);
         }
         // All the exponentiated matrices are summed up into one matrix
         //temporary double are used to avoid error accumulation
         double tempValue[16]= {0,0,0,0,
                                0,0,0,0,
                                0,0,0,0,
                                0,0,0,0
                               };
         for(size_t m=0; m<matrixNumber; ++m)
         {
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
         reg_tool_WriteAffineFile(&outputMatrix,outputName);
      }
   }
   // Create the LTS average matrix
   else if(operation==1)
   {
       //Check the name of the first file to verify if they are analyse or nifti image, as it only works for affines
       std::string n(argv[3]);
       if(     n.find( ".nii.gz") != std::string::npos ||
               n.find( ".nii") != std::string::npos ||
               n.find( ".hdr") != std::string::npos ||
               n.find( ".img") != std::string::npos ||
               n.find( ".img.gz") != std::string::npos)
       {
           reg_print_msg_error("The LTS average method only works with affine transformations.\n");
           return EXIT_FAILURE;
       }
       else
       {
           // input arguments are assumed to be text file name
           // Create an mat44 array to store all input matrices
           const size_t matrixNumber=argc-3;
           mat44 *inputMatrices=(mat44 *)malloc(matrixNumber * sizeof(mat44));
           // Read all the input matrices
           for(size_t m=0; m<matrixNumber; ++m)
           {
               if(FILE *aff=fopen(argv[m+3], "r"))
               {
                   fclose(aff);
               }
               else
               {
                   reg_print_msg_error("The specified input affine file can not be read\n");
                   reg_print_msg_error(argv[m+3]);
                   reg_exit(1);
               }
               // Read the current matrix file
               std::ifstream affineFile;
               affineFile.open(argv[m+3]);
               if(affineFile.is_open())
               {
                   // Transfer the values into the mat44 array
                   int i=0;
                   float value1,value2,value3,value4;
                   while(!affineFile.eof())
                   {
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
           for(size_t m=0; m<matrixNumber; ++m)
           {
               inputMatrices[m] = reg_mat44_logm(&inputMatrices[m]);
           }

           // All the exponentiated matrices are summed up into one matrix
           // temporary double are used to avoid error accumulation

           double percent=0.5f;
           double * weight=(double *)malloc(matrixNumber * sizeof(double));
           double * weight2=(double *)malloc(matrixNumber * sizeof(double));
           for(size_t m=0; m<matrixNumber; ++m)
           {
               weight[m]=1;
               weight2[m]=1;
           }

           mat44 outputMatrix;
           for(int iter=0; iter<10; iter++)
           {
               double tempValue[16]= {0,0,0,0,
                                      0,0,0,0,
                                      0,0,0,0,
                                      0,0,0,0
                                     };

               // All the exponentiated matrices are summed up into one matrix
               // in order to create the average matrix.
               // temporary double are used to avoid error accumulation
               double sumdistance=0;
               for(size_t m=0; m<matrixNumber; ++m)
               {
                   tempValue[ 0]+=weight[m]*(double)inputMatrices[m].m[0][0];
                   tempValue[ 1]+=weight[m]*(double)inputMatrices[m].m[0][1];
                   tempValue[ 2]+=weight[m]*(double)inputMatrices[m].m[0][2];
                   tempValue[ 3]+=weight[m]*(double)inputMatrices[m].m[0][3];
                   tempValue[ 4]+=weight[m]*(double)inputMatrices[m].m[1][0];
                   tempValue[ 5]+=weight[m]*(double)inputMatrices[m].m[1][1];
                   tempValue[ 6]+=weight[m]*(double)inputMatrices[m].m[1][2];
                   tempValue[ 7]+=weight[m]*(double)inputMatrices[m].m[1][3];
                   tempValue[ 8]+=weight[m]*(double)inputMatrices[m].m[2][0];
                   tempValue[ 9]+=weight[m]*(double)inputMatrices[m].m[2][1];
                   tempValue[10]+=weight[m]*(double)inputMatrices[m].m[2][2];
                   tempValue[11]+=weight[m]*(double)inputMatrices[m].m[2][3];
                   tempValue[12]+=weight[m]*(double)inputMatrices[m].m[3][0];
                   tempValue[13]+=weight[m]*(double)inputMatrices[m].m[3][1];
                   tempValue[14]+=weight[m]*(double)inputMatrices[m].m[3][2];
                   tempValue[15]+=weight[m]*(double)inputMatrices[m].m[3][3];
                   sumdistance+=weight[m];
               }
               // Average matrix is computed
               tempValue[ 0] /= (double)sumdistance;
               tempValue[ 1] /= (double)sumdistance;
               tempValue[ 2] /= (double)sumdistance;
               tempValue[ 3] /= (double)sumdistance;
               tempValue[ 4] /= (double)sumdistance;
               tempValue[ 5] /= (double)sumdistance;
               tempValue[ 6] /= (double)sumdistance;
               tempValue[ 7] /= (double)sumdistance;
               tempValue[ 8] /= (double)sumdistance;
               tempValue[ 9] /= (double)sumdistance;
               tempValue[10] /= (double)sumdistance;
               tempValue[11] /= (double)sumdistance;
               tempValue[12] /= (double)sumdistance;
               tempValue[13] /= (double)sumdistance;
               tempValue[14] /= (double)sumdistance;
               tempValue[15] /= (double)sumdistance;

               // The final matrix is exponentiated
               outputMatrix.m[0][0]=(float)tempValue[ 0];
               outputMatrix.m[0][1]=(float)tempValue[ 1];
               outputMatrix.m[0][2]=(float)tempValue[ 2];
               outputMatrix.m[0][3]=(float)tempValue[ 3];
               outputMatrix.m[1][0]=(float)tempValue[ 4];
               outputMatrix.m[1][1]=(float)tempValue[ 5];
               outputMatrix.m[1][2]=(float)tempValue[ 6];
               outputMatrix.m[1][3]=(float)tempValue[ 7];
               outputMatrix.m[2][0]=(float)tempValue[ 8];
               outputMatrix.m[2][1]=(float)tempValue[ 9];
               outputMatrix.m[2][2]=(float)tempValue[10];
               outputMatrix.m[2][3]=(float)tempValue[11];
               outputMatrix.m[3][0]=(float)tempValue[12];
               outputMatrix.m[3][1]=(float)tempValue[13];
               outputMatrix.m[3][2]=(float)tempValue[14];
               outputMatrix.m[3][3]=(float)tempValue[15];

               // The weights are updated based on the
               for(size_t m=0; m<matrixNumber; ++m)
               {
                   weight[m]=1;

                   mat44 Minus=reg_mat44_minus(&(inputMatrices[m]),&outputMatrix);

                   mat44 Minus_transpose;
                   for(int i=0; i<4; ++i)
                   {
                       for(int j=0; j<4; ++j)
                       {
                           Minus_transpose.m[i][j]=Minus.m[j][i];
                       }
                   }
                   mat44 MTM=reg_mat44_mul(&Minus_transpose,&Minus);
                   double trace=0;
                   for(size_t i=0; i<4; ++i)
                   {
                       trace+=MTM.m[i][i];
                   }
                   weight[m]=1/(sqrt(trace));
                   weight2[m]=1/(sqrt(trace));
               }

               reg_heapSort(weight2,matrixNumber);
               for(size_t m=0; m<matrixNumber; ++m)
               {
                   weight[m]=weight[m]>weight2[(int)ceil(matrixNumber*percent)];
               }
               outputMatrix = reg_mat44_expm(&outputMatrix);
           }

           // Free the array containing the input matrices
           free(inputMatrices);
           // The final matrix is saved
           reg_tool_WriteAffineFile(&outputMatrix,outputName);
       }
   }
   else
   {
      /* **** the average image is created after resampling **** */
      // read the reference image
      nifti_image *referenceImage=reg_io_ReadImageFile(argv[3]);
      if(referenceImage==NULL)
      {
         reg_print_msg_error("The reference image cannot be read. Filename:");
         reg_print_msg_error(argv[3]);
         return EXIT_FAILURE;
      }
#ifndef NDEBUG
      reg_print_msg_debug("reg_average: User-specified reference image:");
      reg_print_msg_debug(referenceImage->fname);
#endif
      // Create the average image without demeaning
      if(operation==2)
      {
          // Throw an error
          if ( (argc-4) % 2 != 0)
          {
              reg_print_msg_error("An odd number of transformation/image pairs was provided.");
              return EXIT_FAILURE;
          }
         // Create an average image
         nifti_image *averageImage = nifti_copy_nim_info(referenceImage);
         averageImage->nbyper=sizeof(float);
         averageImage->datatype=NIFTI_TYPE_FLOAT32;
         averageImage->scl_slope=1.f;
         averageImage->scl_inter=0.f;
         averageImage->data=(void *)calloc(averageImage->nvox,
                                           averageImage->nbyper);

         for(int i=4;i<argc;i+=2){
            mat44 *inputTransformationMatrix=NULL;
            nifti_image *inputTransformationImage=NULL;
            // First check if the input filename is an image
            if(reg_isAnImageFileName(argv[i]))
            {
               inputTransformationImage=reg_io_ReadImageFile(argv[i]);
               if(inputTransformationImage==NULL)
               {
                  fprintf(stderr, "[NiftyReg ERROR] Error when reading the provided transformation: %s\n",
                          argv[i]);
                  return 1;
               }
               reg_checkAndCorrectDimension(inputTransformationImage);
            }
            else
            {
               // Read the affine transformation
               inputTransformationMatrix=(mat44 *)malloc(sizeof(mat44));
               reg_tool_ReadAffineFile(inputTransformationMatrix,argv[i]);
            }
            // Generate a deformation field if required
            bool requireDeformationField=false;
            if(inputTransformationMatrix!=NULL)
               requireDeformationField=true;
            else if(inputTransformationImage!=NULL)
               if(inputTransformationImage->intent_p1!=DEF_FIELD &&
                  inputTransformationImage->intent_p1!=DISP_FIELD)
                  requireDeformationField=true;
            nifti_image *deformationField=NULL;
            if(requireDeformationField){
               deformationField=nifti_copy_nim_info(referenceImage);
               deformationField->ndim=deformationField->dim[0]=5;
               deformationField->nt=deformationField->dim[4]=1;
               deformationField->nu=deformationField->dim[5]=deformationField->nz>1?3:2;
               deformationField->nvox=(size_t)deformationField->nx *
                                               deformationField->ny * deformationField->nz *
                                               deformationField->nt * deformationField->nu;
               deformationField->nbyper=sizeof(float);
               deformationField->datatype=NIFTI_TYPE_FLOAT32;
               deformationField->intent_code=NIFTI_INTENT_VECTOR;
               memset(deformationField->intent_name, 0, 16);
               strcpy(deformationField->intent_name,"NREG_TRANS");
               deformationField->scl_slope=1.f;
               deformationField->scl_inter=0.f;
               deformationField->intent_p1=DEF_FIELD;
               deformationField->data=(void *)malloc(deformationField->nvox*deformationField->nbyper);
               if(inputTransformationMatrix!=NULL){
                  reg_affine_getDeformationField(inputTransformationMatrix,
                                                 deformationField);
               }
               else switch(static_cast<int>(inputTransformationImage->intent_p1)){
               case SPLINE_GRID:
                  reg_spline_getDeformationField(inputTransformationImage,
                                                 deformationField,
                                                 NULL,
                                                 false,
                                                 true);
                  break;
               case SPLINE_VEL_GRID:
                  reg_spline_getDefFieldFromVelocityGrid(inputTransformationImage,
                                                         deformationField,
                                                         false);
                  break;
               case DISP_VEL_FIELD:
                  reg_getDeformationFromDisplacement(inputTransformationImage);
               case DEF_VEL_FIELD:
                  reg_defField_getDeformationFieldFromFlowField(inputTransformationImage,
                                                                deformationField,
                                                                false);
                  break;
               }
               if(inputTransformationMatrix!=NULL)
                  free(inputTransformationMatrix);
               if(inputTransformationImage!=NULL)
                  nifti_image_free(inputTransformationImage);
            }
            else{
               // Check the deformation field dimension
               if(deformationField->nx!=referenceImage->nx ||
                     deformationField->ny!=referenceImage->ny ||
                     deformationField->nz!=referenceImage->nz ||
                     deformationField->nu!=(referenceImage->nz>1?3:2)){
                  reg_print_msg_error("The provided def or disp field dimension");
                  reg_print_msg_error("do not match the reference image dimension");
                  char name[255];
                  sprintf(name,"Field: %s", argv[i]);
                  reg_print_msg_error(name);
                  reg_exit(1);
               }
               deformationField=inputTransformationImage;
               if(deformationField->intent_p1==DISP_FIELD)
                  reg_getDeformationFromDisplacement(deformationField);
            }

            // Read the floating image
            nifti_image *floatingImage = reg_io_ReadImageFile(argv[i+1]);
            reg_checkAndCorrectDimension(floatingImage);
            reg_tools_changeDatatype<float>(floatingImage);

            // Create a warped image
            nifti_image *warpedImage = nifti_copy_nim_info(referenceImage);
            warpedImage->nbyper=sizeof(float);
            warpedImage->datatype=NIFTI_TYPE_FLOAT32;
            warpedImage->scl_slope=1.f;
            warpedImage->scl_inter=0.f;
            warpedImage->data=(void *)malloc(warpedImage->nvox*warpedImage->nbyper);
            // Warp the floating image
            reg_resampleImage(floatingImage,
                              warpedImage,
                              deformationField,
                              NULL,
                              3,
                              0);
            nifti_image_free(floatingImage);
            nifti_image_free(deformationField);
            // Normalise the warped image intensity
            //average_norm_intensity<float>(warpedImage);
            // Accumulate the warped image
            reg_tools_addImageToImage(averageImage,warpedImage,averageImage);
            nifti_image_free(warpedImage);
         }
         // Normalise the average image intensity by the number of input images
         float inputImagesNumber = (argc - 4)/2;
         reg_tools_divideValueToImage(averageImage,averageImage,inputImagesNumber);
         // Save the average image
         reg_io_WriteImageFile(averageImage,outputName);
         nifti_image_free(averageImage);
      }
      else if(operation==3)
      {
         // Affine parametrisations are provided
         size_t affineNumber = (argc - 4)/2;
         // All affine matrices are read in
         mat44 *affineMatrices = (mat44 *)malloc(affineNumber*sizeof(mat44));
         for(int i=4, j=0; i<argc; i+=2,++j)
         {
            if(reg_isAnImageFileName(argv[i]))
            {
               reg_print_msg_error("An affine transformation was expected. Filename:");
               reg_print_msg_error(argv[i]);
               return EXIT_FAILURE;
            }
            reg_tool_ReadAffineFile(&affineMatrices[j],argv[i]);
         }
         // The rigid matrices are removed from all affine matrices
         mat44 tempMatrix, averageMatrix;
         memset(&averageMatrix,0,sizeof(mat44));
         for(size_t i=0; i<affineNumber; ++i)
         {
            float qb,qc,qd,qx,qy,qz,qfac;
            nifti_mat44_to_quatern(affineMatrices[i],&qb,&qc,&qd,&qx,&qy,&qz,NULL,NULL,NULL,&qfac);
            tempMatrix=nifti_quatern_to_mat44(qb,qc,qd,qx,qy,qz,1.f,1.f,1.f,qfac);
            tempMatrix=nifti_mat44_inverse(tempMatrix);
            tempMatrix=reg_mat44_mul(&tempMatrix,&affineMatrices[i]);
            tempMatrix = reg_mat44_logm(&tempMatrix);
            averageMatrix = averageMatrix + tempMatrix;
         }
         // The average matrix is normalised
         averageMatrix = reg_mat44_mul(&averageMatrix,1.f/(float)affineNumber);
         // The average matrix is exponentiated
         averageMatrix = reg_mat44_expm(&averageMatrix);
         // The average matrix is inverted
         averageMatrix = nifti_mat44_inverse(averageMatrix);
         // Demean all the input affine matrices
         for(size_t i=0; i<affineNumber; ++i)
         {
            affineMatrices[i] = averageMatrix * affineMatrices[i];
         }
         // Create a deformation field to be used to resample all the floating images
         nifti_image *deformationField = nifti_copy_nim_info(referenceImage);
         deformationField->dim[0]=deformationField->ndim=5;
         deformationField->dim[4]=deformationField->nt=1;
         deformationField->dim[5]=deformationField->nu=referenceImage->nz>1?3:2;
         deformationField->nvox = (size_t)deformationField->nx *
                                  deformationField->ny * deformationField->nz * deformationField->nu;
         if(deformationField->datatype!=NIFTI_TYPE_FLOAT32 || deformationField->datatype!=NIFTI_TYPE_FLOAT64)
         {
            deformationField->datatype=NIFTI_TYPE_FLOAT32;
            deformationField->nbyper=sizeof(float);
         }
         deformationField->scl_slope=1.f;
         deformationField->scl_inter=0.f;
         deformationField->data = (void *)malloc(deformationField->nvox*deformationField->nbyper);
         // Create an average image
         nifti_image *averageImage = nifti_copy_nim_info(referenceImage);
         if(averageImage->datatype!=NIFTI_TYPE_FLOAT32 || averageImage->datatype!=NIFTI_TYPE_FLOAT64)
         {
            averageImage->datatype=NIFTI_TYPE_FLOAT32;
            averageImage->nbyper=sizeof(float);
         }
         averageImage->data = (void *)calloc(averageImage->nvox,averageImage->nbyper);
         // Create a temporary image
         nifti_image *tempImage = nifti_copy_nim_info(averageImage);
         tempImage->scl_slope=1.f;
         tempImage->scl_inter=0.f;
         tempImage->data = (void *)malloc(tempImage->nvox*tempImage->nbyper);
         // warp all floating images and sum them up
         for(int i=5, j=0; i<argc; i+=2,++j)
         {
            nifti_image *floatingImage = reg_io_ReadImageFile(argv[i]);
            if(floatingImage==NULL)
            {
               reg_print_msg_error("The floating image cannot be read. Filename:");
               reg_print_msg_error(argv[i]);
               return EXIT_FAILURE;
            }
            reg_affine_getDeformationField(&affineMatrices[j],deformationField);
            if(floatingImage->datatype!=tempImage->datatype)
            {
               switch(tempImage->datatype)
               {
               case NIFTI_TYPE_FLOAT32:
                  reg_tools_changeDatatype<float>(floatingImage);
                  break;
               case NIFTI_TYPE_FLOAT64:
                  reg_tools_changeDatatype<double>(floatingImage);
                  break;
               }
            }
            reg_resampleImage(floatingImage,tempImage,deformationField,NULL,3,0.f);
//            if(sizeof(PrecisionTYPE)==sizeof(double))
//               average_norm_intensity<double>(tempImage);
//            else average_norm_intensity<float>(tempImage);
            reg_tools_addImageToImage(averageImage,tempImage,averageImage);
            nifti_image_free(floatingImage);
         }
         // Normalise the intensity by the number of images
         reg_tools_divideValueToImage(averageImage,averageImage,(float)affineNumber);
         // Free the allocated arrays and images
         nifti_image_free(deformationField);
         nifti_image_free(tempImage);
         free(affineMatrices);
         // Save the average image
         reg_io_WriteImageFile(averageImage,outputName);
         // Free the average image
         nifti_image_free(averageImage);
      } // -demean1
      else if(operation==4 || operation==5)
      {
         // Compute some constant
         int incrementValue=operation==4?2:3;
         int subjectNumber=(argc-4)/incrementValue;
#ifndef NDEBUG
         char msg[256];
         sprintf(msg,"reg_average: Number of input transformations: %i",subjectNumber);
         reg_print_msg_debug(msg);
#endif
         /* **** Create an average image by demeaning the non-rigid transformation **** */
         // First compute an average field to remove from the final field
         nifti_image *averageField = nifti_copy_nim_info(referenceImage);
         averageField->dim[0]=averageField->ndim=5;
         averageField->dim[4]=averageField->nt=1;
         averageField->dim[5]=averageField->nu=averageField->nz>1?3:2;
         averageField->nvox = (size_t)averageField->nx *
                              averageField->ny * averageField->nz * averageField->nu;
         if(averageField->datatype!=NIFTI_TYPE_FLOAT32 || averageField->datatype!=NIFTI_TYPE_FLOAT64)
         {
            averageField->datatype=NIFTI_TYPE_FLOAT32;
            averageField->nbyper=sizeof(float);
         }
         averageField->data = (void *)calloc(averageField->nvox,averageField->nbyper);
         averageField->scl_slope=1.f;
         averageField->scl_inter=0.f;
         reg_tools_multiplyValueToImage(averageField,averageField,0.f);
         // Iterate over all the transformation parametrisations - Note that I don't store them all to save space

         for(int i=operation; i<argc; i+=incrementValue)
         {
            nifti_image *transformation = reg_io_ReadImageFile(argv[i]);
            if(transformation==NULL)
            {
               reg_print_msg_error("The transformation parametrisation cannot be read. Filename:");
               reg_print_msg_error(argv[i]);
               return EXIT_FAILURE;
            }
            if(transformation->ndim!=5)
            {
               reg_print_msg_error("The specified filename does not seem to be a transformation parametrisation. Filename:");
               reg_print_msg_error(transformation->fname);
               return EXIT_FAILURE;
            }
#ifndef NDEBUG
            reg_print_msg_debug("reg_average: Input non-rigid transformation:");
            reg_print_msg_debug(transformation->fname);
#endif
            // Generate the deformation or flow field
            nifti_image *deformationField = nifti_copy_nim_info(averageField);
            deformationField->data = (void *)malloc(deformationField->nvox*deformationField->nbyper);
            reg_tools_multiplyValueToImage(deformationField,deformationField,0.f);
            deformationField->scl_slope=1.f;
            deformationField->scl_inter=0.f;
            deformationField->intent_p1=DISP_FIELD;
            reg_getDeformationFromDisplacement(deformationField);
            // Generate a deformation field or a flow field depending of the input transformation
            switch(static_cast<int>(transformation->intent_p1))
            {
            case DISP_FIELD:
               reg_getDeformationFromDisplacement(transformation);
            case DEF_FIELD:
               reg_defField_compose(transformation,deformationField,NULL);
               break;
            case SPLINE_GRID:
               reg_spline_getDeformationField(transformation,deformationField,NULL,true,true);
               break;
            case DISP_VEL_FIELD:
               reg_getDeformationFromDisplacement(transformation);
            case DEF_VEL_FIELD:
               reg_defField_compose(transformation,deformationField,NULL);
               break;
            case SPLINE_VEL_GRID:
               reg_spline_getFlowFieldFromVelocityGrid(transformation,deformationField);
#ifndef NDEBUG
               reg_print_msg_debug("reg_average: A dense flow field has been computed from:");
               reg_print_msg_debug(transformation->fname);
#endif
               break;
            default:
               reg_print_msg_error("Unsupported transformation parametrisation type. Filename:");
               reg_print_msg_error(transformation->fname);
               return EXIT_FAILURE;
            }
            // An affine transformation is use to remove the affine component
            if(operation==5 || transformation->num_ext>0)
            {
               mat44 affineTransformation;
               if(transformation->num_ext>0)
               {
                  if(operation==5)
                  {
                     reg_print_msg_warn("The provided non-rigid parametrisation already embbeds an affine transformation");
                     reg_print_msg_warn(transformation->fname);
                  }
                  affineTransformation=*reinterpret_cast<mat44 *>(transformation->ext_list[0].edata);
                  // Note that if the transformation is a flow field, only half-of the affine has be used
                  if(transformation->num_ext>1 &&
                        deformationField->intent_p1!=DEF_VEL_FIELD)
                  {
                     affineTransformation=reg_mat44_mul(
                                             reinterpret_cast<mat44 *>(transformation->ext_list[1].edata),
                                             &affineTransformation);
                  }
               }
               else
               {
                  reg_tool_ReadAffineFile(&affineTransformation,
                                          argv[i-1]);
#ifndef NDEBUG
                  reg_print_msg_debug("reg_average: Input affine transformation. Filename:");
                  reg_print_msg_debug(argv[i-1]);
#endif
               }
               // The affine component is substracted
               nifti_image *tempField = nifti_copy_nim_info(deformationField);
               tempField->data = (void *)malloc(tempField->nvox*tempField->nbyper);
               tempField->scl_slope=1.f;
               tempField->scl_inter=0.f;
               reg_affine_getDeformationField(&affineTransformation,
                                              tempField);
               reg_tools_substractImageToImage(deformationField,tempField,deformationField);
               nifti_image_free(tempField);
            }
            else reg_getDisplacementFromDeformation(deformationField);
            reg_tools_addImageToImage(averageField,deformationField,averageField);
            nifti_image_free(transformation);
            nifti_image_free(deformationField);
         } // iteration over all transformation parametrisation
         // the average def/flow field is normalised by the number of input image
         reg_tools_divideValueToImage(averageField,averageField,subjectNumber);
         // The new de-mean transformation are computed and the floating image resample
         // Create an image to store average image
         nifti_image *averageImage = nifti_copy_nim_info(referenceImage);
         if(averageImage->datatype!=NIFTI_TYPE_FLOAT32 || averageImage->datatype!=NIFTI_TYPE_FLOAT64)
         {
            averageImage->datatype=NIFTI_TYPE_FLOAT32;
            averageImage->nbyper=sizeof(float);
         }
         averageImage->scl_slope=1.f;
         averageImage->scl_inter=0.f;
         averageImage->data = (void *)calloc(averageImage->nvox,averageImage->nbyper);
         // Create a temporary image
         nifti_image *tempImage = nifti_copy_nim_info(averageImage);
         tempImage->data = (void *)malloc(tempImage->nvox*tempImage->nbyper);
         // Iterate over all the transformation parametrisations
         for(int i=operation; i<argc; i+=incrementValue)
         {
            nifti_image *transformation = reg_io_ReadImageFile(argv[i]);
            if(transformation==NULL)
            {
               reg_print_msg_error("The transformation parametrisation cannot be read. Filename:");
               reg_print_msg_error(argv[i]);
               return EXIT_FAILURE;
            }
            if(transformation->ndim!=5)
            {
               reg_print_msg_error("The specified filename does not seem to be a transformation parametrisation. Filename");
               reg_print_msg_error(transformation->fname);
               return EXIT_FAILURE;
            }
#ifndef NDEBUG
            reg_print_msg_debug("reg_average: Demeaning transformation:");
            reg_print_msg_debug(transformation->fname);
#endif
            // Generate the deformation or flow field
            nifti_image *deformationField = nifti_copy_nim_info(averageField);
            deformationField->data = (void *)malloc(deformationField->nvox*deformationField->nbyper);
            reg_tools_multiplyValueToImage(deformationField,deformationField,0.f);
            deformationField->intent_p1=DISP_FIELD;
            reg_getDeformationFromDisplacement(deformationField);
            deformationField->scl_slope=1.f;
            deformationField->scl_inter=0.f;
            // Generate a deformation field or a flow field depending of the input transformation
            switch(static_cast<int>(transformation->intent_p1))
            {
            case DISP_FIELD:
               reg_getDeformationFromDisplacement(transformation);
            case DEF_FIELD:
               reg_defField_compose(transformation,deformationField,NULL);
               break;
            case SPLINE_GRID:
               reg_spline_getDeformationField(transformation,deformationField,NULL,true,true);
               break;
            case DISP_VEL_FIELD:
               reg_getDeformationFromDisplacement(transformation);
            case DEF_VEL_FIELD:
               reg_defField_compose(transformation,deformationField,NULL);
               break;
            case SPLINE_VEL_GRID:
               if(transformation->num_ext>0)
                  nifti_copy_extensions(deformationField,transformation);
               reg_spline_getFlowFieldFromVelocityGrid(transformation,deformationField);
               break;
            default:
               reg_print_msg_error("Unsupported transformation parametrisation type. Filename:");
               reg_print_msg_error(transformation->fname);
               return EXIT_FAILURE;
            }
            // The deformation is de-mean
            if(deformationField->intent_p1==DEF_VEL_FIELD)
            {
               reg_tools_substractImageToImage(deformationField,averageField,deformationField);
               nifti_image *tempDef = nifti_copy_nim_info(deformationField);
               tempDef->data = (void *)malloc(tempDef->nvox*tempDef->nbyper);
               memcpy(tempDef->data,deformationField->data,tempDef->nvox*tempDef->nbyper);
               tempDef->scl_slope=1.f;
               tempDef->scl_inter=0.f;
               reg_defField_getDeformationFieldFromFlowField(tempDef,deformationField,false);
               deformationField->intent_p1=DEF_FIELD;
               nifti_free_extensions(deformationField);
               nifti_image_free(tempDef);
            }
            else reg_tools_substractImageToImage(deformationField,averageField,deformationField);
            // The floating image is resampled
            nifti_image *floatingImage=reg_io_ReadImageFile(argv[i+1]);
            if(floatingImage==NULL)
            {
               reg_print_msg_error("The floating image cannot be read. Filename:");
               reg_print_msg_error(argv[i+1]);
               return EXIT_FAILURE;
            }
            if(floatingImage->datatype!=tempImage->datatype)
            {
               switch(tempImage->datatype)
               {
               case NIFTI_TYPE_FLOAT32:
                  reg_tools_changeDatatype<float>(floatingImage);
                  break;
               case NIFTI_TYPE_FLOAT64:
                  reg_tools_changeDatatype<double>(floatingImage);
                  break;
               }
            }
            reg_resampleImage(floatingImage,tempImage,deformationField,NULL,3,0.f);
#ifndef NDEBUG
            reg_print_msg_debug("reg_average: Warping floating image:");
            reg_print_msg_debug(floatingImage->fname);
            sprintf(msg,"reg_average_%i.nii",i);
            reg_io_WriteImageFile(tempImage,msg);
#endif
            reg_tools_addImageToImage(averageImage,tempImage,averageImage);
            nifti_image_free(floatingImage);
            nifti_image_free(deformationField);
         } // iteration over all transformation parametrisation
         // Normalise the average image by the number of input images
         reg_tools_divideValueToImage(averageImage,averageImage,subjectNumber);
         // Free the allocated field
         nifti_image_free(averageField);
         // Save and free the average image
         reg_io_WriteImageFile(averageImage,outputName);
         nifti_image_free(averageImage);
      } // (operation==4 || operation==5)
      nifti_image_free(referenceImage);
   } // (-demean -avg_tran)

   return EXIT_SUCCESS;
}

#endif
