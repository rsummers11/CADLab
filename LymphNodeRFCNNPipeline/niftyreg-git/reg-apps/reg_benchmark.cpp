/*
*  benchmark.cpp
*
*
*  Created by Marc Modat on 15/11/2009.
*  Copyright (c) 2009, University College London. All rights reserved.
*  Centre for Medical Image Computing (CMIC)
*  See the LICENSE.txt file in the nifty_reg root folder
*
*/

#include "_reg_resampling.h"
#include "_reg_affineTransformation.h"
#include "_reg_bspline.h"
#include "_reg_bspline_comp.h"
#include "_reg_mutualinformation.h"
#include "_reg_ssd.h"
#include "_reg_tools.h"
#include "_reg_blockMatching.h"

#ifdef _USE_CUDA
#include "_reg_cudaCommon.h"
#include "_reg_resampling_gpu.h"
#include "_reg_affineTransformation_gpu.h"
#include "_reg_bspline_gpu.h"
#include "_reg_mutualinformation_gpu.h"
#include "_reg_tools_gpu.h"
#include "_reg_blockMatching_gpu.h"
#endif

#ifdef _WINDOWS
#include <time.h>
#endif

void Usage(char *);

int main(int argc, char **argv)
{
   int dimension = 100;
   float gridSpacing = 10.0f;
   unsigned int binning = 68;
   char *outputFileName = (char *)"benchmark_result.txt";
   bool runGPU=1;

   for(int i=1; i<argc; i++)
   {
      if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
            strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
            strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0)
      {
         Usage(argv[0]);
         return 0;
      }
      else if(strcmp(argv[i], "-dim") == 0)
      {
         dimension=atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-sp") == 0)
      {
         gridSpacing=atof(argv[++i]);
      }
      else if(strcmp(argv[i], "-bin") == 0)
      {
         binning=atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-o") == 0)
      {
         outputFileName=argv[++i];
      }
      else if(strcmp(argv[i], "-cpu") == 0)
      {
         runGPU=0;
      }
      else
      {
         fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
         Usage(argv[0]);
         return 1;
      }
   }

   // The target, source and result images are created
   int dim_img[8];
   dim_img[0]=3;
   dim_img[1]=dimension;
   dim_img[2]=dimension;
   dim_img[3]=dimension;
   dim_img[4]=dim_img[5]=dim_img[6]=dim_img[7]=1;
   nifti_image *targetImage = nifti_make_new_nim(dim_img, NIFTI_TYPE_FLOAT32, true);
   nifti_image *sourceImage = nifti_make_new_nim(dim_img, NIFTI_TYPE_FLOAT32, true);
   nifti_image *resultImage = nifti_make_new_nim(dim_img, NIFTI_TYPE_FLOAT32, true);
   targetImage->sform_code=0;
   sourceImage->sform_code=0;
   resultImage->sform_code=0;
   int *maskImage = (int *)malloc(targetImage->nvox*sizeof(int));

   // The target and source images are filled with random number
   float *targetPtr=static_cast<float *>(targetImage->data);
   float *sourcePtr=static_cast<float *>(sourceImage->data);
   srand((unsigned)time(0));
   for(unsigned int i=0; i<targetImage->nvox; ++i)
   {
      *targetPtr++ = (float)(binning-4)*(float)rand()/(float)RAND_MAX + 2.0f;
      *sourcePtr++ = (float)(binning-4)*(float)rand()/(float)RAND_MAX + 2.0f;
      maskImage[i]=i;
   }

   // Deformation field image is created
   dim_img[0]=5;
   dim_img[1]=dimension;
   dim_img[2]=dimension;
   dim_img[3]=dimension;
   dim_img[5]=3;
   dim_img[4]=dim_img[6]=dim_img[7]=1;
   nifti_image *deformationFieldImage = nifti_make_new_nim(dim_img, NIFTI_TYPE_FLOAT32, true);

   // Joint histogram creation
   double *probaJointHistogram=(double *)malloc(binning*(binning+2)*sizeof(double));
   double *logJointHistogram=(double *)malloc(binning*(binning+2)*sizeof(double));


   // A control point image is created
   dim_img[0]=5;
   dim_img[1]=(int)floor(targetImage->nx*targetImage->dx/gridSpacing)+4;
   dim_img[2]=(int)floor(targetImage->ny*targetImage->dy/gridSpacing)+4;
   dim_img[3]=(int)floor(targetImage->nz*targetImage->dz/gridSpacing)+4;
   dim_img[5]=3;
   dim_img[4]=dim_img[6]=dim_img[7]=1;
   nifti_image *controlPointImage = nifti_make_new_nim(dim_img, NIFTI_TYPE_FLOAT32, true);
   controlPointImage->cal_min=0;
   controlPointImage->cal_max=0;
   controlPointImage->pixdim[0]=1.0f;
   controlPointImage->pixdim[1]=controlPointImage->dx=gridSpacing;
   controlPointImage->pixdim[2]=controlPointImage->dy=gridSpacing;
   controlPointImage->pixdim[3]=controlPointImage->dz=gridSpacing;
   controlPointImage->pixdim[4]=controlPointImage->dt=1.0f;
   controlPointImage->pixdim[5]=controlPointImage->du=1.0f;
   controlPointImage->pixdim[6]=controlPointImage->dv=1.0f;
   controlPointImage->pixdim[7]=controlPointImage->dw=1.0f;
   controlPointImage->qform_code=targetImage->qform_code;
   controlPointImage->sform_code=targetImage->sform_code;
   float qb, qc, qd, qx, qy, qz, dx, dy, dz, qfac;
   nifti_mat44_to_quatern( targetImage->qto_xyz, &qb, &qc, &qd, &qx, &qy, &qz, &dx, &dy, &dz, &qfac);
   controlPointImage->quatern_b=qb;
   controlPointImage->quatern_c=qc;
   controlPointImage->quatern_d=qd;
   controlPointImage->qfac=qfac;
   controlPointImage->qto_xyz = nifti_quatern_to_mat44(qb, qc, qd, qx, qy, qz,
                                controlPointImage->dx, controlPointImage->dy, controlPointImage->dz, qfac);
   float originIndex[3];
   float originReal[3];
   originIndex[0] = -1.0f;
   originIndex[1] = -1.0f;
   originIndex[2] = -1.0f;
   reg_mat44_mul(&(controlPointImage->qto_xyz), originIndex, originReal);
   controlPointImage->qto_xyz.m[0][3] = controlPointImage->qoffset_x = originReal[0];
   controlPointImage->qto_xyz.m[1][3] = controlPointImage->qoffset_y = originReal[1];
   controlPointImage->qto_xyz.m[2][3] = controlPointImage->qoffset_z = originReal[2];
   controlPointImage->qto_ijk = nifti_mat44_inverse(controlPointImage->qto_xyz);

   // Velocity field image
   nifti_image *velocityFieldImage = nifti_copy_nim_info(controlPointImage);
   velocityFieldImage->datatype = NIFTI_TYPE_FLOAT32;
   velocityFieldImage->nbyper = sizeof(float);
   velocityFieldImage->data = (void *)calloc(velocityFieldImage->nvox, velocityFieldImage->nbyper);

   // Different gradient images
   nifti_image *resultGradientImage = nifti_copy_nim_info(deformationFieldImage);
   resultGradientImage->datatype = NIFTI_TYPE_FLOAT32;
   resultGradientImage->nbyper = sizeof(float);
   resultGradientImage->data = (void *)calloc(resultGradientImage->nvox, resultGradientImage->nbyper);
   nifti_image *voxelNMIGradientImage = nifti_copy_nim_info(deformationFieldImage);
   voxelNMIGradientImage->datatype = NIFTI_TYPE_FLOAT32;
   voxelNMIGradientImage->nbyper = sizeof(float);
   voxelNMIGradientImage->data = (void *)calloc(voxelNMIGradientImage->nvox, voxelNMIGradientImage->nbyper);
   nifti_image *nodeNMIGradientImage = nifti_copy_nim_info(controlPointImage);
   nodeNMIGradientImage->datatype = NIFTI_TYPE_FLOAT32;
   nodeNMIGradientImage->nbyper = sizeof(float);
   nodeNMIGradientImage->data = (void *)calloc(nodeNMIGradientImage->nvox, nodeNMIGradientImage->nbyper);

#ifdef _USE_CUDA
   float *targetImageArray_d;
   cudaArray *sourceImageArray_d;
   int *targetMask_d;
   float4 *deformationFieldImageArray_d;
   if(runGPU)
   {
      if(cudaCommon_allocateArrayToDevice<float>(&targetImageArray_d, targetImage->dim)) return 1;
      if(cudaCommon_transferNiftiToArrayOnDevice<float>(&targetImageArray_d, targetImage)) return 1;
      if(cudaCommon_allocateArrayToDevice<float>(&sourceImageArray_d, sourceImage->dim)) return 1;
      if(cudaCommon_transferNiftiToArrayOnDevice<float>(&sourceImageArray_d,sourceImage)) return 1;
      CUDA_SAFE_CALL(cudaMalloc((void **)&targetMask_d, targetImage->nvox*sizeof(int)));
      CUDA_SAFE_CALL(cudaMemcpy(targetMask_d, maskImage, targetImage->nvox*sizeof(int), cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMalloc((void **)&deformationFieldImageArray_d, targetImage->nvox*sizeof(float4)));
   }
#endif

   time_t start,end;
   int minutes, seconds, cpuTime, maxIt;
#ifdef _USE_CUDA
   int gpuTime
#endif

   FILE *outputFile;
   outputFile=fopen(outputFileName, "w");

   /* Functions to be tested
   	- affine deformation field
   	- spline deformation field
   	- linear interpolation
   	- block matching computation
   	- spatial gradient computation
   	- voxel-based NMI gradient computation
   	- node-based NMI gradient computation
        - bending-energy computation
        - bending-energy gradient computation

        - Jacobian-based penalty term computation from CPP
        - Jacobian-based gradient computation from CPP
        - Folding correction from CPP
        - Approximated Jacobian-based penalty term computation from CPP
        - Approximated Jacobian-based gradient computation from CPP
        - Approximated Folding correction from CPP
   */

   // AFFINE DEFORMATION FIELD CREATION
   {
      maxIt=500000 / dimension;
//        maxIt=1;
      mat44 *affineTransformation = (mat44 *)calloc(1,sizeof(mat44));
      affineTransformation->m[0][0]=1.0;
      affineTransformation->m[1][1]=1.0;
      affineTransformation->m[2][2]=1.0;
      affineTransformation->m[3][3]=1.0;
      if(reg_bspline_initialiseControlPointGridWithAffine(affineTransformation, controlPointImage))
         return 1;

      time(&start);
      for(int i=0; i<maxIt; ++i)
      {
         reg_affine_positionField(   affineTransformation,
                                     targetImage,
                                     deformationFieldImage);
      }
      time(&end);
      cpuTime=(end-start);
      minutes = (int)floorf(float(cpuTime)/60.0f);
      seconds = (int)(cpuTime - 60*minutes);
      printf( "CPU - %i affine deformation field computations - %i min %i sec\n", maxIt, minutes, seconds);
      fprintf(outputFile, "CPU - %i affine deformation field computations - %i min %i sec\n", maxIt, minutes, seconds);
#ifdef _USE_CUDA
      if(runGPU)
      {
         time(&start);
         for(int i=0; i<maxIt; ++i)
         {
            reg_affine_positionField_gpu(   affineTransformation,
                                            targetImage,
                                            &deformationFieldImageArray_d);
         }
         time(&end);
         gpuTime=(end-start);
         minutes = (int)floorf(float(gpuTime)/60.0f);
         seconds = (int)(gpuTime - 60*minutes);
         printf("GPU - %i affine deformation field computations - %i min %i sec\n", maxIt, minutes, seconds);
         fprintf(outputFile, "GPU - %i affine deformation field computations - %i min %i sec\n", maxIt, minutes, seconds);
         printf("Affine deformation field ratio - %g time(s)\n", (float)cpuTime/(float)gpuTime);
         fprintf(outputFile, "Affine deformation field ratio - %g time(s)\n\n", (float)cpuTime/(float)gpuTime);
      }
#endif
      printf("Affine deformation done\n\n");
   }

   // SPLINE DEFORMATION FIELD CREATION
#ifdef _USE_CUDA
   float4 *controlPointImageArray_d;
   if(runGPU)
   {
      if(cudaCommon_allocateArrayToDevice<float4>(&controlPointImageArray_d, controlPointImage->dim)) return 1;
      if(cudaCommon_transferNiftiToArrayOnDevice<float4>(&controlPointImageArray_d,controlPointImage)) return 1;
   }
#endif
   {
      maxIt=50000 / dimension;
//        maxIt=1;
      time(&start);
      for(int i=0; i<maxIt; ++i)
      {
         reg_bspline<float>( controlPointImage,
                             targetImage,
                             deformationFieldImage,
                             maskImage,
                             0);
      }
      time(&end);
      cpuTime=(end-start);
      minutes = (int)floorf(float(cpuTime)/60.0f);
      seconds = (int)(cpuTime - 60*minutes);
      printf("CPU - %i spline deformation field computations - %i min %i sec\n", maxIt, minutes, seconds);
      fprintf(outputFile, "CPU - %i spline deformation field computations - %i min %i sec\n", maxIt, minutes, seconds);
#ifdef _USE_CUDA
      if(runGPU)
      {
         time(&start);
         for(int i=0; i<maxIt; ++i)
         {
            reg_bspline_gpu(controlPointImage,
                            targetImage,
                            &controlPointImageArray_d,
                            &deformationFieldImageArray_d,
                            &targetMask_d,
                            targetImage->nvox);
         }
         time(&end);
         gpuTime=(end-start);
         minutes = (int)floorf(float(gpuTime)/60.0f);
         seconds = (int)(gpuTime - 60*minutes);
         printf("GPU - %i spline deformation field computations - %i min %i sec\n", maxIt, minutes, seconds);
         fprintf(outputFile, "GPU - %i spline deformation field computations - %i min %i sec\n", maxIt, minutes, seconds);
         printf("Spline deformation field ratio - %g time(s)\n", (float)cpuTime/(float)gpuTime);
         fprintf(outputFile, "Spline deformation field ratio - %g time(s)\n\n", (float)cpuTime/(float)gpuTime);
      }
#endif
      printf("Spline deformation done\n\n");
   }

   // SCALING-AND-SQUARING APPROACH
#ifdef _USE_CUDA
   float4 *velocityFieldImageArray_d;
   if(runGPU)
   {
      if(cudaCommon_allocateArrayToDevice<float4>(&velocityFieldImageArray_d, velocityFieldImage->dim)) return 1;
      if(cudaCommon_transferNiftiToArrayOnDevice<float4>(&velocityFieldImageArray_d,velocityFieldImage)) return 1;
   }
#endif
   {
      maxIt=20000 / dimension;
//        maxIt=1;
      time(&start);
      for(int i=0; i<maxIt; ++i)
      {
         reg_spline_scaling_squaring(velocityFieldImage,
                                     controlPointImage);
      }
      time(&end);
      cpuTime=(end-start);
      minutes = (int)floorf(float(cpuTime)/60.0f);
      seconds = (int)(cpuTime - 60*minutes);
      printf("CPU - %i scaling-and-squaring - %i min %i sec\n", maxIt, minutes, seconds);
      fprintf(outputFile, "CPU - %i scaling-and-squarings - %i min %i sec\n", maxIt, minutes, seconds);
      time(&start);
#ifdef _USE_CUDA
      if(runGPU)
      {
         for(int i=0; i<maxIt; ++i)
         {
            reg_spline_scaling_squaring_gpu(velocityFieldImage,
                                            controlPointImage,
                                            &velocityFieldImageArray_d,
                                            &controlPointImageArray_d);
         }
         time(&end);
         gpuTime=(end-start);
         minutes = (int)floorf(float(gpuTime)/60.0f);
         seconds = (int)(gpuTime - 60*minutes);
         printf("GPU - %i scaling-and-squaring - %i min %i sec\n", maxIt, minutes, seconds);
         fprintf(outputFile, "GPU - %i scaling-and-squarings - %i min %i sec\n", maxIt, minutes, seconds);
         printf("Scaling-and-squarings ratio - %g time(s)\n", (float)cpuTime/(float)gpuTime);
         fprintf(outputFile, "Scaling-and-squarings ratio - %g time(s)\n\n", (float)cpuTime/(float)gpuTime);
      }
#endif
      printf("scaling-and-squarings done\n\n");
   }

   // LINEAR INTERPOLATION
#ifdef _USE_CUDA
   float *resultImageArray_d;
   if(runGPU)
      if(cudaCommon_allocateArrayToDevice<float>(&resultImageArray_d, targetImage->dim)) return 1;
#endif
   {
      maxIt=100000 / dimension;
//        maxIt=1;
      time(&start);
      for(int i=0; i<maxIt; ++i)
      {
         reg_resampleSourceImage<float>( targetImage,
                                         sourceImage,
                                         resultImage,
                                         deformationFieldImage,
                                         maskImage,
                                         1,
                                         0);
      }
      time(&end);
      cpuTime=(end-start);
      minutes = (int)floorf(float(cpuTime)/60.0f);
      seconds = (int)(cpuTime - 60*minutes);
      printf("CPU - %i linear interpolation computations - %i min %i sec\n", maxIt, minutes, seconds);
      fprintf(outputFile, "CPU - %i linear interpolation computations - %i min %i sec\n", maxIt, minutes, seconds);
#ifdef _USE_CUDA
      if(runGPU)
      {
         time(&start);
         for(int i=0; i<maxIt; ++i)
         {
            reg_resampleSourceImage_gpu(resultImage,
                                        sourceImage,
                                        &resultImageArray_d,
                                        &sourceImageArray_d,
                                        &deformationFieldImageArray_d,
                                        &targetMask_d,
                                        targetImage->nvox,
                                        0);
         }
         time(&end);
         gpuTime=(end-start);
         minutes = (int)floorf(float(gpuTime)/60.0f);
         seconds = (int)(gpuTime - 60*minutes);
         printf("GPU - %i linear interpolation computations - %i min %i sec\n", maxIt, minutes, seconds);
         fprintf(outputFile, "GPU - %i linear interpolation computations - %i min %i sec\n", maxIt, minutes, seconds);
         printf("Linear interpolation ratio - %g time(s)\n", (float)cpuTime/(float)gpuTime);
         fprintf(outputFile, "Linear interpolation ratio - %g time(s)\n\n", (float)cpuTime/(float)gpuTime);
      }
#endif
      printf("Linear interpolation done\n\n");
   }

   // SPATIAL GRADIENT COMPUTATION
#ifdef _USE_CUDA
   float4 *resultGradientArray_d;
   CUDA_SAFE_CALL(cudaMalloc((void **)&resultGradientArray_d, targetImage->nvox*sizeof(float4)));
#endif
   {
      maxIt=100000 / dimension;
//        maxIt=1;
      time(&start);
      for(int i=0; i<maxIt; ++i)
      {
         reg_getSourceImageGradient<float>(  targetImage,
                                             sourceImage,
                                             resultGradientImage,
                                             deformationFieldImage,
                                             maskImage,
                                             1);
      }
      time(&end);
      cpuTime=(end-start);
      minutes = (int)floorf(float(cpuTime)/60.0f);
      seconds = (int)(cpuTime - 60*minutes);
      printf("CPU - %i spatial gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
      fprintf(outputFile, "CPU - %i spatial gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
#ifdef _USE_CUDA
      if(runGPU)
      {
         time(&start);
         for(int i=0; i<maxIt; ++i)
         {
            reg_getSourceImageGradient_gpu( targetImage,
                                            sourceImage,
                                            &sourceImageArray_d,
                                            &deformationFieldImageArray_d,
                                            &resultGradientArray_d,
                                            targetImage->nvox);
         }
         time(&end);
         gpuTime=(end-start);
         minutes = (int)floorf(float(gpuTime)/60.0f);
         seconds = (int)(gpuTime - 60*minutes);
         printf("GPU - %i spatial gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
         fprintf(outputFile, "GPU - %i spatial gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
         printf("Spatial gradient ratio - %g time(s)\n", (float)cpuTime/(float)gpuTime);
         fprintf(outputFile, "Spatial gradient ratio - %g time(s)\n\n", (float)cpuTime/(float)gpuTime);
         cudaCommon_free( &sourceImageArray_d );
      }
#endif
      printf("Spatial gradient done\n\n");
   }
   nifti_image_free(sourceImage);

#ifdef _USE_CUDA
   if(runGPU)
   {
      cudaCommon_free( (void **)&deformationFieldImageArray_d );
   }
#endif

   // JOINT HISTOGRAM COMPUTATION
   double entropies[4];
   {
      reg_getEntropies<double>(   targetImage,
                                  resultImage,
                                  2,
                                  &binning,
                                  &binning,
                                  probaJointHistogram,
                                  logJointHistogram,
                                  entropies,
                                  maskImage);
   }

   // VOXEL-BASED NMI GRADIENT COMPUTATION
#ifdef _USE_CUDA
   float4 *voxelNMIGradientArray_d;
   if(runGPU)
   {
      if(cudaCommon_allocateArrayToDevice(&voxelNMIGradientArray_d, resultImage->dim)) return 1;
   }
#endif
   {
      maxIt=100000 / dimension;
//        maxIt=1;
      time(&start);
      for(int i=0; i<maxIt; ++i)
      {
         reg_getVoxelBasedNMIGradientUsingPW<double>(targetImage,
               resultImage,
               2,
               resultGradientImage,
               &binning,
               &binning,
               logJointHistogram,
               entropies,
               voxelNMIGradientImage,
               maskImage);
      }
      time(&end);
      cpuTime=(end-start);
      minutes = (int)floorf(float(cpuTime)/60.0f);
      seconds = (int)(cpuTime - 60*minutes);
      printf("CPU - %i voxel-based NMI gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
      fprintf(outputFile, "CPU - %i voxel-based NMI gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
#ifdef _USE_CUDA
      float *logJointHistogram_d;
      if(runGPU)
      {
         CUDA_SAFE_CALL(cudaMalloc((void **)&logJointHistogram_d, binning*(binning+2)*sizeof(float)));
         float *tempB=(float *)malloc(binning*(binning+2)*sizeof(float));
         for(int i=0; i<binning*(binning+2); i++)
         {
            tempB[i]=(float)logJointHistogram[i];
         }
         CUDA_SAFE_CALL(cudaMemcpy(logJointHistogram_d, tempB, binning*(binning+2)*sizeof(float), cudaMemcpyHostToDevice));
         free(tempB);
         time(&start);
         for(int i=0; i<maxIt; ++i)
         {
            reg_getVoxelBasedNMIGradientUsingPW_gpu(targetImage,
                                                    resultImage,
                                                    &targetImageArray_d,
                                                    &resultImageArray_d,
                                                    &resultGradientArray_d,
                                                    &logJointHistogram_d,
                                                    &voxelNMIGradientArray_d,
                                                    &targetMask_d,
                                                    targetImage->nvox,
                                                    entropies,
                                                    binning);
         }
         time(&end);
         gpuTime=(end-start);
         minutes = (int)floorf(float(gpuTime)/60.0f);
         seconds = (int)(gpuTime - 60*minutes);
         printf("GPU - %i voxel-based NMI gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
         fprintf(outputFile, "GPU - %i voxel-based NMI gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
         printf("Voxel-based NMI gradient ratio - %g time(s)\n", (float)cpuTime/(float)gpuTime);
         fprintf(outputFile, "Voxel-based NMI gradient ratio - %g time(s)\n\n", (float)cpuTime/(float)gpuTime);
         cudaCommon_free((void **)&logJointHistogram_d);
      }
      CUDA_SAFE_CALL(cudaFree(targetMask_d));
#endif
      printf("Voxel-based NMI gradient done\n\n");
   }

#ifdef _USE_CUDA
   if(runGPU)
   {
      cudaCommon_free((void **)&resultGradientArray_d);
   }
#endif

   // NODE-BASED NMI GRADIENT COMPUTATION
#ifdef _USE_CUDA
   float4 *nodeNMIGradientArray_d;
   if(runGPU)
   {
      if(cudaCommon_allocateArrayToDevice(&nodeNMIGradientArray_d, controlPointImage->dim)) return 1;
   }
#endif
   {
      maxIt=10000 / dimension;
//        maxIt=1;
      int smoothingRadius[3];
      smoothingRadius[0] = (int)floor( 2.0*controlPointImage->dx/targetImage->dx );
      smoothingRadius[1] = (int)floor( 2.0*controlPointImage->dy/targetImage->dy );
      smoothingRadius[2] = (int)floor( 2.0*controlPointImage->dz/targetImage->dz );
      time(&start);
      for(int i=0; i<maxIt; ++i)
      {
         reg_smoothImageForCubicSpline<float>(voxelNMIGradientImage,smoothingRadius);
         reg_voxelCentric2NodeCentric(nodeNMIGradientImage,voxelNMIGradientImage,1.0f);
      }
      time(&end);
      cpuTime=(end-start);
      minutes = (int)floorf(float(cpuTime)/60.0f);
      seconds = (int)(cpuTime - 60*minutes);
      printf("CPU - %i node-based NMI gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
      fprintf(outputFile, "CPU - %i node-based NMI gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
#ifdef _USE_CUDA
      if(runGPU)
      {
         time(&start);
         for(int i=0; i<maxIt; ++i)
         {
            reg_smoothImageForCubicSpline_gpu(  resultImage,
                                                &voxelNMIGradientArray_d,
                                                smoothingRadius);
            reg_voxelCentric2NodeCentric_gpu(   targetImage,
                                                controlPointImage,
                                                &voxelNMIGradientArray_d,
                                                &nodeNMIGradientArray_d,
                                                1.0f);
         }
         time(&end);
         gpuTime=(end-start);
         minutes = (int)floorf(float(gpuTime)/60.0f);
         seconds = (int)(gpuTime - 60*minutes);
         printf("GPU - %i node-based NMI gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
         fprintf(outputFile, "GPU - %i node-based NMI gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
         printf("Node-based NMI gradient ratio - %g time(s)\n", (float)cpuTime/(float)gpuTime);
         fprintf(outputFile, "Node-based NMI gradient ratio - %g time(s)\n\n", (float)cpuTime/(float)gpuTime);
      }
#endif
      printf("Node-based NMI gradient done\n\n");
   }

#ifdef _USE_CUDA
   if(runGPU)
   {
      cudaCommon_free((void **)&voxelNMIGradientArray_d);
      cudaCommon_free((void **)&nodeNMIGradientArray_d);
   }
#endif

   // APPROXIMATED BENDING ENERGY PENALTY TERM COMPUTATION
   {
      maxIt=100000 / dimension;
//        maxIt=1;
      time(&start);
      for(int i=0; i<maxIt; ++i)
      {
         reg_bspline_bendingEnergy<float>(controlPointImage, targetImage,1);
      }
      time(&end);
      cpuTime=(end-start);
      minutes = (int)floorf(float(cpuTime)/60.0f);
      seconds = (int)(cpuTime - 60*minutes);
      printf("CPU - %i BE computations - %i min %i sec\n", maxIt, minutes, seconds);
      fprintf(outputFile, "CPU - %i BE computations - %i min %i sec\n", maxIt, minutes, seconds);
#ifdef _USE_CUDA
      if(runGPU)
      {
         time(&start);
         for(int i=0; i<maxIt; ++i)
         {
            reg_bspline_ApproxBendingEnergy_gpu(controlPointImage,
                                                &controlPointImageArray_d);
         }
         time(&end);
         gpuTime=(end-start);
         minutes = (int)floorf(float(gpuTime)/60.0f);
         seconds = (int)(gpuTime - 60*minutes);
         printf("GPU - %i BE computations - %i min %i sec\n", maxIt, minutes, seconds);
         fprintf(outputFile, "GPU - %i BE computations - %i min %i sec\n", maxIt, minutes, seconds);
         printf("BE  ratio - %g time(s)\n", (float)cpuTime/(float)gpuTime);
         fprintf(outputFile, "BE  ratio - %g time(s)\n\n", (float)cpuTime/(float)gpuTime);
      }
#endif
      printf("BE done\n\n");
   }

   // APPROXIMATED BENDING ENERGY GRADIENT COMPUTATION
   {
      maxIt=1000000 / dimension;
//        maxIt=1;
      time(&start);
      for(int i=0; i<maxIt; ++i)
      {
         reg_bspline_bendingEnergyGradient<float>(   controlPointImage,
               targetImage,
               nodeNMIGradientImage,
               0.01f);
      }
      time(&end);
      cpuTime=(end-start);
      minutes = (int)floorf(float(cpuTime)/60.0f);
      seconds = (int)(cpuTime - 60*minutes);
      printf("CPU - %i BE gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
      fprintf(outputFile, "CPU - %i BE gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
#ifdef _USE_CUDA
      if(runGPU)
      {
         time(&start);
         for(int i=0; i<maxIt; ++i)
         {
            reg_bspline_ApproxBendingEnergyGradient_gpu(targetImage,
                  controlPointImage,
                  &controlPointImageArray_d,
                  &nodeNMIGradientArray_d,
                  0.01f);
         }
         time(&end);
         gpuTime=(end-start);
         minutes = (int)floorf(float(gpuTime)/60.0f);
         seconds = (int)(gpuTime - 60*minutes);
         printf("GPU - %i BE gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
         fprintf(outputFile, "GPU - %i BE gradient computations - %i min %i sec\n", maxIt, minutes, seconds);
         printf("BE gradient ratio - %g time(s)\n", (float)cpuTime/(float)gpuTime);
         fprintf(outputFile, "BE gradient ratio - %g time(s)\n\n", (float)cpuTime/(float)gpuTime);
      }
#endif
      printf("BE gradient done\n\n");
   }

   // JACOBIAN DETERMINANT PENALTY TERM COMPUTATION
   {
      maxIt=10000 / dimension;
      time(&start);
      for(int i=0; i<maxIt; ++i)
      {
         reg_bspline_jacobian<float>(controlPointImage,targetImage,0);
      }
      time(&end);
      cpuTime=(end-start);
      minutes = (int)floorf(float(cpuTime)/60.0f);
      seconds = (int)(cpuTime - 60*minutes);
      printf("CPU - %i |Jac| penalty term computations - %i min %i sec\n", maxIt, minutes, seconds);
      fprintf(outputFile, "CPU - %i |Jac| penalty term computations - %i min %i sec\n", maxIt, minutes, seconds);
#ifdef _USE_CUDA
      if(runGPU)
      {
         time(&start);
         for(int i=0; i<maxIt; ++i)
         {
            reg_bspline_ComputeJacobianPenaltyTerm_gpu(targetImage,controlPointImage,&controlPointImageArray_d,0);
         }
         time(&end);
         gpuTime=(end-start);
         minutes = (int)floorf(float(gpuTime)/60.0f);
         seconds = (int)(gpuTime - 60*minutes);
         printf("GPU - %i |Jac| penalty term computations - %i min %i sec\n", maxIt, minutes, seconds);
         fprintf(outputFile, "GPU - %i |Jac| penalty term computations - %i min %i sec\n", maxIt, minutes, seconds);
         printf("|Jac| penalty term ratio - %g time(s)\n", (float)cpuTime/(float)gpuTime);
         fprintf(outputFile, "|Jac| penalty term ratio - %g time(s)\n\n", (float)cpuTime/(float)gpuTime);
      }
#endif
      printf("|Jac| penalty term done\n\n");
   }

   // APPROXIMATED JACOBIAN DETERMINANT PENALTY TERM COMPUTATION
   {
      maxIt=1000000 / dimension;
      time(&start);
      for(int i=0; i<maxIt; ++i)
      {
         reg_bspline_jacobian<float>(controlPointImage,targetImage,1);
      }
      time(&end);
      cpuTime=(end-start);
      minutes = (int)floorf(float(cpuTime)/60.0f);
      seconds = (int)(cpuTime - 60*minutes);
      printf("CPU - %i Approx. |Jac| penalty term computations - %i min %i sec\n", maxIt, minutes, seconds);
      fprintf(outputFile, "CPU - %i Approx. |Jac| penalty term computations - %i min %i sec\n", maxIt, minutes, seconds);
#ifdef _USE_CUDA
      if(runGPU)
      {
         time(&start);
         for(int i=0; i<maxIt; ++i)
         {
            reg_bspline_ComputeJacobianPenaltyTerm_gpu(targetImage,controlPointImage,&controlPointImageArray_d,1);
         }
         time(&end);
         gpuTime=(end-start);
         minutes = (int)floorf(float(gpuTime)/60.0f);
         seconds = (int)(gpuTime - 60*minutes);
         printf("GPU - %i Approx. |Jac| penalty term computations - %i min %i sec\n", maxIt, minutes, seconds);
         fprintf(outputFile, "GPU - %i Approx. |Jac| penalty term computations - %i min %i sec\n", maxIt, minutes, seconds);
         printf("Approx. |Jac| penalty term ratio - %g time(s)\n", (float)cpuTime/(float)gpuTime);
         fprintf(outputFile, "Approx. |Jac| penalty term ratio - %g time(s)\n\n", (float)cpuTime/(float)gpuTime);
      }
#endif
      printf("Approx. |Jac| penalty term done\n\n");
   }

#ifdef _USE_CUDA
   if(runGPU)
   {
      cudaCommon_free( (void **)&controlPointImageArray_d );
   }
#endif

   // BLOCK MATCHING
   {
      maxIt=2000 / dimension;
//        maxIt=1;
      _reg_blockMatchingParam blockMatchingParams;
      initialise_block_matching_method(   targetImage,
                                          &blockMatchingParams,
                                          100,    // percentage of block kept
                                          50,     // percentage of inlier in the optimisation process
                                          maskImage);
#ifdef _USE_CUDA
      int *activeBlock_d;
      float *targetPosition_d;
      float *resultPosition_d;
      if(runGPU)
      {
         CUDA_SAFE_CALL(cudaMalloc((void **)&activeBlock_d,
                                   blockMatchingParams.blockNumber[0]*blockMatchingParams.blockNumber[1]*blockMatchingParams.blockNumber[2]*sizeof(int)));
         CUDA_SAFE_CALL(cudaMemcpy(activeBlock_d, blockMatchingParams.activeBlock,
                                   blockMatchingParams.blockNumber[0]*blockMatchingParams.blockNumber[1]*blockMatchingParams.blockNumber[2]*sizeof(int),
                                   cudaMemcpyHostToDevice));
         CUDA_SAFE_CALL(cudaMalloc((void **)&targetPosition_d, blockMatchingParams.activeBlockNumber*3*sizeof(float)));
         CUDA_SAFE_CALL(cudaMalloc((void **)&resultPosition_d, blockMatchingParams.activeBlockNumber*3*sizeof(float)));
      }
#endif
      time(&start);
      for(int i=0; i<maxIt; ++i)
      {
         block_matching_method<float>(   targetImage,
                                         resultImage,
                                         &blockMatchingParams,
                                         maskImage);
      }
      time(&end);
      cpuTime=(end-start);
      minutes = (int)floorf(float(cpuTime)/60.0f);
      seconds = (int)(cpuTime - 60*minutes);
      printf("CPU - %i block matching computations - %i min %i sec\n", maxIt, minutes, seconds);
      fprintf(outputFile, "CPU - %i block matching computations - %i min %i sec\n", maxIt, minutes, seconds);
#ifdef _USE_CUDA
      if(runGPU)
      {
         time(&start);
         for(int i=0; i<maxIt; ++i)
         {
            block_matching_method_gpu(  targetImage,
                                        resultImage,
                                        &blockMatchingParams,
                                        &targetImageArray_d,
                                        &resultImageArray_d,
                                        &targetPosition_d,
                                        &resultPosition_d,
                                        &activeBlock_d);
         }
         time(&end);
         gpuTime=(end-start);
         minutes = (int)floorf(float(gpuTime)/60.0f);
         seconds = (int)(gpuTime - 60*minutes);
         printf("GPU - %i block matching computations - %i min %i sec\n", maxIt, minutes, seconds);
         fprintf(outputFile, "GPU - %i block matching computations - %i min %i sec\n", maxIt, minutes, seconds);
         printf("Block-Matching ratio - %g time(s)\n", (float)cpuTime/(float)gpuTime);
         fprintf(outputFile, "Block-Matching ratio - %g time(s)\n\n", (float)cpuTime/(float)gpuTime);
         cudaCommon_free((void **)&targetPosition_d);
         cudaCommon_free((void **)&resultPosition_d);
         cudaCommon_free((void **)&activeBlock_d);
      }
#endif
      printf("Block-matching done\n");
   }

   fclose(outputFile);

   /* Monsieur Propre */
   nifti_image_free(targetImage);
   nifti_image_free(resultImage);
   nifti_image_free(controlPointImage);
   nifti_image_free(deformationFieldImage);
   nifti_image_free(resultGradientImage);
   nifti_image_free(voxelNMIGradientImage);
   nifti_image_free(nodeNMIGradientImage);
   free(maskImage);
   free(probaJointHistogram);
   free(logJointHistogram);

#ifdef _USE_CUDA
   if(runGPU)
   {
      cudaCommon_free( (void **)&targetImageArray_d );
      cudaCommon_free( (void **)&resultImageArray_d );
   }
#endif

   return 0;
}

void Usage(char *exec)
{
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   printf("Usage:\t%s [OPTIONS].\n",exec);
   printf("\t-dim <int>\tImage dimension [100]\n");
   printf("\t-bin <int>\tBin number [68]\n");
   printf("\t-sp <float>\tControl point grid spacing [10]\n");
   printf("\t-o <char*>\t Output file name [benchmark_result.txt]\n");
   printf("\t-cpu\t\t Run only the CPU function\n");
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   return;
}
