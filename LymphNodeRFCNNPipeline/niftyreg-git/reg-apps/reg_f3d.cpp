/*
 *  reg_f3d.cpp
 *
 *
 *  Created by Marc Modat on 26/03/2009.
 *  Copyright (c) 2009, University College London. All rights reserved.
 *  Centre for Medical Image Computing (CMIC)
 *  See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#include "_reg_ReadWriteImage.h"
#include "_reg_f3d2.h"
#include "reg_f3d.h"
#include <float.h>
#include <limits>

#ifdef _USE_CUDA
#   include "_reg_f3d_gpu.h"
#endif

#ifdef _WIN32
#   include <time.h>
#endif

#ifdef _USE_NR_DOUBLE
#   define PrecisionTYPE double
#else
#   define PrecisionTYPE float
#endif

void HelpPenaltyTerm()
{
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   printf("Additional help on the penalty term that have been implemented in F3D\n");
   printf("\t-be\t Bending Energy, sum of the second derivatives of the transformation T\n");
   printf("\t\t\t (d2T/dxx)^2 + (d2T/dyy)^2 + (d2T/dzz)^2 + 2*((d2T/dxy)^2 + (d2T/dyz)^2 + (d2T/dxz)^2)\n");
   printf("\t-le\t Linear Elasticity, 2 parameters weighted differently:\n");
   printf("\t\t\t 1: Squared member of the symmetric part of the Jacobian matrix\n");
   printf("\t\t\t 1: (dTx/dx)^2 + (dTy/dy)^2 + (dTz/dz)^2 + 1/2 * ( (dTx/dy+dTy/dx)^2 +  (dTx/dz+dTz/dx)^2 +  (dTy/dz+dTz/dy)^2 ) \n");
   printf("\t\t\t 2: Divergence\n");
   printf("\t\t\t 2: (dTx/dx)^2 + (dTy/dy)^2 + (dTz/dz)^2\n");
   printf("\t-l2\t Squared Eucliean distance of the displacement field D\n");
   printf("\t\t\t (Dx)^2 + (Dy)^2 + (Dz)^2\n");
   printf("\t-jl\t Penalty term based on the Jacobian determiant |J(T)|. Squared log of the Jacobian determinant\n");
   printf("\t\t\t log^2(|J(T)|)\n");
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   return;
}
void PetitUsage(char *exec)
{
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   fprintf(stderr,"Fast Free-Form Deformation algorithm for non-rigid registration.\n");
   fprintf(stderr,"Usage:\t%s -ref <targetImageName> -flo <sourceImageName> [OPTIONS].\n",exec);
   fprintf(stderr,"\tSee the help for more details (-h).\n");
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   return;
}
void Usage(char *exec)
{
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   printf("Fast Free-Form Deformation algorithm for non-rigid registration.\n");
   printf("This implementation is a re-factoring of Daniel Rueckert' 99 TMI work.\n");
   printf("The code is presented in Modat et al., \"Fast Free-Form Deformation using\n");
   printf("graphics processing units\", CMPB, 2010\n");
   printf("Cubic B-Spline are used to deform a source image in order to optimise a objective function\n");
   printf("based on the Normalised Mutual Information and a penalty term. The penalty term could\n");
   printf("be either the bending energy or the squared Jacobian determinant log.\n");
   printf("This code has been written by Marc Modat (m.modat@ucl.ac.uk), for any comment,\n");
   printf("please contact him.\n");
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   printf("Usage:\t%s -ref <filename> -flo <filename> [OPTIONS].\n",exec);
   printf("\t-ref <filename>\tFilename of the reference image (mandatory)\n");
   printf("\t-flo <filename>\tFilename of the floating image (mandatory)\n");
   printf("\n***************\n*** OPTIONS ***\n***************\n");
   printf("*** Initial transformation options (One option will be considered):\n");
   printf("\t-aff <filename>\t\tFilename which contains an affine transformation (Affine*Reference=Floating)\n");
   printf("\t-incpp <filename>\tFilename ofloatf control point grid input\n\t\t\t\tThe coarse spacing is defined by this file.\n");

   printf("\n*** Output options:\n");
   printf("\t-cpp <filename>\t\tFilename of control point grid [outputCPP.nii]\n");
   printf("\t-res <filename> \tFilename of the resampled image [outputResult.nii]\n");

   printf("\n*** Input image options:\n");
   printf("\t-rmask <filename>\t\tFilename of a mask image in the reference space\n");
   printf("\t-smooR <float>\t\t\tSmooth the reference image using the specified sigma (mm) [0]\n");
   printf("\t-smooF <float>\t\t\tSmooth the floating image using the specified sigma (mm) [0]\n");
   printf("\t--rLwTh <float>\t\t\tLower threshold to apply to the reference image intensities [none]. Identical value for every timepoint.*\n");
   printf("\t--rUpTh <float>\t\t\tUpper threshold to apply to the reference image intensities [none]. Identical value for every timepoint.*\n");
   printf("\t--fLwTh <float>\t\t\tLower threshold to apply to the floating image intensities [none]. Identical value for every timepoint.*\n");
   printf("\t--fUpTh <float>\t\t\tUpper threshold to apply to the floating image intensities [none]. Identical value for every timepoint.*\n");
   printf("\t-rLwTh <timepoint> <float>\tLower threshold to apply to the reference image intensities [none]*\n");
   printf("\t-rUpTh <timepoint> <float>\tUpper threshold to apply to the reference image intensities [none]*\n");
   printf("\t-fLwTh <timepoint> <float>\tLower threshold to apply to the floating image intensities [none]*\n");
   printf("\t-fUpTh <timepoint> <float>\tUpper threshold to apply to the floating image intensities [none]*\n");
   printf("\t* The scl_slope and scl_inter from the nifti header are taken into account for the thresholds\n");

   printf("\n*** Spline options:\n");
   printf("\t-sx <float>\t\tFinal grid spacing along the x axis in mm (in voxel if negative value) [5 voxels]\n");
   printf("\t-sy <float>\t\tFinal grid spacing along the y axis in mm (in voxel if negative value) [sx value]\n");
   printf("\t-sz <float>\t\tFinal grid spacing along the z axis in mm (in voxel if negative value) [sx value]\n");

   printf("\n*** Regularisation options:\n");
   printf("\t-be <float>\t\tWeight of the bending energy penalty term [0.005]\n");
   printf("\t-le <float> <float>\tWeights of linear elasticity penalty term [0.0 0.0]\n");
   printf("\t-l2 <float>\t\tWeights of L2 norm displacement penalty term [0.0]\n");
   printf("\t-jl <float>\t\tWeight of log of the Jacobian determinant penalty term [0.0]\n");
   printf("\t-noAppJL\t\tTo not approximate the JL value only at the control point position\n");



   printf("\n*** Measure of similarity options:\n");
   printf("*** NMI with 64 bins is used expect if specified otherwise\n");
   printf("\t--nmi\t\t\tNMI. Used NMI even when one or several other measures are specified.\n");
   printf("\t--rbn <int>\t\tNMI. Number of bin to use for the reference image histogram. Identical value for every timepoint.\n");
   printf("\t--fbn <int>\t\tNMI. Number of bin to use for the floating image histogram. Identical value for every timepoint.\n");
   printf("\t-rbn <tp> <int>\t\tNMI. Number of bin to use for the reference image histogram for the specified time point.\n");
   printf("\t-rbn <tp> <int>\t\tNMI. Number of bin to use for the floating image histogram for the specified time point.\n");

   printf("\t--lncc <float>\t\tLNCC. Standard deviation of the Gaussian kernel. Identical value for every timepoint\n");
   printf("\t-lncc <tp> <float>\tLNCC. Standard deviation of the Gaussian kernel for the specified timepoint\n");

   printf("\t--ssd\t\t\tSSD. Used for all time points\n");
   printf("\t-ssd <tp>\t\tSSD. Used for the specified timepoint\n");

   printf("\t--kld\t\t\tKLD. Used for all time points\n");
   printf("\t-kld <tp>\t\tKLD. Used for the specified timepoint\n");
   printf("\t* For the Kullback–Leibler divergence, reference and floating are expected to be probabilities\n");

   printf("\t-amc\t\t\tTo use the additive NMI for multichannel data (bivariate NMI by default)\n");

   printf("\n*** Optimisation options:\n");
   printf("\t-maxit <int>\t\tMaximal number of iteration per level [300]\n");
   printf("\t-ln <int>\t\tNumber of level to perform [3]\n");
   printf("\t-lp <int>\t\tOnly perform the first levels [ln]\n");
   printf("\t-nopy\t\t\tDo not use a pyramidal approach\n");
   printf("\t-noConj\t\t\tTo not use the conjuage gradient optimisation but a simple gradient ascent\n");
   printf("\t-pert <int>\t\tTo add perturbation step(s) after each optimisation scheme\n");

//   printf("\n*** F3D_SYM options:\n");
//   printf("\t-sym \t\t\tUse symmetric approach\n");
//   printf("\t-ic <float>\t\tWeight of the inverse consistency penalty term [0.01]\n");

   printf("\n*** F3D2 options:\n");
   printf("\t-vel \t\t\tUse a velocity field integration to generate the deformation\n");
   printf("\t-fmask <filename>\tFilename of a mask image in the floating space\n");

#if defined (_OPENMP)
   printf("\n*** OpenMP-related options:\n");
   printf("\t-omp <int>\t\tNumber of thread to use with OpenMP. [%i]\n",
          omp_get_num_procs());
#endif
#ifdef _USE_CUDA
   printf("\n*** GPU-related options:\n");
   printf("\t-mem\t\t\tDisplay an approximate memory requierment and exit\n");
   printf("\t-gpu \t\t\tTo use the GPU implementation [no]\n");
#endif
   printf("\n*** Other options:\n");
   printf("\t-smoothGrad <float>\tTo smooth the metric derivative (in mm) [0]\n");
   printf("\t-pad <float>\t\tPadding value [nan]\n");
   printf("\t-voff\t\t\tTo turn verbose off\n");

#ifdef _GIT_HASH
   printf("\n\t--version\t\tPrint current source code git hash key and exit\n\t\t\t\t(%s)\n",_GIT_HASH);
#endif
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   printf("For further description of the penalty term, use: %s -helpPenalty\n", exec);
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   return;
}

int main(int argc, char **argv)
{
   if(argc==1)
   {
      PetitUsage(argv[0]);
      return 1;
   }
   time_t start;
   time(&start);
   int verbose=true;

   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
   // Check if any information is required
   for(int i=1; i<argc; i++)
   {
      if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
            strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
            strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0)
      {
         Usage(argv[0]);
         return 0;
      }
      if(strcmp(argv[i], "--xml")==0)
      {
         printf("%s",xml_f3d);
         return 0;
      }
      if(strcmp(argv[i], "-voff")==0)
      {
         verbose=false;
      }
#ifdef _GIT_HASH
      if( strcmp(argv[i], "-version")==0 ||
            strcmp(argv[i], "-Version")==0 ||
            strcmp(argv[i], "-V")==0 ||
            strcmp(argv[i], "-v")==0 ||
            strcmp(argv[i], "--v")==0 ||
            strcmp(argv[i], "--version")==0)
      {
         printf("%s\n",_GIT_HASH);
         return EXIT_SUCCESS;
      }
#endif
      if(strcmp(argv[i], "-helpPenalty")==0)
      {
         HelpPenaltyTerm();
         return 0;
      }
   }
   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
   // Output the command line
#ifdef NDEBUG
   if(verbose)
   {
#endif
      printf("\n[NiftyReg F3D] Command line:\n\t");
      for(int i=0; i<argc; i++)
         printf(" %s", argv[i]);
      printf("\n\n");
#ifdef NDEBUG
   }
#endif
   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
   // Read the reference and floating image
   nifti_image *referenceImage=NULL;
   nifti_image *floatingImage=NULL;
   for(int i=1; i<argc; i++)
   {
      if((strcmp(argv[i],"-ref")==0) || (strcmp(argv[i],"-target")==0) || (strcmp(argv[i],"--ref")==0))
      {
         referenceImage=reg_io_ReadImageFile(argv[++i]);
         if(referenceImage==NULL)
         {
            fprintf(stderr, "Error when reading the reference image %s\n",argv[i-1]);
            return 1;
         }
      }
      if((strcmp(argv[i],"-flo")==0) || (strcmp(argv[i],"-source")==0) || (strcmp(argv[i],"--flo")==0))
      {
         floatingImage=reg_io_ReadImageFile(argv[++i]);
         if(floatingImage==NULL)
         {
            fprintf(stderr, "Error when reading the floating image %s\n",argv[i-1]);
            return 1;
         }
      }
   }
   // Check that both reference and floating image have been defined
   if(referenceImage==NULL)
   {
      fprintf(stderr, "Error. No reference image has been defined\n");
      PetitUsage(argv[0]);
      return 1;
   }
   // Read the floating image
   if(floatingImage==NULL)
   {
      fprintf(stderr, "Error. No floating image has been defined\n");
      PetitUsage(argv[0]);
      return 1;
   }
   //\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
   // Check the type of registration object to create
#ifdef _USE_CUDA
   CUcontext ctx;
#endif // _USE_CUDA
   reg_f3d<PrecisionTYPE> *REG=NULL;
   for(int i=1; i<argc; i++)
   {
      if(strcmp(argv[i], "-vel")==0 || strcmp(argv[i], "--vel")==0)
      {
         REG=new reg_f3d2<PrecisionTYPE>(referenceImage->nt,floatingImage->nt);
         break;
      }
      if(strcmp(argv[i], "-sym")==0 || strcmp(argv[i], "--sym")==0)
      {
         REG=new reg_f3d_sym<PrecisionTYPE>(referenceImage->nt,floatingImage->nt);
         break;
      }
#ifdef _USE_CUDA
      if(strcmp(argv[i], "-gpu")==0 || strcmp(argv[i], "-mem")==0)
      {
         // Set up the cuda card and display some relevant information and check if the card is suitable
         if(cudaCommon_setCUDACard(&ctx, true))
         {
            fprintf(stderr,"\n[NiftyReg CUDA ERROR] Error while detecting a CUDA card\n");
            fprintf(stderr,"[NiftyReg CUDA WARNING] GPU implementation has been turned off.\n");
         }
         else REG=new reg_f3d_gpu(referenceImage->nt,floatingImage->nt);
         break;
      }
#endif // _USE_CUDA
   }
   if(REG==NULL)
      REG=new reg_f3d<PrecisionTYPE>(referenceImage->nt,floatingImage->nt);
   REG->SetReferenceImage(referenceImage);
   REG->SetFloatingImage(floatingImage);

   // Create some pointers that could be used
   mat44 affineMatrix;
   nifti_image *inputCCPImage=NULL;
   nifti_image *referenceMaskImage=NULL;
   nifti_image *floatingMaskImage=NULL;
   char *outputWarpedImageName=NULL;
   char *outputCPPImageName=NULL;
   bool useMeanLNCC=false;
#ifdef _USE_CUDA
   bool checkMemory=false;
#endif // _use_CUDA
   int refBinNumber=0;
   int floBinNumber=0;

   /* read the input parameter */
   for(int i=1; i<argc; i++)
   {

      if(strcmp(argv[i],"-ref")==0 || strcmp(argv[i],"-target")==0 ||
            strcmp(argv[i],"--ref")==0 || strcmp(argv[i],"-flo")==0 ||
            strcmp(argv[i],"-source")==0 || strcmp(argv[i],"--flo")==0 )
      {
         // argument has already been parsed
         ++i;
      }
      else if(strcmp(argv[i], "-voff")==0)
      {
         verbose=false;
         REG->DoNotPrintOutInformation();
      }
      else if(strcmp(argv[i], "-aff")==0 || (strcmp(argv[i],"--aff")==0))
      {
         // Check first if the specified affine file exist
         char *affineTransformationName=argv[++i];
         if(FILE *aff=fopen(affineTransformationName, "r"))
         {
            fclose(aff);
         }
         else
         {
            fprintf(stderr,"The specified input affine file (%s) can not be read\n",
                    affineTransformationName);
            return 1;
         }
         // Read the affine matrix
         reg_tool_ReadAffineFile(&affineMatrix,
                                 affineTransformationName);
         // Send the transformation to the registration object
         REG->SetAffineTransformation(&affineMatrix);
      }
      else if(strcmp(argv[i], "-incpp")==0 || (strcmp(argv[i],"--incpp")==0))
      {
         inputCCPImage=reg_io_ReadImageFile(argv[++i]);
         if(inputCCPImage==NULL)
         {
            fprintf(stderr, "Error when reading the input control point grid image: %s\n",argv[i-1]);
            return 1;
         }
         REG->SetControlPointGridImage(inputCCPImage);
      }
      else if((strcmp(argv[i],"-rmask")==0) || (strcmp(argv[i],"-tmask")==0) || (strcmp(argv[i],"--rmask")==0))
      {
         referenceMaskImage=reg_io_ReadImageFile(argv[++i]);
         if(referenceMaskImage==NULL)
         {
            fprintf(stderr, "Error when reading the reference mask image: %s\n",argv[i-1]);
            return 1;
         }
         REG->SetReferenceMask(referenceMaskImage);
      }
      else if((strcmp(argv[i],"-res")==0) || (strcmp(argv[i],"-result")==0) || (strcmp(argv[i],"--res")==0))
      {
         outputWarpedImageName=argv[++i];
      }
      else if(strcmp(argv[i], "-cpp")==0 || (strcmp(argv[i],"--cpp")==0))
      {
         outputCPPImageName=argv[++i];
      }
      else if(strcmp(argv[i], "-maxit")==0 || strcmp(argv[i], "--maxit")==0)
      {
         REG->SetMaximalIterationNumber(atoi(argv[++i]));
      }
      else if(strcmp(argv[i], "-sx")==0 || strcmp(argv[i], "--sx")==0)
      {
         REG->SetSpacing(0,(float)atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-sy")==0 || strcmp(argv[i], "--sy")==0)
      {
         REG->SetSpacing(1,(float)atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-sz")==0 || strcmp(argv[i], "--sz")==0)
      {
         REG->SetSpacing(2,(float)atof(argv[++i]));
      }
      else if((strcmp(argv[i],"--nmi")==0) )
      {
         int bin=64;
         if(refBinNumber!=0)
            bin=refBinNumber;
         for(int t=0; t<referenceImage->nt; ++t)
            REG->UseNMISetReferenceBinNumber(t,bin);
         bin=64;
         if(floBinNumber!=0)
            bin=floBinNumber;
         for(int t=0; t<floatingImage->nt; ++t)
            REG->UseNMISetFloatingBinNumber(t,bin);
      }
      else if((strcmp(argv[i],"-rbn")==0) || (strcmp(argv[i],"-tbn")==0))
      {
         int tp=atoi(argv[++i]);
         int bin=atoi(argv[++i]);
         refBinNumber=bin;
         REG->UseNMISetReferenceBinNumber(tp,bin);
      }
      else if((strcmp(argv[i],"--rbn")==0) )
      {
         int bin = atoi(argv[++i]);
         refBinNumber=bin;
         for(int t=0; t<referenceImage->nt; ++t)
            REG->UseNMISetReferenceBinNumber(t,bin);
      }
      else if((strcmp(argv[i],"-fbn")==0) || (strcmp(argv[i],"-sbn")==0))
      {
         int tp=atoi(argv[++i]);
         int bin=atoi(argv[++i]);
         floBinNumber=bin;
         REG->UseNMISetFloatingBinNumber(tp,bin);
      }
      else if((strcmp(argv[i],"--fbn")==0) )
      {
         int bin = atoi(argv[++i]);
         floBinNumber=bin;
         for(int t=0; t<floatingImage->nt; ++t)
            REG->UseNMISetFloatingBinNumber(t,bin);
      }
      else if(strcmp(argv[i], "-ln")==0 || strcmp(argv[i], "--ln")==0)
      {
         REG->SetLevelNumber(atoi(argv[++i]));
      }
      else if(strcmp(argv[i], "-lp")==0 || strcmp(argv[i], "--lp")==0)
      {
         REG->SetLevelToPerform(atoi(argv[++i]));
      }
      else if(strcmp(argv[i], "-be")==0 || strcmp(argv[i], "--be")==0)
      {
         REG->SetBendingEnergyWeight(atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-le")==0)
      {
         float val1=atof(argv[++i]);
         float val2=atof(argv[++i]);
         REG->SetLinearEnergyWeights(val1,val2);
      }
      else if(strcmp(argv[i], "-l2")==0 || strcmp(argv[i], "--l2")==0)
      {
         REG->SetL2NormDisplacementWeight(atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-jl")==0 || strcmp(argv[i], "--jl")==0)
      {
         REG->SetJacobianLogWeight(atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-noAppJL")==0 || strcmp(argv[i], "--noAppJL")==0)
      {
         REG->DoNotApproximateJacobianLog();
      }
      else if((strcmp(argv[i],"-smooR")==0) || (strcmp(argv[i],"-smooT")==0) || strcmp(argv[i], "--smooR")==0)
      {
         REG->SetReferenceSmoothingSigma(atof(argv[++i]));
      }
      else if((strcmp(argv[i],"-smooF")==0) || (strcmp(argv[i],"-smooS")==0) || strcmp(argv[i], "--smooF")==0)
      {
         REG->SetFloatingSmoothingSigma(atof(argv[++i]));
      }
      else if((strcmp(argv[i],"-rLwTh")==0) || (strcmp(argv[i],"-tLwTh")==0))
      {
         int tp=atoi(argv[++i]);
         float val=atof(argv[++i]);
         REG->SetReferenceThresholdLow(tp,val);
      }
      else if((strcmp(argv[i],"-rUpTh")==0) || strcmp(argv[i],"-tUpTh")==0)
      {
         int tp=atoi(argv[++i]);
         float val=atof(argv[++i]);
         REG->SetReferenceThresholdUp(tp,val);
      }
      else if((strcmp(argv[i],"-fLwTh")==0) || (strcmp(argv[i],"-sLwTh")==0))
      {
         int tp=atoi(argv[++i]);
         float val=atof(argv[++i]);
         REG->SetFloatingThresholdLow(tp,val);
      }
      else if((strcmp(argv[i],"-fUpTh")==0) || (strcmp(argv[i],"-sUpTh")==0))
      {
         int tp=atoi(argv[++i]);
         float val=atof(argv[++i]);
         REG->SetFloatingThresholdUp(tp,val);
      }
      else if((strcmp(argv[i],"--rLwTh")==0) )
      {
         float threshold = atof(argv[++i]);
         for(int t=0; t<referenceImage->nt; ++t)
            REG->SetReferenceThresholdLow(t,threshold);
      }
      else if((strcmp(argv[i],"--rUpTh")==0) )
      {
         float threshold = atof(argv[++i]);
         for(int t=0; t<referenceImage->nt; ++t)
            REG->SetReferenceThresholdUp(t,threshold);
      }
      else if((strcmp(argv[i],"--fLwTh")==0) )
      {
         float threshold = atof(argv[++i]);
         for(int t=0; t<floatingImage->nt; ++t)
            REG->SetFloatingThresholdLow(t,threshold);
      }
      else if((strcmp(argv[i],"--fUpTh")==0) )
      {
         float threshold = atof(argv[++i]);
         for(int t=0; t<floatingImage->nt; ++t)
            REG->SetFloatingThresholdUp(t,threshold);
      }
      else if(strcmp(argv[i], "-smoothGrad")==0)
      {
         REG->SetGradientSmoothingSigma(atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-ssd")==0)
      {
         REG->UseSSD(atoi(argv[++i]));
      }
      else if(strcmp(argv[i], "--ssd")==0)
      {
         for(int t=0; t<floatingImage->nt; ++t)
            REG->UseSSD(t);
      }
      else if(strcmp(argv[i], "-kld")==0)
      {
         REG->UseKLDivergence(atoi(argv[++i]));
      }
      else if(strcmp(argv[i], "--kld")==0)
      {
         for(int t=0; t<floatingImage->nt; ++t)
            REG->UseKLDivergence(t);
      }
//        else if(strcmp(argv[i], "-amc")==0){ // HERE TODO
//            REG->UseMultiChannelNMI();
//        }
      else if(strcmp(argv[i], "-lncc")==0)
      {
         int tp=atoi(argv[++i]);
         float stdev = atof(argv[++i]);
         REG->UseLNCC(tp,stdev);
      }
      else if(strcmp(argv[i], "--lncc")==0)
      {
         float stdev = (float)atof(argv[++i]);
         if(stdev!=999999){ // Value specified by the CLI - to be ignored
            for(int t=0; t<referenceImage->nt; ++t)
               REG->UseLNCC(t,stdev);
         }
      }
      else if(strcmp(argv[i], "-lnccMean")==0)
      {
         useMeanLNCC=true;
      }
      else if(strcmp(argv[i], "-dti")==0 || strcmp(argv[i], "--dti")==0)
      {
         bool *timePoint = new bool[referenceImage->nt];
         for(int t=0; t<referenceImage->nt; ++t)
            timePoint[t]=false;
         timePoint[atoi(argv[++i])]=true;
         timePoint[atoi(argv[++i])]=true;
         timePoint[atoi(argv[++i])]=true;
         if(referenceImage->nz>1)
         {
            timePoint[atoi(argv[++i])]=true;
            timePoint[atoi(argv[++i])]=true;
            timePoint[atoi(argv[++i])]=true;
         }
         REG->UseDTI(timePoint);
         delete []timePoint;
      }
      else if(strcmp(argv[i], "-pad")==0)
      {
         REG->SetWarpedPaddingValue(atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-nopy")==0 || strcmp(argv[i], "--nopy")==0)
      {
         REG->DoNotUsePyramidalApproach();
      }
      else if(strcmp(argv[i], "-noConj")==0 || strcmp(argv[i], "--noConj")==0)
      {
         REG->DoNotUseConjugateGradient();
      }
      else if(strcmp(argv[i], "-approxGrad")==0 || strcmp(argv[i], "--approxGrad")==0)
      {
         REG->UseApproximatedGradient();
      }
      else if(strcmp(argv[i], "-interp")==0 || strcmp(argv[i], "--interp")==0)
      {
         int interp=atoi(argv[++i]);
         switch(interp)
         {
         case 0:
            REG->UseNeareatNeighborInterpolation();
            break;
         case 1:
            REG->UseLinearInterpolation();
            break;
         default:
            REG->UseCubicSplineInterpolation();
            break;
         }
      }
//        else if(strcmp(argv[i], "-noAppPW")==0){ // HERE TODO
//            parzenWindowApproximation=false;
//        }
      else if((strcmp(argv[i],"-fmask")==0) || (strcmp(argv[i],"-smask")==0) ||
              (strcmp(argv[i],"--fmask")==0) || (strcmp(argv[i],"--smask")==0))
      {
         floatingMaskImage=reg_io_ReadImageFile(argv[++i]);
         if(floatingMaskImage==NULL)
         {
            fprintf(stderr, "Error when reading the floating mask image: %s\n",argv[i-1]);
            return 1;
         }
         REG->SetFloatingMask(floatingMaskImage);
      }
      else if(strcmp(argv[i], "-ic")==0 || strcmp(argv[i], "--ic")==0)
      {
         REG->SetInverseConsistencyWeight(atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-nox") ==0)
      {
         REG->NoOptimisationAlongX();
      }
      else if(strcmp(argv[i], "-noy") ==0)
      {
         REG->NoOptimisationAlongY();
      }
      else if(strcmp(argv[i], "-noz") ==0)
      {
         REG->NoOptimisationAlongZ();
      }
      else if(strcmp(argv[i],"-pert")==0 || strcmp(argv[i],"--pert")==0)
      {
         REG->SetPerturbationNumber((size_t)atoi(argv[++i]));
      }
      else if(strcmp(argv[i], "-nogr") ==0)
      {
         REG->NoGridRefinement();
      }
      else if(strcmp(argv[i], "-gce")==0 || strcmp(argv[i], "--gce")==0)
      {
         REG->UseGradientCumulativeExp();
      }
      else if(strcmp(argv[i], "-bch")==0 || strcmp(argv[i], "--bch")==0)
      {
         REG->UseBCHUpdate(atoi(argv[++i]));
      }
//        else if(strcmp(argv[i], "-iso")==0 || strcmp(argv[i], "--iso")==0){
//            iso=true;
//        }
#if defined (_OPENMP)
      else if(strcmp(argv[i], "-omp")==0 || strcmp(argv[i], "--omp")==0)
      {
         omp_set_num_threads(atoi(argv[++i]));
      }
#endif
#ifdef _USE_CUDA
      else if(strcmp(argv[i], "-mem")==0)
      {
         checkMemory=true;
      }
#endif
      /* All the following arguments should have already been parsed */
      else if(strcmp(argv[i], "-help")!=0 && strcmp(argv[i], "-Help")!=0 &&
      strcmp(argv[i], "-HELP")!=0 && strcmp(argv[i], "-h")!=0 &&
      strcmp(argv[i], "--h")!=0 && strcmp(argv[i], "--help")!=0 &&
      strcmp(argv[i], "--xml")!=0 && strcmp(argv[i], "-version")!=0 &&
      strcmp(argv[i], "-Version")!=0 && strcmp(argv[i], "-V")!=0 &&
      strcmp(argv[i], "-v")!=0 && strcmp(argv[i], "--v")!=0 &&
      strcmp(argv[i], "--version")!=0 && strcmp(argv[i], "-helpPenalty")!=0 &&
#ifdef _USE_CUDA
      strcmp(argv[i], "-gpu")!=0 &&
#endif
      strcmp(argv[i], "-vel")!=0 && strcmp(argv[i], "-sym")!=0)
      {
         fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
         PetitUsage(argv[0]);
         return 1;
      }
   }
   if(useMeanLNCC)
      REG->SetLNCCKernelType(2);

#ifndef NDEBUG
   printf("[NiftyReg DEBUG] *******************************************\n");
   printf("[NiftyReg DEBUG] *******************************************\n");
   printf("[NiftyReg DEBUG] NiftyReg has been compiled in DEBUG mode\n");
   printf("[NiftyReg DEBUG] Please re-run cmake to set the variable\n");
   printf("[NiftyReg DEBUG] CMAKE_BUILD_TYPE to \"Release\" if required\n");
   printf("[NiftyReg DEBUG] *******************************************\n");
   printf("[NiftyReg DEBUG] *******************************************\n");
#endif

#if defined (_OPENMP)
   if(verbose)
   {
      int maxThreadNumber = omp_get_max_threads();
      printf("[NiftyReg F3D] OpenMP is used with %i thread(s)\n", maxThreadNumber);
   }
#endif // _OPENMP

   // Run the registration
#ifdef _USE_CUDA
   if(checkMemory)
   {
      size_t free, total, requiredMemory = REG->CheckMemoryMB();
      cuMemGetInfo(&free, &total);
      printf("[NiftyReg CUDA] The required memory to run the registration is %lu Mb\n",
             (unsigned long int)requiredMemory);
      printf("[NiftyReg CUDA] The GPU card has %lu Mb from which %lu Mb are currenlty free\n",
             (unsigned long int)(total/(1024*1024)), (unsigned long int)(free/(1024*1024)));
   }
   else
   {
#endif
      REG->Run();

      // Save the control point result
      nifti_image *outputControlPointGridImage = REG->GetControlPointPositionImage();
      if(outputCPPImageName==NULL) outputCPPImageName=(char *)"outputCPP.nii";
      memset(outputControlPointGridImage->descrip, 0, 80);
      strcpy (outputControlPointGridImage->descrip,"Control point position from NiftyReg (reg_f3d)");
      if(strcmp("NiftyReg F3D2", REG->GetExecutableName())==0)
         strcpy (outputControlPointGridImage->descrip,"Velocity field grid from NiftyReg (reg_f3d2)");
      reg_io_WriteImageFile(outputControlPointGridImage,outputCPPImageName);
      nifti_image_free(outputControlPointGridImage);
      outputControlPointGridImage=NULL;

      // Save the backward control point result
      if(REG->GetSymmetricStatus())
      {
         // _backward is added to the forward control point grid image name
         std::string b(outputCPPImageName);
         if(b.find( ".nii.gz") != std::string::npos)
            b.replace(b.find( ".nii.gz"),7,"_backward.nii.gz");
         else if(b.find( ".nii") != std::string::npos)
            b.replace(b.find( ".nii"),4,"_backward.nii");
         else if(b.find( ".hdr") != std::string::npos)
            b.replace(b.find( ".hdr"),4,"_backward.hdr");
         else if(b.find( ".img.gz") != std::string::npos)
            b.replace(b.find( ".img.gz"),7,"_backward.img.gz");
         else if(b.find( ".img") != std::string::npos)
            b.replace(b.find( ".img"),4,"_backward.img");
         else if(b.find( ".png") != std::string::npos)
            b.replace(b.find( ".png"),4,"_backward.png");
         else if(b.find( ".nrrd") != std::string::npos)
            b.replace(b.find( ".nrrd"),5,"_backward.nrrd");
         else b.append("_backward.nii");
         nifti_image *outputBackwardControlPointGridImage = REG->GetBackwardControlPointPositionImage();
         memset(outputBackwardControlPointGridImage->descrip, 0, 80);
         strcpy (outputBackwardControlPointGridImage->descrip,"Backward Control point position from NiftyReg (reg_f3d)");
         if(strcmp("NiftyReg F3D2", REG->GetExecutableName())==0)
            strcpy (outputBackwardControlPointGridImage->descrip,"Backward velocity field grid from NiftyReg (reg_f3d2)");
         reg_io_WriteImageFile(outputBackwardControlPointGridImage,b.c_str());
         nifti_image_free(outputBackwardControlPointGridImage);
         outputBackwardControlPointGridImage=NULL;
      }

      // Save the warped image result(s)
      nifti_image **outputWarpedImage=(nifti_image **)malloc(2*sizeof(nifti_image *));
      outputWarpedImage[0]=NULL;
      outputWarpedImage[1]=NULL;
      outputWarpedImage = REG->GetWarpedImage();
      if(outputWarpedImageName==NULL)
         outputWarpedImageName=(char *)"outputResult.nii";
      memset(outputWarpedImage[0]->descrip, 0, 80);
      strcpy (outputWarpedImage[0]->descrip,"Warped image using NiftyReg (reg_f3d)");
      if(strcmp("NiftyReg F3D SYM", REG->GetExecutableName())==0)
      {
         strcpy (outputWarpedImage[0]->descrip,"Warped image using NiftyReg (reg_f3d_sym)");
         strcpy (outputWarpedImage[1]->descrip,"Warped image using NiftyReg (reg_f3d_sym)");
      }
      if(strcmp("NiftyReg F3D2", REG->GetExecutableName())==0)
      {
         strcpy (outputWarpedImage[0]->descrip,"Warped image using NiftyReg (reg_f3d2)");
         strcpy (outputWarpedImage[1]->descrip,"Warped image using NiftyReg (reg_f3d2)");
      }
      if(REG->GetSymmetricStatus())
      {
         if(outputWarpedImage[1]!=NULL)
         {
            std::string b(outputWarpedImageName);
            if(b.find( ".nii.gz") != std::string::npos)
               b.replace(b.find( ".nii.gz"),7,"_backward.nii.gz");
            else if(b.find( ".nii") != std::string::npos)
               b.replace(b.find( ".nii"),4,"_backward.nii");
            else if(b.find( ".hdr") != std::string::npos)
               b.replace(b.find( ".hdr"),4,"_backward.hdr");
            else if(b.find( ".img.gz") != std::string::npos)
               b.replace(b.find( ".img.gz"),7,"_backward.img.gz");
            else if(b.find( ".img") != std::string::npos)
               b.replace(b.find( ".img"),4,"_backward.img");
            else if(b.find( ".png") != std::string::npos)
               b.replace(b.find( ".png"),4,"_backward.png");
            else if(b.find( ".nrrd") != std::string::npos)
               b.replace(b.find( ".nrrd"),5,"_backward.nrrd");
            else b.append("_backward.nii");
            reg_io_WriteImageFile(outputWarpedImage[1],b.c_str());
         }
      }
      reg_io_WriteImageFile(outputWarpedImage[0],outputWarpedImageName);
      if(outputWarpedImage[0]!=NULL)
         nifti_image_free(outputWarpedImage[0]);
      outputWarpedImage[0]=NULL;
      if(outputWarpedImage[1]!=NULL)
         nifti_image_free(outputWarpedImage[1]);
      outputWarpedImage[1]=NULL;
      free(outputWarpedImage);
      outputWarpedImage=NULL;
#ifdef _USE_CUDA
   }
   cudaCommon_unsetCUDACard(&ctx);
#endif
   // Erase the registration object
   delete REG;

   // Clean the allocated images
   if(referenceImage!=NULL) nifti_image_free(referenceImage);
   if(floatingImage!=NULL) nifti_image_free(floatingImage);
   if(inputCCPImage!=NULL) nifti_image_free(inputCCPImage);
   if(referenceMaskImage!=NULL) nifti_image_free(referenceMaskImage);
   if(floatingMaskImage!=NULL) nifti_image_free(floatingMaskImage);

   time_t end;
   time( &end );
   int minutes = (int)floorf(float(end-start)/60.0f);
   int seconds = (int)(end-start - 60*minutes);

#ifdef NDEBUG
   if(verbose)
   {
#endif
      printf("[NiftyReg F3D] Registration Performed in %i min %i sec\n", minutes, seconds);
      printf("[NiftyReg F3D] Have a good day !\n");
#ifdef NDEBUG
   }
#endif
   return 0;
}
