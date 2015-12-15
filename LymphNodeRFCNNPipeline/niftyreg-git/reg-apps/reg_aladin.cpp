/**
 * @file reg_aladin.cpp
 * @author Marc Modat, David C Cash and Pankaj Daga
 * @date 12/08/2009
 *
 * Copyright (c) 2009, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

#ifndef _MM_ALADIN_CPP
#define _MM_ALADIN_CPP

#include "_reg_ReadWriteImage.h"
#include "_reg_aladin_sym.h"
#include "_reg_tools.h"
#include "reg_aladin.h"

#ifdef _WIN32
#   include <time.h>
#endif

#ifdef _USE_NR_DOUBLE
#   define PrecisionTYPE double
#else
#   define PrecisionTYPE float
#endif

void PetitUsage(char *exec)
{
   fprintf(stderr,"\n");
   fprintf(stderr,"reg_aladin\n");
   fprintf(stderr,"Usage:\t%s -ref <referenceImageName> -flo <floatingImageName> [OPTIONS].\n",exec);
   fprintf(stderr,"\tSee the help for more details (-h).\n");
   fprintf(stderr,"\n");
   return;
}
void Usage(char *exec)
{
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   printf("Block Matching algorithm for global registration.\n");
   printf("Based on Ourselin et al., \"Reconstructing a 3D structure from serial histological sections\",\n");
   printf("Image and Vision Computing, 2001\n");
   printf("This code has been written by Marc Modat (m.modat@ucl.ac.uk) and Pankaj Daga,\n");
   printf("for any comment, please contact them.\n");
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   printf("Usage:\t%s -ref <filename> -flo <filename> [OPTIONS].\n",exec);
   printf("\t-ref <filename>\tReference image filename (also called Target or Fixed) (mandatory)\n");
   printf("\t-flo <filename>\tFloating image filename (also called Source or moving) (mandatory)\n");

   printf("\n* * OPTIONS * *\n");
   printf("\t-noSym \t\t\tThe symmetric version of the algorithm is used by default. Use this flag to disable it.\n");
   printf("\t-rigOnly\t\tTo perform a rigid registration only. (Rigid+affine by default)\n");
   printf("\t-affDirect\t\tDirectly optimize 12 DoF affine. (Default is rigid initially then affine)\n");

   printf("\t-aff <filename>\t\tFilename which contains the output affine transformation. [outputAffine.txt]\n");
   printf("\t-inaff <filename>\tFilename which contains an input affine transformation. (Affine*Reference=Floating) [none]\n");

   printf("\t-rmask <filename>\tFilename of a mask image in the reference space.\n");
   printf("\t-fmask <filename>\tFilename of a mask image in the floating space. (Only used when symmetric turned on)\n");
   printf("\t-res <filename>\t\tFilename of the resampled image. [outputResult.nii]\n");

   printf("\t-maxit <int>\t\tMaximal number of iterations of the trimmed least square approach to perform per level. [5]\n");
   printf("\t-ln <int>\t\tNumber of levels to use to generate the pyramids for the coarse-to-fine approach. [3]\n");
   printf("\t-lp <int>\t\tNumber of levels to use to run the registration once the pyramids have been created. [ln]\n");

   printf("\t-smooR <float>\t\tStandard deviation in mm (voxel if negative) of the Gaussian kernel used to smooth the Reference image. [0]\n");
   printf("\t-smooF <float>\t\tStandard deviation in mm (voxel if negative) of the Gaussian kernel used to smooth the Floating image. [0]\n");
   printf("\t-refLowThr <float>\tLower threshold value applied to the reference image. [0]\n");
   printf("\t-refUpThr <float>\tUpper threshold value applied to the reference image. [0]\n");
   printf("\t-floLowThr <float>\tLower threshold value applied to the floating image. [0]\n");
   printf("\t-floUpThr <float>\tUpper threshold value applied to the floating image. [0]\n");

   printf("\t-nac\t\t\tUse the nifti header origin to initialise the transformation. (Image centres are used by default)\n");
   printf("\t-cog\t\t\tUse the input masks centre of mass to initialise the transformation. (Image centres are used by default)\n");
   printf("\t-interp\t\t\tInterpolation order to use internally to warp the floating image.\n");
   printf("\t-iso\t\t\tMake floating and reference images isotropic if required.\n");

   printf("\t-pv <int>\t\tPercentage of blocks to use in the optimisation scheme. [50]\n");
   printf("\t-pi <int>\t\tPercentage of blocks to consider as inlier in the optimisation scheme. [50]\n");
   printf("\t-speeeeed\t\tGo faster\n");
#if defined (_OPENMP)
   printf("\t-omp <int>\t\tNumber of thread to use with OpenMP. [%i]\n",
          omp_get_num_procs());
#endif
   printf("\t-voff\t\t\tTurns verbose off [on]\n");
#ifdef _GIT_HASH
   printf("\n\t--version\t\tPrint current source code git hash key and exit\n\t\t\t\t(%s)\n",_GIT_HASH);
#endif
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

   int symFlag=1;

   char *referenceImageName=NULL;
   int referenceImageFlag=0;

   char *floatingImageName=NULL;
   int floatingImageFlag=0;

   char *outputAffineName=NULL;
   int outputAffineFlag=0;

   char *inputAffineName=NULL;
   int inputAffineFlag=0;

   char *referenceMaskName=NULL;
   int referenceMaskFlag=0;

   char *floatingMaskName=NULL;
   int floatingMaskFlag=0;

   char *outputResultName=NULL;
   int outputResultFlag=0;

   int maxIter=5;
   int nLevels=3;
   int levelsToPerform=std::numeric_limits<int>::max();
   int affineFlag=1;
   int rigidFlag=1;
   int blockStepSize=1;
   int blockPercentage=50;
   float inlierLts=50.0f;
   int alignCentre=1;
   int alignCentreOfGravity=0;
   int interpolation=1;
   float floatingSigma=0.0;
   float referenceSigma=0.0;

   float referenceLowerThr=-std::numeric_limits<PrecisionTYPE>::max();
   float referenceUpperThr=std::numeric_limits<PrecisionTYPE>::max();
   float floatingLowerThr=-std::numeric_limits<PrecisionTYPE>::max();
   float floatingUpperThr=std::numeric_limits<PrecisionTYPE>::max();

   bool iso=false;
   bool verbose=true;

   /* read the input parameter */
   for(int i=1; i<argc; i++)
   {
      if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 ||
            strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 ||
            strcmp(argv[i], "--h")==0 || strcmp(argv[i], "--help")==0)
      {
         Usage(argv[0]);
         return 0;
      }
      else if(strcmp(argv[i], "--xml")==0)
      {
         printf("%s",xml_aladin);
         return 0;
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
         return 0;
      }
#endif
      else if(strcmp(argv[i], "-ref")==0 || strcmp(argv[i], "-target")==0 || strcmp(argv[i], "--ref")==0)
      {
         referenceImageName=argv[++i];
         referenceImageFlag=1;
      }
      else if(strcmp(argv[i], "-flo")==0 || strcmp(argv[i], "-source")==0 || strcmp(argv[i], "--flo")==0)
      {
         floatingImageName=argv[++i];
         floatingImageFlag=1;
      }
      else if(strcmp(argv[i], "-noSym")==0 || strcmp(argv[i], "--noSym")==0)
      {
         symFlag=0;
      }
      else if(strcmp(argv[i], "-aff")==0 || strcmp(argv[i], "--aff")==0)
      {
         outputAffineName=argv[++i];
         outputAffineFlag=1;
      }
      else if(strcmp(argv[i], "-inaff")==0 || strcmp(argv[i], "--inaff")==0)
      {
         inputAffineName=argv[++i];
         inputAffineFlag=1;
      }
      else if(strcmp(argv[i], "-rmask")==0 || strcmp(argv[i], "-tmask")==0 || strcmp(argv[i], "--rmask")==0)
      {
         referenceMaskName=argv[++i];
         referenceMaskFlag=1;
      }
      else if(strcmp(argv[i], "-fmask")==0 || strcmp(argv[i], "-smask")==0 || strcmp(argv[i], "--fmask")==0)
      {
         floatingMaskName=argv[++i];
         floatingMaskFlag=1;
      }
      else if(strcmp(argv[i], "-res")==0 || strcmp(argv[i], "-result")==0 || strcmp(argv[i], "--res")==0)
      {
         outputResultName=argv[++i];
         outputResultFlag=1;
      }
      else if(strcmp(argv[i], "-maxit")==0 || strcmp(argv[i], "--maxit")==0)
      {
         maxIter = atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-ln")==0 || strcmp(argv[i], "--ln")==0)
      {
         nLevels=atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-lp")==0 || strcmp(argv[i], "--lp")==0)
      {
         levelsToPerform=atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-smooR")==0 || strcmp(argv[i], "-smooT")==0 || strcmp(argv[i], "--smooR")==0)
      {
         referenceSigma = (float)(atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-smooF")==0 || strcmp(argv[i], "-smooS")==0 || strcmp(argv[i], "--smooF")==0)
      {
         floatingSigma=(float)(atof(argv[++i]));
      }
      else if(strcmp(argv[i], "-rigOnly")==0 || strcmp(argv[i], "--rigOnly")==0)
      {
         rigidFlag=1;
         affineFlag=0;
      }
      else if(strcmp(argv[i], "-affDirect")==0 || strcmp(argv[i], "--affDirect")==0)
      {
         rigidFlag=0;
         affineFlag=1;
      }
      else if(strcmp(argv[i], "-nac")==0 || strcmp(argv[i], "--nac")==0)
      {
         alignCentre=0;
      }
      else if(strcmp(argv[i], "-cog")==0 || strcmp(argv[i], "--cog")==0)
      {
         alignCentre=0;
         alignCentreOfGravity=1;
      }
      else if(strcmp(argv[i], "-%v")==0 || strcmp(argv[i], "-pv")==0 || strcmp(argv[i], "--pv")==0)
      {
         float value=atof(argv[++i]);
         if(value<0.f || value>100.f){
            reg_print_msg_error("The variance argument is expected to be between 0 and 100");
            return EXIT_FAILURE;
         }
         blockPercentage=value;
      }
      else if(strcmp(argv[i], "-%i")==0 || strcmp(argv[i], "-pi")==0 || strcmp(argv[i], "--pi")==0)
      {
         float value=atof(argv[++i]);
         if(value<0.f || value>100.f){
            reg_print_msg_error("The inlier argument is expected to be between 0 and 100");
            return EXIT_FAILURE;
         }
         inlierLts=value;
      }
      else if(strcmp(argv[i], "-speeeeed")==0 || strcmp(argv[i], "--speeed")==0)
      {
         blockStepSize=2;
      }
      else if(strcmp(argv[i], "-interp")==0 || strcmp(argv[i], "--interp")==0)
      {
         interpolation=atoi(argv[++i]);
      }
      else if(strcmp(argv[i], "-refLowThr")==0 || strcmp(argv[i], "--refLowThr")==0)
      {
         referenceLowerThr=atof(argv[++i]);
      }
      else if(strcmp(argv[i], "-refUpThr")==0 || strcmp(argv[i], "--refUpThr")==0)
      {
         referenceUpperThr=atof(argv[++i]);
      }
      else if(strcmp(argv[i], "-floLowThr")==0 || strcmp(argv[i], "--floLowThr")==0)
      {
         floatingLowerThr=atof(argv[++i]);
      }
      else if(strcmp(argv[i], "-floUpThr")==0 || strcmp(argv[i], "--floUpThr")==0)
      {
         floatingUpperThr=atof(argv[++i]);
      }
      else if(strcmp(argv[i], "-iso")==0 || strcmp(argv[i], "--iso")==0)
      {
         iso=true;
      }
      else if(strcmp(argv[i], "-voff")==0 || strcmp(argv[i], "--voff")==0)
      {
         verbose=false;
      }
#if defined (_OPENMP)
      else if(strcmp(argv[i], "-omp")==0 || strcmp(argv[i], "--omp")==0)
      {
         omp_set_num_threads(atoi(argv[++i]));
      }
#endif
      else
      {
         fprintf(stderr,"Err:\tParameter %s unknown.\n",argv[i]);
         PetitUsage(argv[0]);
         return 1;
      }
   }

   if(!referenceImageFlag || !floatingImageFlag)
   {
      fprintf(stderr,"Err:\tThe reference and the floating image have to be defined.\n");
      PetitUsage(argv[0]);
      return 1;
   }

//   // Update the CLI progress bar that the registration has started
//   startProgress("reg_aladin");

   // Output the command line
#ifdef NDEBUG
   if(verbose)
   {
#endif
      printf("\n[NiftyReg ALADIN] Command line:\n\t");
      for(int i=0; i<argc; i++)
         printf(" %s", argv[i]);
      printf("\n\n");
#ifdef NDEBUG
   }
#endif

   reg_aladin<PrecisionTYPE> *REG;
   if(symFlag)
   {
      REG = new reg_aladin_sym<PrecisionTYPE>;
      if ( (referenceMaskFlag && !floatingMaskName) || (!referenceMaskFlag && floatingMaskName) )
      {
         fprintf(stderr,"[NiftyReg Warning] You have one image mask option turned on but not the other.\n");
         fprintf(stderr,"[NiftyReg Warning] This will affect the degree of symmetry achieved.\n");
      }
   }
   else
   {
      REG = new reg_aladin<PrecisionTYPE>;
      if (floatingMaskFlag)
      {
         fprintf(stderr,"Note: Floating mask flag only used in symmetric method. Ignoring this option\n");
      }
   }

   /* Read the reference image and check its dimension */
   nifti_image *referenceHeader = reg_io_ReadImageFile(referenceImageName);
   if(referenceHeader == NULL)
   {
      fprintf(stderr,"* ERROR Error when reading the reference  image: %s\n",referenceImageName);
      return 1;
   }

   /* Read the floating image and check its dimension */
   nifti_image *floatingHeader = reg_io_ReadImageFile(floatingImageName);
   if(floatingHeader == NULL)
   {
      fprintf(stderr,"* ERROR Error when reading the floating image: %s\n",floatingImageName);
      return 1;
   }

   // Set the reference and floating images
   nifti_image *isoRefImage=NULL;
   nifti_image *isoFloImage=NULL;
   if(iso)
   {
      // make the images isotropic if required
      isoRefImage=reg_makeIsotropic(referenceHeader,1);
      isoFloImage=reg_makeIsotropic(floatingHeader,1);
      REG->SetInputReference(isoRefImage);
      REG->SetInputFloating(isoFloImage);
   }
   else
   {
      REG->SetInputReference(referenceHeader);
      REG->SetInputFloating(floatingHeader);
   }

   /* read the reference mask image */
   nifti_image *referenceMaskImage=NULL;
   nifti_image *isoRefMaskImage=NULL;
   if(referenceMaskFlag)
   {
      referenceMaskImage = reg_io_ReadImageFile(referenceMaskName);
      if(referenceMaskImage == NULL)
      {
         fprintf(stderr,"* ERROR Error when reading the reference mask image: %s\n",referenceMaskName);
         return 1;
      }
      /* check the dimension */
      for(int i=1; i<=referenceHeader->dim[0]; i++)
      {
         if(referenceHeader->dim[i]!=referenceMaskImage->dim[i])
         {
            fprintf(stderr,"* ERROR The reference image and its mask do not have the same dimension\n");
            return 1;
         }
      }
      if(iso)
      {
         // make the image isotropic if required
         isoRefMaskImage=reg_makeIsotropic(referenceMaskImage,0);
         REG->SetInputMask(isoRefMaskImage);
      }
      else REG->SetInputMask(referenceMaskImage);
   }
   /* Read the floating mask image */
   nifti_image *floatingMaskImage=NULL;
   nifti_image *isoFloMaskImage=NULL;
   if(floatingMaskFlag && symFlag)
   {
      floatingMaskImage = reg_io_ReadImageFile(floatingMaskName);
      if(floatingMaskImage == NULL)
      {
         fprintf(stderr,"* ERROR Error when reading the floating mask image: %s\n",floatingMaskName);
         return 1;
      }
      /* check the dimension */
      for(int i=1; i<=floatingHeader->dim[0]; i++)
      {
         if(floatingHeader->dim[i]!=floatingMaskImage->dim[i])
         {
            fprintf(stderr,"* ERROR The floating image and its mask do not have the same dimension\n");
            return 1;
         }
      }
      if(iso)
      {
         // make the image isotropic if required
         isoFloMaskImage=reg_makeIsotropic(floatingMaskImage,0);
         REG->SetInputFloatingMask(isoFloMaskImage);
      }
      else REG->SetInputFloatingMask(floatingMaskImage);
   }


//   // Update the CLI progress bar
//   progressXML(2, "Input data ready...");

   REG->SetMaxIterations(maxIter);
   REG->SetNumberOfLevels(nLevels);
   REG->SetLevelsToPerform(levelsToPerform);
   REG->SetReferenceSigma(referenceSigma);
   REG->SetFloatingSigma(floatingSigma);
   REG->SetAlignCentre(alignCentre);
   REG->SetAlignCentreGravity(alignCentreOfGravity);
   REG->SetPerformAffine(affineFlag);
   REG->SetPerformRigid(rigidFlag);
   REG->SetBlockStepSize(blockStepSize);
   REG->SetBlockPercentage(blockPercentage);
   REG->SetInlierLts(inlierLts);
   REG->SetInterpolation(interpolation);

   if (referenceLowerThr != referenceUpperThr)
   {
      REG->SetReferenceLowerThreshold(referenceLowerThr);
      REG->SetReferenceUpperThreshold(referenceUpperThr);
   }

   if (floatingLowerThr != floatingUpperThr)
   {
      REG->SetFloatingLowerThreshold(floatingLowerThr);
      REG->SetFloatingUpperThreshold(floatingUpperThr);
   }

   if(REG->GetLevelsToPerform() > REG->GetNumberOfLevels())
      REG->SetLevelsToPerform(REG->GetNumberOfLevels());

   // Set the input affine transformation if defined
   if(inputAffineFlag==1)
      REG->SetInputTransform(inputAffineName);

   // Set the verbose type
   REG->SetVerbose(verbose);

   // Run the registration
   REG->Run();

   // The warped image is saved
   if(iso)
   {
      REG->SetInputReference(referenceHeader);
      REG->SetInputFloating(floatingHeader);
   }
   nifti_image *outputResultImage=REG->GetFinalWarpedImage();
   if(!outputResultFlag) outputResultName=(char *)"outputResult.nii";
   reg_io_WriteImageFile(outputResultImage,outputResultName);
   nifti_image_free(outputResultImage);

   /* The affine transformation is saved */
   if(outputAffineFlag)
      reg_tool_WriteAffineFile(REG->GetTransformationMatrix(), outputAffineName);
   else reg_tool_WriteAffineFile(REG->GetTransformationMatrix(), (char *)"outputAffine.txt");

//   // Tell the CLI that we finished
//   closeProgress("reg_aladin", "Normal exit");


   nifti_image_free(referenceHeader);
   nifti_image_free(floatingHeader);
   if(isoRefImage!=NULL)
      nifti_image_free(isoRefImage);
   if(isoFloImage!=NULL)
      nifti_image_free(isoFloImage);
   if(referenceMaskImage!=NULL)
      nifti_image_free(referenceMaskImage);
   if(floatingMaskImage!=NULL)
      nifti_image_free(floatingMaskImage);
   if(isoRefMaskImage!=NULL)
      nifti_image_free(isoRefMaskImage);
   if(isoFloMaskImage!=NULL)
      nifti_image_free(isoFloMaskImage);

   delete REG;
#ifdef NDEBUG
   if(verbose)
   {
#endif
      time_t end;
      time(&end);
      int minutes=(int)floorf((end-start)/60.0f);
      int seconds=(int)(end-start - 60*minutes);
      printf("Registration Performed in %i min %i sec\n", minutes, seconds);
      printf("Have a good day !\n");
#ifdef NDEBUG
   }
#endif

   return 0;
}

#endif
