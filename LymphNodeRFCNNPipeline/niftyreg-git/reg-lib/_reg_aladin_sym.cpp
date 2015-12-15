#include "_reg_aladin_sym.h"
#ifndef _REG_ALADIN_SYM_CPP
#define _REG_ALADIN_SYM_CPP

/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_aladin_sym<T>::reg_aladin_sym ()
   :reg_aladin<T>::reg_aladin()
{
   this->ExecutableName=(char*) "reg_aladin_sym";

   this->InputFloatingMask=NULL;
   this->FloatingMaskPyramid=NULL;
   this->CurrentFloatingMask=NULL;
   this->BackwardActiveVoxelNumber=NULL;

   this->BackwardDeformationFieldImage=NULL;
   this->CurrentBackwardWarped=NULL;
   this->BackwardTransformationMatrix=new mat44;

   this->FloatingUpperThreshold=std::numeric_limits<T>::max();
   this->FloatingLowerThreshold=-std::numeric_limits<T>::max();

#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_aladin_sym constructor called\n");
#endif

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
reg_aladin_sym<T>::~reg_aladin_sym()
{
   this->ClearBackwardWarpedImage();
   this->ClearBackwardDeformationField();

   if(this->BackwardTransformationMatrix!=NULL)
      delete this->BackwardTransformationMatrix;
   this->BackwardTransformationMatrix=NULL;

   if(this->FloatingMaskPyramid!=NULL)
   {
      for(unsigned int i=0; i<this->LevelsToPerform; ++i)
      {
         if(this->FloatingMaskPyramid[i]!=NULL)
         {
            free(this->FloatingMaskPyramid[i]);
            this->FloatingMaskPyramid[i]=NULL;
         }
      }
      free(this->FloatingMaskPyramid);
      this->FloatingMaskPyramid=NULL;
   }
   free(this->BackwardActiveVoxelNumber);
   this->BackwardActiveVoxelNumber=NULL;

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::SetInputFloatingMask(nifti_image *m)
{
   this->InputFloatingMask = m;
   return;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::AllocateBackwardWarpedImage()
{
   if(this->CurrentReference==NULL || this->CurrentFloating==NULL)
   {
      fprintf(stderr,"[NiftyReg ERROR] reg_aladin_sym::AllocateBackwardWarpedImage()\n");
      fprintf(stderr,"[NiftyReg ERROR] Reference and FLoating images are not defined. Exit.\n");
      reg_exit(1);
   }
   this->ClearBackwardWarpedImage();
   this->CurrentBackwardWarped=nifti_copy_nim_info(this->CurrentFloating);
   this->CurrentBackwardWarped->dim[0]=this->CurrentBackwardWarped->ndim=this->CurrentReference->ndim;
   this->CurrentBackwardWarped->dim[4]=this->CurrentBackwardWarped->nt=this->CurrentReference->nt;
   this->CurrentBackwardWarped->pixdim[4]=this->CurrentBackwardWarped->dt=1.0;
   this->CurrentBackwardWarped->nvox =
      (size_t)this->CurrentBackwardWarped->nx *
      (size_t)this->CurrentBackwardWarped->ny *
      (size_t)this->CurrentBackwardWarped->nz *
      (size_t)this->CurrentBackwardWarped->nt;
   this->CurrentBackwardWarped->datatype=this->CurrentReference->datatype;
   this->CurrentBackwardWarped->nbyper=this->CurrentReference->nbyper;
   this->CurrentBackwardWarped->data = (void*) calloc(this->CurrentBackwardWarped->nvox,this->CurrentBackwardWarped->nbyper);

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::ClearBackwardWarpedImage()
{
   if(this->CurrentBackwardWarped!=NULL)
      nifti_image_free(this->CurrentBackwardWarped);
   this->CurrentBackwardWarped=NULL;

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::AllocateBackwardDeformationField()
{
   if(this->CurrentFloating==NULL)
   {
      fprintf(stderr,"[NiftyReg ERROR] reg_aladin_sym::AllocateBackwardDeformationField()\n");
      fprintf(stderr,"[NiftyReg ERROR] Floating image is not defined. Exit.\n");
      reg_exit(1);
   }
   this->ClearBackwardDeformationField();
   this->BackwardDeformationFieldImage = nifti_copy_nim_info(this->CurrentFloating);
   this->BackwardDeformationFieldImage->dim[0]=this->BackwardDeformationFieldImage->ndim=5;
   this->BackwardDeformationFieldImage->dim[4]=this->BackwardDeformationFieldImage->nt=1;
   this->BackwardDeformationFieldImage->pixdim[4]=this->BackwardDeformationFieldImage->dt=1.0;
   if(this->CurrentFloating->nz==1)
      this->BackwardDeformationFieldImage->dim[5]=this->BackwardDeformationFieldImage->nu=2;
   else this->BackwardDeformationFieldImage->dim[5]=this->BackwardDeformationFieldImage->nu=3;
   this->BackwardDeformationFieldImage->pixdim[5]=this->BackwardDeformationFieldImage->du=1.0;
   this->BackwardDeformationFieldImage->dim[6]=this->BackwardDeformationFieldImage->nv=1;
   this->BackwardDeformationFieldImage->pixdim[6]=this->BackwardDeformationFieldImage->dv=1.0;
   this->BackwardDeformationFieldImage->dim[7]=this->BackwardDeformationFieldImage->nw=1;
   this->BackwardDeformationFieldImage->pixdim[7]=this->BackwardDeformationFieldImage->dw=1.0;
   this->BackwardDeformationFieldImage->nvox =
      (size_t)this->BackwardDeformationFieldImage->nx *
      (size_t)this->BackwardDeformationFieldImage->ny *
      (size_t)this->BackwardDeformationFieldImage->nz *
      (size_t)this->BackwardDeformationFieldImage->nt *
      (size_t)this->BackwardDeformationFieldImage->nu;
   this->BackwardDeformationFieldImage->nbyper = sizeof(T);
   if(sizeof(T)==4)
      this->BackwardDeformationFieldImage->datatype = NIFTI_TYPE_FLOAT32;
   else if(sizeof(T)==8)
      this->BackwardDeformationFieldImage->datatype = NIFTI_TYPE_FLOAT64;
   else
   {
      fprintf(stderr,"[NiftyReg ERROR] reg_aladin_sym::AllocateBackwardDeformationField()\n");
      fprintf(stderr,"[NiftyReg ERROR] Only float or double are expected for the deformation field. Exit.\n");
      reg_exit(1);
   }
   this->BackwardDeformationFieldImage->data = (void *)calloc(this->BackwardDeformationFieldImage->nvox, this->BackwardDeformationFieldImage->nbyper);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::ClearBackwardDeformationField()
{
   if(this->BackwardDeformationFieldImage!=NULL)
      nifti_image_free(this->BackwardDeformationFieldImage);
   this->BackwardDeformationFieldImage=NULL;
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::InitialiseRegistration()
{

#ifndef NDEBUG
   printf("[NiftyReg DEBUG] reg_aladin_sym::InitialiseRegistration() called\n");
#endif

   reg_aladin<T>::InitialiseRegistration();
   this->FloatingMaskPyramid = (int **) malloc(this->LevelsToPerform*sizeof(int *));
   this->BackwardActiveVoxelNumber= (int *)malloc(this->LevelsToPerform*sizeof(int));
   if (this->InputFloatingMask!=NULL)
   {
      reg_createMaskPyramid<T>(this->InputFloatingMask,
                               this->FloatingMaskPyramid,
                               this->NumberOfLevels,
                               this->LevelsToPerform,
                               this->BackwardActiveVoxelNumber);
   }
   else
   {
      for(unsigned int l=0; l<this->LevelsToPerform; ++l)
      {
         this->BackwardActiveVoxelNumber[l]=this->FloatingPyramid[l]->nx*this->FloatingPyramid[l]->ny*this->FloatingPyramid[l]->nz;
         this->FloatingMaskPyramid[l]=(int *)calloc(this->BackwardActiveVoxelNumber[l],sizeof(int));
      }
   }

   // CHECK THE THRESHOLD VALUES TO UPDATE THE MASK
   if(this->FloatingUpperThreshold!=std::numeric_limits<T>::max())
   {
      for(unsigned int l=0; l<this->LevelsToPerform; ++l)
      {
         T *refPtr = static_cast<T *>(this->FloatingPyramid[l]->data);
         int *mskPtr = this->FloatingMaskPyramid[l];
         size_t removedVoxel=0;
         for(size_t i=0;
               i<(size_t)this->FloatingPyramid[l]->nx*this->FloatingPyramid[l]->ny*this->FloatingPyramid[l]->nz;
               ++i)
         {
            if(mskPtr[i]>-1)
            {
               if(refPtr[i]>this->FloatingUpperThreshold)
               {
                  ++removedVoxel;
                  mskPtr[i]=-1;
               }
            }
         }
         this->BackwardActiveVoxelNumber[l] -= removedVoxel;
      }
   }
   if(this->FloatingLowerThreshold!=-std::numeric_limits<T>::max())
   {
      for(unsigned int l=0; l<this->LevelsToPerform; ++l)
      {
         T *refPtr = static_cast<T *>(this->FloatingPyramid[l]->data);
         int *mskPtr = this->FloatingMaskPyramid[l];
         size_t removedVoxel=0;
         for(size_t i=0;
               i<(size_t)this->FloatingPyramid[l]->nx*this->FloatingPyramid[l]->ny*this->FloatingPyramid[l]->nz;
               ++i)
         {
            if(mskPtr[i]>-1)
            {
               if(refPtr[i]<this->FloatingLowerThreshold)
               {
                  ++removedVoxel;
                  mskPtr[i]=-1;
               }
            }
         }
         this->BackwardActiveVoxelNumber[l] -= removedVoxel;
      }
   }

   if(this->AlignCentreGravity && this->InputTransformName==NULL)
   {
      if(!this->InputReferenceMask && !this->InputFloatingMask){
         reg_print_msg_error("The masks' centre of gravity can only be used when two masks are specified");
         reg_exit(1);
      }
      float referenceCentre[3]={0,0,0};
      float referenceCount=0;
      reg_tools_changeDatatype<float>(this->InputReferenceMask);
      float *refMaskPtr=static_cast<float *>(this->InputReferenceMask->data);
      size_t refIndex=0;
      for(int z=0;z<this->InputReferenceMask->nz;++z){
         for(int y=0;y<this->InputReferenceMask->ny;++y){
            for(int x=0;x<this->InputReferenceMask->nx;++x){
               if(refMaskPtr[refIndex]!=0.f){
                  referenceCentre[0]+=x;
                  referenceCentre[1]+=y;
                  referenceCentre[2]+=z;
                  referenceCount++;
               }
               refIndex++;
            }
         }
      }
      referenceCentre[0]/=referenceCount;
      referenceCentre[1]/=referenceCount;
      referenceCentre[2]/=referenceCount;
      float refCOG[3];
      if(this->InputReference->sform_code>0)
         reg_mat44_mul(&(this->InputReference->sto_xyz),referenceCentre,refCOG);

      float floatingCentre[3]={0,0,0};
      float floatingCount=0;
      reg_tools_changeDatatype<float>(this->InputFloatingMask);
      float *floMaskPtr=static_cast<float *>(this->InputFloatingMask->data);
      size_t floIndex=0;
      for(int z=0;z<this->InputFloatingMask->nz;++z){
         for(int y=0;y<this->InputFloatingMask->ny;++y){
            for(int x=0;x<this->InputFloatingMask->nx;++x){
               if(floMaskPtr[floIndex]!=0.f){
                  floatingCentre[0]+=x;
                  floatingCentre[1]+=y;
                  floatingCentre[2]+=z;
                  floatingCount++;
               }
               floIndex++;
            }
         }
      }
      floatingCentre[0]/=floatingCount;
      floatingCentre[1]/=floatingCount;
      floatingCentre[2]/=floatingCount;
      float floCOG[3];
      if(this->InputFloating->sform_code>0)
         reg_mat44_mul(&(this->InputFloating->sto_xyz),floatingCentre,floCOG);
      reg_mat44_eye(this->TransformationMatrix);
      this->TransformationMatrix->m[0][3]=floCOG[0]-refCOG[0];
      this->TransformationMatrix->m[1][3]=floCOG[1]-refCOG[1];
      this->TransformationMatrix->m[2][3]=floCOG[2]-refCOG[2];
   }

   //TransformationMatrix maps the initial transform from reference to floating.
   //Invert to get initial transformation from floating to reference
   *(this->BackwardTransformationMatrix) = nifti_mat44_inverse(*(this->TransformationMatrix));
   //Future todo: Square root of transform gives halfway space for both images
   //Forward will be to transform CurrentReference to HalfwayFloating
   //Backward will be to transform CurrentFloating to HalfwayReference
   //Forward transform will update Halfway Reference to new halfway space image
   //Backward transform will update Halfway Floating to new halfway space image
   //*(this->BackwardTransformationMatrix) = reg_mat44_sqrt(this->BackwardTransformationMatrix);
   //*(this->TransformationMatrix) = reg_mat44_sqrt(this->TransformationMatrix);

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::InitialiseBlockMatching(int CurrentPercentageOfBlockToUse)
{
   //Perform conventional block matchingin initialisation
   //Then initialise for the backward case.
   //TODO: If using halfway space, then we need to initialize the block matching
   //so that the HalfwayFloating (Warped) now serves as the target
   //and the Mask is correct as well
   //Then do the same thing for the block matching algorithm
   reg_aladin<T>::InitialiseBlockMatching(CurrentPercentageOfBlockToUse);
   initialise_block_matching_method(this->CurrentFloating,
                                    &this->BackwardBlockMatchingParams,
                                    CurrentPercentageOfBlockToUse,    // percentage of block kept
                                    this->InlierLts,         // percentage of inlier in the optimisation process
                                    this->BlockStepSize,
                                    this->CurrentFloatingMask,
                                    false // GPU is not used here
                                   );
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::SetCurrentImages()
{
   reg_aladin<T>::SetCurrentImages();
   this->CurrentFloatingMask=this->FloatingMaskPyramid[this->CurrentLevel];
   this->AllocateBackwardWarpedImage();
   this->AllocateBackwardDeformationField();
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::GetBackwardDeformationField()
{
   reg_affine_getDeformationField(this->BackwardTransformationMatrix,
                                  this->BackwardDeformationFieldImage);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::GetWarpedImage(int interp)
{
   reg_aladin<T>::GetWarpedImage(interp);
   this->GetBackwardDeformationField();
   //TODO: This needs correction, otherwise we are transforming an image that has already been warped
   reg_resampleImage(this->CurrentReference,
                     this->CurrentBackwardWarped,
                     this->BackwardDeformationFieldImage,
                     this->CurrentFloatingMask,
                     interp,
                     0);
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::UpdateTransformationMatrix(int type)
{
   // Update first the forward transformation matrix
   block_matching_method(this->CurrentReference,
                         this->CurrentWarped,
                         &this->blockMatchingParams,
                         this->CurrentReferenceMask);
   if(type==RIGID)
   {
      optimize(&this->blockMatchingParams,
               this->TransformationMatrix,
               RIGID);
   }
   else
   {
      optimize(&this->blockMatchingParams,
               this->TransformationMatrix,
               AFFINE);
   }
   // Update now the backward transformation matrix
   block_matching_method(this->CurrentFloating,
                         this->CurrentBackwardWarped,
                         &this->BackwardBlockMatchingParams,
                         this->CurrentFloatingMask);
   if(type==RIGID)
   {
      optimize(&this->BackwardBlockMatchingParams,
               this->BackwardTransformationMatrix,
               RIGID);
   }
   else
   {
      optimize(&this->BackwardBlockMatchingParams,
               this->BackwardTransformationMatrix,
               AFFINE);
   }
   // Forward and backward matrix are inverted
   mat44 fInverted = nifti_mat44_inverse(*(this->TransformationMatrix));
   mat44 bInverted = nifti_mat44_inverse(*(this->BackwardTransformationMatrix));

   // We average the forward and inverted backward matrix
   *(this->TransformationMatrix)=reg_mat44_avg2(this->TransformationMatrix,
                                                &bInverted
                                                );
   // We average the inverted forward and backward matrix
   *(this->BackwardTransformationMatrix)=reg_mat44_avg2(&fInverted,
                                                        this->BackwardTransformationMatrix
                                                        );
   for(int i=0;i<3;++i){
      this->TransformationMatrix->m[3][i]=0.f;
      this->BackwardTransformationMatrix->m[3][i]=0.f;
   }
   this->TransformationMatrix->m[3][3]=1.f;
   this->BackwardTransformationMatrix->m[3][3]=1.f;
#ifndef NDEBUG
   reg_mat44_disp(this->TransformationMatrix, (char *)"[DEBUG] updated forward transformation matrix");
   reg_mat44_disp(this->BackwardTransformationMatrix, (char *)"[DEBUG] updated backward transformation matrix");
#endif
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::ClearCurrentInputImage()
{
   reg_aladin<T>::ClearCurrentInputImage();
   if(this->FloatingMaskPyramid[this->CurrentLevel]!=NULL)
      free(this->FloatingMaskPyramid[this->CurrentLevel]);
   this->FloatingMaskPyramid[this->CurrentLevel]=NULL;
   this->CurrentFloatingMask=NULL;

   this->ClearBackwardWarpedImage();
   this->ClearBackwardDeformationField();
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::DebugPrintLevelInfoStart()
{
   printf("[%s] Current level %i / %i\n", this->ExecutableName, this->CurrentLevel+1, this->NumberOfLevels);
   printf("[%s] reference image size: \t%ix%ix%i voxels\t%gx%gx%g mm\n", this->ExecutableName,
          this->CurrentReference->nx, this->CurrentReference->ny, this->CurrentReference->nz,
          this->CurrentReference->dx, this->CurrentReference->dy, this->CurrentReference->dz);
   printf("[%s] floating image size: \t%ix%ix%i voxels\t%gx%gx%g mm\n", this->ExecutableName,
          this->CurrentFloating->nx, this->CurrentFloating->ny, this->CurrentFloating->nz,
          this->CurrentFloating->dx, this->CurrentFloating->dy, this->CurrentFloating->dz);
   if(this->CurrentReference->nz==1)
      printf("[%s] Block size = [4 4 1]\n", this->ExecutableName);
   else printf("[%s] Block size = [4 4 4]\n", this->ExecutableName);
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");
   printf("[%s] Forward Block number = [%i %i %i]\n", this->ExecutableName, this->blockMatchingParams.blockNumber[0],
          this->blockMatchingParams.blockNumber[1], this->blockMatchingParams.blockNumber[2]);
   printf("[%s] Backward Block number = [%i %i %i]\n", this->ExecutableName, this->BackwardBlockMatchingParams.blockNumber[0],
          this->BackwardBlockMatchingParams.blockNumber[1], this->BackwardBlockMatchingParams.blockNumber[2]);
   reg_mat44_disp(this->TransformationMatrix,
                  (char *)"[reg_aladin_sym] Initial forward transformation matrix:");
   reg_mat44_disp(this->BackwardTransformationMatrix,
                  (char *)"[reg_aladin_sym] Initial backward transformation matrix:");
   printf("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n");

}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
template <class T>
void reg_aladin_sym<T>::DebugPrintLevelInfoEnd()
{
   reg_mat44_disp(this->TransformationMatrix,
                  (char *)"[reg_aladin_sym] Final forward transformation matrix:");
   reg_mat44_disp(this->BackwardTransformationMatrix,
                  (char *)"[reg_aladin_sym] Final backward transformation matrix:");
}
/* \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ */
#endif //REG_ALADIN_SYM_CPP
