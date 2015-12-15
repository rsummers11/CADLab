#include "_reg_ReadWriteImage.h"
#include "_reg_f3d2.h"
#include "_reg_tools.h"

#define EPS 0.000001

int main(int argc, char **argv)
{

   if(argc!=5)
   {
      fprintf(stderr, "Usage: %s <refImage> <floImage> <affineMatrix> <expectedControlPointGrid>\n", argv[0]);
      return EXIT_FAILURE;
   }

   char *inputRefImageName=argv[1];
   char *inputFloImageName=argv[2];
   char *inputMatFileName=argv[3];
   char *inputControlPointGridFileName=argv[4];

   // Read the input reference image
   nifti_image *referenceImage = reg_io_ReadImageFile(inputRefImageName);
   if(referenceImage==NULL){
      reg_print_msg_error("The input reference image could not be read");
      return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(referenceImage);
   // Read the input reference image
   nifti_image *floatingImage = reg_io_ReadImageFile(inputFloImageName);
   if(floatingImage==NULL){
      reg_print_msg_error("The input floating image could not be read");
      return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(floatingImage);
   // Read the input affine matrix
   mat44 *inputMatrix=(mat44 *)malloc(sizeof(mat44));
   reg_tool_ReadAffineFile(inputMatrix, inputMatFileName);
   // Read the input control point grid image
   nifti_image *inputControlPointGridImage = reg_io_ReadImageFile(inputControlPointGridFileName);
   if(inputControlPointGridImage==NULL){
      reg_print_msg_error("The input control point grid image could not be read");
      return EXIT_FAILURE;
   }
   reg_tools_changeDatatype<float>(inputControlPointGridImage);

   // Run the affine registration
   reg_f3d2<float> *nonlinear=new reg_f3d2<float>(referenceImage->nt,floatingImage->nt);
   nonlinear->SetReferenceImage(referenceImage);
   nonlinear->SetFloatingImage(floatingImage);
   nonlinear->SetAffineTransformation(inputMatrix);
   nonlinear->Run();

   // Check the control point grid dimension
   if(nonlinear->GetControlPointPositionImage()->nx != inputControlPointGridImage->nx ||
      nonlinear->GetControlPointPositionImage()->ny != inputControlPointGridImage->ny ||
      nonlinear->GetControlPointPositionImage()->nz != inputControlPointGridImage->nz ||
      nonlinear->GetControlPointPositionImage()->nt != inputControlPointGridImage->nt ||
      nonlinear->GetControlPointPositionImage()->nu != inputControlPointGridImage->nu){
      reg_print_msg_error("The input and recovered control point grid images do not have corresponding sizes");
      return EXIT_FAILURE;
   }

   // Compute the difference between the computed and inputed deformation field
   reg_tools_substractImageToImage(inputControlPointGridImage,
                                   nonlinear->GetControlPointPositionImage(),
                                   inputControlPointGridImage);
   reg_tools_abs_image(inputControlPointGridImage);
   double max_difference=reg_tools_getMaxValue(inputControlPointGridImage);

   // Cleaning up
   nifti_image_free(referenceImage);
   nifti_image_free(floatingImage);
   nifti_image_free(inputControlPointGridImage);
   delete nonlinear;
   free(inputMatrix);

   if(max_difference>EPS){
      fprintf(stderr, "reg_test_fullSymNonlinear error too large: %g (>%g)\n",
              max_difference, EPS);
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

