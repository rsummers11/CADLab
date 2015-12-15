#include "_reg_ReadWriteImage.h"
#include "_reg_localTransformation.h"
#include "_reg_tools.h"

#define EPS 0.000001

int main(int argc, char **argv)
{
   if(argc!=4)
   {
      fprintf(stderr, "Usage: %s <refImage> <inputControlPointGrid> <expectedField>\n", argv[0]);
      return EXIT_FAILURE;
   }

   char *inputRefImageName=argv[1];
   char *inputCPPImageName=argv[2];
   char *inputDefImageName=argv[3];

   // Read the input reference image
   nifti_image *referenceImage = reg_io_ReadImageHeader(inputRefImageName);
   if(referenceImage==NULL){
      reg_print_msg_error("The input reference image could not be read");
      return EXIT_FAILURE;
   }
   // Read the input deformation field image image
   nifti_image *controlPointGridImage = reg_io_ReadImageFile(inputCPPImageName);
   if(controlPointGridImage==NULL){
      reg_print_msg_error("The input control point grid image could not be read");
      return EXIT_FAILURE;
   }
   // Read the input deformation field image image
   nifti_image *inputDeformationField = reg_io_ReadImageFile(inputDefImageName);
   if(inputDeformationField==NULL){
      reg_print_msg_error("The input deformation field image could not be read");
      return EXIT_FAILURE;
   }
   // Check the dimension of the input images
   if(referenceImage->nx != inputDeformationField->nx ||
      referenceImage->ny != inputDeformationField->ny ||
      referenceImage->nz != inputDeformationField->nz ||
      (referenceImage->nz>1?3:2) != inputDeformationField->nu){
      reg_print_msg_error("The input reference and deformation field images do not have corresponding sizes");
      return EXIT_FAILURE;
   }

   // Create a deformation field
   nifti_image *test_field=nifti_copy_nim_info(inputDeformationField);
   test_field->data=(void *)malloc(test_field->nvox*test_field->nbyper);

   // Compute the non-linear deformation field
   memset(test_field->data, 0, test_field->nvox*test_field->nbyper);
   reg_getDeformationFromDisplacement(test_field);
   reg_spline_getDeformationField(controlPointGridImage,
                                  test_field,
                                  NULL,
                                  true,
                                  true);

   // Compute the difference between the computed and inputed deformation field
   reg_tools_substractImageToImage(inputDeformationField,test_field,test_field);
   reg_tools_abs_image(test_field);
   double max_difference=reg_tools_getMaxValue(test_field);

   nifti_image_free(referenceImage);
   nifti_image_free(controlPointGridImage);
   nifti_image_free(inputDeformationField);
   nifti_image_free(test_field);

   if(max_difference>EPS){
      fprintf(stderr, "reg_test_nonlinear_deformation_field error too large: %g (>%g)\n",
              max_difference, EPS);
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
