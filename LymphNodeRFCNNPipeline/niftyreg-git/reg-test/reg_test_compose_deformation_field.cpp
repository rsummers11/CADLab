#include "_reg_ReadWriteImage.h"
#include "_reg_localTransformation.h"
#include "_reg_tools.h"

#define EPS 0.000001

int main(int argc, char **argv)
{
   if(argc!=3)
   {
      fprintf(stderr, "Usage: %s <inputDefField> <expectedField>\n", argv[0]);
      return EXIT_FAILURE;
   }

   char *inputDefFieldImageName=argv[1];
   char *inputComFieldImageName=argv[2];

   // Read the input deformation field image image
   nifti_image *inputDeformationField = reg_io_ReadImageFile(inputDefFieldImageName);
   if(inputDeformationField==NULL){
      reg_print_msg_error("The input deformation field image could not be read");
      return EXIT_FAILURE;
   }
   nifti_image *inputComFieldImage = reg_io_ReadImageFile(inputComFieldImageName);
   if(inputComFieldImage==NULL){
      reg_print_msg_error("The input composed deformation field image could not be read");
      return EXIT_FAILURE;
   }
   // Check the dimension of the input images
   if(inputDeformationField->nx != inputComFieldImage->nx ||
      inputDeformationField->ny != inputComFieldImage->ny ||
      inputDeformationField->nz != inputComFieldImage->nz ||
      inputDeformationField->nu != inputComFieldImage->nu){
      reg_print_msg_error("The input deformation field images do not have corresponding sizes");
      return EXIT_FAILURE;
   }

   // Create a deformation field
   nifti_image *test_field=nifti_copy_nim_info(inputDeformationField);
   test_field->data=(void *)malloc(test_field->nvox*test_field->nbyper);
   memcpy(test_field->data, inputDeformationField->data, test_field->nvox*test_field->nbyper);

   // Compute the non-linear deformation field
   reg_defField_compose(inputDeformationField,
                        test_field,
                        NULL);

   // Compute the difference between the computed and inputed deformation field
   reg_tools_substractImageToImage(inputComFieldImage,test_field,test_field);
   reg_tools_abs_image(test_field);
   double max_difference=reg_tools_getMaxValue(test_field);

   nifti_image_free(inputDeformationField);
   nifti_image_free(inputComFieldImage);
   nifti_image_free(test_field);

   if(max_difference>EPS){
      fprintf(stderr, "reg_test_compose_deformation_field error too large: %g (>%g)\n",
              max_difference, EPS);
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
