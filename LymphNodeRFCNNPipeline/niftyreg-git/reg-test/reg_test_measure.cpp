#include <limits>

#include "_reg_ReadWriteImage.h"
#include "_reg_tools.h"
#include "_reg_nmi.h"
#include "_reg_ssd.h"
#include "_reg_lncc.h"

#define LNCC_VALUE_2D 0.691751
#define NMI_VALUE_2D  1.116657
#define SSD_VALUE_2D -0.0367104

#define LNCC_VALUE_3D 0.872809
#define NMI_VALUE_3D  1.149116
#define SSD_VALUE_3D -0.017684

#define EPS 0.000001

int main(int argc, char **argv)
{

   if(argc!=4)
   {
      fprintf(stderr, "Usage: %s <refImage> <warImage> <LNCC|NMI|SSD>\n", argv[0]);
      return EXIT_FAILURE;
   }

   char *inputRefImageName=argv[1];
   char *inputWarImageName=argv[2];
   char *measure_type=argv[3];

   /* Read the reference image */
   nifti_image *refImage = reg_io_ReadImageFile(inputRefImageName);
   if(refImage == NULL)
   {
      fprintf(stderr,"[NiftyReg ERROR] Error when reading the reference image: %s\n",
              inputRefImageName);
      return EXIT_FAILURE;
   }
   reg_checkAndCorrectDimension(refImage);
   reg_tools_changeDatatype<float>(refImage);

   /* Read the warped image */
   nifti_image *warImage = reg_io_ReadImageFile(inputWarImageName);
   if(warImage == NULL)
   {
      fprintf(stderr,"[NiftyReg ERROR] Error when reading the floating image: %s\n",
              inputWarImageName);
      return EXIT_FAILURE;
   }
   reg_checkAndCorrectDimension(warImage);
   reg_tools_changeDatatype<float>(warImage);

   // Check if the input images have the same size
   for(int i=0;i<8;++i){
      if(refImage->dim[i]!=warImage->dim[i])
      {
         reg_print_msg_error("reg_test_measure: The input images do not have the same size");
         return EXIT_FAILURE;
      }
   }

   int *mask_image=(int *)calloc(refImage->nvox,sizeof(int));

   int return_value = EXIT_SUCCESS;

   /* Compute the LNCC if required */
   if(strcmp(measure_type, "LNCC")==0)
   {
      reg_lncc *measure_object=new reg_lncc();
      for(int i=0;i<refImage->nt;++i)
         measure_object->SetActiveTimepoint(i);
      measure_object->InitialiseMeasure(refImage,
                                        warImage,
                                        mask_image,
                                        warImage,
                                        NULL,
                                        NULL);
      double measure=measure_object->GetSimilarityMeasureValue();
      printf("reg_test_measure: LNCC value %iD = %g\n",
             (refImage->nz>1?3:2), measure);
      double expectedValue = LNCC_VALUE_2D;
      if(refImage->nz>1)
         expectedValue = LNCC_VALUE_3D;
      if(fabs(measure-expectedValue)>EPS)
      {
         printf("reg_test_measure: Incorrect measure value %g (!=%g)\n",
                measure, expectedValue);
         return_value = EXIT_FAILURE;
      }
      delete measure_object;
   }
   /* Compute the NMI if required */
   else if(strcmp(measure_type, "NMI")==0)
   {
      reg_nmi *measure_object=new reg_nmi();
      for(int i=0;i<refImage->nt;++i)
         measure_object->SetActiveTimepoint(i);
      measure_object->InitialiseMeasure(refImage,
                                        warImage,
                                        mask_image,
                                        warImage,
                                        NULL,
                                        NULL);
      double measure=measure_object->GetSimilarityMeasureValue();
      printf("reg_test_measure: NMI value %iD = %.7g\n",
             (refImage->nz>1?3:2), measure);
      double expectedValue = NMI_VALUE_2D;
      if(refImage->nz>1)
         expectedValue = NMI_VALUE_3D;
      if(fabs(measure-expectedValue)>EPS)
      {
         printf("reg_test_measure: Incorrect measure value %g (%g)\n",
                measure, fabs(measure-expectedValue));
         return_value = EXIT_FAILURE;
      }
      delete measure_object;
   }
   /* Compute the SSD if required */
   else if(strcmp(measure_type, "SSD")==0)
   {
      reg_ssd *measure_object=new reg_ssd();
      for(int i=0;i<refImage->nt;++i)
         measure_object->SetActiveTimepoint(i);
      measure_object->InitialiseMeasure(refImage,
                                        warImage,
                                        mask_image,
                                        warImage,
                                        NULL,
                                        NULL);
      double measure=measure_object->GetSimilarityMeasureValue();
      printf("reg_test_measure: SSD value %iD = %g\n",
             (refImage->nz>1?3:2), measure);
      double expectedValue = SSD_VALUE_2D;
      if(refImage->nz>1)
         expectedValue = SSD_VALUE_3D;
      if(fabs(measure-expectedValue)>EPS)
      {
         printf("reg_test_measure: Incorrect measure value %g (!=%g)\n",
                measure, expectedValue);
         return_value = EXIT_FAILURE;
      }
      delete measure_object;
   }
   else
   {
      reg_print_msg_error("reg_test_measure: Unknown measure type");
      return_value= EXIT_FAILURE;
   }

   // Free the allocated images
   nifti_image_free(refImage);
   nifti_image_free(warImage);
   free(mask_image);

   return return_value;
}
