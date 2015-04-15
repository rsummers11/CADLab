/*=========================================================================
 *
 *  Copyright National Institutes of Health
 *
 *  Unless required by applicable law or agreed to in writing, this software
 *  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 *  ANY KIND, either express or implied.
 *
 *=========================================================================*/
#include <iomanip>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkExtractImageFilter.h>
#include <itkFileTools.h>
#include <itkIntensityWindowingImageFilter.h>
#include <itkMath.h>
#include <itksys/SystemTools.hxx>
#include <itkDirectory.h>

const unsigned int Dimension = 3;
#include <nihHelperFunctions.h>


static const char * argv_hc[] = {"*.exe",
	"inputImage.nii.gz", //inputImage
  "/tmp/slices", //outputDir
  "-100",
  "200",
  "256", // outSize
  ".png", //outExtension
};

#define HARDCODEDPARAMS 0

typedef int InputPixelType; // 8-bit png image
typedef unsigned char OutputPixelType; // 8-bit images
//typedef unsigned short OutputPixelType; // 16-bit png images
//typedef unsigned char SegmentationPixelType;
//typedef itk::RGBPixel<OutputPixelType> RGBPixelType;

typedef itk::Image<InputPixelType, 3>  InputImageType;
typedef itk::Image<InputPixelType, 2> SliceImageType;
typedef itk::Image<OutputPixelType, 2> OutputImageType;

//nih::InterpolationType interpolationType = nih::LINEAR;
nih::InterpolationType interpolationType = nih::BSPLINE;

typedef itk::IntensityWindowingImageFilter<InputImageType, InputImageType> WindowingFilterType;

int main(int argc, const char **argv)
{

#if HARDCODEDPARAMS
	argv = argv_hc;
	argc = sizeof(argv_hc)/sizeof(const char*);
#endif

  // check input parameters
  if ( !((argc==7) || (argc==9)) )
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " inputImageFilename outputPrefix winMin winMax outputImageSize outExtension [startSlice (opt.), endSlice (opt.)]." << std::endl;
    std::cout << "  but had only " << argc << " input:" << std::endl;
    for (unsigned int i=0; i<argc; i++)
    {
      std::cout << "    " << i+1 << ".: " << argv[i] << std::endl;
    }
    return EXIT_FAILURE;
  }

  unsigned int i = 0;
  std::string inputImageFilename = argv[++i];
  std::string outputPrefix = argv[++i];
  const InputPixelType  winMin = atof( argv[++i] );
  const InputPixelType  winMax = atof( argv[++i] );
  const unsigned int outputImageSize = atoi( argv[++i] );
  std::string outExtension = argv[++i];

  const OutputPixelType desiredMin = 0.0;
  const OutputPixelType desiredMax = itk::NumericTraits< OutputPixelType >::max(); // intensity range for PNG images

  typedef itk::ImageFileReader< InputImageType >  ImageReaderType;
  itk::SmartPointer<ImageReaderType> reader = ImageReaderType::New();

  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  itk::SmartPointer<WriterType> writer = WriterType::New();

  printf(" Reading image %s ...\n", inputImageFilename.c_str() );
  // Read intensity image
  reader->SetFileName( inputImageFilename.c_str() );
  try
  {
    reader->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
  }

  // windowing options
  WindowingFilterType::Pointer windowFilter = WindowingFilterType::New();
  windowFilter->SetInput( reader->GetOutput() );
  windowFilter->SetWindowMinimum(winMin);
  windowFilter->SetWindowMaximum(winMax);
  windowFilter->SetOutputMinimum( desiredMin );
  windowFilter->SetOutputMaximum( desiredMax );
  try
  {
    windowFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
  }
  itk::SmartPointer<InputImageType> image = windowFilter->GetOutput();

  //create output directory
  std::string outputDir = nih::getPath(outputPrefix);
  std::cout << "  Creating output directory at: " << outputDir << std::endl;
  itk::FileTools::CreateDirectory( outputDir.c_str() );

  // Write image slices
  InputImageType::SizeType ImageSize = image->GetLargestPossibleRegion().GetSize();
  printf(" image size is [%d, %d, %d]\n", (int)ImageSize[0], (int)ImageSize[1], (int)ImageSize[2]);

  int startSlice = 0;
  int endSlice = ImageSize[2]-1;
  if (argc==9)
  {
    startSlice = atoi( argv[++i] );
    endSlice = atoi( argv[++i] );
  }
  // if any negative start or end slice, assume last slice is meant
  if (startSlice<0)
  {
    startSlice=ImageSize[2]-3;
  }
  if (endSlice<0) 
  {
    endSlice=ImageSize[2]-1;
  }

  std::string outputImageName;
  char count_char[100];

  OutputImageType::SizeType outSize;
  outSize[0] = outputImageSize;
  outSize[1] = outputImageSize;
  std::cout << std::endl << " wrote " << ImageSize[2] << " image slices to " << outputDir << std::endl;
  unsigned int zcounter = 0;
  printf(" Extract slices between %d and %d ...\n", (int)startSlice, (int)endSlice);
  for (unsigned int z = startSlice; z <= endSlice; z++)
  {
    zcounter++;
    sprintf(count_char, "_z%03d", z); // sprintf_s in windows?
    outputImageName = outputPrefix + itksys::SystemTools::GetFilenameWithoutExtension(inputImageFilename.c_str()).c_str() + count_char + outExtension;
    writer->SetInput(
      nih::rescaleImage2D<SliceImageType,OutputImageType>( nih::getImageSlice<InputImageType,SliceImageType>(image, z), outSize )
    );
    writer->SetFileName( outputImageName );

    std::cout << z << ", ";
    try
    {
      writer->Update();
    }
    catch( itk::ExceptionObject & excep )
    {
      std::cerr << "Exception caught !" << std::endl;
      std::cerr << excep << std::endl;
      return EXIT_FAILURE;
    }
  }// for (unsigned int z = 0; z < ImageSize[2]; z++)

  std::cout << std::endl << " wrote " << zcounter << " image slices to " << outputDir << std::endl;

  return EXIT_SUCCESS;
}
