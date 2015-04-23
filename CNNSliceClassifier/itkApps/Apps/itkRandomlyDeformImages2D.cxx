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
#include <itkImageRegionIterator.h>
#include <itkBinaryImageToLabelMapFilter.h>
#include <itkLabelMapToLabelImageFilter.h>
#include <itkLabelStatisticsImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkFileTools.h>
#include <itkRegionOfInterestImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkIntensityWindowingImageFilter.h>
#include <itkLabelOverlapMeasuresImageFilter.h>
#include <itkMath.h>
#include <itksys/SystemTools.hxx>
#include <itkDirectory.h>

const unsigned int Dimension = 2;
#include <nihHelperFunctions.h>


static const char * argv_hc[] = {"*.exe",
	"D:/HolgerRoth/data/Liver/LiverImages/Training512", //inputDir
  ".png", //searchString
  "D:/HolgerRoth/data/Liver/LiverImages/Training512_t1_r1_d1_6", //outputDir
  "256", // outSize
//	"D:/HolgerRoth/data/Liver/LiverImages/Training", //inputDir
//  ".png", //searchString
//  "D:/HolgerRoth/data/Liver/LiverImages/Training_t1_r1_d1", //outputDir
  "1", // Ntranslations
  "1", // Nrotations
  "1", // Nnonrigiddeforms
  "80.0", //translation_max [mm]
  "5.0", //rotation_max [degrees]
  "5", // Nnonrigid_points
  "15.0", //nonrigid_deform_max [mm]
  "1e-2" // siffness
};

#define HARDCODEDPARAMS 0

typedef unsigned char InputPixelType; // 8-bit png image
typedef unsigned char OutputPixelType; // 8-bit png image
//typedef unsigned short OutputPixelType; // 16-bit png images
//typedef unsigned char SegmentationPixelType;
//typedef itk::RGBPixel<OutputPixelType> RGBPixelType;

typedef itk::Image<InputPixelType, Dimension>  InputImageType;
typedef itk::Image<OutputPixelType, Dimension> OutputImageType;

//nih::InterpolationType interpolationType = nih::LINEAR;
nih::InterpolationType interpolationType = nih::BSPLINE;

int main(int argc, const char **argv)
{

#if HARDCODEDPARAMS
	argv = argv_hc;
	argc = sizeof(argv_hc)/sizeof(const char*);
#endif

  // check input parameters
  if (argc != 13)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " inputDir searchString[e.g. '.png'] outputFilenamePrefix outSize Ntranslations Nrotations Nnonrigiddeforms translation_max [mm] rotation_max [degrees] Nnonrigid_points nonrigid_deform_max [mm] siffness." << std::endl;
    std::cout << "  but had only " << argc << " input:" << std::endl;
    for (unsigned int i=0; i<argc; i++)
    {
      std::cout << "    " << i+1 << ".: " << argv[i] << std::endl;
    }
    return EXIT_FAILURE;
  }

  std::string inputDir = argv[1];
  std::string searchString = argv[2];
  std::string outputFilenamePrefix = argv[3];
  const unsigned int outSize = atoi( argv[4] );
  const unsigned int Ntranslations = atoi( argv[5] );
  const unsigned int Nrotations = atoi( argv[6] );
  const unsigned int Nnonrigiddeforms = atoi( argv[7] );
  const double translation_max = atof( argv[8] );
  const double rotation_max = atof( argv[9] );
  const unsigned int Nnonrigid_points = atoi( argv[10] );
  const double nonrigid_deform_max = atof( argv[11] );
  const double siffness = atof( argv[12] );

  unsigned long image_count = 0;
  unsigned long deform_count = 0;

  typedef itk::ImageFileReader< InputImageType >  ImageReaderType;
  itk::SmartPointer<ImageReaderType> reader = ImageReaderType::New();

  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  itk::SmartPointer<WriterType> writer = WriterType::New();

  // Find file names
  itk::SmartPointer<itk::Directory> dir = itk::Directory::New();

  if ( !dir->Load( inputDir.c_str() ) )
  {
      std::cerr << "Directory " << inputDir.c_str() << " cannot be read!";
      if ( itksys::SystemTools::FileIsDirectory( inputDir.c_str() ) )
      {
              std::cerr << "    " << inputDir.c_str() << " is no directory!";
      }
      return EXIT_FAILURE;
  }

  //create output directory
  std::string outDir = nih::getPath(outputFilenamePrefix);
  std::cout << "  Creating output directory at: " << outDir << std::endl;
  itk::FileTools::CreateDirectory( outDir.c_str() );

  std::string inputImageName;
  std::string outputImageName;
  char count_char[100];
  double *translation = new double[3];
  double *rotationAxis = new double[3];
  rotationAxis[0] = 0.0;
  rotationAxis[1] = 0.0;
  rotationAxis[2] = 1.0; // Z-axis
  double rotationAngle = 0.0;
  itk::SmartPointer<InputImageType> image = NULL;
  itk::SmartPointer<InputImageType> deformed = NULL;

  OutputImageType::SizeType outImageSize;
  outImageSize[0] = outSize;
  outImageSize[1] = outSize;
  for ( unsigned long i = 0; i < dir->GetNumberOfFiles(); i++ )
  {
    // Only read files with searchString
    if ( std::string(dir->GetFile(i)).find( searchString ) !=std::string::npos )
    {
      image_count++;
      inputImageName = inputDir + "/" + dir->GetFile(i);
      printf(" Reading image %d of %d: %s ...\n", i+1, dir->GetNumberOfFiles(), inputImageName.c_str() );
      // Read intensity image
      reader->SetFileName( inputImageName );
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
      image = reader->GetOutput();

      // Deformations //
      for (unsigned int t=0; t<=Ntranslations; ++t)
      {
        if (t>0) // randomly deform
        {
          translation = nih::getRandomUniformPoint< double >( translation ); // random direction
          translation[0] = translation[0] * translation_max;
          translation[1] = translation[1] * translation_max;
          deform_count++;
          printf(" %d.: %d. translation by [%g, %g]...\n", i+1, t, translation[0], translation[1]);
          deformed = nih::translateImage<InputImageType,OutputImageType, Dimension>(image, translation, interpolationType);
        }
        else // no deform
        {
          deformed = image;
        }
        for (unsigned int r=0; r<=Nrotations; ++r)
        {
          if (r>0) // randomly deform
          {
            deform_count++;
            rotationAngle = nih::getRandomVariateUniformDouble( rotation_max );
            printf(" %d.: %d. rotation around [%g, %g, %g] by %g degrees...\n", i+1, r, rotationAxis[0], rotationAxis[1], rotationAxis[2], rotationAngle);
            deformed = nih::rotateImageAroundCenter<OutputImageType,OutputImageType, Dimension>(deformed, rotationAxis, rotationAngle, interpolationType);
          }
          for (unsigned int d=0; d<=Nnonrigiddeforms; ++d)
          {
            if (d>0) // randomly deform
            {
              deform_count++;
              printf(" %d.: %d. rotation\n", i+1, d);
              deformed = nih::randomWarpImage2D<OutputImageType,OutputImageType>(deformed, Nnonrigid_points, nonrigid_deform_max, siffness);
            }

            // Write image patch


            sprintf(count_char, "_t%03d_r%03d_d%03d", t, r, d);
            outputImageName = outputFilenamePrefix + itksys::SystemTools::GetFilenameWithoutExtension(dir->GetFile(i)).c_str() + count_char + ".png";
            writer->SetInput( nih::rescaleImage2D<OutputImageType,OutputImageType>(deformed, outImageSize) );
            writer->SetFileName( outputImageName );
            std::cout << " writing image patch to " << outputImageName << std::endl;
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
          } // Nnonrigiddeforms
        } // Nrotations
      } // Ntranslations

    } // if ( strcmp(dir.GetFile(i), searchString.c_str()) == 0 )
  } // for ( unsigned long i = 0; i < dir.GetNumberOfFiles(); i++ )

  printf(" read %d images...\n", image_count);
  printf(" saved %d deformed prefix %s...\n", deform_count);

  return EXIT_SUCCESS;
}
