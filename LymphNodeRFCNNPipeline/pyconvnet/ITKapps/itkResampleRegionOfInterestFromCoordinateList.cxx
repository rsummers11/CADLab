/*=========================================================================
 *
 *  Copyright National Institutes of Health
 *
 *  Unless required by applicable law or agreed to in writing, this software
 *  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF 
 *  ANY KIND, either express or implied.
 *
 *=========================================================================*/

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkIsotropicWindowingResampleImageFilter.h>
#include <itkRegionOfInterestImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkConstantPadImageFilter.h>
#include <itkImageDuplicator.h>
#include <itkFlatStructuringElement.h>
#include <itkBinaryDilateImageFilter.h>
#include <itkBinaryNotImageFilter.h>
#include <itkAndImageFilter.h>
#include <itkFileTools.h>
#include <itkTimeProbe.h>
#include <itkMath.h.>

#include <nihHelperFunctions.h>
/*
static const char * argv_hc[] = {"*.exe",
	"D:/HolgerRoth/data/LymphNodes/test/U12GV2G1/U12GV2G1.nii.gz", //inputImageFile
	"D:/HolgerRoth/data/LymphNodes/test/U12GV2G1/test_hc/U12GV2G1_win_iso_hc", //outputFilenamePrefix
  "-101", //lower[HU]
  "201", //upper[HU]
  "D:/HolgerRoth/data/LymphNodes/test/U12GV2G1/U12GV2G1_lymphnodes_physicalPoints.txt", //roiCentersFilename
  "30", //cubeSize[mm] 
  "32", //numberROIvoxels
  "2", //numberRandomTranslations
  "3.0", // displacementLimitRandomTranslations[mm]
  "2" // numberRandomRotations
};
*/
/*
static const char * argv_hc[] = {"*.exe",
	"D:/HolgerRoth/data/Spine/BoneLesions/BoneLesions_NII/normal_case1/normal_case1.nii.gz",
  "D:/HolgerRoth/data/Spine/BoneLesions/BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered/normal_cases/normal_case1/normal_case1_normal_case1_s30mm_neg_boneCADe",
  "0",
  "500",
  "D:/HolgerRoth/data/Spine/BoneLesions/BoneLesions_NII/normal_case1/normal_case1_neg_boneCADe_physicalPoints.txt",
  "30",
  "32",
  "5",
  "3.0",
  "5",
  "BSPLINE"
};
*/
#define HARDCODEDPARAMS 0

bool WRITE_ISOMETRIC_IMAGE = false;
bool WRITE_SEGMENTATION_IMAGE = false;

const unsigned int Dimension = 3;

typedef float InputPixelType;
typedef unsigned short OutputPixelType; // 16-bit png images
//typedef unsigned char OutputPixelType; // 8-bit png image
typedef unsigned char SegmentationPixelType; 
typedef itk::RGBPixel<OutputPixelType> RGBPixelType;

typedef itk::Image<InputPixelType, Dimension>                                   InputImageType;
typedef itk::Image<OutputPixelType, Dimension>                                  OutputImageType;
typedef itk::Image<SegmentationPixelType, Dimension>                            SegmentationImageType;
typedef itk::Image<RGBPixelType, 2>                                             RGBImageType;
typedef itk::Image<OutputPixelType, 2>                                            GrayImageType;
typedef itk::ImageFileReader<InputImageType>                                    ReaderType;
typedef itk::ImageFileWriter<OutputImageType>                                   WriterType;
typedef itk::ImageFileWriter<RGBImageType>                                      RGBWriterType;
typedef itk::ImageFileWriter<GrayImageType>                                     GrayWriterType;
typedef itk::ImageFileWriter<SegmentationImageType>                             SegmentationWriterType;
typedef itk::IsotropicWindowingResampleImageFilter<InputImageType, OutputImageType>      IsoFilterType;
typedef itk::RegionOfInterestImageFilter<OutputImageType, OutputImageType>      RoiFilterType;
typedef itk::BinaryThresholdImageFilter<OutputImageType, SegmentationImageType> BinaryThresholdFilterType;
typedef itk::ConstantPadImageFilter<OutputImageType, OutputImageType>           PadImageFilterType;

const SegmentationPixelType insideValue = 255;
const SegmentationPixelType outsideValue = 0;

int main(int argc, const char **argv)
{

#if HARDCODEDPARAMS
	argv = argv_hc;
	argc = sizeof(argv_hc)/sizeof(const char*);
#endif

  // check input parameters
  if( argc != 13 )
  {
    std::cout << "argc = " << argc << std::endl;
    for (unsigned int i=0; i<argc; i++)
    {
      std::cout << i+1 << ".: " << argv[i] << std::endl;
    }
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " inputImageFile outputFilenamePrefix lower[HU] upper[HU] roiCentersFilename cubeSize[mm] numberROIvoxels numberRandomTranslations displacementLimitRandomTranslations[mm] numberRandomRotations interpolationType{NEAREST,LINEAR,BSPLINE} transformType{XYZ,XY}." << std::endl;
    return EXIT_FAILURE;
  }

  const char* InputFilename = argv[1];
  std::string OutputFilenamePrefix = argv[2];
  std::string OutputFilename;
  const InputPixelType  winMin = atof( argv[3] );
  const InputPixelType  winMax = atof( argv[4] );
  const OutputPixelType desiredMin = 0.0;
  const OutputPixelType desiredMax = itk::NumericTraits< OutputPixelType >::max(); // intensity range for PNG images
  const OutputPixelType lowerThreshold = desiredMin + 1;
  const OutputPixelType upperThreshold = desiredMax;

  const char* roiCentersFilename = argv[5];
  const float cubeSize = atof( argv[6] ); // in mm
  const unsigned int numberROIvoxels = atoi( argv[7] ); 

  const unsigned int numberRandomTranslations = atoi( argv[8] ); 
  const double displacementLimitRandomTranslations = atof( argv[9] );  
  const unsigned int numberRandomRotations = atoi( argv[10] );
  
  const char* interpTypeChoice = argv[11];
  nih::InterpolationType interpolationType;
  if ( _strcmpi(interpTypeChoice,"NEAREST") == 0 )
  {
    interpolationType = nih::NEAREST;
  }
  else if ( _strcmpi(interpTypeChoice,"LINEAR") == 0 )
  {
    interpolationType = nih::LINEAR;
  }
  else if ( _strcmpi(interpTypeChoice,"BSPLINE") == 0 )
  {
    interpolationType = nih::BSPLINE;
  }
  else
  {
    std::cerr << " No such interpolation type: " << interpTypeChoice << std::endl;
    return EXIT_FAILURE;
  }

  const char* transformType = argv[12];
  bool ONLY_XY;
  if ( _strcmpi(transformType,"XY") == 0 )
  {
    std::cout << " Transformation are computed in XY ONLY." << std::endl;
    ONLY_XY = true;
  }
  else if ( _strcmpi(transformType,"XYZ") == 0 )
  {
    std::cout << " Transformation are computed in XYZ." << std::endl;
    ONLY_XY = false;
  }
  else
  {
    std::cerr << " No such transformation type: " << transformType << std::endl;
    return EXIT_FAILURE;    
  }

  printf("Number random transformations:\n  %d Translations, %d Rotations...\n",numberRandomTranslations,numberRandomRotations);
  printf("  displacementLimitRandomTranslations = %f\n",displacementLimitRandomTranslations);

  //** RUN PIPELINE **//
  double isoSpacing = cubeSize/numberROIvoxels;

  //create output directory
  std::string outDir = nih::getPath(OutputFilenamePrefix); 
  std::cout << "  Creating output directory at: " << outDir << std::endl;
  itk::FileTools::CreateDirectoryA( outDir.c_str() );

  //read image
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( InputFilename );
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

  // Make image isometric
  IsoFilterType::Pointer isoFilter = IsoFilterType::New();
  isoFilter->SetInput( reader->GetOutput() );
  // windowing options
  isoFilter->SetInputWinMin(winMin);
  isoFilter->SetInputWinMax(winMax);
  isoFilter->SetOutputWinMin( desiredMin );
  isoFilter->SetOutputWinMax( desiredMax );
  // isometric spacing options
  isoFilter->SetInterpolationType( interpolationType );
  isoFilter->SetIsoSpacing( isoSpacing );
  try
  {
    isoFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
  }

  OutputImageType::Pointer isoImage = isoFilter->GetOutput();
  OutputImageType::SizeType isoImageSize = isoImage->GetLargestPossibleRegion().GetSize();

  printf("  isoImageSize before padding [%d, %d, %d]\n", isoImageSize[0], isoImageSize[1], isoImageSize[2]);
  // Zero pad iso image to ensure not extracting ROIs outside of image boundary
  //unsigned long paddingSize = (displacementLimitRandomTranslations/isoSpacing) + numberROIvoxels + 1; // to be on the save side
  unsigned long paddingSize = 1.5*numberROIvoxels + 2;
  std::cout << "  paddingSize = " << paddingSize << std::endl;
  const unsigned long Padding[Dimension] = { paddingSize, paddingSize, paddingSize };
  PadImageFilterType::Pointer paddingFilter = PadImageFilterType::New();
  paddingFilter->SetInput(isoImage);
  paddingFilter->SetConstant( itk::NumericTraits< OutputPixelType >::quiet_NaN()  );
  paddingFilter->SetPadLowerBound(Padding);
  paddingFilter->SetPadUpperBound(Padding);
  try
  {
    paddingFilter->UpdateLargestPossibleRegion();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
  }
  isoImage = paddingFilter->GetOutput();
  isoImageSize = isoImage->GetLargestPossibleRegion().GetSize();
  printf("  isoImageSize after padding [%d, %d, %d]\n", isoImageSize[0], isoImageSize[1], isoImageSize[2]);

  // segment tissue
  printf("  thresholding between %d and %d.\n", lowerThreshold, upperThreshold);
  BinaryThresholdFilterType::Pointer thresholdFilter = BinaryThresholdFilterType::New();
  thresholdFilter->SetInput(isoImage);
  thresholdFilter->SetLowerThreshold(lowerThreshold);
  thresholdFilter->SetUpperThreshold(upperThreshold);
  thresholdFilter->SetInsideValue( insideValue );
  thresholdFilter->SetOutsideValue( outsideValue ); 
  try
  {
    thresholdFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
  }
  SegmentationImageType::Pointer tissueImage = thresholdFilter->GetOutput();
  // Duplicate tissueImage to label used positive ROIs
  typedef itk::ImageDuplicator< SegmentationImageType > DuplicatorType;
  DuplicatorType::Pointer duplicateFilter = DuplicatorType::New();
  duplicateFilter->SetInputImage( tissueImage );
  duplicateFilter->Update();
  try
  {
    duplicateFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
  }
  SegmentationImageType::Pointer posRoiImage = duplicateFilter->GetOutput();
  // set all voxels to 'outside'
  posRoiImage->FillBuffer( outsideValue); 

  // write isometric image
  WriterType::Pointer writer = WriterType::New();
  RGBWriterType::Pointer rgbWriter = RGBWriterType::New();
  GrayWriterType::Pointer grayWriter = GrayWriterType::New();

  if (WRITE_ISOMETRIC_IMAGE)
  {
    writer->SetInput( isoImage );
    OutputFilename = OutputFilenamePrefix + "_isoImage.nii.gz";
    writer->SetFileName( OutputFilename );
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
  }

  // ROI parameters
  // Extract region of interest
  RoiFilterType::Pointer superRoiFilter = RoiFilterType::New();
  RoiFilterType::Pointer subRoiFilter   = RoiFilterType::New();
  superRoiFilter->SetInput( isoImage );

  OutputImageType::Pointer    superROIImage;
  OutputImageType::RegionType superROIRegion;
  OutputImageType::IndexType  superROIIndex;
  OutputImageType::SizeType   superRoiSize;
  superRoiSize[0] = 3*numberROIvoxels;
  superRoiSize[1] = 3*numberROIvoxels;
  superRoiSize[2] = 3*numberROIvoxels;
  superROIRegion.SetSize( superRoiSize );
  std::cout << "  superRoiSize = " << superRoiSize << std::endl;

  OutputImageType::Pointer    subROIImage;
  OutputImageType::RegionType subROIRegion;
  OutputImageType::IndexType  subROIIndex;
  OutputImageType::SizeType   subRoiSize;
  subRoiSize[0] = numberROIvoxels;
  subRoiSize[1] = numberROIvoxels;
  subRoiSize[2] = numberROIvoxels;
  subROIRegion.SetSize( subRoiSize );
  std::cout << "  subRoiSize = " << subRoiSize << std::endl;

  subROIIndex[0] = numberROIvoxels;
  subROIIndex[1] = numberROIvoxels;
  subROIIndex[2] = numberROIvoxels;

  // read roi center points text file
  std::ifstream iFile( roiCentersFilename );
  if (!iFile)
  {
    std::cerr << " Error: could not find " << roiCentersFilename << "!" << std::endl;
  }

  char count_char[100];
  double physPt0, physPt1, physPt2;
  OutputImageType::PointType roiCenterPhys;

  OutputImageType::IndexType startIdx;
  unsigned int NumberROIs = 0;

  //===================================================================
  // Generate Regions of Interest
  //===================================================================

  /* While there is still a line. */
  while (iFile >> physPt0 >> physPt1 >> physPt2)
  {
    //printf("  %d: roiCenter at [%g, %g, %g] mm\n", NumberROIs, physPt0, physPt1, physPt2);
    roiCenterPhys[0] = physPt0 - cubeSize/2;
    roiCenterPhys[1] = physPt1 - cubeSize/2;
    roiCenterPhys[2] = physPt2 - cubeSize/2;

    isoImage->TransformPhysicalPointToIndex( roiCenterPhys, startIdx );
    // shift by one (+1 seems to align the regions better!)
    startIdx[0] = startIdx[0];
    startIdx[1] = startIdx[1];
    startIdx[2] = startIdx[2] - 1;
    printf("  %d: [%g, %g, %g] mm\t maps to index [%d, %d, %d].\n", NumberROIs, 
            roiCenterPhys[0], roiCenterPhys[1], roiCenterPhys[2],
                  startIdx[0], startIdx[1], startIdx[2]);

    subROIRegion.SetIndex( startIdx );
    if (WRITE_SEGMENTATION_IMAGE)
    {
      // "label" ROI regions in posRoiImage in ROIs
      itk::ImageRegionIterator<SegmentationImageType> roiIterator(posRoiImage, subROIRegion);  
      while(!roiIterator.IsAtEnd())
      {
        roiIterator.Set( insideValue );
        ++roiIterator;
      }
    }

    // extract a SUPER region of interest
    superROIIndex[0] = startIdx[0] - numberROIvoxels;
    superROIIndex[1] = startIdx[1] - numberROIvoxels;
    superROIIndex[2] = startIdx[2] - numberROIvoxels;
    superROIRegion.SetIndex( superROIIndex );

    superRoiFilter->SetRegionOfInterest(superROIRegion);
    try
    {
      superRoiFilter->Update();
      //superRoiFilter->UpdateLargestPossibleRegion();
    }
    catch( itk::ExceptionObject & excep )
    {
      std::cerr << "Exception caught !" << std::endl;
      std::cerr << excep << std::endl;
      std::cout << " superROIRegion.GetIndex() = " << superROIRegion.GetIndex() << std::endl;
      std::cout << " superROIRegion.Size() = " << superROIRegion.GetSize() << std::endl;
      std::cout << " superRoiFilter-InputImageSize = " << superRoiFilter->GetInput()->GetLargestPossibleRegion().GetSize() << std::endl;
      return EXIT_FAILURE;
    }
    superROIImage = superRoiFilter->GetOutput();

    OutputImageType::PointType randVec;
    double *rotationAxis = new double[3];
    double rotationAngle = 0.0;

    for (unsigned int t = 0; t < numberRandomTranslations; t++)
    {
      if (t>0) // initially, stay on centroit
      {
        // convert mm to voxels
        randVec = nih::getRandomUniformPoint< OutputImageType >(); // random direction
        double random_length = nih::getRandomVariateUniformDouble( 1.0 ) *displacementLimitRandomTranslations;
        randVec[0] = randVec[0] * random_length;
        randVec[1] = randVec[1] * random_length;
        randVec[2] = randVec[2] * random_length;
        if (ONLY_XY)
        {
          randVec[2] = 0.0;
        }
        double shift_distance = nih::getPointNorm< OutputImageType >(randVec);
        if (shift_distance > cubeSize/2)
        {
          std::cerr << "  shifting distance should not be larger than half the cube size (" 
            << shift_distance << " > " << cubeSize/2 << ")!" << std::endl;
        }
        else
        {
          //printf("    -> %d. region shifted by [%g, %g, %g] mm...\n",  t, randVec[0], randVec[1], randVec[2]);
          superROIImage = nih::translateImage<OutputImageType,OutputImageType>(superROIImage, randVec.GetDataPointer(), interpolationType);
        }
      }

      for (unsigned int r = 0; r < numberRandomRotations; r++)
      {
        // rotate ROI

        if (r>0) // no ration at first iteration
        {
          if (ONLY_XY)
          {
            rotationAxis[0] = 0.0;
            rotationAxis[1] = 0.0;
            rotationAxis[2] = 1.0;
          }
          else
          {
            rotationAxis[0] = nih::getRandomVariateUniformDouble( 1.0 );
            rotationAxis[1] = nih::getRandomVariateUniformDouble( 1.0 );
            rotationAxis[2] = nih::getRandomVariateUniformDouble( 1.0 );
          }
          rotationAngle = nih::getRandomVariateUniformDouble( 360.0 );

          /*printf("    -> %d. region rotated by %g degrees around [%g, %g, %g]...\n", 
            r, rotationAngle, rotationAxis[0], rotationAxis[1], rotationAxis[2]);*/

          // rotate
          superROIImage = nih::rotateImageAroundCenter<OutputImageType, OutputImageType>(superROIImage, rotationAxis, rotationAngle, interpolationType);

        } // if (r>0)

        // get rotated ROI from "larger" translated and rotated ROI
        subROIRegion.SetIndex( subROIIndex );

        subRoiFilter->SetInput( superROIImage ); 
        subRoiFilter->SetRegionOfInterest( subROIRegion );
        try
        {
          subRoiFilter->Update();
        }
        catch( itk::ExceptionObject & excep )
        {
          std::cerr << "Exception caught !" << std::endl;
          std::cerr << excep << std::endl;
          return EXIT_FAILURE;
        }
        subROIImage = subRoiFilter->GetOutput();
        
        //****************** WRITE OUTPUTS **************************//
        sprintf_s(count_char, "_ROI%05d_t%05d_r%05d", NumberROIs, t, r);

        // write ROI output and centroid text file
        if (r==0 && t==0) // only non-transformed ROI
        {
          // ROI volume
          writer->SetInput( subROIImage );
          OutputFilename = OutputFilenamePrefix + count_char + ".nii.gz";
          writer->SetFileName( OutputFilename );
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
        }

        // Text centroids 
        OutputFilename = OutputFilenamePrefix + count_char + ".txt";
        std::ofstream oCentroidsFile( OutputFilename );
        oCentroidsFile << physPt0 << " "
                       << physPt1 << " "
                       << physPt2 << std::endl;
        oCentroidsFile << randVec[0] << " "
                       << randVec[1] << " "
                       << randVec[2] << std::endl;
        oCentroidsFile << rotationAxis[0] << " "
                       << rotationAxis[1] << " "
                       << rotationAxis[2] << " "
                       << rotationAngle << std::endl;
        oCentroidsFile.close();

        // extract RGB image patch
        if (ONLY_XY)
        {
          rgbWriter->SetInput( nih::getRGBImagePatchXY<OutputImageType,RGBImageType>( subROIImage ) );
          OutputFilename = OutputFilenamePrefix + count_char + "_AxRGB.png";
        }
        else
        {
          rgbWriter->SetInput( nih::getRGBImagePatch<OutputImageType,RGBImageType>( subROIImage ) );
          OutputFilename = OutputFilenamePrefix + count_char + "_AxCoSa.png";
        }
        rgbWriter->SetFileName( OutputFilename );
        try
        {
          rgbWriter->Update();
        }
        catch( itk::ExceptionObject & excep )
        {
          std::cerr << "Exception caught !" << std::endl;
          std::cerr << excep << std::endl;
          return EXIT_FAILURE;
        }

        // extract axial/Z image patch
        grayWriter->SetInput( nih::getGrayImagePatch<OutputImageType,GrayImageType>( subROIImage, 2 ) );
        OutputFilename = OutputFilenamePrefix + count_char + "_AX.png";
        grayWriter->SetFileName( OutputFilename );
        try
        {
          grayWriter->Update();
        }
        catch( itk::ExceptionObject & excep )
        {
          std::cerr << "Exception caught !" << std::endl;
          std::cerr << excep << std::endl;
          return EXIT_FAILURE;
        }

        // extract CORONAL/X image patch
        grayWriter->SetInput( nih::getGrayImagePatch<OutputImageType,GrayImageType>( subROIImage, 0 ) );
        OutputFilename = OutputFilenamePrefix + count_char + "_CO.png";
        grayWriter->SetFileName( OutputFilename );
        try
        {
          grayWriter->Update();
        }
        catch( itk::ExceptionObject & excep )
        {
          std::cerr << "Exception caught !" << std::endl;
          std::cerr << excep << std::endl;
          return EXIT_FAILURE;
        }

        // extract SAGITTAL/Y image patch
        grayWriter->SetInput( nih::getGrayImagePatch<OutputImageType,GrayImageType>( subROIImage, 1 ) );
        OutputFilename = OutputFilenamePrefix + count_char + "_SA.png";
        grayWriter->SetFileName( OutputFilename );
        try
        {
          grayWriter->Update();
        }
        catch( itk::ExceptionObject & excep )
        {
          std::cerr << "Exception caught !" << std::endl;
          std::cerr << excep << std::endl;
          return EXIT_FAILURE;
        }
      } /* for (unsigned int r = 0; r < numberRandomTranslations; r++) */

    } /* for (unsigned int t = 0; t < numberRandomTranslations; t++) */

    if ( (NumberROIs%100)==0 && (NumberROIs>0) )
    {
      printf("  %d ROIs so far...\n", NumberROIs);
    }

    NumberROIs++;
  } /* while (iFile >> physPt0 >> physPt1 >> physPt2) */
  iFile.close();

  unsigned int numberTotalT = 0; 
  unsigned int numberTotalR = 0;
  if (numberRandomTranslations>1)
  {
    numberTotalT = numberRandomTranslations*NumberROIs;
  }
  if (numberRandomRotations>1)
  {
    numberTotalR = numberRandomRotations*NumberROIs;
  }  
  printf("  Found %d ROIs of size [%d, %d, %d] (%d translations, %d rotations).\n", 
    NumberROIs, numberROIvoxels, numberROIvoxels, numberROIvoxels, 
    numberTotalT,
    numberTotalR);

  // write segmentation output
  if (WRITE_SEGMENTATION_IMAGE)
  {
    SegmentationWriterType::Pointer segWriter = SegmentationWriterType::New();
    segWriter->SetInput( tissueImage );
    OutputFilename = OutputFilenamePrefix + "_Tissue.nii.gz";
    segWriter->SetFileName( OutputFilename );
    try
    {
      segWriter->Update();
    }
    catch( itk::ExceptionObject & excep )
    {
      std::cerr << "Exception caught !" << std::endl;
      std::cerr << excep << std::endl;
      return EXIT_FAILURE;
    }

    // write segmentation output
    segWriter->SetInput( posRoiImage );
    OutputFilename = OutputFilenamePrefix + "_extractedRegionsOfInterest.nii.gz";
    segWriter->SetFileName( OutputFilename );
    try
    {
      segWriter->Update();
    }
    catch( itk::ExceptionObject & excep )
    {
      std::cerr << "Exception caught !" << std::endl;
      std::cerr << excep << std::endl;
      return EXIT_FAILURE;
    }
  }
 
  std::cout << "Results save with prefix " << OutputFilenamePrefix << std::endl;

  return EXIT_SUCCESS;
}


