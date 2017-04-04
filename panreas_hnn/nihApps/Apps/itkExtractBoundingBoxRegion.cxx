/*=========================================================================
 *
 *  Copyright National Institutes of Health
 *
 *  Unless required by applicable law or agreed to in writing, this software
 *  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 *  ANY KIND, either express or implied.
 *
 *=========================================================================*/

#include <iostream>
#include <fstream>
#include <string>

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkNumericTraits.h>
#include <itkFileTools.h>
#include <itkLabelGeometryImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkExtractImageFilter.h>

const unsigned int Dimension = 3;
#include <nihHelperFunctions.h>

static const char * argv_hc[] = {"*.exe",
	"/home/rothhr/Data/Pancreas/MICCAI2016/PancreasRawData/Training/img/img0001.nii.gz",
  "/home/rothhr/Data/Pancreas/MICCAI2016/PancreasRawData/Training/label/label0001.nii.gz",
	"/home/rothhr/Data/Pancreas/MICCAI2016/PancreasRawData/Training/img-cropped/img0001.nii.gz",
  "/home/rothhr/Data/Pancreas/MICCAI2016/PancreasRawData/Training/label-cropped/label0001.nii.gz",
};

#define HARDCODEDPARAMS 0

typedef int PixelType; // Images, segmentation
typedef itk::Image<PixelType, Dimension>  ImageType;

typedef int LabelType;
typedef itk::Image<LabelType, Dimension>  LabelImageType;

typedef itk::ImageFileReader<ImageType>                              ReaderType;
typedef itk::ImageFileReader<LabelImageType>                              LabelReaderType;
typedef itk::ImageFileWriter<ImageType>                              WriterType;
typedef itk::ImageFileWriter<LabelImageType>                              LabelWriterType;

int main(int argc, const char **argv)
{
  int BACKGROUND = 0;
  int FOREGROUND = 1;


#if HARDCODEDPARAMS
	argv = argv_hc;
	argc = sizeof(argv_hc)/sizeof(const char*);
#endif

  // check input parameters
  if( argc != 5 && argc != 6 )
  {
    std::cout << "argc = " << argc << std::endl;
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "  ImageFilename LabelFilename CroppedImageFilename CroppedLabelFilename [margin_mm]" << std::endl;
    return EXIT_FAILURE;
  }

  // parse inputs
  unsigned int idx = 0;
  const char* ImageFilename = argv[++idx];
  const char* LabelFilename = argv[++idx];
  std::string CroppedImageFilename = argv[++idx];
  std::string CroppedLabelFilename = argv[++idx];
  float margin_mm = 0.0;
  if (argc>5)
  {
    margin_mm = atof(argv[++idx]);
  }
  std::cout << "Extract bouding box using " << margin_mm << " mm margins..." << std::endl;

  //** RUN PIPELINE **//
  //read image
  ReaderType::Pointer imageReader = ReaderType::New();
  std::cout << "Read image " << ImageFilename << std::endl;
  imageReader->SetFileName( ImageFilename );
  imageReader->Update();
  itk::SmartPointer<ImageType> intensityImage = imageReader->GetOutput();

  //read label
  LabelReaderType::Pointer labelReader = LabelReaderType::New();
  std::cout << "Read label " << LabelFilename << std::endl;
  labelReader->SetFileName( LabelFilename );
  labelReader->Update();
  itk::SmartPointer<LabelImageType> labelImage = labelReader->GetOutput();

  // threshold image to get one label only (for bounding box computation)!
  typedef itk::BinaryThresholdImageFilter <LabelImageType, LabelImageType>
    BinaryThresholdImageFilterType;

  BinaryThresholdImageFilterType::Pointer thresholdFilter
    = BinaryThresholdImageFilterType::New();
  thresholdFilter->SetInput(labelImage);
  thresholdFilter->SetLowerThreshold(BACKGROUND+1); //>= lowerThreshold
  thresholdFilter->SetInsideValue(FOREGROUND);
  thresholdFilter->SetOutsideValue(BACKGROUND);

  // get bounding box
  typedef itk::LabelGeometryImageFilter< LabelImageType > LabelGeometryImageFilterType;
  LabelGeometryImageFilterType::Pointer labelGeometryImageFilter = LabelGeometryImageFilterType::New();
  labelGeometryImageFilter->SetInput( thresholdFilter->GetOutput() );

  labelGeometryImageFilter->Update();
  LabelGeometryImageFilterType::LabelsType allLabels =
    labelGeometryImageFilter->GetLabels();

  int labelValue = FOREGROUND;
  std::cout << "\tLabel: " << labelValue << std::endl;
  std::cout << "\tBounding box (min-max pairs): "
              << labelGeometryImageFilter->GetBoundingBox(labelValue) << std::endl;
  std::cout << "\tBounding box size: "
              << labelGeometryImageFilter->GetBoundingBoxSize(labelValue) << std::endl;

  LabelImageType::SpacingType labelSpacing = labelImage->GetSpacing();
  LabelImageType::SizeType labelImageSize = labelImage->GetLargestPossibleRegion().GetSize();
  LabelImageType::IndexType margin_vx;
  for (unsigned int i=0; i<Dimension; i++)
  {
    margin_vx[i] = std::ceil(margin_mm/labelSpacing[i]);
  }
  // Extract bounding box regions
  ImageType::IndexType desiredStart;
  desiredStart[0] = labelGeometryImageFilter->GetBoundingBox(labelValue)[0] - margin_vx[0]; // minx
  desiredStart[1] = labelGeometryImageFilter->GetBoundingBox(labelValue)[2] - margin_vx[1]; // miny
  desiredStart[2] = labelGeometryImageFilter->GetBoundingBox(labelValue)[4] - margin_vx[2]; // minz
  // adjust margins to image dims
  for (unsigned int i=0; i<Dimension; i++)
  {
    if (desiredStart[i]<0)
    {
      desiredStart[i] = 0;
    }
  }

  ImageType::SizeType desiredSize = labelGeometryImageFilter->GetBoundingBoxSize(labelValue);
  //add margin to desired size
  for (unsigned int i=0; i<Dimension; i++)
  {
    if ( (desiredStart[i] + desiredSize[i] + 2*margin_vx[i]) <labelImageSize[i])
    {
      desiredSize[i] += 2*margin_vx[i];
    }
    else
    {
      desiredSize[i] = labelImageSize[i] - desiredStart[i];
    }
  }

  ImageType::RegionType desiredRegion(desiredStart, desiredSize);

  // extract and write cropped images
  typedef itk::ExtractImageFilter< ImageType, ImageType > ExtractFilterType;
  typedef itk::ExtractImageFilter< LabelImageType, LabelImageType > LabelExtractFilterType;

  // image
  itk::SmartPointer<ExtractFilterType> extractFilter = ExtractFilterType::New();
  extractFilter->SetExtractionRegion(desiredRegion);
  extractFilter->SetInput(intensityImage);
#if ITK_VERSION_MAJOR >= 4
  extractFilter->SetDirectionCollapseToIdentity(); // This is required.
#endif

  std::cout << "  Original image sum " << nih::getImageSum<ImageType>(intensityImage)
      << " vs. cropped image sum " << nih::getImageSum<ImageType>(extractFilter->GetOutput()) << std::endl;

  //create output directory
  std::string outDir = nih::getPath(CroppedImageFilename);
  itk::FileTools::CreateDirectory( outDir.c_str() );

  itk::SmartPointer<WriterType> writer = WriterType::New();
  writer->SetInput( extractFilter->GetOutput() );
  writer->SetFileName( CroppedImageFilename );
  writer->Update();
  std::cout << " -> saved cropped image as " << CroppedImageFilename << std::endl;

  // label
  itk::SmartPointer<LabelExtractFilterType> labelExtractFilter = LabelExtractFilterType::New();
  labelExtractFilter->SetExtractionRegion(desiredRegion);
  labelExtractFilter->SetInput(labelImage);
#if ITK_VERSION_MAJOR >= 4
  labelExtractFilter->SetDirectionCollapseToIdentity(); // This is required.
#endif

  std::cout << "  Original label sum " << nih::getImageSum<LabelImageType>(labelImage)
      << " vs. cropped label sum " << nih::getImageSum<LabelImageType>(labelExtractFilter->GetOutput()) << std::endl;

  //create output directory
  outDir = nih::getPath(CroppedLabelFilename);
  itk::FileTools::CreateDirectory( outDir.c_str() );

  itk::SmartPointer<LabelWriterType> labelWriter = LabelWriterType::New();
  labelWriter->SetInput( labelExtractFilter->GetOutput() );
  labelWriter->SetFileName( CroppedLabelFilename );
  labelWriter->Update();
  std::cout << " -> saved cropped label as " << CroppedLabelFilename << std::endl;

  // save bounding box
  std::ofstream log((CroppedLabelFilename+"BoundingBox.txt").c_str());
  log << "desiredStart: " << desiredStart << std::endl;
  log << "desiredSize: " << desiredSize << std::endl;
  log.close();

  return EXIT_SUCCESS;
}
