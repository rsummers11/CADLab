#pragma once
#ifndef _NIHHELPERFUNCTIONS_H_
#define _NIHHELPERFUNCTIONS_H_

#if defined(_WIN32) || defined(_WIN64)
  #define snprintf _snprintf
  #define vsnprintf _vsnprintf
  #define strcasecmp _stricmp
  #define strncasecmp _strnicmp
#endif

#include <iostream>
#include <fstream>
#include <vector>

#include <itkImage.h>
#include <itkExtractImageFilter.h>
#include <itkMersenneTwisterRandomVariateGenerator.h>
#include <itkImageFunction.h>
#include <itkResampleImageFilter.h>
#include <itkAffineTransform.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkBSplineInterpolateImageFunction.h>
#include <itkConstantBoundaryCondition.h>
#include <itkWindowedSincInterpolateImageFunction.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkBinaryDilateImageFilter.h>
#include <itkBinaryNotImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkAndImageFilter.h>
#include <itkLabelOverlapMeasuresImageFilter.h>
#include <itkVector.h>
#include <itkWarpImageFilter.h>
#include <itkMaskImageFilter.h>
#include <itkImageAdaptor.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkChangeInformationImageFilter.h>
#include <itkSignedMaurerDistanceMapImageFilter.h>
#include <itkPasteImageFilter.h>
#include <itkIntensityWindowingImageFilter.h>
#include <itkFileTools.h>
#include <itkStatisticsImageFilter.h>
#include <itkCropImageFilter.h>
#if ITK_VERSION_MAJOR < 4
  #include <itkDeformationFieldSource.h>
#else
  #include <itkLandmarkDisplacementFieldSource.h>
#endif
#if ITK_VERSION_MAJOR < 4
  #include <itkCompose3DCovariantVectorImageFilter.h>
#else
  #include <itkComposeImageFilter.h>
#endif

namespace nih
{

//================================================================================================//
//  nih::InterpolationType
//================================================================================================//
enum InterpolationType { NEAREST, LINEAR, BSPLINE };


//================================================================================================//
//  nih::getImageMinimum
//================================================================================================//
template <class TImageType>
double getImageMinimum(typename TImageType::Pointer Image)
{
  typedef itk::StatisticsImageFilter<TImageType> StatisticsImageFilterType;
  itk::SmartPointer<StatisticsImageFilterType> statisticsImageFilter
            = StatisticsImageFilterType::New ();
  statisticsImageFilter->SetInput(Image);
  statisticsImageFilter->Update();

  return statisticsImageFilter->GetMinimum();
} //  nih::getImageMinimum

//================================================================================================//
//  nih::getImageMaximum
//================================================================================================//
template <class TImageType>
double getImageMaximum(typename TImageType::Pointer Image)
{
  typedef itk::StatisticsImageFilter<TImageType> StatisticsImageFilterType;
  itk::SmartPointer<StatisticsImageFilterType> statisticsImageFilter
            = StatisticsImageFilterType::New ();
  statisticsImageFilter->SetInput(Image);
  statisticsImageFilter->Update();

  return statisticsImageFilter->GetMaximum();
} //  nih::getImageMaximum

//================================================================================================//
//  nih::getImageMean
//================================================================================================//
template <class TImageType>
double getImageMean(typename TImageType::Pointer Image)
{
  typedef itk::StatisticsImageFilter<TImageType> StatisticsImageFilterType;
  itk::SmartPointer<StatisticsImageFilterType> statisticsImageFilter
            = StatisticsImageFilterType::New ();
  statisticsImageFilter->SetInput(Image);
  statisticsImageFilter->Update();

  return statisticsImageFilter->GetMean();
} //  nih::getImageMean

//================================================================================================//
//  nih::getImageSum
//================================================================================================//
template <class TImageType>
double getImageSum(typename TImageType::Pointer Image)
{
  typedef itk::StatisticsImageFilter<TImageType> StatisticsImageFilterType;
  itk::SmartPointer<StatisticsImageFilterType> statisticsImageFilter
            = StatisticsImageFilterType::New ();
  statisticsImageFilter->SetInput(Image);
  statisticsImageFilter->Update();

  return statisticsImageFilter->GetSum();
} //  nih::getImageSum

//================================================================================================//
//  nih::getImageSigma
//================================================================================================//
template <class TImageType>
double getImageSigma(typename TImageType::Pointer Image)
{
  typedef itk::StatisticsImageFilter<TImageType> StatisticsImageFilterType;
  itk::SmartPointer<StatisticsImageFilterType> statisticsImageFilter
            = StatisticsImageFilterType::New ();
  statisticsImageFilter->SetInput(Image);
  statisticsImageFilter->Update();

  return statisticsImageFilter->GetSigma();
} //  nih::getImageSigma

//================================================================================================//
//  nih::readPointData
//================================================================================================//
std::vector< std::vector<double> > readPointData( const char *filename )
{
  std::cout << " Reading 3D point data from: " << filename << std::endl;
  std::vector<double> point(3);
  std::vector< std::vector<double> > points;

 // read roi center points text file
  std::ifstream iFile( filename );
  if (!iFile)
  {
    std::cerr << " Error: could not find " << filename << "!" << std::endl;
  }
  unsigned int Npoints = 0;
  double pt0, pt1, pt2;
  while (iFile >> pt0 >> pt1 >> pt2)
  {
    Npoints++;

    point[0] = pt0;
    point[1] = pt1;
    point[2] = pt2;

    printf("  %d.: (%g, %g, %g)\n", Npoints,
            pt0, pt1, pt2);

    points.push_back( point );
  }

  iFile.close();

  return points;
}

//================================================================================================//
//  nih::compareImageSizes (return true if the same size)
//================================================================================================//
template <class TSizeTypeA, class TSizeTypeB>
bool compareImageSizes(TSizeTypeA sizeA, TSizeTypeB sizeB)
{
  bool isSame = false;

  // check size 2D
  if (sizeA[0]==sizeB[0] && sizeA[1]==sizeB[1])
  {
    isSame = true;
  }
  // check size 3D
  if (sizeA[2]==sizeB[2])
  {
    isSame = true;
  }

  return isSame;
}

//================================================================================================//
//  nih::compareImageDims (return true if the same size and spacing)
//================================================================================================//
template <class TImageTypeA, class TImageTypeB>
bool compareImageDims(typename TImageTypeA::Pointer A, typename TImageTypeB::Pointer B)
{
  bool isSame = false;

  // check size
  typename TImageTypeA::SizeType    sizeA    = A->GetLargestPossibleRegion().GetSize();
  typename TImageTypeA::SpacingType spacingA = A->GetSpacing();

  typename TImageTypeB::SizeType    sizeB    = B->GetLargestPossibleRegion().GetSize();
  typename TImageTypeB::SpacingType spacingB = B->GetSpacing();
  if (sizeA[0]==sizeB[0] && sizeA[1]==sizeB[1] && sizeA[2]==sizeB[2] &&
      spacingA[0]==spacingB[0] && spacingA[1]==spacingB[1] && spacingA[2]==spacingB[2])
  {
    isSame = true;
  }

  return isSame;
}

//================================================================================================//
//  nih::composeRGBImage
//================================================================================================//
template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer composeRGBImage(typename TInputImageType::Pointer R, typename TInputImageType::Pointer G, typename TInputImageType::Pointer B)
{
  typedef itk::Image<typename TInputImageType::PixelType, 2  >                             SliceImageType;
  typedef itk::ExtractImageFilter<TInputImageType, SliceImageType >              SliceFilterType;
  #if ITK_VERSION_MAJOR < 4
    typedef itk::Compose3DCovariantVectorImageFilter<OutputSliceImageType, RGBImageType> ComposeImageFilterType;
  #else
    typedef itk::ComposeImageFilter<SliceImageType, TOutputImageType> ComposeImageFilterType;
  #endif

  // Compose RGB image
  typename TInputImageType::PointType origin;
  origin.Fill(0.0);
  R->SetOrigin( origin );
  G->SetOrigin( origin );
  B->SetOrigin( origin );

  typename TInputImageType::SpacingType spacing;
  spacing.Fill(1.0);
  R->SetSpacing( spacing );
  G->SetSpacing( spacing );
  B->SetSpacing( spacing );

  itk::SmartPointer<ComposeImageFilterType> composeFilter = ComposeImageFilterType::New();
  composeFilter->SetInput1( R );
  composeFilter->SetInput2( G );
  composeFilter->SetInput3( B );
  try
  {
    composeFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    exit(EXIT_FAILURE);
  }

  return composeFilter->GetOutput();
}

//================================================================================================//
//  nih::getRGBchannel
//================================================================================================//
template <class TRGBImageType, class TChannelImageType>
typename TChannelImageType::Pointer getRGBchannel(typename TRGBImageType::Pointer rgbImage, unsigned int channel)
{
  itk::SmartPointer<TChannelImageType> channelImage = TChannelImageType::New();
  channelImage->SetRegions( rgbImage->GetLargestPossibleRegion() );
  channelImage->CopyInformation( rgbImage );
  channelImage->Allocate();
  // select channels:
  // channel = 0 -> Red
  // channel = 1 -> Green
  // channel = 2 -> Blue

  // copy values from input image
  itk::ImageRegionConstIterator< TRGBImageType > rgbit(rgbImage, rgbImage->GetLargestPossibleRegion());
  itk::ImageRegionIterator< TChannelImageType >     cit(channelImage, channelImage->GetLargestPossibleRegion());
  switch (channel)
  {
    case 0:
      for (rgbit.GoToBegin(), rgbit.GoToBegin(); !rgbit.IsAtEnd(); ++rgbit, ++cit )
      {
        cit.Set( rgbit.Value().GetRed() );
      }
      break;
    case 1:
      for (rgbit.GoToBegin(), rgbit.GoToBegin(); !rgbit.IsAtEnd(); ++rgbit, ++cit )
      {
        cit.Set( rgbit.Value().GetGreen() );
      }
      break;
    case 2:
      for (rgbit.GoToBegin(), rgbit.GoToBegin(); !rgbit.IsAtEnd(); ++rgbit, ++cit )
      {
        cit.Set( rgbit.Value().GetBlue() );
      }
      break;
    default:
      std::cerr << " RGB channels can only be accessed with channel = 0, 1, 2." << std::endl;
      exit(EXIT_FAILURE);
  }

  return channelImage;
}

//================================================================================================//
//  nih::rescaleImage2D
//================================================================================================//
template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer rescaleImage2D(typename TInputImageType::Pointer input, typename TOutputImageType::SizeType outputSize)//, nih::InterpolationType interpolationType)
{
  const unsigned int Dimension = 2;

  if ( compareImageSizes<typename TInputImageType::SizeType, typename TOutputImageType::SizeType>(input->GetLargestPossibleRegion().GetSize(), outputSize) )
  {
    typedef itk::CastImageFilter< TInputImageType, TOutputImageType > CastFilterType;
    typename CastFilterType::Pointer castFilter = CastFilterType::New();
    castFilter->SetInput( input );
    try
    {
      castFilter->Update();
    }
    catch( itk::ExceptionObject & excep )
    {
      std::cerr << "Exception caught !" << std::endl;
      std::cerr << excep << std::endl;
      return NULL;
    }
    return castFilter->GetOutput();
  }

  typename TInputImageType::SizeType inputSize = input->GetLargestPossibleRegion().GetSize();
  //printf(" warping image size [%d,%d] to size [%d,%d] ...\n", inputSize[0],inputSize[1], outputSize[0],outputSize[1]);

  // Resize
  typename TOutputImageType::SpacingType outputSpacing;
  outputSpacing[0] = input->GetSpacing()[0] * (static_cast<double>(inputSize[0]-1) / (static_cast<double>(outputSize[0])) );
  outputSpacing[1] = input->GetSpacing()[1] * (static_cast<double>(inputSize[1]-1) / (static_cast<double>(outputSize[1])) );

  typedef itk::AffineTransform< double, Dimension > TransformType;

  typedef itk::ResampleImageFilter<TInputImageType, TOutputImageType> ResampleImageFilterType;
  itk::SmartPointer<ResampleImageFilterType> resample = ResampleImageFilterType::New();

  // Physical space coordinate of origin for X and Y
  resample->SetInput(input);
  resample->SetOutputOrigin( input->GetOrigin() );
  resample->SetOutputDirection( input->GetDirection() );
  resample->SetSize(outputSize);
  resample->SetOutputSpacing(outputSpacing);
  resample->SetTransform(TransformType::New());
  // select interpolation type
  /*if (interpolationType == nih::NEAREST)
  {
    std::cout << "  ... using nearest neighbor interpolation..." << std::endl;
    typedef itk::NearestNeighborInterpolateImageFunction< TInputImageType, double >  InterpolatorType;
    resample->SetInterpolator( InterpolatorType::New() );
  }
  else if (interpolationType == nih::LINEAR)
  {
    std::cout << "  ... using linear interpolation..." << std::endl;
    typedef itk::LinearInterpolateImageFunction< TInputImageType, double >  InterpolatorType;
    resample->SetInterpolator( InterpolatorType::New() );
  }
  else if (interpolationType == nih::BSPLINE) // best trade-off between accuracy and computational cost
  {
    std::cout << "  ... using B-Spline interpolation..." << std::endl;
    typedef itk::BSplineInterpolateImageFunction< TInputImageType, double > InterpolatorType;
    resample->SetInterpolator( InterpolatorType::New() );
  }
  else
  {
    std::cerr << "[Error] nih::rotateImageAroundCenter : No such interpolation type! " << interpolationType << std::endl;
    exit(EXIT_FAILURE);
  }*/
  //resample->SetDefaultPixelValue( itk::NumericTraits< typename TOutputImageType::PixelType >::quiet_NaN() );
  try
  {
    //resample->Update();
    resample->UpdateLargestPossibleRegion();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  return resample->GetOutput();
} // nih::rescaleImage2D

//================================================================================================//
//  nih::rescaleRGBImage2D
//================================================================================================//
template <class TRGBImageType, class TChannelImageType>
typename TRGBImageType::Pointer rescaleRGBImage2D(typename TRGBImageType::Pointer input, typename TRGBImageType::SizeType outputSize)//, nih::InterpolationType interpolationType)
{
  itk::SmartPointer<TChannelImageType> R = nih::rescaleImage2D<TChannelImageType,TChannelImageType>
    (getRGBchannel<TRGBImageType, TChannelImageType>(input,0), outputSize);
  itk::SmartPointer<TChannelImageType> G = nih::rescaleImage2D<TChannelImageType,TChannelImageType>
    (getRGBchannel<TRGBImageType, TChannelImageType>(input,1), outputSize);
  itk::SmartPointer<TChannelImageType> B = nih::rescaleImage2D<TChannelImageType,TChannelImageType>
    (getRGBchannel<TRGBImageType, TChannelImageType>(input,2), outputSize);

  return composeRGBImage<TChannelImageType, TRGBImageType>(R, G, B);
}

//================================================================================================//
//  nih::mergeRegions (return merged bounding box)
//================================================================================================//
template <class TRegionType, const unsigned int Dimensions>
TRegionType mergeRegions(TRegionType regionA, TRegionType regionB)
{
  typename TRegionType::SizeType  sizeA  = regionA.GetSize();
  typename TRegionType::IndexType indexA = regionA.GetIndex();
  typename TRegionType::IndexType endA = regionA.GetIndex();

  typename TRegionType::SizeType sizeB   = regionB.GetSize();
  typename TRegionType::IndexType indexB = regionB.GetIndex();
  typename TRegionType::IndexType endB = regionA.GetIndex();

  TRegionType regionAB;
  typename TRegionType::SizeType sizeAB;
  typename TRegionType::IndexType indexAB;

  // index
  for (unsigned int i=0; i<Dimensions; i++)
  {
    if (indexA[i] < indexB[i])
    {
      indexAB[i] = indexA[i];
    }
    else
    {
      indexAB[i] = indexB[i];
    }
  }

  // size
  for (unsigned int i=0; i<Dimensions; i++)
  {
    endA[i] = indexA[i] + sizeA[i];
    endB[i] = indexB[i] + sizeB[i];
    if (endA[i] > endB[i])
    {
      sizeAB[i] = endA[i] - indexAB[i];
    }
    else
    {
      sizeAB[i] = endB[i] - indexAB[i];
    }
  }

  regionAB.SetIndex(indexAB);
  regionAB.SetSize(sizeAB);

  return regionAB;
}

//================================================================================================//
//  nih::getNegativeROIcandidatesMask
//================================================================================================//
template <class TImageType>
typename TImageType::Pointer getNegativeROIcandidatesMask(typename TImageType::Pointer roiImage, typename TImageType::Pointer tissueImage, float radius)
{
  const typename TImageType::PixelType insideValue  = 0;
  const typename TImageType::PixelType outsideValue = 255;

  std::cout << "  getNegativeROIcandidatesMask with box element of size " << radius << " [voxels]." << std::endl;
  // Dilate region of roiImage by box of size 'radius'
  typedef itk::FlatStructuringElement< 3 > StructuringElementType;
  StructuringElementType::RadiusType elementRadius;
  elementRadius.Fill( radius );

  StructuringElementType structuringElement = StructuringElementType::Box(elementRadius);

  typedef itk::BinaryDilateImageFilter <TImageType, TImageType, StructuringElementType>
    dilateImageFilterType;

  itk::SmartPointer<dilateImageFilterType> dilateFilter = dilateImageFilterType::New();
  dilateFilter->SetInput( roiImage );
  dilateFilter->SetKernel(structuringElement);
  dilateFilter->SetDilateValue( insideValue );
  try
  {
    dilateFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  // Invert roiImage for AND operation
  typedef itk::BinaryNotImageFilter< TImageType > BinaryNotImageFilterType;
  itk::SmartPointer<BinaryNotImageFilterType> binaryNotFilter = BinaryNotImageFilterType::New();
  binaryNotFilter->SetInput( roiImage );
  try
  {
    binaryNotFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  // 1. AND (two images at a time): Dilated ROIs AND Tissue
  typedef itk::AndImageFilter< TImageType > AndImageFilterType;
  itk::SmartPointer<AndImageFilterType> andFilter = AndImageFilterType::New();
  andFilter->SetInput1( dilateFilter->GetOutput() );
  andFilter->SetInput2( tissueImage );
  try
  {
    andFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  // 2. AND (two images at a time): 1. AND output with NOT postive ROIs to give negative ROI candidates
  andFilter->SetInput1( andFilter->GetOutput() );
  andFilter->SetInput2( binaryNotFilter->GetOutput() );
  try
  {
    andFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  return andFilter->GetOutput();
}

//================================================================================================//
//  nih::getRGBImagePatch
//================================================================================================//
template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer getRGBImagePatch(typename TInputImageType::Pointer roiImage, typename TOutputImageType::SizeType outputPatchSize)
{
  typename TInputImageType::SizeType size = roiImage->GetLargestPossibleRegion().GetSize();
  if (size[2]<=1)
  {
    std::cerr << " input image needs to be a volume but is of size: " << roiImage->GetLargestPossibleRegion().GetSize() << std::endl;
    exit(EXIT_FAILURE);
  }

#if _DEBUG
  // Write the deformation field
    typedef itk::ImageFileWriter< TInputImageType  > WriterType;
    itk::SmartPointer<WriterType> writer =
      WriterType::New();
    writer->SetInput (  roiImage );
    writer->SetFileName( "/home/rothhr/Data/Spine/PostElemFxs/edge/DEVEL/tripatch/rois/DISPLFX_only_25mm_step1_scales1/roiImage.nii.gz" );
    writer->Update();
#endif

  typedef itk::Image<typename TInputImageType::PixelType, 2  >                   SliceImageType;
  typedef itk::ExtractImageFilter<TInputImageType, SliceImageType >              SliceFilterType;
  #if ITK_VERSION_MAJOR < 4
    typedef itk::Compose3DCovariantVectorImageFilter<OutputSliceImageType, RGBImageType> ComposeImageFilterType;
  #else
    typedef itk::ComposeImageFilter<SliceImageType, TOutputImageType> ComposeImageFilterType;
  #endif

  unsigned int slicingDim = 0;

  itk::SmartPointer<SliceFilterType> slicingFilterZ = SliceFilterType::New();
  slicingFilterZ->InPlaceOn();
  slicingFilterZ->SetDirectionCollapseToIdentity(); // direction does not matter for RGB images?

  itk::SmartPointer<SliceFilterType> slicingFilterX = SliceFilterType::New();
  slicingFilterX->InPlaceOn();
  slicingFilterX->SetDirectionCollapseToIdentity(); // direction does not matter for RGB images?

  itk::SmartPointer<SliceFilterType> slicingFilterY = SliceFilterType::New();
  slicingFilterY->InPlaceOn();
  slicingFilterY->SetDirectionCollapseToIdentity(); // direction does not matter for RGB images?

  typename TInputImageType::SizeType slicingSize;
  typename TInputImageType::IndexType slicingStart;

  typename TInputImageType::RegionType slicingRegion;

  // Z slice (axial)
  slicingDim = 2;
  slicingStart.Fill( 0 );
  slicingSize = roiImage->GetLargestPossibleRegion().GetSize();
  slicingStart[slicingDim] = slicingSize[slicingDim]/2;
  slicingSize[slicingDim] = 0; // Reduce to 2 dimensions
  slicingRegion.SetSize( slicingSize );
  slicingRegion.SetIndex( slicingStart );
  slicingFilterZ->SetInput( roiImage );
  slicingFilterZ->SetExtractionRegion( slicingRegion );
  try
  {
    slicingFilterZ->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "slicingFilterZ: Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    exit(EXIT_FAILURE);
  }

  // X slice (coronal?)
  slicingDim = 0;
  slicingStart.Fill( 0 );
  slicingSize = roiImage->GetLargestPossibleRegion().GetSize();
  slicingStart[slicingDim] = slicingSize[slicingDim]/2;
  slicingSize[slicingDim] = 0; // Reduce to 2 dimensions
  slicingRegion.SetSize( slicingSize );
  slicingRegion.SetIndex( slicingStart );
  slicingFilterX->SetInput( roiImage );
  slicingFilterX->SetExtractionRegion( slicingRegion );
  try
  {
    slicingFilterX->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "slicingFilterX: Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    exit(EXIT_FAILURE);
  }

  // Y slice (sagittal?)
  slicingDim = 1;
  slicingStart.Fill( 0 );
  slicingSize = roiImage->GetLargestPossibleRegion().GetSize();
  slicingStart[slicingDim] = slicingSize[slicingDim]/2;
  slicingSize[slicingDim] = 0; // Reduce to 2 dimensions
  slicingRegion.SetSize( slicingSize );
  slicingRegion.SetIndex( slicingStart );
  slicingFilterY->SetInput( roiImage );
  slicingFilterY->SetExtractionRegion( slicingRegion );
  try
  {
    slicingFilterY->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "slicingFilterY: Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    exit(EXIT_FAILURE);
  }

  // Compose RGB image
  itk::SmartPointer<SliceImageType> sliceZ = nih::rescaleImage2D<SliceImageType,SliceImageType>( slicingFilterZ->GetOutput(), outputPatchSize);
  itk::SmartPointer<SliceImageType> sliceX = nih::rescaleImage2D<SliceImageType,SliceImageType>( slicingFilterX->GetOutput(), outputPatchSize);
  itk::SmartPointer<SliceImageType> sliceY = nih::rescaleImage2D<SliceImageType,SliceImageType>( slicingFilterY->GetOutput(), outputPatchSize);

  typename SliceImageType::PointType origin;
  origin.Fill(0.0);
  sliceZ->SetOrigin( origin );
  sliceX->SetOrigin( origin );
  sliceY->SetOrigin( origin );

  typename SliceImageType::SpacingType spacing;
  spacing.Fill(1.0);
  sliceZ->SetSpacing( spacing );
  sliceX->SetSpacing( spacing );
  sliceY->SetSpacing( spacing );

  itk::SmartPointer<ComposeImageFilterType> composeFilter = ComposeImageFilterType::New();
  composeFilter->SetInput1( sliceZ );
  composeFilter->SetInput2( sliceX );
  composeFilter->SetInput3( sliceY );
  try
  {
    composeFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    exit(EXIT_FAILURE);
  }

  return composeFilter->GetOutput();
}
// no outputPatchSize given: use same as input size
template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer getRGBImagePatch(typename TInputImageType::Pointer roiImage)
{
  typename TOutputImageType::SizeType outputPatchSize;
  typename TInputImageType::SizeType inSize = roiImage->GetLargestPossibleRegion().GetSize();
  outputPatchSize[0] = inSize[0];
  outputPatchSize[1] = inSize[1];

  return nih::getRGBImagePatch<TInputImageType, TOutputImageType>(roiImage, outputPatchSize);
}
//================================================================================================//
//  nih::getRGBImagePatchXY
//================================================================================================//
template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer getRGBImagePatchXY(typename TInputImageType::Pointer roiImage)
{
  typedef itk::Image<typename TInputImageType::PixelType, 2  >                             SliceImageType;
  typedef itk::ExtractImageFilter< TInputImageType, SliceImageType >              SliceFilterType;
  #if ITK_VERSION_MAJOR < 4
    typedef itk::Compose3DCovariantVectorImageFilter<OutputSliceImageType, RGBImageType> ComposeImageFilterType;
  #else
    typedef itk::ComposeImageFilter<SliceImageType, TOutputImageType> ComposeImageFilterType;
  #endif

  unsigned int slicingDim = 0;

  itk::SmartPointer<SliceFilterType> slicingFilterZ = SliceFilterType::New();
  slicingFilterZ->InPlaceOn();
  slicingFilterZ->SetDirectionCollapseToIdentity(); // direction does not matter for RGB images?

  typename TInputImageType::SizeType slicingSize;
  typename TInputImageType::IndexType slicingStart;

  typename TInputImageType::RegionType slicingRegion;

  // Z slice (axial)
  slicingDim = 2;
  slicingSize = roiImage->GetLargestPossibleRegion().GetSize();
  slicingStart.Fill( 0 );
  slicingStart[slicingDim] = slicingSize[slicingDim]/2 - 1;
  slicingSize[slicingDim] = 0;
  slicingRegion.SetSize( slicingSize );
  slicingRegion.SetIndex( slicingStart );
  slicingFilterZ->SetInput( roiImage );
  slicingFilterZ->SetExtractionRegion( slicingRegion );
  try
  {
    slicingFilterZ->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  // Compose RGB image
  itk::SmartPointer<SliceImageType> sliceZ = slicingFilterZ->GetOutput();

  typename SliceImageType::PointType origin;
  origin.Fill(0.0);

  sliceZ->SetOrigin( origin );

  itk::SmartPointer<ComposeImageFilterType> composeFilter = ComposeImageFilterType::New();
  composeFilter->SetInput1( sliceZ );
  composeFilter->SetInput2( sliceZ );
  composeFilter->SetInput3( sliceZ );
  try
  {
    composeFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  return composeFilter->GetOutput();
}

//================================================================================================//
//  nih::getGrayImagePatch (middle slice of slicingDim)
//================================================================================================//
template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer getGrayImagePatch(typename TInputImageType::Pointer roiImage, unsigned int slicingDim)
{
  //typedef itk::Image<TInputImageType::PixelType, 2  >                             SliceImageType;
  typedef itk::ExtractImageFilter< TInputImageType, TOutputImageType >              SliceFilterType;

  typename SliceFilterType::Pointer slicingFilter = SliceFilterType::New();
  slicingFilter->InPlaceOn();
  slicingFilter->SetDirectionCollapseToIdentity(); // direction does not matter for RGB/gray images?

  typename TInputImageType::SizeType slicingSize;
  typename TInputImageType::IndexType slicingStart;

  typename TInputImageType::RegionType slicingRegion;

  slicingSize = roiImage->GetLargestPossibleRegion().GetSize();
  slicingStart.Fill( 0 );
  slicingStart[slicingDim] = slicingSize[slicingDim]/2;// - 1;
  slicingSize[slicingDim] = 0;
  slicingRegion.SetSize( slicingSize );
  slicingRegion.SetIndex( slicingStart );
  slicingFilter->SetInput( roiImage );
  slicingFilter->SetExtractionRegion( slicingRegion );
  try
  {
    slicingFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "getGrayImagePatch: Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  return slicingFilter->GetOutput();
}

//================================================================================================//
//  nih::getImageSlice
//================================================================================================//
template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer getImageSlice(typename TInputImageType::Pointer roiImage, unsigned int slice, unsigned int slicingDim=2)
{
  //Axial (default): slicingDim = 2;
  // X slice: slicingDim = 0;
  // Y slice: slicingDim = 1;

  typedef itk::ExtractImageFilter< TInputImageType, TOutputImageType > SliceFilterType;

  itk::SmartPointer<SliceFilterType> slicingFilter = SliceFilterType::New();
  slicingFilter->InPlaceOn();
  slicingFilter->SetDirectionCollapseToIdentity(); // direction does not matter for RGB images?

  typename TInputImageType::RegionType slicingRegion = roiImage->GetLargestPossibleRegion();
  typename TInputImageType::IndexType  slicingIndex  = slicingRegion.GetIndex();
  typename TInputImageType::SizeType   slicingSize   = slicingRegion.GetSize();
  if ((slice<0) || (slice>=slicingSize[slicingDim]))
  {
    std::cerr << " Slice not within range: " << slice << " not within [0," << slicingSize[slicingDim] << "]" << std::endl;
    return NULL;
  }
  slicingIndex[slicingDim] = slicingIndex[slicingDim] + slice;
  slicingSize[slicingDim] = 0;
  slicingRegion.SetSize( slicingSize );
  slicingRegion.SetIndex( slicingIndex );
  slicingFilter->SetInput( roiImage );
  slicingFilter->SetExtractionRegion( slicingRegion );
  try
  {
    slicingFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "getImageSlice: Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  return slicingFilter->GetOutput();

} // nih::getImageSlice

//================================================================================================//
//  nih::getPath
//================================================================================================//
std::string getPath(std::string s)
{
  std::size_t found = s.find_last_of('/'); // linux
  if (found == std::string::npos)
  {
    found = s.find_last_of('\\'); // windows
  }
  if (found == std::string::npos)
  {
    found = s.find_last_of(':'); // mac
  }
  return s.substr(0,found);
}

//================================================================================================//
//  nih::getRandomVariateUniformDouble
//================================================================================================//
double getRandomVariateUniformDouble(double limit)
{
  limit = abs(limit);
  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator GeneratorType;
  itk::SmartPointer<GeneratorType> generator = GeneratorType::New();

  generator->Initialize();

  return generator->GetUniformVariate(-1*limit, limit);
}

//================================================================================================//
//  nih::getRandomVariateInt
//================================================================================================//
int getRandomVariateInt(int limit)
{
  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator GeneratorType;
  itk::SmartPointer<GeneratorType> generator = GeneratorType::New();

  generator->Initialize();

  return generator->GetIntegerVariate( limit ); // Get an int between 0 and limit (inclusive - e.g. limit=5 is sample from the set {0,1,2,3,4,5})
}

//================================================================================================//
//  nih::getPointNorm
//================================================================================================//
template <class TImageType>
double getPointNorm( typename TImageType::PointType p )
{
   return sqrt( pow(p[0],2) + pow(p[1],2) + pow(p[2],2) );
}

//================================================================================================//
//  nih::getPointNorm
//================================================================================================//
template <class T>
double getPointNorm( T* p, const unsigned int Dimension=3 )
{
  double sum = 0.0;
  for (unsigned int i = 0; i<Dimension; i++)
  {
    sum += pow(p[i],2);
  }
  return sqrt( sum );
}

//================================================================================================//
//  nih::normalizePoint
//================================================================================================//
template <class TImageType>
typename TImageType::PointType normalizePoint( typename TImageType::PointType p )
{
   double n = getPointNorm< TImageType >( p );
   p[0] = p[0]/n;
   p[1] = p[1]/n;
   p[2] = p[2]/n;

   return p;
}

//================================================================================================//
//  nih::normalizePoint
//================================================================================================//
template <class T>
T* normalizePoint( double *p, const unsigned int Dimension=3 )
{
   double n = getPointNorm< T >( p );
   for (unsigned int i=0; i<Dimension; i++)
   {
      p[i] = p[i]/n;
   }

   return p;
}

//================================================================================================//
//  nih::getRandomUniformPoint
//================================================================================================//
template <class TImageType>
typename TImageType::PointType getRandomUniformPoint( const unsigned int Dimension=3 )
{
  typename TImageType::PointType v;
  for (unsigned int i = 0; i<Dimension; i++)
  {
    v[i] = getRandomVariateUniformDouble( 1.0 );
  }

  return normalizePoint< TImageType >( v );
}

//================================================================================================//
//  nih::getRandomUniformPoint
//================================================================================================//
template <class T>
T* getRandomUniformPoint( T *v, const unsigned int Dimension=3 )
{
  for (unsigned int i = 0; i<Dimension; i++)
  {
    v[i] = getRandomVariateUniformDouble( 1.0 );
  }

  return normalizePoint< T >( v );
}

//================================================================================================//
//  nih::getImageCenter
//================================================================================================//
template <class TImageType>
typename TImageType::PointType getImageCenter(typename TImageType::Pointer image)
{
  // this is equivalent to origin[1] + spacing[1] * size[1] / 2.0
  // see ITK guide, section 6.9. Geometric Transformations
  typedef itk::ImageFunction<TImageType, TImageType> Superclass;

  typename TImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();
  typename Superclass::ContinuousIndexType halfsize;
  typename Superclass::PointType center;

  for (unsigned int i=0; i<image->GetImageDimension(); i++)
  {
    halfsize[i] = size[i]/2.0;
  }

  image->TransformContinuousIndexToPhysicalPoint(halfsize, center);


  //printf("  getImageCenter: center at [%g, %g, %g] (image size is [%d, %d, %d]).\n",
    //center[0], center[1], center[2], size[0], size[1], size[2]);
  return center;
}

//================================================================================================//
//  nih::rotateImageAroundCenter
//================================================================================================//
template <class TInputImageType, class TOutputImageType, unsigned int Dimensions>
typename TOutputImageType::Pointer rotateImageAroundCenter
  (typename TInputImageType::Pointer image, double *axis, double angle_in_degrees, nih::InterpolationType interpolationType )
{
  typedef itk::ResampleImageFilter<TInputImageType, TOutputImageType >            RotationFilterType;
  typedef itk::AffineTransform< double, Dimensions >                               TransformType;

  // best possible interpolator for grid data but slow (see ITK guide sec. 8.9.4 'Windowed Sinc Interpolation')
  /*typedef itk::ConstantBoundaryCondition< TInputImageType > BoundaryConditionType;
  const unsigned int WindowRadius = 5;
  typedef itk::Function::HammingWindowFunction<WindowRadius> WindowFunctionType;
  typedef itk::WindowedSincInterpolateImageFunction<
            TInputImageType, WindowRadius, WindowFunctionType,
            BoundaryConditionType, double  >                 InterpolatorType;*/

  itk::SmartPointer<RotationFilterType> rotationFilter = RotationFilterType::New();
  itk::SmartPointer<TransformType> transform = TransformType::New();

  // select interpolation type
  if (interpolationType == nih::NEAREST)
  {
    //std::cout << "  ... using nearest neighbor interpolation..." << std::endl;
    typedef itk::NearestNeighborInterpolateImageFunction< TInputImageType, double >  InterpolatorType;
    itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
    rotationFilter->SetInterpolator( interpolator );
  }
  else if (interpolationType == nih::LINEAR)
  {
    //std::cout << "  ... using linear interpolation..." << std::endl;
    typedef itk::LinearInterpolateImageFunction< TInputImageType, double >  InterpolatorType;
    itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
    rotationFilter->SetInterpolator( interpolator );
  }
  else if (interpolationType == nih::BSPLINE) // best trade-off between accuracy and computational cost
  {
    //std::cout << "  ... using B-Spline interpolation..." << std::endl;
    typedef itk::BSplineInterpolateImageFunction< TInputImageType, double > InterpolatorType;
    itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
    rotationFilter->SetInterpolator( interpolator );
  }
  else
  {
    std::cerr << "[Error] nih::rotateImageAroundCenter : No such interpolation type! " << interpolationType << std::endl;
    exit(EXIT_FAILURE);
  }

  // set default pixel value (used if interpolation sample is outside image domain)
//  rotationFilter->SetDefaultPixelValue( itk::NumericTraits< typename TOutputImageType::PixelType >::quiet_NaN() );
  //rotationFilter->SetDefaultPixelValue( nih::getImageMean<TInputImageType>(image) );
  rotationFilter->SetDefaultPixelValue( nih::getImageMinimum<TInputImageType>(image) );

  rotationFilter->SetOutputSpacing(   image->GetSpacing() );
  rotationFilter->SetOutputDirection( image->GetDirection() );
  rotationFilter->SetSize(            image->GetLargestPossibleRegion().GetSize() );
  rotationFilter->SetOutputOrigin(    image->GetOrigin() );

  // translate region to physical origin (0,0,0)
  typename TOutputImageType::PointType center = nih::getImageCenter< TOutputImageType >( image );
  typename TransformType::OutputVectorType translation1;
  translation1[0] = -1*center[0];
  translation1[1] = -1*center[1];
  if (Dimensions==3)
  {
    translation1[2] = -1*center[2];
  }
  transform->Translate( translation1 );

  const double degreesToRadians = vcl_atan(1.0) / 45.0;
  double radians = angle_in_degrees * degreesToRadians;

  typename TransformType::OutputVectorType rotationAxis;
  rotationAxis[0] = axis[0];
  rotationAxis[1] = axis[1];
  if (Dimensions==3)
  {
    rotationAxis[2] = axis[2];
    transform->Rotate3D( rotationAxis, radians, false ); // false, to indicate that the rotation should be applied 'after' the current transform content.
  }
  else if (Dimensions==2)
  {
    transform->Rotate2D( radians, false ); // false, to indicate that the rotation should be applied 'after' the current transform content.
  }
  else
  {
    std::cerr << "[Error] nih::rotateImageAroundCenter :  rotations in other than 2 or 3 dimensions are not defined! " << std::endl;
    exit(EXIT_FAILURE);
  }

  // translate back to the previous location
  typename TransformType::OutputVectorType translation2;
  translation2[0] = -1*translation1[0];
  translation2[1] = -1*translation1[1];
  if (Dimensions==3)
  {
    translation2[2] = -1*translation1[2];
  }
  transform->Translate( translation2, false );

  // apply transform to resampling filter
  rotationFilter->SetInput( image );
  rotationFilter->SetTransform( transform );
  try
  {
    rotationFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  return rotationFilter->GetOutput();
}

//================================================================================================//
//  nih::rotateRGBImageAroundCenter
//================================================================================================//
template <class TRGBImageType, class TChannelImageType, unsigned int Dimensions>
typename TRGBImageType::Pointer rotateRGBImageAroundCenter
  (typename TRGBImageType::Pointer image, double *axis, double angle_in_degrees, nih::InterpolationType interpolationType )
{
  itk::SmartPointer<TChannelImageType> R = nih::rotateImageAroundCenter<TChannelImageType,TChannelImageType, Dimensions>
    (getRGBchannel<TRGBImageType, TChannelImageType>(image,0), axis, angle_in_degrees, interpolationType);
  itk::SmartPointer<TChannelImageType> G = nih::rotateImageAroundCenter<TChannelImageType,TChannelImageType, Dimensions>
    (getRGBchannel<TRGBImageType, TChannelImageType>(image,1), axis, angle_in_degrees, interpolationType);
  itk::SmartPointer<TChannelImageType> B = nih::rotateImageAroundCenter<TChannelImageType,TChannelImageType, Dimensions>
    (getRGBchannel<TRGBImageType, TChannelImageType>(image,2), axis, angle_in_degrees, interpolationType);

  return composeRGBImage<TChannelImageType, TRGBImageType>(R, G, B);
}

//================================================================================================//
//  nih::translateImage
//================================================================================================//
template <class TInputImageType, class TOutputImageType, unsigned int Dimensions >
typename TOutputImageType::Pointer translateImage
  (typename TInputImageType::Pointer image, double *translation)
{
  return translateImage<TInputImageType, TOutputImageType, Dimensions>(image, translation, nih::BSPLINE) ;
}

template <class TInputImageType, class TOutputImageType, unsigned int Dimensions >
typename TOutputImageType::Pointer translateImage(
  typename TInputImageType::Pointer image, double *translation, nih::InterpolationType interpolationType)
{
  typedef itk::ResampleImageFilter<TInputImageType, TOutputImageType >            TranslationFilterType;
  typedef itk::AffineTransform< double, Dimensions >                               TransformType;

  // best possible interpolator for grid data but slow (see ITK guide sec. 8.9.4 'Windowed Sinc Interpolation')
  /*typedef itk::ConstantBoundaryCondition< TInputImageType > BoundaryConditionType;
  const unsigned int WindowRadius = 5;
  typedef itk::Function::HammingWindowFunction<WindowRadius> WindowFunctionType;
  typedef itk::WindowedSincInterpolateImageFunction<
            TInputImageType, WindowRadius, WindowFunctionType,
            BoundaryConditionType, double  >                 InterpolatorType;*/

  itk::SmartPointer<TranslationFilterType> translationFilter = TranslationFilterType::New();
  itk::SmartPointer<TransformType> transform = TransformType::New();

  // select interpolation type
  if (interpolationType == nih::NEAREST)
  {
    //std::cout << "  ... using nearest neighbor interpolation..." << std::endl;
    typedef itk::NearestNeighborInterpolateImageFunction< TInputImageType, double >  InterpolatorType;
    itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
    translationFilter->SetInterpolator( interpolator );
  }
  else if (interpolationType == nih::LINEAR)
  {
    //std::cout << "  ... using linear interpolation..." << std::endl;
    typedef itk::LinearInterpolateImageFunction< TInputImageType, double >  InterpolatorType;
    itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
    translationFilter->SetInterpolator( interpolator );
  }
  else if (interpolationType == nih::BSPLINE) // best trade-off between accuracy and computational cost
  {
    //std::cout << "  ... using B-Spline interpolation..." << std::endl;
    typedef itk::BSplineInterpolateImageFunction< TInputImageType, double > InterpolatorType;
    itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
    translationFilter->SetInterpolator( interpolator );
  }
  else
  {
    std::cerr << "Error!: No such interpolation type! " << interpolationType << std::endl;
    exit(-1);
  }

  //translationFilter->SetDefaultPixelValue( itk::NumericTraits< typename TOutputImageType::PixelType >::quiet_NaN() );

  translationFilter->SetOutputSpacing(   image->GetSpacing() );
  translationFilter->SetOutputDirection( image->GetDirection() );
  translationFilter->SetSize(            image->GetLargestPossibleRegion().GetSize() );
  translationFilter->SetOutputOrigin(    image->GetOrigin() );

  // translate region to physical origin (0,0,0)
  typename TOutputImageType::PointType center = nih::getImageCenter< TOutputImageType >( image );
  typename TransformType::OutputVectorType translation1;
  for (unsigned int i=0; i<Dimensions; i++)
  {
    translation1[i] = translation[i];
  }
  transform->Translate( translation1 );

  // apply transform to resampling filter
  translationFilter->SetInput( image );
  translationFilter->SetTransform( transform );
  try
  {
    translationFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  return translationFilter->GetOutput();
}

//================================================================================================//
//  nih::translateRGBImage
//================================================================================================//
template <class TRGBImageType, class TChannelImageType, unsigned int Dimensions >
typename TRGBImageType::Pointer translateRGBImage(
  typename TRGBImageType::Pointer image, double *translation, nih::InterpolationType interpolationType)
{
  itk::SmartPointer<TChannelImageType> R = nih::translateImage<TChannelImageType,TChannelImageType, Dimensions>
    (getRGBchannel<TRGBImageType, TChannelImageType>(image,0), translation, interpolationType);
  itk::SmartPointer<TChannelImageType> G = nih::translateImage<TChannelImageType,TChannelImageType, Dimensions>
    (getRGBchannel<TRGBImageType, TChannelImageType>(image,1), translation, interpolationType);
  itk::SmartPointer<TChannelImageType> B = nih::translateImage<TChannelImageType,TChannelImageType, Dimensions>
    (getRGBchannel<TRGBImageType, TChannelImageType>(image,2), translation, interpolationType);

  return composeRGBImage<TChannelImageType, TRGBImageType>(R, G, B);
}

//================================================================================================//
//  nih::randomWarpRGBImage2D
//================================================================================================//
template <class TRGBImageType, class TChannelImageType>
typename TRGBImageType::Pointer randomWarpRGBImage2D(typename TRGBImageType::Pointer movingImage, unsigned int Npoints, double deform_length, double stiffness=1e-6)
{
  if ( (Npoints==0) || (deform_length<1e-6) )
  {
    std::cout << " no randomWarpImage2D performed. " << std::endl;
    return movingImage;
  }

  typedef   float VectorComponentType;

  typedef   itk::Vector< VectorComponentType, 2 >    VectorType;
  typedef   itk::Image< VectorType,  2 >   DeformationFieldType;

  typename TRGBImageType::SizeType movingSize = movingImage->GetLargestPossibleRegion().GetSize();

#if ITK_VERSION_MAJOR < 4
  typedef itk::DeformationFieldSource<DeformationFieldType>  DeformationFieldSourceType;
#else
  typedef itk::LandmarkDisplacementFieldSource<DeformationFieldType>  DeformationFieldSourceType;
#endif
  itk::SmartPointer<DeformationFieldSourceType> deformationFieldSource = DeformationFieldSourceType::New();
  deformationFieldSource->SetOutputSpacing( movingImage->GetSpacing() );
  deformationFieldSource->SetOutputOrigin(  movingImage->GetOrigin() );
  deformationFieldSource->SetOutputRegion(  movingImage->GetLargestPossibleRegion() );
  deformationFieldSource->SetOutputDirection( movingImage->GetDirection() );

  deformationFieldSource->GetKernelTransform()->SetStiffness( stiffness ); // doesn't seem to make a lot of difference
  //printf("Spline stiffness = %g\n", deformationFieldSource->GetKernelTransform()->GetStiffness() ); // default is 0.0

  //  Create source and target landmarks.
  typedef DeformationFieldSourceType::LandmarkContainerPointer   LandmarkContainerPointer;
  typedef DeformationFieldSourceType::LandmarkContainer          LandmarkContainerType;
  typedef DeformationFieldSourceType::LandmarkPointType          LandmarkPointType;

  itk::SmartPointer<LandmarkContainerType> sourceLandmarks = LandmarkContainerType::New();
  itk::SmartPointer<LandmarkContainerType> targetLandmarks = LandmarkContainerType::New();

  LandmarkPointType sourcePoint;
  LandmarkPointType targetPoint;

  // insert landmarks along that fix the border
  unsigned int Nborder = 1;
  unsigned int landmarkCount = 0;
  // along x1
  double xstep = itk::Math::Ceil<double>((double)movingSize[0]/3);
  double ystep = itk::Math::Ceil<double>((double)movingSize[0]/3);
  for (double i=0; i<movingSize[0]; i=i+xstep)
  {
    // x1-ystep
    for (double j=1; j<=Nborder; j++)
    {
      sourcePoint[0] = i;
      sourcePoint[1] = 0-j*ystep;
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, sourcePoint );
      landmarkCount++;
    }
    // x1
    sourcePoint[0] = i;
    sourcePoint[1] = 0;
    sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
    targetLandmarks->InsertElement( landmarkCount, sourcePoint );
    landmarkCount++;
    //printf(" %d. adding source: [%g, %g] (border)...\n", landmarkCount, sourcePoint[0], sourcePoint[1]);
    //printf(" %d. adding target: [%g, %g] (border)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
    // x2
    sourcePoint[0] = i;
    sourcePoint[1] = movingSize[1];
    sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
    targetLandmarks->InsertElement( landmarkCount, sourcePoint );
    landmarkCount++;
    // x2+ystep
    for (double j=1; j<=Nborder; j++)
    {
      sourcePoint[0] = i;
      sourcePoint[1] = movingSize[1]+j*ystep;
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, sourcePoint );
      landmarkCount++;
    }
    //printf(" %d. adding source: [%g, %g] (border)...\n", landmarkCount, sourcePoint[0], sourcePoint[1]);
    //printf(" %d. adding target: [%g, %g] (border)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
  }
  // along y
  //for (double i=ystep; i<movingSize[1]-ystep; i=i+ystep) // exclude 'corners'
  for (double i=0; i<movingSize[1]; i=i+ystep)
  {
    // y1-xstep
    for (double j=1; j<=Nborder; j++)
    {
      sourcePoint[0] = 0-j*xstep;
      sourcePoint[1] = i;
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, sourcePoint );
      landmarkCount++;
    }
    // y1
    sourcePoint[0] = 0;
    sourcePoint[1] = i;
    sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
    targetLandmarks->InsertElement( landmarkCount, sourcePoint );
    landmarkCount++;
    //printf(" %d. adding source: [%g, %g] (border)...\n", landmarkCount, sourcePoint[0], sourcePoint[1]);
    //printf(" %d. adding target: [%g, %g] (border)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
    // y2
    sourcePoint[0] = movingSize[0];
    sourcePoint[1] = i;
    sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
    targetLandmarks->InsertElement( landmarkCount, sourcePoint );
    landmarkCount++;
    // y2+xstep
    for (double j=1; j<=Nborder; j++)
    {
      sourcePoint[0] = movingSize[0]+j*xstep;
      sourcePoint[1] = i;
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, sourcePoint );
      landmarkCount++;
    }
    //printf(" %d. adding source: [%g, %g] (border)...\n", landmarkCount, sourcePoint[0], sourcePoint[1]);
    //printf(" %d. adding target: [%g, %g] (border)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
  }

  // add random deformations in center of image
  double N = sqrt((double)Npoints);
  xstep = movingSize[0]/N;
  ystep = movingSize[1]/N;
  double xcenter = movingSize[0]/2;
  double ycenter = movingSize[1]/2;
  for (int x=-N/2; x<=N/2; x++)
  {
    for (int y=-N/2; y<=N/2; y++)
    {
      //std::cout << "x = " << x << std::endl;
      //std::cout << "y = " << y << std::endl;
      sourcePoint[0] = xcenter + x*xstep;
      sourcePoint[1] = ycenter + y*ystep;
      targetPoint[0] = sourcePoint[0] + nih::getRandomVariateUniformDouble( deform_length );
      targetPoint[1] = sourcePoint[1] + nih::getRandomVariateUniformDouble( deform_length );
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, targetPoint );
      landmarkCount++;
      //printf(" %d. adding source: [%g, %g] (random)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
      //printf(" %d. adding target: [%g, %g] (random)...\n", landmarkCount,  targetPoint[0], targetPoint[1]);
    }
  }

  unsigned int NLandmarks = landmarkCount;
  //printf(" warping image (size [%d, %d]) with %d random vectors...\n", movingSize[0], movingSize[1], NLandmarks);

#if _DEBUG
  std::string debug_dir = "C:/HR/Data/Pancreas/SPIE/deform_examples/def5";
  // save landmarks
  std::ofstream sourceLandmarkFile;
  std::ofstream targetLandmarkFile;
  sourceLandmarkFile.open ( debug_dir + "/source_landmarks.txt");
  targetLandmarkFile.open ( debug_dir + "/target_landmarks.txt");
  for (unsigned int i=0; i<NLandmarks; i++)
  {
      sourceLandmarkFile << sourceLandmarks->GetElement(i)[0] << " " << sourceLandmarks->GetElement(i)[1] << std::endl;
      targetLandmarkFile << targetLandmarks->GetElement(i)[0] << " " << targetLandmarks->GetElement(i)[1] << std::endl;
  }
  sourceLandmarkFile.close();
  targetLandmarkFile.close();
#endif

  // get deformation field from landmarks
  deformationFieldSource->SetSourceLandmarks( sourceLandmarks.GetPointer() );
  deformationFieldSource->SetTargetLandmarks( targetLandmarks.GetPointer() );
  try
  {
    deformationFieldSource->UpdateLargestPossibleRegion();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

#if _DEBUG
  // Write the deformation field
    typedef itk::ImageFileWriter<  DeformationFieldType  > WriterType;
    itk::SmartPointer<WriterType> writer =
      WriterType::New();
    writer->SetInput (  deformationFieldSource->GetOutput() );
    writer->SetFileName( debug_dir + "/deform.mha" );
    writer->Update();
#endif

  typedef itk::WarpImageFilter< TChannelImageType, TChannelImageType, DeformationFieldType  >  WarpImageFilterType;
  itk::SmartPointer<WarpImageFilterType> warpImageFilterR = WarpImageFilterType::New();
  itk::SmartPointer<WarpImageFilterType> warpImageFilterG = WarpImageFilterType::New();
  itk::SmartPointer<WarpImageFilterType> warpImageFilterB = WarpImageFilterType::New();

  //typedef itk::NearestNeighborInterpolateImageFunction< TChannelImageType, double > InterpolatorType;
  typedef itk::LinearInterpolateImageFunction< TChannelImageType, double >          InterpolatorType;
  //typedef itk::BSplineInterpolateImageFunction< TChannelImageType, double >          InterpolatorType; // best trade-off between accuracy and computational cost

  itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
  warpImageFilterR->SetInterpolator( interpolator );
  warpImageFilterG->SetInterpolator( interpolator );
  warpImageFilterB->SetInterpolator( interpolator );
  warpImageFilterR->SetOutputSpacing( deformationFieldSource->GetOutput()->GetSpacing() );
  warpImageFilterG->SetOutputSpacing( deformationFieldSource->GetOutput()->GetSpacing() );
  warpImageFilterB->SetOutputSpacing( deformationFieldSource->GetOutput()->GetSpacing() );
  warpImageFilterR->SetOutputOrigin(  deformationFieldSource->GetOutput()->GetOrigin() );
  warpImageFilterG->SetOutputOrigin(  deformationFieldSource->GetOutput()->GetOrigin() );
  warpImageFilterB->SetOutputOrigin(  deformationFieldSource->GetOutput()->GetOrigin() );
#if ITK_VERSION_MAJOR < 4
  warpImageFilterR->SetDeformationField( deformationFieldSource->GetOutput() );
  warpImageFilterG->SetDeformationField( deformationFieldSource->GetOutput() );
  warpImageFilterB->SetDeformationField( deformationFieldSource->GetOutput() );
#else
  warpImageFilterR->SetDisplacementField( deformationFieldSource->GetOutput() );
  warpImageFilterG->SetDisplacementField( deformationFieldSource->GetOutput() );
  warpImageFilterB->SetDisplacementField( deformationFieldSource->GetOutput() );
#endif
  warpImageFilterR->SetInput( getRGBchannel<TRGBImageType, TChannelImageType>(movingImage,0) );
  warpImageFilterG->SetInput( getRGBchannel<TRGBImageType, TChannelImageType>(movingImage,1) );
  warpImageFilterB->SetInput( getRGBchannel<TRGBImageType, TChannelImageType>(movingImage,2) );
  try
  {
    warpImageFilterR->Update();
    warpImageFilterG->Update();
    warpImageFilterB->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  return composeRGBImage<TChannelImageType, TRGBImageType>(warpImageFilterR->GetOutput(), warpImageFilterG->GetOutput(), warpImageFilterB->GetOutput());
}

//================================================================================================//
//  nih::warpRGBImage
//================================================================================================//
template <class TChannelImageType, class TRGBImageType, class TDeformationFieldType>
typename TRGBImageType::Pointer warpRGBImage(typename TRGBImageType::Pointer movingImage, typename TDeformationFieldType::Pointer deformationField, nih::InterpolationType interpolationType=nih::LINEAR)
{
  // warp RGB image
  typedef itk::WarpImageFilter< TChannelImageType, TChannelImageType, TDeformationFieldType  >  WarpImageFilterType;
  itk::SmartPointer<WarpImageFilterType> warpImageFilterR = WarpImageFilterType::New();
  itk::SmartPointer<WarpImageFilterType> warpImageFilterG = WarpImageFilterType::New();
  itk::SmartPointer<WarpImageFilterType> warpImageFilterB = WarpImageFilterType::New();
  // select interpolationType1
  if (interpolationType == nih::NEAREST)
  {
    //std::cout << "  ... using nearest neighbor interpolation..." << std::endl;
    typedef itk::NearestNeighborInterpolateImageFunction< TChannelImageType, double >  InterpolatorType;
    itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
    warpImageFilterR->SetInterpolator( interpolator );
    warpImageFilterG->SetInterpolator( interpolator );
    warpImageFilterB->SetInterpolator( interpolator );
  }
  else if (interpolationType == nih::LINEAR)
  {
    //std::cout << "  ... using linear interpolation..." << std::endl;
    typedef itk::LinearInterpolateImageFunction< TChannelImageType, double >  InterpolatorType;
    itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
    warpImageFilterR->SetInterpolator( interpolator );
    warpImageFilterG->SetInterpolator( interpolator );
    warpImageFilterB->SetInterpolator( interpolator );
  }
  /*else if (interpolationType == nih::BSPLINE) // best trade-off between accuracy and computational cost
  {
    //std::cout << "  ... using B-Spline interpolation..." << std::endl;
    typedef itk::BSplineInterpolateImageFunction< TChannelImageType, double > InterpolatorType;
    itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
    warpImageFilterR->SetInterpolator( interpolator );
    warpImageFilterG->SetInterpolator( interpolator );
    warpImageFilterB->SetInterpolator( interpolator );
  }*/
  else
  {
    std::cerr << "[Error] nih::rotateImageAroundCenter : No such interpolation type! " << interpolationType << std::endl;
    exit(EXIT_FAILURE);
  }
  warpImageFilterR->SetOutputSpacing( deformationField->GetSpacing() );
  warpImageFilterG->SetOutputSpacing( deformationField->GetSpacing() );
  warpImageFilterB->SetOutputSpacing( deformationField->GetSpacing() );
  warpImageFilterR->SetOutputOrigin(  deformationField->GetOrigin() );
  warpImageFilterG->SetOutputOrigin(  deformationField->GetOrigin() );
  warpImageFilterB->SetOutputOrigin(  deformationField->GetOrigin() );
#if ITK_VERSION_MAJOR < 4
  warpImageFilterR->SetDeformationField( deformationField );
  warpImageFilterG->SetDeformationField( deformationField );
  warpImageFilterB->SetDeformationField( deformationField );
#else
  warpImageFilterR->SetDisplacementField( deformationField );
  warpImageFilterG->SetDisplacementField( deformationField );
  warpImageFilterB->SetDisplacementField( deformationField );
#endif
  warpImageFilterR->SetInput( getRGBchannel<TRGBImageType, TChannelImageType>(movingImage,0) );
  warpImageFilterG->SetInput( getRGBchannel<TRGBImageType, TChannelImageType>(movingImage,1) );
  warpImageFilterB->SetInput( getRGBchannel<TRGBImageType, TChannelImageType>(movingImage,2) );
  try
  {
    warpImageFilterR->Update();
    warpImageFilterG->Update();
    warpImageFilterB->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  return composeRGBImage<TChannelImageType, TRGBImageType>(warpImageFilterR->GetOutput(), warpImageFilterG->GetOutput(), warpImageFilterB->GetOutput());
} // nih::warpRGBImage


//================================================================================================//
//  nih::randomWarpRGBImagePair
//================================================================================================//
template <class TRGBImageType, class TChannelImageType>
typename TRGBImageType::Pointer randomWarpRGBImagePair(typename TRGBImageType::Pointer movingImage1, typename TRGBImageType::Pointer &outImage1,
                                                       typename TRGBImageType::Pointer movingImage2, typename TRGBImageType::Pointer &outImage2,
                                                       unsigned int Npoints, double deform_length, double stiffness=1e-6,
                                                       nih::InterpolationType interpolationType1=nih::LINEAR, nih::InterpolationType interpolationType2=nih::LINEAR)
{
  if ( (Npoints==0) || (deform_length<1e-6) )
  {
    std::cout << " no randomWarpRGBImagePair performed. " << std::endl;
    outImage1 = movingImage1;
    outImage2 = movingImage2;
  }

  typedef   float VectorComponentType;

  typedef   itk::Vector< VectorComponentType, 2 >    VectorType;
  typedef   itk::Image< VectorType,  2 >   DeformationFieldType;

  typename TRGBImageType::SizeType movingSize = movingImage1->GetLargestPossibleRegion().GetSize();

#if ITK_VERSION_MAJOR < 4
  typedef itk::DeformationFieldSource<DeformationFieldType>  DeformationFieldSourceType;
#else
  typedef itk::LandmarkDisplacementFieldSource<DeformationFieldType>  DeformationFieldSourceType;
#endif
  itk::SmartPointer<DeformationFieldSourceType> deformationFieldSource = DeformationFieldSourceType::New();
  deformationFieldSource->SetOutputSpacing( movingImage1->GetSpacing() );
  deformationFieldSource->SetOutputOrigin(  movingImage1->GetOrigin() );
  deformationFieldSource->SetOutputRegion(  movingImage1->GetLargestPossibleRegion() );
  deformationFieldSource->SetOutputDirection( movingImage1->GetDirection() );

  deformationFieldSource->GetKernelTransform()->SetStiffness( stiffness ); // doesn't seem to make a lot of difference
  //printf("Spline stiffness = %g\n", deformationFieldSource->GetKernelTransform()->GetStiffness() ); // default is 0.0

  //  Create source and target landmarks.
  typedef DeformationFieldSourceType::LandmarkContainerPointer   LandmarkContainerPointer;
  typedef DeformationFieldSourceType::LandmarkContainer          LandmarkContainerType;
  typedef DeformationFieldSourceType::LandmarkPointType          LandmarkPointType;

  itk::SmartPointer<LandmarkContainerType> sourceLandmarks = LandmarkContainerType::New();
  itk::SmartPointer<LandmarkContainerType> targetLandmarks = LandmarkContainerType::New();

  LandmarkPointType sourcePoint;
  LandmarkPointType targetPoint;

  // insert landmarks along that fix the border
  unsigned int Nborder = 1;
  unsigned int landmarkCount = 0;
  // along x1
  double xstep = itk::Math::Ceil<double>((double)movingSize[0]/3);
  double ystep = itk::Math::Ceil<double>((double)movingSize[0]/3);
  for (double i=0; i<movingSize[0]; i=i+xstep)
  {
    // x1-ystep
    for (double j=1; j<=Nborder; j++)
    {
      sourcePoint[0] = i;
      sourcePoint[1] = 0-j*ystep;
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, sourcePoint );
      landmarkCount++;
    }
    // x1
    sourcePoint[0] = i;
    sourcePoint[1] = 0;
    sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
    targetLandmarks->InsertElement( landmarkCount, sourcePoint );
    landmarkCount++;
    //printf(" %d. adding source: [%g, %g] (border)...\n", landmarkCount, sourcePoint[0], sourcePoint[1]);
    //printf(" %d. adding target: [%g, %g] (border)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
    // x2
    sourcePoint[0] = i;
    sourcePoint[1] = movingSize[1];
    sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
    targetLandmarks->InsertElement( landmarkCount, sourcePoint );
    landmarkCount++;
    // x2+ystep
    for (double j=1; j<=Nborder; j++)
    {
      sourcePoint[0] = i;
      sourcePoint[1] = movingSize[1]+j*ystep;
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, sourcePoint );
      landmarkCount++;
    }
    //printf(" %d. adding source: [%g, %g] (border)...\n", landmarkCount, sourcePoint[0], sourcePoint[1]);
    //printf(" %d. adding target: [%g, %g] (border)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
  }
  // along y
  //for (double i=ystep; i<movingSize[1]-ystep; i=i+ystep) // exclude 'corners'
  for (double i=0; i<movingSize[1]; i=i+ystep)
  {
    // y1-xstep
    for (double j=1; j<=Nborder; j++)
    {
      sourcePoint[0] = 0-j*xstep;
      sourcePoint[1] = i;
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, sourcePoint );
      landmarkCount++;
    }
    // y1
    sourcePoint[0] = 0;
    sourcePoint[1] = i;
    sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
    targetLandmarks->InsertElement( landmarkCount, sourcePoint );
    landmarkCount++;
    //printf(" %d. adding source: [%g, %g] (border)...\n", landmarkCount, sourcePoint[0], sourcePoint[1]);
    //printf(" %d. adding target: [%g, %g] (border)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
    // y2
    sourcePoint[0] = movingSize[0];
    sourcePoint[1] = i;
    sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
    targetLandmarks->InsertElement( landmarkCount, sourcePoint );
    landmarkCount++;
    // y2+xstep
    for (double j=1; j<=Nborder; j++)
    {
      sourcePoint[0] = movingSize[0]+j*xstep;
      sourcePoint[1] = i;
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, sourcePoint );
      landmarkCount++;
    }
    //printf(" %d. adding source: [%g, %g] (border)...\n", landmarkCount, sourcePoint[0], sourcePoint[1]);
    //printf(" %d. adding target: [%g, %g] (border)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
  }

  // add random deformations in center of image
  double N = sqrt((double)Npoints);
  xstep = movingSize[0]/N;
  ystep = movingSize[1]/N;
  double xcenter = movingSize[0]/2;
  double ycenter = movingSize[1]/2;
  for (int x=-N/2; x<=N/2; x++)
  {
    for (int y=-N/2; y<=N/2; y++)
    {
      //std::cout << "x = " << x << std::endl;
      //std::cout << "y = " << y << std::endl;
      sourcePoint[0] = xcenter + x*xstep;
      sourcePoint[1] = ycenter + y*ystep;
      targetPoint[0] = sourcePoint[0] + nih::getRandomVariateUniformDouble( deform_length );
      targetPoint[1] = sourcePoint[1] + nih::getRandomVariateUniformDouble( deform_length );
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, targetPoint );
      landmarkCount++;
      //printf(" %d. adding source: [%g, %g] (random)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
      //printf(" %d. adding target: [%g, %g] (random)...\n", landmarkCount,  targetPoint[0], targetPoint[1]);
    }
  }

  unsigned int NLandmarks = landmarkCount;
  //printf(" warping image (size [%d, %d]) with %d random vectors...\n", movingSize[0], movingSize[1], NLandmarks);

#if _DEBUG
  std::string debug_dir = "C:/HR/Data/Pancreas/SPIE/deform_examples/def5";
  // save landmarks
  std::ofstream sourceLandmarkFile;
  std::ofstream targetLandmarkFile;
  sourceLandmarkFile.open ( debug_dir + "/source_landmarks.txt");
  targetLandmarkFile.open ( debug_dir + "/target_landmarks.txt");
  for (unsigned int i=0; i<NLandmarks; i++)
  {
      sourceLandmarkFile << sourceLandmarks->GetElement(i)[0] << " " << sourceLandmarks->GetElement(i)[1] << std::endl;
      targetLandmarkFile << targetLandmarks->GetElement(i)[0] << " " << targetLandmarks->GetElement(i)[1] << std::endl;
  }
  sourceLandmarkFile.close();
  targetLandmarkFile.close();
#endif

  // get deformation field from landmarks
  deformationFieldSource->SetSourceLandmarks( sourceLandmarks.GetPointer() );
  deformationFieldSource->SetTargetLandmarks( targetLandmarks.GetPointer() );
  try
  {
    deformationFieldSource->UpdateLargestPossibleRegion();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

#if _DEBUG
  // Write the deformation field
    typedef itk::ImageFileWriter<  DeformationFieldType  > WriterType;
    itk::SmartPointer<WriterType> writer =
      WriterType::New();
    writer->SetInput (  deformationFieldSource->GetOutput() );
    writer->SetFileName( debug_dir + "/deform.mha" );
    writer->Update();
#endif

  outImage1 = nih::warpRGBImage<TChannelImageType,TRGBImageType,DeformationFieldType>(movingImage1, deformationFieldSource->GetOutput(), interpolationType1);
  outImage2 = nih::warpRGBImage<TChannelImageType,TRGBImageType,DeformationFieldType>(movingImage2, deformationFieldSource->GetOutput(), interpolationType2);
}

//================================================================================================//
//  nih::randomWarpImage2D
//================================================================================================//
template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer randomWarpImage2D(typename TInputImageType::Pointer movingImage, unsigned int Npoints, double deform_length, double stiffness=1e-6)
{
  if ( (Npoints==0) || (deform_length<1e-6) )
  {
    std::cout << " no randomWarpImage2D performed. " << std::endl;
    return movingImage;
  }

  typedef   float VectorComponentType;

  typedef   itk::Vector< VectorComponentType, 2 >    VectorType;
  typedef   itk::Image< VectorType,  2 >   DeformationFieldType;

  typename TInputImageType::SizeType movingSize = movingImage->GetLargestPossibleRegion().GetSize();

#if ITK_VERSION_MAJOR < 4
  typedef itk::DeformationFieldSource<DeformationFieldType>  DeformationFieldSourceType;
#else
  typedef itk::LandmarkDisplacementFieldSource<DeformationFieldType>  DeformationFieldSourceType;
#endif
  itk::SmartPointer<DeformationFieldSourceType> deformationFieldSource = DeformationFieldSourceType::New();
  deformationFieldSource->SetOutputSpacing( movingImage->GetSpacing() );
  deformationFieldSource->SetOutputOrigin(  movingImage->GetOrigin() );
  deformationFieldSource->SetOutputRegion(  movingImage->GetLargestPossibleRegion() );
  deformationFieldSource->SetOutputDirection( movingImage->GetDirection() );

  deformationFieldSource->GetKernelTransform()->SetStiffness( stiffness ); // doesn't seem to make a lot of difference
  printf("Spline stiffness = %g\n", deformationFieldSource->GetKernelTransform()->GetStiffness() ); // default is 0.0

  //  Create source and target landmarks.
  typedef DeformationFieldSourceType::LandmarkContainerPointer   LandmarkContainerPointer;
  typedef DeformationFieldSourceType::LandmarkContainer          LandmarkContainerType;
  typedef DeformationFieldSourceType::LandmarkPointType          LandmarkPointType;

  itk::SmartPointer<LandmarkContainerType> sourceLandmarks = LandmarkContainerType::New();
  itk::SmartPointer<LandmarkContainerType> targetLandmarks = LandmarkContainerType::New();

  LandmarkPointType sourcePoint;
  LandmarkPointType targetPoint;

  // insert landmarks along that fix the border
  unsigned int Nborder = 1;
  unsigned int landmarkCount = 0;
  // along x1
  double xstep = itk::Math::Ceil<double>((double)movingSize[0]/3);
  double ystep = itk::Math::Ceil<double>((double)movingSize[0]/3);
  for (double i=0; i<movingSize[0]; i=i+xstep)
  {
    // x1-ystep
    for (double j=1; j<=Nborder; j++)
    {
      sourcePoint[0] = i;
      sourcePoint[1] = 0-j*ystep;
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, sourcePoint );
      landmarkCount++;
    }
    // x1
    sourcePoint[0] = i;
    sourcePoint[1] = 0;
    sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
    targetLandmarks->InsertElement( landmarkCount, sourcePoint );
    landmarkCount++;
    //printf(" %d. adding source: [%g, %g] (border)...\n", landmarkCount, sourcePoint[0], sourcePoint[1]);
    //printf(" %d. adding target: [%g, %g] (border)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
    // x2
    sourcePoint[0] = i;
    sourcePoint[1] = movingSize[1];
    sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
    targetLandmarks->InsertElement( landmarkCount, sourcePoint );
    landmarkCount++;
    // x2+ystep
    for (double j=1; j<=Nborder; j++)
    {
      sourcePoint[0] = i;
      sourcePoint[1] = movingSize[1]+j*ystep;
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, sourcePoint );
      landmarkCount++;
    }
    //printf(" %d. adding source: [%g, %g] (border)...\n", landmarkCount, sourcePoint[0], sourcePoint[1]);
    //printf(" %d. adding target: [%g, %g] (border)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
  }
  // along y
  //for (double i=ystep; i<movingSize[1]-ystep; i=i+ystep) // exclude 'corners'
  for (double i=0; i<movingSize[1]; i=i+ystep)
  {
    // y1-xstep
    for (double j=1; j<=Nborder; j++)
    {
      sourcePoint[0] = 0-j*xstep;
      sourcePoint[1] = i;
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, sourcePoint );
      landmarkCount++;
    }
    // y1
    sourcePoint[0] = 0;
    sourcePoint[1] = i;
    sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
    targetLandmarks->InsertElement( landmarkCount, sourcePoint );
    landmarkCount++;
    //printf(" %d. adding source: [%g, %g] (border)...\n", landmarkCount, sourcePoint[0], sourcePoint[1]);
    //printf(" %d. adding target: [%g, %g] (border)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
    // y2
    sourcePoint[0] = movingSize[0];
    sourcePoint[1] = i;
    sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
    targetLandmarks->InsertElement( landmarkCount, sourcePoint );
    landmarkCount++;
    // y2+xstep
    for (double j=1; j<=Nborder; j++)
    {
      sourcePoint[0] = movingSize[0]+j*xstep;
      sourcePoint[1] = i;
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, sourcePoint );
      landmarkCount++;
    }
    //printf(" %d. adding source: [%g, %g] (border)...\n", landmarkCount, sourcePoint[0], sourcePoint[1]);
    //printf(" %d. adding target: [%g, %g] (border)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
  }

  // add random deformations in center of image
  double N = sqrt((double)Npoints);
  xstep = movingSize[0]/N;
  ystep = movingSize[1]/N;
  double xcenter = movingSize[0]/2;
  double ycenter = movingSize[1]/2;
  for (int x=-N/2; x<=N/2; x++)
  {
    for (int y=-N/2; y<=N/2; y++)
    {
      //std::cout << "x = " << x << std::endl;
      //std::cout << "y = " << y << std::endl;
      sourcePoint[0] = xcenter + x*xstep;
      sourcePoint[1] = ycenter + y*ystep;
      targetPoint[0] = sourcePoint[0] + nih::getRandomVariateUniformDouble( deform_length );
      targetPoint[1] = sourcePoint[1] + nih::getRandomVariateUniformDouble( deform_length );
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, targetPoint );
      landmarkCount++;
      //printf(" %d. adding source: [%g, %g] (random)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
      //printf(" %d. adding target: [%g, %g] (random)...\n", landmarkCount,  targetPoint[0], targetPoint[1]);
    }
  }

  unsigned int NLandmarks = landmarkCount;
  printf(" warping image (size [%d, %d]) with %d random vectors...\n", (int)movingSize[0], (int)movingSize[1], (int)NLandmarks);

#if _DEBUG
  std::string debug_dir = "C:/HR/Data/Pancreas/SPIE/deform_examples/def5";
  // save landmarks
  std::ofstream sourceLandmarkFile;
  std::ofstream targetLandmarkFile;
  sourceLandmarkFile.open ( debug_dir + "/source_landmarks.txt");
  targetLandmarkFile.open ( debug_dir + "/target_landmarks.txt");
  for (unsigned int i=0; i<NLandmarks; i++)
  {
      sourceLandmarkFile << sourceLandmarks->GetElement(i)[0] << " " << sourceLandmarks->GetElement(i)[1] << std::endl;
      targetLandmarkFile << targetLandmarks->GetElement(i)[0] << " " << targetLandmarks->GetElement(i)[1] << std::endl;
  }
  sourceLandmarkFile.close();
  targetLandmarkFile.close();
#endif

  // get deformation field from landmarks
  deformationFieldSource->SetSourceLandmarks( sourceLandmarks.GetPointer() );
  deformationFieldSource->SetTargetLandmarks( targetLandmarks.GetPointer() );
  try
  {
    deformationFieldSource->UpdateLargestPossibleRegion();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

#if _DEBUG
  // Write the deformation field
    typedef itk::ImageFileWriter<  DeformationFieldType  > WriterType;
    itk::SmartPointer<WriterType> writer =
      WriterType::New();
    writer->SetInput (  deformationFieldSource->GetOutput() );
    writer->SetFileName( debug_dir + "/deform.mha" );
    writer->Update();
#endif

  typedef itk::WarpImageFilter< TInputImageType, TOutputImageType, DeformationFieldType  >  WarpImageFilterType;
  itk::SmartPointer<WarpImageFilterType> warpImageFilter = WarpImageFilterType::New();

  //typedef itk::NearestNeighborInterpolateImageFunction< TInputImageType, double > InterpolatorType;
  typedef itk::LinearInterpolateImageFunction< TInputImageType, double >          InterpolatorType;
  //typedef itk::BSplineInterpolateImageFunction< TInputImageType, double >          InterpolatorType; // best trade-off between accuracy and computational cost

  itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
  warpImageFilter->SetInterpolator( interpolator );
  warpImageFilter->SetOutputSpacing( deformationFieldSource->GetOutput()->GetSpacing() );
  warpImageFilter->SetOutputOrigin(  deformationFieldSource->GetOutput()->GetOrigin() );
#if ITK_VERSION_MAJOR < 4
  warpImageFilter->SetDeformationField( deformationFieldSource->GetOutput() );
#else
  warpImageFilter->SetDisplacementField( deformationFieldSource->GetOutput() );
#endif
  warpImageFilter->SetInput( movingImage );
  try
  {
    warpImageFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  return warpImageFilter->GetOutput();
} // nih::warpImage

//================================================================================================//
//  nih::randomWarpImagePair2D
//================================================================================================//
template <class TInputImageType, class TOutputImageType>
int randomWarpImagePair2D(typename TInputImageType::Pointer movingImage1, typename TOutputImageType::Pointer &outImage1,
                          typename TInputImageType::Pointer movingImage2, typename TOutputImageType::Pointer &outImage2,
                          unsigned int Npoints, double deform_length, double stiffness=1e-6,
                          nih::InterpolationType interpolationType1=nih::LINEAR, nih::InterpolationType interpolationType2=nih::LINEAR)
{
  if ( (Npoints==0) || (deform_length<1e-6) )
  {
    std::cout << " no randomWarpImage2D performed. " << std::endl;
    outImage1 = movingImage1;
    outImage2 = movingImage2;
  }

  typedef   float VectorComponentType;

  typedef   itk::Vector< VectorComponentType, 2 >    VectorType;
  typedef   itk::Image< VectorType,  2 >   DeformationFieldType;

  typename TInputImageType::SizeType movingSize = movingImage1->GetLargestPossibleRegion().GetSize();
  typename TInputImageType::SizeType movingSize2 = movingImage2->GetLargestPossibleRegion().GetSize();

  if (!compareImageSizes<typename TInputImageType::SizeType, typename TInputImageType::SizeType>(movingSize, movingSize2))
  {
    std::cerr << "[ERROR]: randomWarpImagePair2D: expecting two images of same size!" << std::endl;
    return EXIT_FAILURE;
  }

#if ITK_VERSION_MAJOR < 4
  typedef itk::DeformationFieldSource<DeformationFieldType>  DeformationFieldSourceType;
#else
  typedef itk::LandmarkDisplacementFieldSource<DeformationFieldType>  DeformationFieldSourceType;
#endif
  itk::SmartPointer<DeformationFieldSourceType> deformationFieldSource = DeformationFieldSourceType::New();
  deformationFieldSource->SetOutputSpacing( movingImage1->GetSpacing() );
  deformationFieldSource->SetOutputOrigin(  movingImage1->GetOrigin() );
  deformationFieldSource->SetOutputRegion(  movingImage1->GetLargestPossibleRegion() );
  deformationFieldSource->SetOutputDirection( movingImage1->GetDirection() );

  deformationFieldSource->GetKernelTransform()->SetStiffness( stiffness ); // doesn't seem to make a lot of difference
  printf("Spline stiffness = %g\n", deformationFieldSource->GetKernelTransform()->GetStiffness() ); // default is 0.0

  //  Create source and target landmarks.
  typedef DeformationFieldSourceType::LandmarkContainerPointer   LandmarkContainerPointer;
  typedef DeformationFieldSourceType::LandmarkContainer          LandmarkContainerType;
  typedef DeformationFieldSourceType::LandmarkPointType          LandmarkPointType;

  itk::SmartPointer<LandmarkContainerType> sourceLandmarks = LandmarkContainerType::New();
  itk::SmartPointer<LandmarkContainerType> targetLandmarks = LandmarkContainerType::New();

  LandmarkPointType sourcePoint;
  LandmarkPointType targetPoint;

  // insert landmarks along that fix the border
  unsigned int Nborder = 1;
  unsigned int landmarkCount = 0;
  // along x1
  double xstep = itk::Math::Ceil<double>((double)movingSize[0]/3);
  double ystep = itk::Math::Ceil<double>((double)movingSize[1]/3);
  for (double i=0; i<movingSize[0]; i=i+xstep)
  {
    // x1-ystep
    for (double j=1; j<=Nborder; j++)
    {
      sourcePoint[0] = i;
      sourcePoint[1] = 0-j*ystep;
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, sourcePoint );
      landmarkCount++;
    }
    // x1
    sourcePoint[0] = i;
    sourcePoint[1] = 0;
    sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
    targetLandmarks->InsertElement( landmarkCount, sourcePoint );
    landmarkCount++;
    //printf(" %d. adding source: [%g, %g] (border)...\n", landmarkCount, sourcePoint[0], sourcePoint[1]);
    //printf(" %d. adding target: [%g, %g] (border)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
    // x2
    sourcePoint[0] = i;
    sourcePoint[1] = movingSize[1];
    sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
    targetLandmarks->InsertElement( landmarkCount, sourcePoint );
    landmarkCount++;
    // x2+ystep
    for (double j=1; j<=Nborder; j++)
    {
      sourcePoint[0] = i;
      sourcePoint[1] = movingSize[1]+j*ystep;
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, sourcePoint );
      landmarkCount++;
    }
    //printf(" %d. adding source: [%g, %g] (border)...\n", landmarkCount, sourcePoint[0], sourcePoint[1]);
    //printf(" %d. adding target: [%g, %g] (border)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
  }
  // along y
  //for (double i=ystep; i<movingSize[1]-ystep; i=i+ystep) // exclude 'corners'
  for (double i=0; i<movingSize[1]; i=i+ystep)
  {
    // y1-xstep
    for (double j=1; j<=Nborder; j++)
    {
      sourcePoint[0] = 0-j*xstep;
      sourcePoint[1] = i;
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, sourcePoint );
      landmarkCount++;
    }
    // y1
    sourcePoint[0] = 0;
    sourcePoint[1] = i;
    sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
    targetLandmarks->InsertElement( landmarkCount, sourcePoint );
    landmarkCount++;
    //printf(" %d. adding source: [%g, %g] (border)...\n", landmarkCount, sourcePoint[0], sourcePoint[1]);
    //printf(" %d. adding target: [%g, %g] (border)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
    // y2
    sourcePoint[0] = movingSize[0];
    sourcePoint[1] = i;
    sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
    targetLandmarks->InsertElement( landmarkCount, sourcePoint );
    landmarkCount++;
    // y2+xstep
    for (double j=1; j<=Nborder; j++)
    {
      sourcePoint[0] = movingSize[0]+j*xstep;
      sourcePoint[1] = i;
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, sourcePoint );
      landmarkCount++;
    }
    //printf(" %d. adding source: [%g, %g] (border)...\n", landmarkCount, sourcePoint[0], sourcePoint[1]);
    //printf(" %d. adding target: [%g, %g] (border)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
  }

  // add random deformations in center of image
  double N = sqrt((double)Npoints);
  xstep = movingSize[0]/N;
  ystep = movingSize[1]/N;
  double xcenter = movingSize[0]/2;
  double ycenter = movingSize[1]/2;
  for (int x=-N/2; x<=N/2; x++)
  {
    for (int y=-N/2; y<=N/2; y++)
    {
      //std::cout << "x = " << x << std::endl;
      //std::cout << "y = " << y << std::endl;
      sourcePoint[0] = xcenter + x*xstep;
      sourcePoint[1] = ycenter + y*ystep;
      targetPoint[0] = sourcePoint[0] + nih::getRandomVariateUniformDouble( deform_length );
      targetPoint[1] = sourcePoint[1] + nih::getRandomVariateUniformDouble( deform_length );
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, targetPoint );
      landmarkCount++;
      //printf(" %d. adding source: [%g, %g] (random)...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
      //printf(" %d. adding target: [%g, %g] (random)...\n", landmarkCount,  targetPoint[0], targetPoint[1]);
    }
  }

  unsigned int NLandmarks = landmarkCount;
  printf(" warping image (size [%d, %d]) with %d random vectors and max of %g...\n",
      (int)movingSize[0], (int)movingSize[1], (int)NLandmarks, deform_length);

#if _DEBUG
  std::string debug_dir = "C:/HR/Data/Pancreas/SPIE/deform_examples/def5";
  // save landmarks
  std::ofstream sourceLandmarkFile;
  std::ofstream targetLandmarkFile;
  sourceLandmarkFile.open ( debug_dir + "/source_landmarks.txt");
  targetLandmarkFile.open ( debug_dir + "/target_landmarks.txt");
  for (unsigned int i=0; i<NLandmarks; i++)
  {
      sourceLandmarkFile << sourceLandmarks->GetElement(i)[0] << " " << sourceLandmarks->GetElement(i)[1] << std::endl;
      targetLandmarkFile << targetLandmarks->GetElement(i)[0] << " " << targetLandmarks->GetElement(i)[1] << std::endl;
  }
  sourceLandmarkFile.close();
  targetLandmarkFile.close();
#endif

  // get deformation field from landmarks
  deformationFieldSource->SetSourceLandmarks( sourceLandmarks.GetPointer() );
  deformationFieldSource->SetTargetLandmarks( targetLandmarks.GetPointer() );
  try
  {
    deformationFieldSource->UpdateLargestPossibleRegion();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
  }

#if _DEBUG
  // Write the deformation field
    typedef itk::ImageFileWriter<  DeformationFieldType  > WriterType;
    itk::SmartPointer<WriterType> writer =
      WriterType::New();
    writer->SetInput (  deformationFieldSource->GetOutput() );
    writer->SetFileName( debug_dir + "/deform.mha" );
    writer->Update();
#endif

  typedef itk::WarpImageFilter< TInputImageType, TOutputImageType, DeformationFieldType  >  WarpImageFilterType;
  itk::SmartPointer<WarpImageFilterType> warpImageFilter1 = WarpImageFilterType::New();
  itk::SmartPointer<WarpImageFilterType> warpImageFilter2 = WarpImageFilterType::New();

  //typedef itk::NearestNeighborInterpolateImageFunction< TInputImageType, double > InterpolatorType;
  //typedef itk::LinearInterpolateImageFunction< TInputImageType, double >          InterpolatorType;
  //typedef itk::BSplineInterpolateImageFunction< TInputImageType, double >          InterpolatorType; // best trade-off between accuracy and computational cost

  // select interpolationType1
  if (interpolationType1 == nih::NEAREST)
  {
    //std::cout << "  ... using nearest neighbor interpolation..." << std::endl;
    typedef itk::NearestNeighborInterpolateImageFunction< TInputImageType, double >  InterpolatorType;
    itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
    warpImageFilter1->SetInterpolator( interpolator );
  }
  else if (interpolationType1 == nih::LINEAR)
  {
    //std::cout << "  ... using linear interpolation..." << std::endl;
    typedef itk::LinearInterpolateImageFunction< TInputImageType, double >  InterpolatorType;
    itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
    warpImageFilter1->SetInterpolator( interpolator );
  }
  else if (interpolationType1 == nih::BSPLINE) // best trade-off between accuracy and computational cost
  {
    //std::cout << "  ... using B-Spline interpolation..." << std::endl;
    typedef itk::BSplineInterpolateImageFunction< TInputImageType, double > InterpolatorType;
    itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
    warpImageFilter1->SetInterpolator( interpolator );
  }
  else
  {
    std::cerr << "[Error] nih::rotateImageAroundCenter : No such interpolation type! " << interpolationType1 << std::endl;
    exit(EXIT_FAILURE);
  }
  warpImageFilter1->SetOutputSpacing( deformationFieldSource->GetOutput()->GetSpacing() );
  warpImageFilter1->SetOutputOrigin(  deformationFieldSource->GetOutput()->GetOrigin() );
#if ITK_VERSION_MAJOR < 4
  warpImageFilter1->SetDeformationField( deformationFieldSource->GetOutput() );
#else
  warpImageFilter1->SetDisplacementField( deformationFieldSource->GetOutput() );
#endif

  // select interpolationType1
  if (interpolationType2 == nih::NEAREST)
  {
    //std::cout << "  ... using nearest neighbor interpolation..." << std::endl;
    typedef itk::NearestNeighborInterpolateImageFunction< TInputImageType, double >  InterpolatorType;
    itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
    warpImageFilter2->SetInterpolator( interpolator );
  }
  else if (interpolationType2 == nih::LINEAR)
  {
    //std::cout << "  ... using linear interpolation..." << std::endl;
    typedef itk::LinearInterpolateImageFunction< TInputImageType, double >  InterpolatorType;
    itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
    warpImageFilter2->SetInterpolator( interpolator );
  }
  else if (interpolationType2 == nih::BSPLINE) // best trade-off between accuracy and computational cost
  {
    //std::cout << "  ... using B-Spline interpolation..." << std::endl;
    typedef itk::BSplineInterpolateImageFunction< TInputImageType, double > InterpolatorType;
    itk::SmartPointer<InterpolatorType> interpolator = InterpolatorType::New();
    warpImageFilter2->SetInterpolator( interpolator );
  }
  else
  {
    std::cerr << "[Error] nih::rotateImageAroundCenter : No such interpolation type! " << interpolationType2 << std::endl;
    exit(EXIT_FAILURE);
  }
  warpImageFilter2->SetOutputSpacing( deformationFieldSource->GetOutput()->GetSpacing() );
  warpImageFilter2->SetOutputOrigin(  deformationFieldSource->GetOutput()->GetOrigin() );
#if ITK_VERSION_MAJOR < 4
  warpImageFilter2->SetDeformationField( deformationFieldSource->GetOutput() );
#else
  warpImageFilter2->SetDisplacementField( deformationFieldSource->GetOutput() );
#endif

  // warp image1
  warpImageFilter1->SetInput( movingImage1 );
  try
  {
    warpImageFilter1->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
  }
  outImage1 = warpImageFilter1->GetOutput();

  // warp image2
  warpImageFilter2->SetInput( movingImage2 );
  try
  {
    warpImageFilter2->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
  }
  outImage2 = warpImageFilter2->GetOutput();
} // nih::randomlyWarpImagePair2D

//================================================================================================//
//  nih::getSegmentationOverlap2D
//================================================================================================//
template <class TInputImageType>
double getSegmentationOverlap2D(typename TInputImageType::Pointer segSource, typename TInputImageType::Pointer segTarget)
{
  typename TInputImageType::PixelType backgroundValue = 0;
  typename TInputImageType::PixelType labelValue      = 1;

  typename TInputImageType::SizeType sizeSource = segSource->GetLargestPossibleRegion().GetSize();
  typename TInputImageType::SizeType sizeTarget = segTarget->GetLargestPossibleRegion().GetSize();
  // Check if images are consistent
  if ( (sizeSource[0]!=sizeTarget[0]) || (sizeSource[1]!=sizeTarget[1]) || (sizeSource[2]!=sizeTarget[2]) )
  {
    std::cerr << " [ERROR] getSegmentationOverlap2D: Image dimensions of image and label image does not fit!" << std::endl;
    return EXIT_FAILURE;
  }

  // Make sure there is only one label in each image
  itk::ImageRegionIterator<TInputImageType> segSourceIterator(segSource, segSource->GetLargestPossibleRegion());
  itk::ImageRegionIterator<TInputImageType> segTargetIterator(segTarget, segTarget->GetLargestPossibleRegion());
  while(!segSourceIterator.IsAtEnd())
  {
    if (segSourceIterator.Get() != backgroundValue)
    {
      segSourceIterator.Set( labelValue );
    }
    if (segTargetIterator.Get() != backgroundValue)
    {
      segTargetIterator.Set( labelValue );
    }
    ++segSourceIterator;
    ++segTargetIterator;
  }

  typedef itk::LabelOverlapMeasuresImageFilter<TInputImageType> OverlapMeasureFilterType;
  itk::SmartPointer<OverlapMeasureFilterType> overlapFilter = OverlapMeasureFilterType::New();

  overlapFilter->SetSourceImage( segSource );
  overlapFilter->SetTargetImage( segTarget ); // ground truth
  try
  {
    overlapFilter->UpdateLargestPossibleRegion();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }
  //std::cout << "labelValue:      " << labelValue << std::endl;
  //std::cout << "Target:          " << overlapFilter->GetTargetOverlap( labelValue ) << std::endl;
  //std::cout << "Union (jaccard): " << overlapFilter->GetUnionOverlap( labelValue ) << std::endl;
  //std::cout << "Mean (dice):     " << overlapFilter->GetMeanOverlap( labelValue ) << std::endl;
  //std::cout << "Volume sim.:     " << overlapFilter->GetVolumeSimilarity( labelValue ) << std::endl;
  //std::cout << "False negative:  " << overlapFilter->GetFalseNegativeError( labelValue ) << std::endl;
  //std::cout << "False positive:  " << overlapFilter->GetFalsePositiveError( labelValue ) << std::endl;
  //std::cout << std::endl;

  return overlapFilter->GetMeanOverlap( labelValue );
} // nih::getSegmentationOverlap2D

//================================================================================================//
//  nih::checkPairwiseOverlap
//================================================================================================//
template <class TImageTypeSource, class TImageTypeTarget>
int checkPairwiseOverlap(typename TImageTypeSource::Pointer Source, typename TImageTypeTarget::Pointer Target, unsigned int label_i, unsigned int label_j)
{
  /* PairwiseTypes */
  //PairwiseType = 1: both source labels are    "Inside" target mask.
  //PairwiseType = 2: both source labels are    "Outside" target mask.
  //PairwiseType = 3: both source labels are on "Border" target mask.

  // check if overlapping with target in label_i and label_j
  typename TImageTypeTarget::PixelType backgroundValue = 0;

  int PairwiseType = -1;

  // get target values insided source label i and j
  double target_i = 0;
  double target_j = 0;
  unsigned int target_i_count = 0;
  unsigned int target_j_count = 0;
  itk::ImageRegionIterator<TImageTypeSource> sourceIterator(Source, Source->GetLargestPossibleRegion());
  itk::ImageRegionIterator<TImageTypeTarget> targetIterator(Target, Target->GetLargestPossibleRegion());
  while(!sourceIterator.IsAtEnd())
  {
    if (sourceIterator.Get() == label_i)
    {
      target_i += targetIterator.Get();
      target_i_count++;
    }
    if (sourceIterator.Get() == label_j)
    {
      target_j += targetIterator.Get();
      target_j_count++;
    }
    ++sourceIterator;
    ++targetIterator;
  }
  // check if all target pixels are of same value
  target_i = target_i/target_i_count;
  target_j = target_j/target_j_count;
  if ( target_i/(int)target_i > 1.0 )
  {
    std::cerr << " not all target pixels are constant within label_i! " << target_i << std::endl;
    exit(EXIT_FAILURE);
  }
  if ( target_j/(int)target_j > 1.0 )
  {
    std::cerr << " not all target pixels are constant within label_j! " << target_j << std::endl;
    exit(EXIT_FAILURE);
  }

  // get PairwiseType
  if ( (target_i>0) && (target_j>0) )
  {
    PairwiseType = 1; // Inside
  }
  else if ( (target_i==0) && (target_j==0) )
  {
    PairwiseType = 2; // Outside
  }
  else if ( ( (target_i >0) && (target_j==0) ) ||
            ( (target_i==0) && (target_j >0) ) )
  {
    PairwiseType = 3; // Border
  }
  else
  {
    std::cerr << " Could not find PairwiseType! " << std::endl;
    exit(EXIT_FAILURE);
  }

  #if _DEBUG
  //if (PairwiseType==1) // Inside
  //if (PairwiseType==2) // Outside
  if (PairwiseType==3) // Border
  {
    std::string debug_dir = "C:/HR/Data/Pancreas/MICCAI/slic_pairwise_prob_DEVEL";
    // Write the source image
    typedef itk::ImageFileWriter< typename TImageTypeSource  > WriterSourceType;
    itk::SmartPointer<WriterSourceType> writerSource = WriterSourceType::New();
    writerSource->SetInput ( Source );
    writerSource->SetFileName( debug_dir + "/source.nii.gz" );
    writerSource->Update();
    // Write the target image
    typedef itk::ImageFileWriter< typename TImageTypeTarget  > WriterTargetType;
    itk::SmartPointer<WriterTargetType> writerTarget = WriterTargetType::New();
    writerTarget->SetInput ( Target );
    writerTarget->SetFileName( debug_dir + "/target.nii.gz" );
    writerTarget->Update();
  }
  #endif

  return PairwiseType;
} // nih::getSegmentationOverlap2D


//================================================================================================//
//  nih::dilateBinaryImage
//================================================================================================//
template <class TImageType, unsigned int Dimensions>
typename TImageType::Pointer dilateBinaryImage(typename TImageType::Pointer image, const unsigned int radius)
{
  typedef itk::BinaryBallStructuringElement<typename TImageType::PixelType, Dimensions> KernelType;

  KernelType ball;
  typename KernelType::SizeType ballSize;
  for (unsigned int i=0; i<Dimensions; i++)
  {
    ballSize[i] = radius;
  }
  ball.SetRadius(ballSize);
  ball.CreateStructuringElement();

  typedef itk::BinaryDilateImageFilter <TImageType, TImageType, KernelType>
          BinaryDilateImageFilterType;

  itk::SmartPointer<BinaryDilateImageFilterType> dilateFilter
          = BinaryDilateImageFilterType::New();
  dilateFilter->SetInput(image);
  dilateFilter->SetKernel(ball);
  try
  {
    std::cout << " nih:dilateBinaryImage with radius " << radius << std::endl;
    dilateFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  return dilateFilter->GetOutput();
} // nih::dilateBinaryImage

//================================================================================================//
//  nih::maskImage
//================================================================================================//
template <class TImageType, class TMaskType>
typename TImageType::Pointer maskImage(typename TImageType::Pointer Image, typename TMaskType::Pointer Mask)
{
  typedef itk::MaskImageFilter< TImageType, TMaskType > MaskFilterType;
  itk::SmartPointer< MaskFilterType > maskFilter = MaskFilterType::New();
  maskFilter->SetInput(Image);
  maskFilter->SetMaskImage(Mask);
  try
  {
    maskFilter->UpdateLargestPossibleRegion();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }
  //Mask->Print(std::cout);

  return maskFilter->GetOutput();
} // nih::maskImage

//================================================================================================//
//  nih::readImage
//================================================================================================//
template <class TImageType>
bool readImage(itk::SmartPointer<TImageType> & image, const char *file)
{
  typedef typename TImageType::PixelType PixelType;
  enum { ImageDimension = TImageType::ImageDimension };
  if( std::string(file).length() < 3 )
    {
    std::cerr << " bad file name " << std::string(file) << std::endl;
    image = NULL;
    return EXIT_FAILURE;
    }

  typedef itk::ImageFileReader<TImageType>                 ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( file );
  std::cout << "  Reading image from " << file << std::endl;
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

  return EXIT_SUCCESS;
}

//================================================================================================//
//  nih::writeImage
//================================================================================================//
template <class TImageType>
bool writeImage(itk::SmartPointer<TImageType> & image, const char *file)
{
  typedef typename TImageType::PixelType PixelType;
  enum { ImageDimension = TImageType::ImageDimension };
  if( std::string(file).length() < 3 )
    {
    std::cerr << " bad file name " << std::string(file) << std::endl;
    image = NULL;
    return EXIT_FAILURE;
    }

  typedef itk::ImageFileWriter<TImageType>              WriterType;
  itk::SmartPointer<WriterType> writer = WriterType::New();
  writer->SetFileName( file );
  writer->SetInput( image );
  std::cout << "  Writing image to " << file << std::endl;
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

  return EXIT_SUCCESS;
}

//================================================================================================//
//  nih::binaryThresholdImage
//================================================================================================//
template <class TImage>
typename TImage::Pointer binaryThresholdImage(
  typename TImage::PixelType lower,
  typename TImage::PixelType upper,
  typename TImage::PixelType replaceVal, typename TImage::Pointer input)
{
  typedef typename TImage::PixelType PixelType;
  // Begin Threshold Image
  typedef itk::BinaryThresholdImageFilter<TImage, TImage> InputThresholderType;
  typename InputThresholderType::Pointer inputThresholder =
    InputThresholderType::New();

  inputThresholder->SetInput( input );
  inputThresholder->SetInsideValue(  replaceVal );
  unsigned int outVal = 0;
  if( (float) replaceVal == (float) -1 )
    {
    outVal = 1;
    }
  inputThresholder->SetOutsideValue( outVal );

  if( upper < lower )
    {
    upper = 255;
    }
  inputThresholder->SetLowerThreshold( (PixelType) lower );
  inputThresholder->SetUpperThreshold( (PixelType) upper );
  inputThresholder->Update();

  return inputThresholder->GetOutput();
}

//================================================================================================//
//  nih::distanceToSegmentation (NOTE: assumes input point in physical coordinates!)
//================================================================================================//
template <class TImageType, const unsigned int Dimensions>
std::vector<float> distanceToSegmentation(typename TImageType::Pointer image, std::vector< std::vector<double> > points)
{
  typedef itk::Image<float, Dimensions>  FloatImageType;
  typename TImageType::PointType pointPhys;
  typename TImageType::IndexType pointIndex;

  std::cout << " compute signed maurer distance on image " << image->GetLargestPossibleRegion().GetSize() << std::endl;
  typedef  itk::SignedMaurerDistanceMapImageFilter< TImageType, FloatImageType  > SignedMaurerDistanceMapImageFilterType;
  itk::SmartPointer<SignedMaurerDistanceMapImageFilterType> distanceMapImageFilter =
    SignedMaurerDistanceMapImageFilterType::New();
  distanceMapImageFilter->SetInput(image);
  try
  {
    distanceMapImageFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    exit(EXIT_FAILURE);
  }

#if _DEBUG
 	typedef itk::ImageFileWriter< FloatImageType > WriterType;
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName( "C:/tmp/maurer_distance.nii.gz" );
	writer->SetInput( distanceMapImageFilter->GetOutput() );
	try
	{
  	writer->Update();
	}
	catch( itk::ExceptionObject & error )
	{
		std::cerr << "Error: " << error << std::endl;
		return EXIT_FAILURE;
	}
#endif

  // check distance of points to segmentation
  unsigned int Npoints = points.size();
  std::vector<float> distances;
  for (unsigned int p=0; p<Npoints; p++)
  {
    pointPhys[0] = points[p][0]; // mm
    pointPhys[1] = points[p][1]; // mm
    pointPhys[2] = points[p][2]; // mm

    image->TransformPhysicalPointToIndex( pointPhys, pointIndex ); // convert to index
    if ( image->GetLargestPossibleRegion().IsInside( pointIndex ) )
    {
      distances.push_back(
            distanceMapImageFilter->GetOutput()->GetPixel(pointIndex)
        );
    }
    else // else assume 'infinite' distance
    {
      distances.push_back( std::numeric_limits<float>::infinity() );
    }
  }

  return distances;
}

//================================================================================================//
//  nih::getSlicesAndSave
//================================================================================================//
template <class TInputImageType, class TOutputImageType>
int getSlicesAndSave(typename TInputImageType::Pointer roiImage,
                     typename TInputImageType::PixelType winMin, typename TInputImageType::PixelType winMax,
                     unsigned int outputImageSize,
                     std::string outputPrefix)
{
  typedef itk::ImageFileWriter<TOutputImageType> WriterType;
  itk::SmartPointer<WriterType> writer = WriterType::New();

  typedef itk::IntensityWindowingImageFilter<TInputImageType, TInputImageType> WindowingFilterType;

  // windowing options
  itk::SmartPointer<WindowingFilterType> windowFilter = WindowingFilterType::New();
  windowFilter->SetInput( roiImage );
  windowFilter->SetWindowMinimum(winMin);
  windowFilter->SetWindowMaximum(winMax);
  windowFilter->SetOutputMinimum( 0.0 );
  windowFilter->SetOutputMaximum( itk::NumericTraits< typename TOutputImageType::PixelType >::max() );
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
  itk::SmartPointer<TInputImageType> image = windowFilter->GetOutput();

  //create output directory
  std::string outputDir = nih::getPath(outputPrefix);
  std::cout << "  Creating output directory at: " << outputDir << std::endl;
  itk::FileTools::CreateDirectory( outputDir.c_str() );

  // Write image slices
  typename TInputImageType::SizeType ImageSize = image->GetLargestPossibleRegion().GetSize();
  printf(" image size is [%d, %d, %d]\n", (int)ImageSize[0], (int)ImageSize[1], (int)ImageSize[2]);

  std::string outputImageName;
  char count_char[100];

  typename TOutputImageType::SizeType outSize;
  outSize[0] = outputImageSize;
  outSize[1] = outputImageSize;
  std::cout << std::endl << " wrote " << ImageSize[2] << " image slices to " << outputDir << std::endl;
  unsigned int zcounter = 0;
  printf(" Extract slices between %d and %d ...\n", 1, (int)ImageSize[2]);
  for (unsigned int z = 0; z < ImageSize[2]; z++)
  {
    std::cout << z << ", ";
    zcounter++;
    sprintf(count_char, "_z%03d", z); // sprintf_s in windows?
    outputImageName = outputPrefix + count_char + ".jpg";
    writer->SetInput(
      nih::rescaleImage2D<TOutputImageType,TOutputImageType>( nih::getImageSlice<TInputImageType,TOutputImageType>(image, z), outSize )
    );
//    writer->SetInput( nih::getImageSlice<TInputImageType,TOutputImageType>(image, z) );

    writer->SetFileName( outputImageName );

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
} // nih::getSlicesAndSave()

//================================================================================================//
//  nih::getSlicesAndSave
//================================================================================================//
template <class TInputImageType, class TOutputImageType>
int getSlicesAndSaveLabelBasedOnPoints(typename TInputImageType::Pointer roiImage,
                     std::vector< std::vector<double> > points,
                     unsigned int distance2sliceForPos,
                     typename TInputImageType::PixelType winMin, typename TInputImageType::PixelType winMax,
                     unsigned int outputImageSize,
                     std::string outputPrefix)
{
  typename TInputImageType::PointType pointPhys;
  typename TInputImageType::IndexType pointIndex;
  unsigned int Npoints = points.size();

  typedef itk::ImageFileWriter<TOutputImageType> WriterType;
  itk::SmartPointer<WriterType> writer = WriterType::New();

  typedef itk::IntensityWindowingImageFilter<TInputImageType, TInputImageType> WindowingFilterType;

  // windowing options
  itk::SmartPointer<WindowingFilterType> windowFilter = WindowingFilterType::New();
  windowFilter->SetInput( roiImage );
  windowFilter->SetWindowMinimum(winMin);
  windowFilter->SetWindowMaximum(winMax);
  windowFilter->SetOutputMinimum( 0.0 );
  windowFilter->SetOutputMaximum( itk::NumericTraits< typename TOutputImageType::PixelType >::max() );
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
  itk::SmartPointer<TInputImageType> image = windowFilter->GetOutput();

  //create output directory
  std::string outputDir = nih::getPath(outputPrefix);
  std::cout << "  Creating output directory at: " << outputDir << std::endl;
  itk::FileTools::CreateDirectory( outputDir.c_str() );

  // Write image slices
  typename TInputImageType::SizeType ImageSize = image->GetLargestPossibleRegion().GetSize();
  printf(" image size is [%d, %d, %d]\n", (int)ImageSize[0], (int)ImageSize[1], (int)ImageSize[2]);

  std::string outputImageName;
  char count_char[100];
  unsigned int pos_count = 0;
  unsigned int neg_count = 0;

  std::string class_str;
  typename TOutputImageType::SizeType outSize;
  outSize[0] = outputImageSize;
  outSize[1] = outputImageSize;
  std::cout << std::endl << " use distance2sliceForPos = " << distance2sliceForPos << std::endl;
  unsigned int zcounter = 0;
  printf(" Extract slices between %d and %d ...\n", 1, (int)ImageSize[2]);
  for (unsigned int z = 0; z < ImageSize[2]; z++)
  {
    std::cout << z << ", ";

    class_str = "neg"; // default if Npoints = 0
    for (unsigned int p=0; p<Npoints; p++)
    {
      pointPhys[0] = points[p][0]; // mm
      pointPhys[1] = points[p][1]; // mm
      pointPhys[2] = points[p][2]; // mm

      image->TransformPhysicalPointToIndex( pointPhys, pointIndex ); // convert to index

      // is point close to current slice?
      double point2slice = abs( pointIndex[2] - z );
      std::cout << "-> point2slice = " <<  point2slice << std::endl;
      if (point2slice <= distance2sliceForPos)
      {
        std::cout << "\tFOUND POS FRACTURE: " << pointPhys[2] << ", slice " << z << ": "
          << " distance " << point2slice << " <= " <<  distance2sliceForPos << " slice(s).." << std::endl;
        class_str = "pos";
        pos_count++;
      }
      else
      {
        class_str = "neg";
        neg_count++;
      }
    }

    zcounter++;
    sprintf(count_char, "_z%03d_", z); // sprintf_s in windows?
    outputImageName = outputPrefix + count_char + class_str + ".jpg";
    writer->SetInput(
      nih::rescaleImage2D<TOutputImageType,TOutputImageType>( nih::getImageSlice<TInputImageType,TOutputImageType>(image, z), outSize )
    );
//    writer->SetInput( nih::getImageSlice<TInputImageType,TOutputImageType>(image, z) );

    writer->SetFileName( outputImageName );

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

  printf("Found %d (of %d)\t POSITIVE region(s) of interest.\n", pos_count, Npoints);
  printf("Found %d        \t NEGATIVE region(s) of interest.\n", neg_count);

  return EXIT_SUCCESS;
} // nih::getSlicesAndSaveLabelBasedOnPoints()

//================================================================================================//
//  nih::padImageWithZeroOffset (via image pasting rather than padding)
// NOTE: constantpad filter adds negative index as start index and causes region of interest filter to not work!
//================================================================================================//
template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer padImageWithZeroOffset(
            typename TInputImageType::Pointer image, const unsigned long Padding[3], typename TOutputImageType::PixelType paddingValue)
{
  printf("  before padding: index [%d, %d, %d], size [%d, %d, %d]\n",
              image->GetLargestPossibleRegion().GetIndex()[0], image->GetLargestPossibleRegion().GetIndex()[1], image->GetLargestPossibleRegion().GetIndex()[2],
              image->GetLargestPossibleRegion().GetSize()[0], image->GetLargestPossibleRegion().GetSize()[1], image->GetLargestPossibleRegion().GetSize()[2]);

  typename TInputImageType::SizeType inSize =  image->GetLargestPossibleRegion().GetSize();

  // Create padded image
  typename TOutputImageType::RegionType outRegion;
  typename TOutputImageType::IndexType outIndex;
  outIndex.Fill( 0 );
  typename TOutputImageType::SizeType outSize;
  outSize[0] = inSize[0] + 2*Padding[0];
  outSize[1] = inSize[1] + 2*Padding[1];
  outSize[2] = inSize[2] + 2*Padding[2];
  outRegion.SetIndex( outIndex );
  outRegion.SetSize( outSize );

  itk::SmartPointer<TOutputImageType> paddedImage = TOutputImageType::New();
  paddedImage->SetRegions( outRegion );
  paddedImage->Allocate();

  // paste original image
  typename TOutputImageType::IndexType destinationIndex;
  destinationIndex[0] = Padding[0];
  destinationIndex[1] = Padding[1];
  destinationIndex[2] = Padding[2];

  typedef itk::PasteImageFilter <TInputImageType, TOutputImageType> PasteImageFilterType;
  itk::SmartPointer<PasteImageFilterType> pasteFilter = PasteImageFilterType::New ();
  pasteFilter->SetSourceImage(image);
  pasteFilter->SetDestinationImage(paddedImage);
  pasteFilter->SetSourceRegion(image->GetLargestPossibleRegion());
  pasteFilter->SetDestinationIndex(destinationIndex);
  try
  {
    pasteFilter->Update();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "pasteFilter: Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    exit(EXIT_FAILURE);
  }
  paddedImage = pasteFilter->GetOutput();

  printf("  after padding: index [%d, %d, %d], size [%d, %d, %d]\n",
              paddedImage->GetLargestPossibleRegion().GetIndex()[0], paddedImage->GetLargestPossibleRegion().GetIndex()[1], paddedImage->GetLargestPossibleRegion().GetIndex()[2],
              paddedImage->GetLargestPossibleRegion().GetSize()[0], paddedImage->GetLargestPossibleRegion().GetSize()[1], paddedImage->GetLargestPossibleRegion().GetSize()[2]);

#if _DEBUG
  // Write the deformation field
    typedef itk::ImageFileWriter< TOutputImageType  > WriterType;
    itk::SmartPointer<WriterType> writer = WriterType::New();
    writer->SetInput (  paddedImage );
    writer->SetFileName( "/home/rothhr/Data/Spine/PostElemFxs/edge/DEVEL/tripatch/paddedImage.nii.gz" );
    writer->Update();
#endif

  return paddedImage;
} // nih::padImageWithZeroOffset

//================================================================================================//
//  nih::cropImage
//================================================================================================//
template <class TInputImageType,class TOutputImageType>
typename TOutputImageType::Pointer cropImage(typename TInputImageType::Pointer Image, typename TInputImageType::SizeType cropSize)
{
  typedef itk::CropImageFilter <TInputImageType, TOutputImageType>
    CropImageFilterType;

  itk::SmartPointer<CropImageFilterType> cropFilter
    = CropImageFilterType::New();
  cropFilter->SetInput(Image);

  cropFilter->SetUpperBoundaryCropSize(cropSize);
  cropFilter->SetLowerBoundaryCropSize(cropSize);

  cropFilter->Update();

  return cropFilter->GetOutput();
} //  nih::cropImage

//================================================================================================//
//  nih::saveToMitkMPS
/*/================================================================================================//
template <class PointSetType>
int saveToMitkMPS(typename PointSetType pointSet, const char *filename)
{
  // MITK point set
  std::ofstream oMPSFile;
  int mitkSpecification = 0;
  int mitkTimeSeries = 0;
  oMPSFile.open( filename );
  oMPSFile << "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>" << std::endl;
  oMPSFile << "<point_set_file>" << std::endl;
  oMPSFile << "  <file_version>0.1</file_version>" << std::endl;
  oMPSFile << "  <point_set>" << std::endl;
  oMPSFile << "    <time_series>" << std::endl;
  oMPSFile << "      <time_series_id>" << mitkTimeSeries << "</time_series_id>" << std::endl;
  for(unsigned int i=0; i<typename pointSet.size(); i++)
  {
    point = pointSet[i];
    oMPSFile << "      <point>" << std::endl;
    oMPSFile << "        <id>" << idx_count << "</id>" << std::endl;
    oMPSFile << "        <specification>" << mitkSpecification << "</specification>" << std::endl;
    oMPSFile << "        <x>" << point[0] << "</x>" << std::endl;
    oMPSFile << "        <y>" << point[1] << "</y>" << std::endl;
    oMPSFile << "        <z>" << point[2] << "</z>" << std::endl;
    oMPSFile << "      </point>" << std::endl;
  }
  oMPSFile << "    </time_series>" << std::endl;
  oMPSFile << "  </point_set>" << std::endl;
  oMPSFile << "</point_set_file>" << std::endl;
  oMPSFile.close();

  return EXIT_SUCCESS;
} // nih::saveToMitkMPS */

} /* end namespace nih */

#endif
