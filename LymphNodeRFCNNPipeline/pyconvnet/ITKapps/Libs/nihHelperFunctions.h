#pragma once
#ifndef _NIHHELPERFUNCTIONS_H_
#define _NIHHELPERFUNCTIONS_H_

#include <iostream>
#include <fstream>

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
#include <itkVector.h>
#include <itkWarpImageFilter.h>
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
//  nih::getNegativeROIcandidatesMask
//================================================================================================//
template <class TImageType>
typename TImageType::Pointer getNegativeROIcandidatesMask(typename TImageType::Pointer roiImage, typename TImageType::Pointer tissueImage, float radius)
{
  std::cout << "  getNegativeROIcandidatesMask with box element of size " << radius << " [voxels]." << std::endl;
  // Dilate region of roiImage by box of size 'radius'
  typedef itk::FlatStructuringElement< Dimension > StructuringElementType;
  StructuringElementType::RadiusType elementRadius;
  elementRadius.Fill( radius );
 
  StructuringElementType structuringElement = StructuringElementType::Box(elementRadius);
 
  typedef itk::BinaryDilateImageFilter <SegmentationImageType, SegmentationImageType, StructuringElementType>
    dilateImageFilterType;
 
  dilateImageFilterType::Pointer dilateFilter = dilateImageFilterType::New();
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
  typedef itk::BinaryNotImageFilter< SegmentationImageType > BinaryNotImageFilterType;
  BinaryNotImageFilterType::Pointer binaryNotFilter = BinaryNotImageFilterType::New();
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
  typedef itk::AndImageFilter< SegmentationImageType > AndImageFilterType;
  AndImageFilterType::Pointer andFilter = AndImageFilterType::New();
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
typename TOutputImageType::Pointer getRGBImagePatch(typename TInputImageType::Pointer roiImage)
{
  typedef itk::Image<TInputImageType::PixelType, 2  >                             SliceImageType;
  typedef itk::ExtractImageFilter< TInputImageType, SliceImageType >              SliceFilterType;
  #if ITK_VERSION_MAJOR < 4
    typedef itk::Compose3DCovariantVectorImageFilter<OutputSliceImageType, RGBImageType> ComposeImageFilterType;
  #else
    typedef itk::ComposeImageFilter<SliceImageType, TOutputImageType> ComposeImageFilterType;
  #endif

  unsigned int slicingDim = 0;  

  SliceFilterType::Pointer slicingFilterZ = SliceFilterType::New();
  slicingFilterZ->InPlaceOn();
  slicingFilterZ->SetDirectionCollapseToIdentity(); // direction does not matter for RGB images?

  SliceFilterType::Pointer slicingFilterX = SliceFilterType::New();
  slicingFilterX->InPlaceOn();
  slicingFilterX->SetDirectionCollapseToIdentity(); // direction does not matter for RGB images?

  SliceFilterType::Pointer slicingFilterY = SliceFilterType::New();
  slicingFilterY->InPlaceOn();
  slicingFilterY->SetDirectionCollapseToIdentity(); // direction does not matter for RGB images?

  TInputImageType::SizeType slicingSize;
  TInputImageType::IndexType slicingStart;
  
  TInputImageType::RegionType slicingRegion;

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

  // X slice
  slicingDim = 0; 
  slicingSize = roiImage->GetLargestPossibleRegion().GetSize();
  slicingStart.Fill( 0 );
  slicingStart[slicingDim] = slicingSize[slicingDim]/2 - 1;
  slicingSize[slicingDim] = 0; 
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
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  // Y slice
  slicingDim = 1; 
  slicingSize = roiImage->GetLargestPossibleRegion().GetSize();
  slicingStart.Fill( 0 );
  slicingStart[slicingDim] = slicingSize[slicingDim]/2 - 1;
  slicingSize[slicingDim] = 0; 
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
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  // Compose RGB image
  SliceImageType::Pointer sliceZ = slicingFilterZ->GetOutput();
  SliceImageType::Pointer sliceX = slicingFilterX->GetOutput();
  SliceImageType::Pointer sliceY = slicingFilterY->GetOutput();

  SliceImageType::PointType origin;
  origin.Fill(0.0);
  
  sliceZ->SetOrigin( origin );
  sliceX->SetOrigin( origin );
  sliceY->SetOrigin( origin );

  ComposeImageFilterType::Pointer composeFilter = ComposeImageFilterType::New();
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
    return NULL;
  }

  return composeFilter->GetOutput();
}

//================================================================================================//
//  nih::getRGBImagePatchXY
//================================================================================================//
template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer getRGBImagePatchXY(typename TInputImageType::Pointer roiImage)
{
  typedef itk::Image<TInputImageType::PixelType, 2  >                             SliceImageType;
  typedef itk::ExtractImageFilter< TInputImageType, SliceImageType >              SliceFilterType;
  #if ITK_VERSION_MAJOR < 4
    typedef itk::Compose3DCovariantVectorImageFilter<OutputSliceImageType, RGBImageType> ComposeImageFilterType;
  #else
    typedef itk::ComposeImageFilter<SliceImageType, TOutputImageType> ComposeImageFilterType;
  #endif

  unsigned int slicingDim = 0;  

  SliceFilterType::Pointer slicingFilterZ = SliceFilterType::New();
  slicingFilterZ->InPlaceOn();
  slicingFilterZ->SetDirectionCollapseToIdentity(); // direction does not matter for RGB images?

  TInputImageType::SizeType slicingSize;
  TInputImageType::IndexType slicingStart;
  
  TInputImageType::RegionType slicingRegion;

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
  SliceImageType::Pointer sliceZ = slicingFilterZ->GetOutput();

  SliceImageType::PointType origin;
  origin.Fill(0.0);
  
  sliceZ->SetOrigin( origin );

  ComposeImageFilterType::Pointer composeFilter = ComposeImageFilterType::New();
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
//  nih::getGrayImagePatch
//================================================================================================//
template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer getGrayImagePatch(typename TInputImageType::Pointer roiImage, unsigned int slicingDim)
{
  //typedef itk::Image<TInputImageType::PixelType, 2  >                             SliceImageType;
  typedef itk::ExtractImageFilter< TInputImageType, TOutputImageType >              SliceFilterType;

  SliceFilterType::Pointer slicingFilter = SliceFilterType::New();
  slicingFilter->InPlaceOn();
  slicingFilter->SetDirectionCollapseToIdentity(); // direction does not matter for RGB/gray images?

  TInputImageType::SizeType slicingSize;
  TInputImageType::IndexType slicingStart;
  
  TInputImageType::RegionType slicingRegion;

  slicingSize = roiImage->GetLargestPossibleRegion().GetSize();
  slicingStart.Fill( 0 );
  slicingStart[slicingDim] = slicingSize[slicingDim]/2 - 1;
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
    std::cerr << "Exception caught !" << std::endl;
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

  SliceFilterType::Pointer slicingFilter = SliceFilterType::New();
  slicingFilter->InPlaceOn();
  slicingFilter->SetDirectionCollapseToIdentity(); // direction does not matter for RGB images?

  TInputImageType::SizeType slicingSize;
  TInputImageType::IndexType slicingStart;
  
  TInputImageType::RegionType slicingRegion;

  slicingSize = roiImage->GetLargestPossibleRegion().GetSize();
  if ((slice<0) || (slice>=slicingSize[slicingDim]))
  {
    std::cerr << " Slice not within range: " << slice << " not within [0," << slicingSize[slicingDim] << "]" << std::endl;
    return NULL;
  }
  slicingStart.Fill( 0 );
  slicingStart[slicingDim] = slice;
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
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return NULL;
  }

  return slicingFilter->GetOutput();

} // nih::getImageSlice

//================================================================================================//
//  nih::getNegativeROIcandidatesMask
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
  GeneratorType::Pointer generator = GeneratorType::New();
 
  generator->Initialize();
  
  return generator->GetUniformVariate(-1*limit, limit);
}
 
//================================================================================================//
//  nih::getRandomVariateInt
//================================================================================================//
int getRandomVariateInt(int limit)
{
  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator GeneratorType;
  GeneratorType::Pointer generator = GeneratorType::New();
 
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
//  nih::getRandomUniformPoint
//================================================================================================//
template <class TImageType>
typename TImageType::PointType getRandomUniformPoint()
{
  TImageType::PointType v;
  v[0] = getRandomVariateUniformDouble( 1.0 );
  v[1] = getRandomVariateUniformDouble( 1.0 );
  v[2] = getRandomVariateUniformDouble( 1.0 );

  return normalizePoint< TImageType >( v );
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

  TImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();
  Superclass::ContinuousIndexType halfsize;
  Superclass::PointType center;

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
template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer rotateImageAroundCenter
  (typename TInputImageType::Pointer image, double *axis, double angle_in_degrees)
{
  return rotateImageAroundCenter<TInputImageType, TOutputImageType>(image, axis, nih::BSPLINE) ;
}

template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer rotateImageAroundCenter
  (typename TInputImageType::Pointer image, double *axis, double angle_in_degrees, nih::InterpolationType interpolationType)
{
  typedef itk::ResampleImageFilter<TInputImageType, TOutputImageType >            RotationFilterType;
  typedef itk::AffineTransform< double, Dimension >                               TransformType;
  //typedef itk::NearestNeighborInterpolateImageFunction< TInputImageType, double > InterpolatorType; 
  //typedef itk::LinearInterpolateImageFunction< TInputImageType, double >          InterpolatorType; 
  //typedef itk::BSplineInterpolateImageFunction< TInputImageType, double >          InterpolatorType; // best trade-off between accuracy and computational cost 
  
  // best possible interpolator for grid data but slow (see ITK guide sec. 8.9.4 'Windowed Sinc Interpolation')
  /*typedef itk::ConstantBoundaryCondition< TInputImageType > BoundaryConditionType;
  const unsigned int WindowRadius = 5;
  typedef itk::Function::HammingWindowFunction<WindowRadius> WindowFunctionType;
  typedef itk::WindowedSincInterpolateImageFunction<
            TInputImageType, WindowRadius, WindowFunctionType,
            BoundaryConditionType, double  >                 InterpolatorType;*/

  RotationFilterType::Pointer rotationFilter = RotationFilterType::New();
  TransformType::Pointer transform = TransformType::New();

  // select interpolation type
  if (interpolationType == nih::NEAREST)
  {
    //std::cout << "  ... using nearest neighbor interpolation..." << std::endl;
    typedef itk::NearestNeighborInterpolateImageFunction< TInputImageType, double >  InterpolatorType;
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    rotationFilter->SetInterpolator( interpolator );
  }
  else if (interpolationType == nih::LINEAR)
  {
    //std::cout << "  ... using linear interpolation..." << std::endl;
    typedef itk::LinearInterpolateImageFunction< TInputImageType, double >  InterpolatorType;
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    rotationFilter->SetInterpolator( interpolator );
  }
  else if (interpolationType == nih::BSPLINE) // best trade-off between accuracy and computational cost 
  {
    //std::cout << "  ... using B-Spline interpolation..." << std::endl;
    typedef itk::BSplineInterpolateImageFunction< TInputImageType, double > InterpolatorType; 
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    rotationFilter->SetInterpolator( interpolator );
  }
  else
  {
    std::cerr << "Error!: No such interpolation type! " << interpolationType << std::endl;
    exit(-1);
  }

  rotationFilter->SetDefaultPixelValue( itk::NumericTraits< typename TOutputImageType::PixelType >::quiet_NaN() );

  rotationFilter->SetOutputSpacing(   image->GetSpacing() );
  rotationFilter->SetOutputDirection( image->GetDirection() );
  rotationFilter->SetSize(            image->GetLargestPossibleRegion().GetSize() );
  rotationFilter->SetOutputOrigin(    image->GetOrigin() );
        
  // translate region to physical origin (0,0,0)
  TOutputImageType::PointType center = nih::getImageCenter< TOutputImageType >( image );
  TransformType::OutputVectorType translation1;
  translation1[0] = -1*center[0]; 
  translation1[1] = -1*center[1];
  translation1[2] = -1*center[2];
  transform->Translate( translation1 );

  const double degreesToRadians = vcl_atan(1.0) / 45.0;
  double radians = angle_in_degrees * degreesToRadians;
      
  TransformType::OutputVectorType rotationAxis;
  rotationAxis[0] = axis[0];
  rotationAxis[1] = axis[1];
  rotationAxis[2] = axis[2];
  transform->Rotate3D( rotationAxis, radians, false ); // false, to indicate that the rotation should be applied 'after' the current transform content.

  // translate back to the previous location
  TransformType::OutputVectorType translation2;
  translation2[0] = -1*translation1[0];
  translation2[1] = -1*translation1[1];
  translation2[2] = -1*translation1[2];
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
//  nih::translateImage
//================================================================================================//
template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer translateImage
  (typename TInputImageType::Pointer image, double *translation)
{
  return translateImage<TInputImageType, TOutputImageType>(image, translation, nih::BSPLINE) ;
}

template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer translateImage(
  typename TInputImageType::Pointer image, double *translation, nih::InterpolationType interpolationType)
{
  typedef itk::ResampleImageFilter<TInputImageType, TOutputImageType >            RotationFilterType;
  typedef itk::AffineTransform< double, Dimension >                               TransformType;

  // best possible interpolator for grid data but slow (see ITK guide sec. 8.9.4 'Windowed Sinc Interpolation')
  /*typedef itk::ConstantBoundaryCondition< TInputImageType > BoundaryConditionType;
  const unsigned int WindowRadius = 5;
  typedef itk::Function::HammingWindowFunction<WindowRadius> WindowFunctionType;
  typedef itk::WindowedSincInterpolateImageFunction<
            TInputImageType, WindowRadius, WindowFunctionType,
            BoundaryConditionType, double  >                 InterpolatorType;*/

  RotationFilterType::Pointer translationFilter = RotationFilterType::New();
  TransformType::Pointer transform = TransformType::New();

  // select interpolation type
  if (interpolationType == nih::NEAREST)
  {
    //std::cout << "  ... using nearest neighbor interpolation..." << std::endl;
    typedef itk::NearestNeighborInterpolateImageFunction< TInputImageType, double >  InterpolatorType;
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    translationFilter->SetInterpolator( interpolator );
  }
  else if (interpolationType == nih::LINEAR)
  {
    //std::cout << "  ... using linear interpolation..." << std::endl;
    typedef itk::LinearInterpolateImageFunction< TInputImageType, double >  InterpolatorType;
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    translationFilter->SetInterpolator( interpolator );
  }
  else if (interpolationType == nih::BSPLINE) // best trade-off between accuracy and computational cost 
  {
    //std::cout << "  ... using B-Spline interpolation..." << std::endl;
    typedef itk::BSplineInterpolateImageFunction< TInputImageType, double > InterpolatorType; 
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    translationFilter->SetInterpolator( interpolator );
  }
  else
  {
    std::cerr << "Error!: No such interpolation type! " << interpolationType << std::endl;
    exit(-1);
  }

  translationFilter->SetDefaultPixelValue( itk::NumericTraits< typename TOutputImageType::PixelType >::quiet_NaN() );

  translationFilter->SetOutputSpacing(   image->GetSpacing() );
  translationFilter->SetOutputDirection( image->GetDirection() );
  translationFilter->SetSize(            image->GetLargestPossibleRegion().GetSize() );
  translationFilter->SetOutputOrigin(    image->GetOrigin() );
        
  // translate region to physical origin (0,0,0)
  TOutputImageType::PointType center = nih::getImageCenter< TOutputImageType >( image );
  TransformType::OutputVectorType translation1;
  translation1[0] = translation[0]; 
  translation1[1] = translation[1];
  translation1[2] = translation[2];
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
//  nih::warpImage
//================================================================================================//
template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer randomWarpImage2D(typename TInputImageType::Pointer movingImage, unsigned int Npoints=0)
{
  typedef   float VectorComponentType;
 
  typedef   itk::Vector< VectorComponentType, 2 >    VectorType;
  typedef   itk::Image< VectorType,  2 >   DeformationFieldType;
 
  TInputImageType::SizeType movingSize = movingImage->GetLargestPossibleRegion().GetSize();
 
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
  
  deformationFieldSource->GetKernelTransform()->SetStiffness( 1e-6 ); // doesn't seem to make a lot of difference
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
  xstep = movingSize[0]/sqrt((double)Npoints);
  ystep = movingSize[1]/sqrt((double)Npoints);
  for (double x=xstep; x<movingSize[0]-xstep; x=x+xstep)
  {
    for (double y=ystep; y<movingSize[1]-ystep; y=y+ystep)
    {
      sourcePoint[0] = x;
      sourcePoint[1] = y;
      targetPoint[0] = x + nih::getRandomVariateUniformDouble( 0.05*xstep );
      targetPoint[1] = y + nih::getRandomVariateUniformDouble( 0.05*ystep );
      sourceLandmarks->InsertElement( landmarkCount, sourcePoint );
      targetLandmarks->InsertElement( landmarkCount, targetPoint );
      landmarkCount++;
      //printf(" %d. adding source: [%g, %g]...\n", landmarkCount,  sourcePoint[0], sourcePoint[1]);
      //printf(" %d. adding target: [%g, %g]...\n", landmarkCount,  targetPoint[0], targetPoint[1]);
    }
  }
  
  unsigned int NLandmarks = landmarkCount;
  printf(" warping image (size [%d, %d]) with %d random vectors...", movingSize[0], movingSize[1], NLandmarks);

#if HARDCODEDPARAMS
  // save landmarks
  std::ofstream sourceLandmarkFile;
  std::ofstream targetLandmarkFile;
  sourceLandmarkFile.open ("D:/HolgerRoth/data/Pancreas/SLIClabels/1002/source_landmarks.txt");
  targetLandmarkFile.open ("D:/HolgerRoth/data/Pancreas/SLIClabels/1002/target_landmarks.txt");
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

#if HARDCODEDPARAMS
  // Write the deformation field
    typedef itk::ImageFileWriter<  DeformationFieldType  > WriterType;
    WriterType::Pointer writer =
      WriterType::New();
    writer->SetInput (  deformationFieldSource->GetOutput() );
    writer->SetFileName( "D:/HolgerRoth/data/Pancreas/SLIClabels/1002/deform.mha" );
    writer->Update();
#endif

  typedef itk::WarpImageFilter< TInputImageType, TOutputImageType, DeformationFieldType  >  WarpImageFilterType;
  itk::SmartPointer<WarpImageFilterType> warpImageFilter = WarpImageFilterType::New();
 
  //typedef itk::NearestNeighborInterpolateImageFunction< TInputImageType, double > InterpolatorType; 
  //typedef itk::LinearInterpolateImageFunction< TInputImageType, double >          InterpolatorType; 
  typedef itk::BSplineInterpolateImageFunction< TInputImageType, double >          InterpolatorType; // best trade-off between accuracy and computational cost 
 
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
//  nih::rescaleImage2D
//================================================================================================//
template <class TInputImageType, class TOutputImageType>
typename TOutputImageType::Pointer rescaleImage2D(typename TInputImageType::Pointer input, typename TOutputImageType::SizeType outputSize)
{
  const unsigned int Dimension = 2;

  TInputImageType::SizeType inputSize = input->GetLargestPossibleRegion().GetSize();
  printf(" warping image size [%d,%d] to size [%d,%d] ...\n",
    inputSize[0],inputSize[1],
    outputSize[0],outputSize[1]);
 
  // Resize
  TOutputImageType::SpacingType outputSpacing;
  outputSpacing[0] = input->GetSpacing()[0] * (static_cast<double>(inputSize[0]) / static_cast<double>(outputSize[0]));
  outputSpacing[1] = input->GetSpacing()[1] * (static_cast<double>(inputSize[1]) / static_cast<double>(outputSize[1]));
 
  typedef itk::AffineTransform< double, Dimension > TransformType;
  typedef itk::NearestNeighborInterpolateImageFunction<
    TInputImageType, double > InterpolatorType;
  //typedef itk::LinearInterpolateImageFunction<
  //    TInputImageType, double > InterpolatorType;

  typedef itk::ResampleImageFilter<TInputImageType, TOutputImageType> ResampleImageFilterType;
  itk::SmartPointer<ResampleImageFilterType> resample = ResampleImageFilterType::New();

  // Physical space coordinate of origin for X and Y
  resample->SetInput(input);
  resample->SetOutputOrigin( input->GetOrigin() );
  resample->SetOutputDirection( input->GetDirection() );
  resample->SetSize(outputSize);
  resample->SetOutputSpacing(outputSpacing);
  resample->SetTransform(TransformType::New());
  resample->SetInterpolator(InterpolatorType::New());
  resample->SetDefaultPixelValue( itk::NumericTraits< typename TOutputImageType::PixelType >::quiet_NaN() );
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
//  nih::getSegmentationOverlap2D
//================================================================================================//
template <class TInputImageType>
double getSegmentationOverlap2D(typename TInputImageType::Pointer segSource, typename TInputImageType::Pointer segTarget)
{
  typename TInputImageType::PixelType backgroundValue = 0;
  typename TInputImageType::PixelType labelValue      = 1;

  TInputImageType::SizeType sizeSource = segSource->GetLargestPossibleRegion().GetSize();
  TInputImageType::SizeType sizeTarget = segTarget->GetLargestPossibleRegion().GetSize();
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
  std::cout << "labelValue:      " << labelValue << std::endl;
  std::cout << "Target:          " << overlapFilter->GetTargetOverlap( labelValue ) << std::endl;
  std::cout << "Union (jaccard): " << overlapFilter->GetUnionOverlap( labelValue ) << std::endl;
  std::cout << "Mean (dice):     " << overlapFilter->GetMeanOverlap( labelValue ) << std::endl;
  std::cout << "Volume sim.:     " << overlapFilter->GetVolumeSimilarity( labelValue ) << std::endl;
  std::cout << "False negative:  " << overlapFilter->GetFalseNegativeError( labelValue ) << std::endl;
  std::cout << "False positive:  " << overlapFilter->GetFalsePositiveError( labelValue ) << std::endl;
  std::cout << std::endl;

  return overlapFilter->GetMeanOverlap( labelValue );
} // nih::getSegmentationOverlap2D

} /* end namespace nih */

#endif
