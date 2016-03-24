/*=========================================================================
 *
 *  Copyright National Institutes of Health
 *
 *  Unless required by applicable law or agreed to in writing, this software
 *  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
 *  ANY KIND, either express or implied.
 *
 *=========================================================================*/

#include <itkImageToImageFilter.h>
#include <itkResampleImageFilter.h>
#include <itkIdentityTransform.h>
#include <itkIntensityWindowingImageFilter.h>
#include <itkNumericTraits.h>

#include "nihHelperFunctions.h"

namespace itk {
template <class TInputImageType, class TOutputImageType>
class IsotropicWindowingResampleImageFilter :
    public ImageToImageFilter<TInputImageType, TOutputImageType>
{
public:
  typedef IsotropicWindowingResampleImageFilter               Self;
  typedef ImageToImageFilter<TInputImageType,TInputImageType> Superclass;
  typedef SmartPointer<Self>                        Pointer;
  typedef SmartPointer<const Self>                  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods).   */
  itkTypeMacro(IsotropicWindowingResampleImageFilter, ImageToImageFilter);

  /** Image dimension. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImageType::ImageDimension);

  void PrintSelf( std::ostream& os, Indent indent ) const;

  typedef typename TInputImageType::PixelType PixelType;

  /** Type of the input image */
  typedef TInputImageType                       InputImageType;
  typedef typename InputImageType::Pointer      InputImagePointer;
  typedef typename InputImageType::ConstPointer InputImageConstPointer;
  typedef typename InputImageType::RegionType   InputImageRegionType;
  typedef typename InputImageType::PixelType    InputImagePixelType;
  typedef typename InputImageType::SpacingType  InputImageSpacingType;
  typedef typename InputImageType::PointType    InputImagePointType;

  typedef typename NumericTraits<InputImagePixelType>::RealType    RealType;
  /** Define the image type for internal computations
      RealType is usually 'double' in NumericTraits.
      Here we prefer float in order to save memory.  */
  typedef float InternalRealType;
  typedef Image< InternalRealType, TInputImageType::ImageDimension > RealImageType;
  typedef typename RealImageType::Pointer RealImagePointer;

  /** Type of the output image */
  typedef TOutputImageType                      OutputImageType;
  typedef typename OutputImageType::Pointer     OutputImagePointer;
  typedef typename OutputImageType::RegionType  OutputImageRegionType;
  typedef typename OutputImageType::PixelType   OutputImagePixelType;
  typedef typename OutputImageType::IndexType   OutputImageIndexType;
  typedef typename OutputImageType::PointType   OutputImagePointType;

  /** Set and Get macros */
  itkGetMacro( InputWinMin, InputImagePixelType);
  itkSetMacro( InputWinMin, InputImagePixelType);
  itkGetMacro( InputWinMax, InputImagePixelType);
  itkSetMacro( InputWinMax, InputImagePixelType);
  itkGetMacro( OutputWinMin, OutputImagePixelType);
  itkSetMacro( OutputWinMin, OutputImagePixelType);
  itkGetMacro( OutputWinMax, OutputImagePixelType);
  itkSetMacro( OutputWinMax, OutputImagePixelType);
  itkGetMacro( IsoSpacing, double);
  itkSetMacro( IsoSpacing, double);
  itkSetMacro( InterpolationType, nih::InterpolationType);

protected:
  IsotropicWindowingResampleImageFilter(); // Default: use B-Spline interpolation

  typedef IntensityWindowingImageFilter< InputImageType, RealImageType >   IntensityFilterType;
  typedef ResampleImageFilter< RealImageType, OutputImageType >            ResampleFilterType;

  void GenerateData();
private:
  nih::InterpolationType m_InterpolationType;

  IsotropicWindowingResampleImageFilter(Self&);   // intentionally not implemented
  void operator=(const Self&);          // intentionally not implemented

  typename IntensityFilterType::Pointer     m_IntensityFilter;
  typename ResampleFilterType::Pointer      m_ResampleFilter;

  InputImagePixelType   m_InputWinMin;
  InputImagePixelType   m_InputWinMax;
  OutputImagePixelType   m_OutputWinMin;
  OutputImagePixelType   m_OutputWinMax;

  double m_IsoSpacing;
};
} /* namespace itk */

/* -----------------------------------------------------------------------
   Constructor()
   ----------------------------------------------------------------------- */
//  The constructor sets up the pipeline, which involves creating the
//  stages, connecting them together, and setting default parameters.
namespace itk
{
//  Software Guide : BeginCodeSnippet
template <class TInputImageType, class TOutputImageType>
IsotropicWindowingResampleImageFilter<TInputImageType,TOutputImageType>
::IsotropicWindowingResampleImageFilter()
{
  m_InputWinMin = -100; // HU
  m_InputWinMax =  200; // HU

  m_OutputWinMin = 0.0;
  m_OutputWinMax = 1.0;

  m_IsoSpacing = 1.0;

  m_IntensityFilter = IntensityFilterType::New();
  m_ResampleFilter  = ResampleFilterType::New();

  m_InterpolationType = nih::BSPLINE;

  // pipeline after m_IntensityFilter->SetInput( this->GetInput() ), called in GenerateData()
  m_ResampleFilter->SetInput( m_IntensityFilter->GetOutput() ); // smoothing not neccessary using B-spline interpolation
}

/* -----------------------------------------------------------------------
   GenerateData()
   ----------------------------------------------------------------------- */
//  This is where the composite magic happens.  First,
//  we connect the first component filter to the inputs of the composite
//  filter (the actual input, supplied by the upstream stage).  Then we
//  graft the output of the last stage onto the output of the composite,
//  which ensures the filter regions are updated.  We force the composite
//  pipeline to be processed by calling \code{Update()} on the final stage,
//  then graft the output back onto the output of the enclosing filter, so
//  it has the result available to the downstream filter.
template <class TInputImageType, class TOutputImageType>
void
IsotropicWindowingResampleImageFilter<TInputImageType,TOutputImageType>::
GenerateData()
{
  std::cout << "	scaling intensity [ " << m_InputWinMin << ", " << m_InputWinMax << "] to window " <<
    "[" << m_OutputWinMin << ", " << m_OutputWinMax << "]" << std::endl;
  m_IntensityFilter->SetWindowMinimum( m_InputWinMin );
  m_IntensityFilter->SetWindowMaximum( m_InputWinMax );
  m_IntensityFilter->SetOutputMinimum( m_OutputWinMin );
  m_IntensityFilter->SetOutputMaximum( m_OutputWinMax );
  m_IntensityFilter->SetInput( this->GetInput() );

  InputImageConstPointer inputImage = this->GetInput();
  typename TInputImageType::SpacingType inputSpacing = inputImage->GetSpacing();
  printf("	input spacing is [%g, %g, %g]\n", inputSpacing[0], inputSpacing[1], inputSpacing[2]);

  // compute size of isometric image
  typedef itk::IdentityTransform< double, TInputImageType::ImageDimension >  TransformType;

  typename TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();
  m_ResampleFilter->SetTransform( transform );

  // select interpolation type
  if (m_InterpolationType == nih::NEAREST)
  {
    std::cout << "  ... using nearest neighbor interpolation..." << std::endl;
    typedef itk::NearestNeighborInterpolateImageFunction< RealImageType, double >  InterpolatorType;
    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
    m_ResampleFilter->SetInterpolator( interpolator );
  }
  else if (m_InterpolationType == nih::LINEAR)
  {
    std::cout << "  ... using linear interpolation..." << std::endl;
    typedef itk::LinearInterpolateImageFunction< RealImageType, double >  InterpolatorType;
    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
    m_ResampleFilter->SetInterpolator( interpolator );
  }
  else if (m_InterpolationType == nih::BSPLINE) // best trade-off between accuracy and computational cost
  {
    std::cout << "  ... using B-Spline interpolation..." << std::endl;
    typedef itk::BSplineInterpolateImageFunction< RealImageType, double > InterpolatorType;
    typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
    m_ResampleFilter->SetInterpolator( interpolator );
  }
  else
  {
    std::cerr << "Error!: No such interpolation type! " << m_InterpolationType << std::endl;
    exit(-1);
  }

  m_ResampleFilter->SetDefaultPixelValue( m_OutputWinMax ); // highlight regions without source

  printf("	target isometric spacing is [%g, %g, %g]\n", m_IsoSpacing, m_IsoSpacing, m_IsoSpacing);
  typename OutputImageType::SpacingType spacing;
  spacing[0] = m_IsoSpacing;
  spacing[1] = m_IsoSpacing;
  spacing[2] = m_IsoSpacing;

  m_ResampleFilter->SetOutputSpacing( spacing );
  m_ResampleFilter->SetOutputOrigin( inputImage->GetOrigin() );
  m_ResampleFilter->SetOutputDirection( inputImage->GetDirection() );
  typename InputImageType::SizeType   inputSize =
                    inputImage->GetLargestPossibleRegion().GetSize();

  typedef typename OutputImageType::SizeType::SizeValueType SizeValueType;
  printf("	input size is [%d, %d, %d]\n", (int)inputSize[0], (int)inputSize[1], (int)inputSize[2]);

  const double dx = inputSize[0] * inputSpacing[0] / m_IsoSpacing;
  const double dy = inputSize[1] * inputSpacing[1] / m_IsoSpacing;

  const double dz = (inputSize[2] - 1 ) * inputSpacing[2] / m_IsoSpacing;
  typename OutputImageType::SizeType   size;

  size[0] = static_cast<SizeValueType>( dx );
  size[1] = static_cast<SizeValueType>( dy );
  size[2] = static_cast<SizeValueType>( dz );
  printf("	output size is [%d, %d, %d]\n", (int)size[0], (int)size[1], (int)size[2]);

  m_ResampleFilter->SetSize( size );

  m_ResampleFilter->GraftOutput( this->GetOutput() );
  try
  {
    /** Running a pipeline where the LargestPossibleRegion in the pipeline
    * is expected to change on consecutive runs. The pipeline does not
    * detect this condition, and it will throw an exception. In this case,
    * an UpdateLargestPossibleRegion() call is required instead of Update(). */
    m_ResampleFilter->UpdateLargestPossibleRegion();
  }
  catch( itk::ExceptionObject & excep )
  {
    std::cerr << "Exception caught !" << std::endl;
    std::cerr << excep << std::endl;
  }
  this->GraftOutput( m_ResampleFilter->GetOutput() );
}

/* -----------------------------------------------------------------------
   PrintSelf()
   ----------------------------------------------------------------------- */
//  Finally we define the \code{PrintSelf} method, which (by convention)
//  prints the filter parameters.  Note how it invokes the superclass to
//  print itself first, and also how the indentation prefixes each line.
template <class TInputImageType, class TOutputImageType>
void
IsotropicWindowingResampleImageFilter<TInputImageType, TOutputImageType>::
PrintSelf( std::ostream& os, Indent indent ) const
{
  //Superclass::PrintSelf(os,indent);
  os
    << indent << "IsotropicWindowingResampleImageFilter: iso spacing is " << m_IsoSpacing
    << std::endl;
}
} /* end namespace itk */


