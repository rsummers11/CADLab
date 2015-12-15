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
#include <itkNumericTraits.h>
#include <itkBSplineInterpolateImageFunction.h>

namespace itk {
template <class TInputImageType, class TOutputImageType>
class IsotropicResampleImageFilter :
    public ImageToImageFilter<TInputImageType, TOutputImageType>
{
public:
  typedef IsotropicResampleImageFilter               Self;
  typedef ImageToImageFilter<TInputImageType,TInputImageType> Superclass;
  typedef SmartPointer<Self>                        Pointer;
  typedef SmartPointer<const Self>                  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods).   */
  itkTypeMacro(IsotropicResampleImageFilter, ImageToImageFilter);
  
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
  itkGetMacro( IsoSpacing, double);
  itkSetMacro( IsoSpacing, double);
  itkGetMacro( OutputWinMax, OutputImagePixelType);
  itkSetMacro( OutputWinMax, OutputImagePixelType);

protected:
  IsotropicResampleImageFilter();

  typedef ResampleImageFilter< InputImageType, OutputImageType >            ResampleFilterType;

  void GenerateData();
private:
  IsotropicResampleImageFilter(Self&);   // intentionally not implemented
  void operator=(const Self&);          // intentionally not implemented

  typename ResampleFilterType::Pointer      m_ResampleFilter;

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
IsotropicResampleImageFilter<TInputImageType,TOutputImageType>
::IsotropicResampleImageFilter()
{
  m_IsoSpacing = 1.0;

  m_ResampleFilter  = ResampleFilterType::New();
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
IsotropicResampleImageFilter<TInputImageType,TOutputImageType>::
GenerateData()
{
  m_ResampleFilter->SetInput( this->GetInput() ); 

  InputImageConstPointer inputImage = this->GetInput();
  TInputImageType::SpacingType inputSpacing = inputImage->GetSpacing();
  printf("	input spacing is [%g, %g, %g]\n", inputSpacing[0], inputSpacing[1], inputSpacing[2]);

  // compute size of isometric image
  typedef itk::IdentityTransform< double, TInputImageType::ImageDimension >  TransformType;

  TransformType::Pointer transform = TransformType::New();
  transform->SetIdentity();
  m_ResampleFilter->SetTransform( transform );
  //typedef itk::LinearInterpolateImageFunction< RealImageType, double >  InterpolatorType;
  typedef itk::BSplineInterpolateImageFunction< InputImageType, double > InterpolatorType; // best trade-off between accuracy and computational cost 
  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  m_ResampleFilter->SetInterpolator( interpolator );
  m_ResampleFilter->SetDefaultPixelValue( m_OutputWinMax ); // highlight regions without source

  printf("	target isometric spacing is [%g, %g, %g]\n", m_IsoSpacing, m_IsoSpacing, m_IsoSpacing);
  OutputImageType::SpacingType spacing;
  spacing[0] = m_IsoSpacing;
  spacing[1] = m_IsoSpacing;
  spacing[2] = m_IsoSpacing;

  m_ResampleFilter->SetOutputSpacing( spacing );
  m_ResampleFilter->SetOutputOrigin( inputImage->GetOrigin() );
  m_ResampleFilter->SetOutputDirection( inputImage->GetDirection() );
  InputImageType::SizeType   inputSize =
                    inputImage->GetLargestPossibleRegion().GetSize();

  typedef OutputImageType::SizeType::SizeValueType SizeValueType;
  printf("	input size is [%d, %d, %d]\n", inputSize[0], inputSize[1], inputSize[2]);

  const double dx = inputSize[0] * inputSpacing[0] / m_IsoSpacing;
  const double dy = inputSize[1] * inputSpacing[1] / m_IsoSpacing;

  const double dz = (inputSize[2] - 1 ) * inputSpacing[2] / m_IsoSpacing;
  OutputImageType::SizeType   size;
  
  size[0] = static_cast<SizeValueType>( dx );
  size[1] = static_cast<SizeValueType>( dy );
  size[2] = static_cast<SizeValueType>( dz );
  printf("	output size is [%d, %d, %d]\n", size[0], size[1], size[2]);

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
IsotropicResampleImageFilter<TInputImageType, TOutputImageType>::
PrintSelf( std::ostream& os, Indent indent ) const
{
  //Superclass::PrintSelf(os,indent);
  os
    << indent << "IsotropicResampleImageFilter: iso spacing is " << m_IsoSpacing
    << std::endl;
}
} /* end namespace itk */


