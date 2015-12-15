// Simple program for ITK image read/write in C++
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImage.h"
#include "itkRGBPixel.h"
int main (int argc, char *argv[])
{
  const unsigned int Dimensions = 3;
  typedef short PixelType;
  typedef itk::Image< PixelType, Dimensions >       ImageType;
  typedef itk::ImageFileReader< ImageType >  ReaderType;
  typedef itk::ImageFileWriter< ImageType >  WriterType;

   if (argc < 2)
   {
               fprintf(stdout,"Usage: %s inputImagename outputImagename\n",argv[0]);
               return 1;
   }

  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();
  reader->SetFileName( argv[1] );
  writer->SetFileName( argv[2] );
  writer->SetInput( reader->GetOutput() );
  try 
  { 
        writer->Update(); 
  } 
  catch( itk::ExceptionObject & err ) 
  { 
               std::cerr << err << std::endl; 
               return 1;
  } 
   return 0;
}
