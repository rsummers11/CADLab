// converts a file of voxel indices to physical image space coordinates

#include <iostream>
#include <fstream>

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImage.h>
//#include <itkRGBPixel.h>

int main (int argc, char *argv[])
{
  const unsigned int dimension = 3;
  typedef unsigned char PixelType; // doesn't matter in this case, we are just interested in the geometry file of the image header (maybe we could just read the header?)
  typedef itk::Image< PixelType, dimension > ImageType;
  typedef itk::ImageFileReader< ImageType >  ReaderType;
  typedef itk::ImageFileWriter< ImageType >  WriterType;

  if (argc != 4)
  {
              fprintf(stdout,"Usage: %s inputImagename indicesFilename outPhysicalPointsFileprefix (generates two output files prefix.mps and prefix.txt)\n", argv[0]);
              return EXIT_FAILURE;
  }
  const char* inImageFilename = argv[1];
  const char* inIndicesFilename = argv[2];
  std::string outPhysicalPointsFileprefix = argv[3];
  std::string outMPS = outPhysicalPointsFileprefix + ".mps";
  std::string outTXT = outPhysicalPointsFileprefix + ".txt";
  // read image
  itk::SmartPointer< ReaderType > reader = ReaderType::New();
  reader->SetFileName( inImageFilename );
  try 
  { 
    printf(" reading file from %s ...\n", inImageFilename);
	  reader->Update(); 
  } 
  catch( itk::ExceptionObject & err ) 
  { 
    std::cerr << err << std::endl; 
    return 1;
  } 
  itk::SmartPointer< ImageType > image = reader->GetOutput();

  // read indices text file
  std::ifstream iFile( inIndicesFilename );
  if (!iFile)
  {
    std::cerr << " Error: could not find " << inIndicesFilename << "!" << std::endl;
  }
  std::string line;

  std::ofstream oMPSFile( outMPS.c_str() );
  std::ofstream oTXTFile( outTXT.c_str() );
 
  ImageType::IndexType index;
  ImageType::PointType physicalPoint;

  /* While there is still a line. */
  unsigned int idx_count = 0;
  int idx0, idx1, idx2;

  // MITK point set
  int mitkSpecification = 0;
  int mitkTimeSeries = 0;
  oMPSFile << "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>" << std::endl;
  oMPSFile << "<point_set_file>" << std::endl;
  oMPSFile << "  <file_version>0.1</file_version>" << std::endl;
  oMPSFile << "  <point_set>" << std::endl;
  oMPSFile << "    <time_series>" << std::endl;
  oMPSFile << "      <time_series_id>" << mitkTimeSeries << "</time_series_id>" << std::endl;
  while (iFile >> idx0 >> idx1 >> idx2)
  {
      /* Printing goes here. */
      std::cout << line << std::endl;

      index[0] = idx0;
      index[1] = idx1;
      index[2] = idx2;

      image->TransformIndexToPhysicalPoint(index,physicalPoint);
      printf("  %d: index [%d, %d, %d] maps to [%g, %g, %g] mm.\n", idx_count+1, 
                          index[0], index[1], index[2],
                                 physicalPoint[0], physicalPoint[1], physicalPoint[2]);

      oMPSFile << "      <point>" << std::endl;
      oMPSFile << "        <id>" << idx_count << "</id>" << std::endl;
      oMPSFile << "        <specification>" << mitkSpecification << "</specification>" << std::endl;
      oMPSFile << "        <x>" << physicalPoint[0] << "</x>" << std::endl;
      oMPSFile << "        <y>" << physicalPoint[1] << "</y>" << std::endl;
      oMPSFile << "        <z>" << physicalPoint[2] << "</z>" << std::endl;
      oMPSFile << "      </point>" << std::endl;


      // NOTE: space delimiter is important to be later read with std::ifstream::operator ('\t' does not work for some reason...)
      oTXTFile << physicalPoint[0] << " " << physicalPoint[1] << " " << physicalPoint[2] << std::endl;

      idx_count++;
  }
  oMPSFile << "    </time_series>" << std::endl; 
  oMPSFile << "  </point_set>" << std::endl;
  oMPSFile << "</point_set_file>" << std::endl;

  iFile.close();
  oMPSFile.close();
  oTXTFile.close();

  std::cout << "  wrote physical coordinates to " << outPhysicalPointsFileprefix << " (*.mps and *.txt)" << std::endl;
  return 0;
}

