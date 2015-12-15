// converts a file of voxel indices to physical image space coordinates

#include <iostream>
#include <fstream>

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImage.h>
//#include <itkRGBPixel.h>

#include <nihHelperFunctions.h>

int main (int argc, char *argv[])
{
  const unsigned int dimension = 3;
  typedef unsigned char PixelType; // doesn't matter in this case, we are just interested in the geometry file of the image header (maybe we could just read the header?)
  typedef itk::Image< PixelType, dimension > ImageType;
  typedef itk::ImageFileReader< ImageType >  ReaderType;
  typedef itk::ImageFileWriter< ImageType >  WriterType;

  if ( (argc != 3) && (argc != 4) )
  {
              fprintf(stdout,"Usage: %s inRoiListFilename outPhysicalPointsFileprefix (generates two output files prefix.mps (if SAVE_MPS=1[default]) and prefix.txt) SAVE_MPS[0,1]\n", argv[0]);
              return EXIT_FAILURE;
  }
  const char* inRoiListFilename = argv[1];
  std::string outPhysicalPointsFileprefix = argv[2];
  bool SAVE_MPS = true;
  if (argc==4)
  {
    SAVE_MPS = atoi( argv[3] );
  }

  std::string outMPS = outPhysicalPointsFileprefix + ".mps";
  std::string outTXT = outPhysicalPointsFileprefix + ".txt";
  
  itk::SmartPointer< ReaderType > reader = ReaderType::New();
  std::string roiFilename;

  // read indices text file
  std::ifstream iFile( inRoiListFilename );
  if (!iFile)
  {
    std::cerr << " Error: could not find " << inRoiListFilename << "!" << std::endl;
  }
  std::string line;

  ImageType::PointType physicalPoint;

  /* While there is still a line. */
  unsigned int idx_count = 0;

  // MITK point set
  std::ofstream oMPSFile;
  int mitkSpecification = 0;
  int mitkTimeSeries = 0;
  if (SAVE_MPS)
  {
    oMPSFile.open( outMPS.c_str() );
    oMPSFile << "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>" << std::endl;
    oMPSFile << "<point_set_file>" << std::endl;
    oMPSFile << "  <file_version>0.1</file_version>" << std::endl;
    oMPSFile << "  <point_set>" << std::endl;
    oMPSFile << "    <time_series>" << std::endl;
    oMPSFile << "      <time_series_id>" << mitkTimeSeries << "</time_series_id>" << std::endl;
  }
  std::ofstream oTXTFile( outTXT.c_str() );

  while (iFile >> roiFilename)
  {
      // read current ROI
      reader->SetFileName( roiFilename );
      try 
      { 
        //printf(" reading file from %s ...\n", roiFilename.c_str());
	      reader->Update(); 
      } 
      catch( itk::ExceptionObject & err ) 
      { 
        std::cerr << err << std::endl; 
        return 1;
      } 
      itk::SmartPointer< ImageType > roiImage = reader->GetOutput();

      ImageType::PointType physicalPoint = nih::getImageCenter< ImageType >( roiImage );
      
      //printf("  %d. ROI: center at [%g, %g, %g] mm.\n", idx_count+1, 
      //                           physicalPoint[0], physicalPoint[1], physicalPoint[2]);

      if (SAVE_MPS)
      {
        oMPSFile << "      <point>" << std::endl;
        oMPSFile << "        <id>" << idx_count << "</id>" << std::endl;
        oMPSFile << "        <specification>" << mitkSpecification << "</specification>" << std::endl;
        oMPSFile << "        <x>" << physicalPoint[0] << "</x>" << std::endl;
        oMPSFile << "        <y>" << physicalPoint[1] << "</y>" << std::endl;
        oMPSFile << "        <z>" << physicalPoint[2] << "</z>" << std::endl;
        oMPSFile << "      </point>" << std::endl;
      }
      // NOTE: space delimiter is important to be later read with std::ifstream::operator ('\t' does not work for some reason...)
      oTXTFile << physicalPoint[0] << " " << physicalPoint[1] << " " << physicalPoint[2] << std::endl;

      idx_count++;
  }
  if (SAVE_MPS)
  {
    oMPSFile << "    </time_series>" << std::endl; 
    oMPSFile << "  </point_set>" << std::endl;
    oMPSFile << "</point_set_file>" << std::endl;
    oMPSFile.close();
  }

  iFile.close();
  oTXTFile.close();

  std::cout << "  wrote " << idx_count << " physical coordinates to " << outPhysicalPointsFileprefix << " (*.mps and *.txt)" << std::endl;
  return 0;
}

