/*-
 * Nathan Lay
 * Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
 * National Institutes of Health
 * March 2017
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <string>
#include <vector>

#include "itkImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

void Trim(std::string &strString);
std::vector<std::string> SplitString(const std::string &strValue, const std::string &strDelim);

bool FileExists(const std::string &strPath);
bool IsFolder(const std::string &strPath);
bool RmDir(const std::string &strPath);
bool MkDir(const std::string &strPath);
bool Unlink(const std::string &strPath);
bool Copy(const std::string &strFrom, const std::string &strTo, bool bReplace = false);
bool Rename(const std::string &strFrom, const std::string &strTo, bool bReplace = false);
void USleep(unsigned int uiMicroSeconds);

std::string BaseName(std::string strPath);
std::string DirName(std::string strPath);

void SanitizeFileName(std::string &strFileName); // Does NOT operate on paths
void FindFiles(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFiles, bool bRecursive = false);
void FindFolders(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFolders, bool bRecursive = false);
void FindDicomFolders(const char *p_cDir, const char *p_cPattern, std::vector<std::string> &vFolders, bool bRecursive = false);

// Use LoadImg since Windows #defines LoadImage ... lame
template<typename PixelType, unsigned int Dimension>
typename itk::Image<PixelType, Dimension>::Pointer LoadImg(const std::string &strPath);

template<typename PixelType, unsigned int Dimension>
bool SaveImg(typename itk::Image<PixelType, Dimension>::Pointer p_clImage, const std::string &strPath, bool bCompress = true);

template<typename PixelType, unsigned int Dimension>
typename itk::Image<PixelType, Dimension>::Pointer LoadDicomImage(const std::string &strPath, const std::string &strSeriesUID = std::string());

template<typename PixelType, unsigned int Dimension>
typename itk::Image<PixelType, Dimension>::Pointer LoadImg(const std::string &strPath) {
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::ImageFileReader<ImageType> ReaderType;

  typename ReaderType::Pointer p_clReader = ReaderType::New();

  p_clReader->SetFileName(strPath);

  try {
    p_clReader->Update();
  }
  catch (itk::ExceptionObject &e) {
    std::cerr << "Error: " << e << std::endl;
    return typename ImageType::Pointer();
  }

  return p_clReader->GetOutput();
}

template<typename PixelType, unsigned int Dimension>
bool SaveImg(typename itk::Image<PixelType, Dimension>::Pointer p_clImage, const std::string &strPath, bool bCompress) {
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::ImageFileWriter<ImageType> WriterType;

  typename WriterType::Pointer p_clWriter = WriterType::New();

  p_clWriter->SetFileName(strPath);
  p_clWriter->SetUseCompression(bCompress);
  p_clWriter->SetInput(p_clImage);

  try {
    p_clWriter->Update();
  }
  catch (itk::ExceptionObject &e) {
    std::cerr << "Error: " << e << std::endl;
    return false;
  }

  return true;
}

template<typename PixelType, unsigned int Dimension>
typename itk::Image<PixelType, Dimension>::Pointer LoadDicomImage(const std::string &strPath, const std::string &strSeriesUID) {
  typedef itk::Image<PixelType, Dimension> ImageType;
  typedef itk::GDCMImageIO ImageIOType;
  typedef itk::GDCMSeriesFileNames FileNameGeneratorType;

  if (!FileExists(strPath)) // File or folder must exist
    return typename ImageType::Pointer();

  ImageIOType::Pointer p_clImageIO = ImageIOType::New();

  p_clImageIO->LoadPrivateTagsOn();
  p_clImageIO->KeepOriginalUIDOn();

  if (Dimension == 2) {
    // Read a 2D image
    typedef itk::ImageFileReader<ImageType> ReaderType;

    if (IsFolder(strPath)) // Must be a file
      return typename ImageType::Pointer();
    
    if (!p_clImageIO->CanReadFile(strPath.c_str()))
      return typename ImageType::Pointer();

    typename ReaderType::Pointer p_clReader = ReaderType::New();

    p_clReader->SetImageIO(p_clImageIO);
    p_clReader->SetFileName(strPath);

    try {
      p_clReader->Update();
    }
    catch (itk::ExceptionObject &e) {
      std::cerr << "Error: " << e << std::endl;
      return typename ImageType::Pointer();
    }

    return p_clReader->GetOutput();
  }

  // Passed a file, read the series UID (ignore the one provided, if any)
  if (!IsFolder(strPath)) {

    if (!p_clImageIO->CanReadFile(strPath.c_str()))
      return typename ImageType::Pointer();

    p_clImageIO->SetFileName(strPath.c_str());

    try {
      p_clImageIO->ReadImageInformation();
    }
    catch (itk::ExceptionObject &e) {
      std::cerr << "Error: " << e << std::endl;
      return typename ImageType::Pointer();
    }

    const itk::MetaDataDictionary &clDicomTags = p_clImageIO->GetMetaDataDictionary();

    std::string strTmpSeriesUID;
    if (!itk::ExposeMetaData(clDicomTags, "0020|000e", strTmpSeriesUID))
      return typename ImageType::Pointer();

    Trim(strTmpSeriesUID);

    return LoadDicomImage<PixelType, Dimension>(DirName(strPath), strTmpSeriesUID); // Call this function again
  }

  FileNameGeneratorType::Pointer p_clFileNameGenerator = FileNameGeneratorType::New();

  // Use the ACTUAL series UID ... not some custom ITK concatenations of lots of junk.
  p_clFileNameGenerator->SetUseSeriesDetails(false); 
  p_clFileNameGenerator->SetDirectory(strPath);

  if (strSeriesUID.empty()) {
    // Passed a folder but no series UID ... pick the first series UID
    const FileNameGeneratorType::SeriesUIDContainerType &vSeriesUIDs = p_clFileNameGenerator->GetSeriesUIDs();

    if (vSeriesUIDs.empty())
      return typename ImageType::Pointer();

    // Use first series UID
    return LoadDicomImage<PixelType, Dimension>(strPath, vSeriesUIDs[0]);
  }

  const FileNameGeneratorType::FileNamesContainerType &vDicomFiles = p_clFileNameGenerator->GetFileNames(strSeriesUID);

  if (vDicomFiles.empty())
    return typename ImageType::Pointer();

  // Read 3D or higher (but 4D probably doesn't work correctly)
  typedef itk::ImageSeriesReader<ImageType> ReaderType;

  typename ReaderType::Pointer p_clReader = ReaderType::New();

  p_clReader->SetImageIO(p_clImageIO);
  p_clReader->SetFileNames(vDicomFiles);

  try {
    p_clReader->Update();
  }
  catch (itk::ExceptionObject &e) {
    std::cerr << "Error: " << e << std::endl;
    return typename ImageType::Pointer();
  }

  return p_clReader->GetOutput();
}

#endif // !COMMON_H
