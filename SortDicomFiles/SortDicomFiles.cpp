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

#include <cstdlib>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <limits>
#include <string>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include "Common.h"
#include "bsdgetopt.h"
#include "strcasestr.h"

// ITK stuff
#include "itkGDCMImageIO.h"
#include "itkMetaDataDictionary.h"
#include "itkMetaDataObject.h"
#include "itkRGBPixel.h"
#include "itkRGBAPixel.h"

#ifdef USE_MD5
#include "itkTestingHashImageFilter.h"
#endif // USE_MD5

#include "gdcmBase64.h"
#include "gdcmCSAHeader.h"
#include "gdcmCSAElement.h"

std::map<std::string, std::string> g_mNameToTagMap;
std::unordered_map<std::string, itk::ImageBase<3>::Pointer> g_mSeriesUIDToImageBase;

#ifdef USE_MD5
std::unordered_map<std::string, std::string> g_mSeriesUIDToMD5;
#endif // USE_MD5

void Usage(const char *p_cArg0) {
  std::cerr << "Usage: " << p_cArg0 << " [-cehlr] folder|filePattern [folder2|filePattern2 ...] destinationPattern" << std::endl;
  std::cerr << "\nOptions:" << std::endl;
  std::cerr << "-c -- Copy instead of move." << std::endl;
  std::cerr << "-e -- Try to remove empty folders." << std::endl;
  std::cerr << "-l -- List supported patterns." << std::endl;
  std::cerr << "-h -- This help message." << std::endl;
  std::cerr << "-r -- Search for DICOMs recursively." << std::endl;
  exit(1);
}

void PrintSupportedPatterns() {
  std::cout << "Supported patterns:" << std::endl;

  for (auto itr = g_mNameToTagMap.begin(); itr != g_mNameToTagMap.end(); ++itr)
    std::cout << '<' << itr->first << "> (" << itr->second << ')' << std::endl;

  std::cout << "<diffusion b-value> (0018|9087 or vendor-specific)" << std::endl;
  std::cout << "<z spacing> (z voxel spacing)" << std::endl;
  std::cout << "<z coordinate> (z voxel coordinate)" << std::endl;
  std::cout << "<z origin> (z patient origin)" << std::endl;
  std::cout << "<x dim> (x voxel dimension)" << std::endl;
  std::cout << "<y dim> (y voxel dimension)" << std::endl;
  std::cout << "<z dim> (z voxel dimension)" << std::endl;

#ifdef USE_MD5
  std::cout << "<slice md5> (MD5 hash of pixel data)" << std::endl;
  std::cout << "<volume md5> (MD5 hash of voxel data)" << std::endl;
#endif // USE_MD5

  std::cout << "<file> (file's basename)" << std::endl;
  std::cout << "<folder> (file's dirname)" << std::endl;
}

bool IsHexDigit(char c);
bool ParseITKTag(const std::string &strKey, uint16_t &ui16Group, uint16_t &ui16Element);

template<typename ValueType>
bool ExposeCSAMetaData(gdcm::CSAHeader &clHeader, const char *p_cKey, ValueType &value);

template<>
bool ExposeCSAMetaData<std::string>(gdcm::CSAHeader &clHeader, const char *p_cKey, std::string &strValue);
 
bool GetCSAHeaderFromElement(const itk::MetaDataDictionary &clDicomTags, const std::string &strKey, gdcm::CSAHeader &clCSAHeader);

void EnsureRmDir(const std::string &strPath);
void MakeFoldersForFile(const std::string &strPath);
std::string MakeFailedValue(const std::string &strTag);
std::string MakeValue(const std::string &strTag, const itk::MetaDataDictionary &clDicomTags, const std::string &strPath = "");
bool ProcessString(std::string &strPattern, const itk::MetaDataDictionary &clDicomTags, const std::string &strPath = "");
bool MoveDicomFile(const std::string &strFileName, const std::string &strPattern, bool bCopy);

// Special handling
itk::ImageBase<3>::Pointer LoadImageBase(const itk::MetaDataDictionary &clDicomTags, const std::string &strPath);
std::string ComputeZCoordinate(const itk::MetaDataDictionary &clDicomTags, const std::string &strPath);
std::string ComputeZOrigin(const itk::MetaDataDictionary &clDicomTags, const std::string &strPath);
std::string ComputeZSpacing(const itk::MetaDataDictionary &clDicomTags, const std::string &strPath);
std::string ComputeDimension(const itk::MetaDataDictionary &clDicomTags, const std::string &strPath, int iDim);
std::string ComputeDiffusionBValue(const itk::MetaDataDictionary &clDicomTags);
std::string ComputeDiffusionBValueSiemens(const itk::MetaDataDictionary &clDicomTags);
std::string ComputeDiffusionBValueGE(const itk::MetaDataDictionary &clDicomTags);
std::string ComputeDiffusionBValueProstateX(const itk::MetaDataDictionary &clDicomTags); // Same as Skyra and Verio
std::string ComputeDiffusionBValuePhilips(const itk::MetaDataDictionary &clDicomTags);

#ifdef USE_MD5
template<typename PixelType, unsigned int Dimension>
std::string ComputeMD5HashHelper(const std::string &strFilePath);

template<unsigned int Dimension>
std::string ComputeMD5Hash(const itk::MetaDataDictionary &clDicomTags, const std::string &strFilePath);
#endif // USE_MD5

int main(int argc, char **argv) {
  const char * const p_cArg0 = argv[0];

  // Populate the description -> tag map

  // http://dicom.nema.org/medical/dicom/current/output/chtml/part06/chapter_6.html

  g_mNameToTagMap["patient id"] = "0010|0020";
  g_mNameToTagMap["patient name"] = "0010|0010";
  g_mNameToTagMap["series description"] = "0008|103e"; // Must be lowercase hex!
  g_mNameToTagMap["series number"] = "0020|0011";
  g_mNameToTagMap["sequence name"] = "0018|0024"; // Used in ProstateX for encoding b-value
  g_mNameToTagMap["instance number"] = "0020|0013";
  g_mNameToTagMap["instance uid"] = "0008|0018";
  g_mNameToTagMap["study date"] = "0008|0020";
  g_mNameToTagMap["accession number"] = "0008|0050";
  g_mNameToTagMap["series uid"] = "0020|000e";
  g_mNameToTagMap["body part examined"] = "0018|0015";

  bool bEraseFolders = false;
  bool bRecursive = false;
  bool bCopy = false;

  int c = 0;
  while ((c = getopt(argc, argv, "cehlr")) != -1) {
    switch (c) {
    case 'c':
      bCopy = true;
      break;
    case 'h':
      Usage(p_cArg0);
      break;
    case 'e':
      bEraseFolders = true;
      break;
    case 'l':
      PrintSupportedPatterns();
      return 0;
    case 'r':
      bRecursive = true;
      break;
    case '?':
    default:
      Usage(p_cArg0); // Exits
      break;
    }
  }

  argc -= optind;
  argv += optind;

  if (argc < 2)
    Usage(p_cArg0); // Exits

  const std::string strDestPattern = argv[argc-1];

  --argc;

  std::vector<std::string> vFiles;

  for (int i = 0; i < argc; ++i) {
    const char * const p_cFile = argv[i];

    if (strpbrk(p_cFile, "?*") != nullptr) {
      // DOS wildcard pattern
      FindFiles(DirName(p_cFile).c_str(), p_cFile, vFiles, bRecursive);
    }
    else if (IsFolder(p_cFile)) {
      // Directory
      FindFiles(p_cFile, "*", vFiles, bRecursive);
    }
    else {
      // Individual file
      vFiles.push_back(p_cFile);
    }
  }

  if (bCopy)
    std::cout << "Copying DICOM files to '" << strDestPattern << "' ..." << std::endl;
  else
    std::cout << "Moving DICOM files to '" << strDestPattern << "' ..." << std::endl;

  for (size_t i = 0; i < vFiles.size(); ++i)
    MoveDicomFile(vFiles[i], strDestPattern, bCopy);

  if (bEraseFolders) {
    std::unordered_set<std::string> sFolders;
    for (size_t i = 0; i < vFiles.size(); ++i) {
#if defined(_WIN32)
      std::vector<std::string> vParts = SplitString(vFiles[i], "/\\");
#elif defined(__unix__)
      std::vector<std::string> vParts = SplitString(vFiles[i], "/");
#endif

      vParts.pop_back();
      std::string strFolder;
      for (size_t j = 0; j < vParts.size(); ++j) {
        strFolder += vParts[j];
        strFolder += '/';
        sFolders.insert(strFolder);
      }
    }

    std::vector<std::string> vFolders(sFolders.begin(), sFolders.end());

    std::sort(vFolders.begin(), vFolders.end(),
      [](const std::string &strString1, const std::string &strString2) -> bool {
        return strString1.size() > strString2.size();
      });

    for (size_t i = 0; i < vFolders.size(); ++i)
      EnsureRmDir(vFolders[i]);
  }

  std::cout << "Done." << std::endl;

  return 0;
}

void EnsureRmDir(const std::string &strPath) {
  const int iMaxTries = 100;

  if (RmDir(strPath)) {
    int i = 0;

    for (i = 0; i < iMaxTries && FileExists(strPath); ++i)
      USleep(100000);

    if (i < iMaxTries)
      std::cout << "Removed empty folder '" << strPath << "'." << std::endl;
  }
}

void MakeFoldersForFile(const std::string &strPath) {
#if defined(_WIN32)
  std::vector<std::string> vFolders = SplitString(strPath, "/\\");
#elif defined(__unix__)
  std::vector<std::string> vFolders = SplitString(strPath, "/");
#endif 

  vFolders.pop_back(); // Pop off the filename

  std::string strTmpPath;

  for (size_t i = 0; i < vFolders.size(); ++i) {
    strTmpPath += vFolders[i];
    strTmpPath += '/';

    if (MkDir(strTmpPath))
      std::cout << "Created folder '" << strTmpPath << "'." << std::endl;
  }
}

std::string MakeFailedValue(const std::string &strTag) {
  std::string strValue = "EMPTY_";

  for (size_t i = 0; i < strTag.size(); ++i) {
    switch (strTag[i]) {
    case ' ':
      strValue.push_back('_');
      break;
    default:
      strValue.push_back(std::toupper(strTag[i]));
      break;
    }
  }

  return strValue;
}

std::string MakeValue(const std::string &strTag, const itk::MetaDataDictionary &clDicomTags, const std::string &strPath) {
  if (strTag == "file") {
    const std::string strValue = BaseName(strPath);
    return strValue.size() > 0 ? strValue : MakeFailedValue(strTag);
  }
  else if (strTag == "folder") {
    const std::string strValue = DirName(strPath);
    return strValue.size() > 0 ? strValue : MakeFailedValue(strTag);
  }
  else if (strTag == "diffusion b-value") {
    const std::string strValue = ComputeDiffusionBValue(clDicomTags);
    return strValue.size() > 0 ? strValue : MakeFailedValue(strTag); 
  }
  else if (strTag == "z spacing") {
    const std::string strValue = ComputeZSpacing(clDicomTags, DirName(strPath));
    return strValue.size() > 0 ? strValue : MakeFailedValue(strTag); 
  }
  else if (strTag == "z coordinate") {
    const std::string strValue = ComputeZCoordinate(clDicomTags, DirName(strPath));
    return strValue.size() > 0 ? strValue : MakeFailedValue(strTag); 
  }
  else if (strTag == "z origin") {
    const std::string strValue = ComputeZOrigin(clDicomTags, DirName(strPath));
    return strValue.size() > 0 ? strValue : MakeFailedValue(strTag); 
  }
  else if (strTag == "x dim") {
    const std::string strValue = ComputeDimension(clDicomTags, DirName(strPath), 0);
    return strValue.size() > 0 ? strValue : MakeFailedValue(strTag); 
  }
  else if (strTag == "y dim") {
    const std::string strValue = ComputeDimension(clDicomTags, DirName(strPath), 1);
    return strValue.size() > 0 ? strValue : MakeFailedValue(strTag); 
  }
  else if (strTag == "z dim") {
    const std::string strValue = ComputeDimension(clDicomTags, DirName(strPath), 2);
    return strValue.size() > 0 ? strValue : MakeFailedValue(strTag); 
  }
#ifdef USE_MD5
  else if (strTag == "slice md5") {
    const std::string strValue = ComputeMD5Hash<2>(clDicomTags, strPath);
    return strValue.size() > 0 ? strValue : MakeFailedValue(strTag); 
  }
  else if (strTag == "volume md5") {
    const std::string strValue = ComputeMD5Hash<3>(clDicomTags, strPath);
    return strValue.size() > 0 ? strValue : MakeFailedValue(strTag); 
  }
#endif // USE_MD5
  else {
    auto itr = g_mNameToTagMap.find(strTag);

    if (itr == g_mNameToTagMap.end())
      return std::string();

    const std::string &strDicomTag = itr->second;

    std::string strValue;
    if (!itk::ExposeMetaData(clDicomTags, strDicomTag, strValue))
      return MakeFailedValue(strTag);

    Trim(strValue);
    SanitizeFileName(strValue);

    return strValue;
  }

  return std::string(); // Not reached
}

bool ProcessString(std::string &strPattern, const itk::MetaDataDictionary &clDicomTags, const std::string &strPath) {
  // Sanity check
  int iParity = 0;

  // NOTE: Nesting <> is not supported

  for (size_t i = 0; i < strPattern.size() && (iParity == 0 || iParity == 1); ++i) {
    switch (strPattern[i]) {
    case '<':
      ++iParity;
      break;
    case '>':
      --iParity;
      break;
    }
  }

  if (iParity != 0) {
    std::cerr << "Error: Mismatch between '<' and '>' in '" << strPattern << "'. Aborting ..." << std::endl;
    return false;
  }

  std::string strTag, strValue;

  size_t p = 0;
  while (p  < strPattern.size()) {
    p = strPattern.find('<', p); // < can never be the last character from sanity check

    if (p == std::string::npos) // Nothing left to do
      break;

    size_t q = strPattern.find('>', p+1); // > must accompany a found < as per sanity check

    strTag = strPattern.substr(p+1, q-p-1);
    strValue = MakeValue(strTag, clDicomTags, strPath);

    if (strValue.empty()) {
      std::cerr << "Error: Failed to process tag '" << strTag << "'. Aborting ..." << std::endl;
      return false;
    }

    strPattern.replace(p, q+1-p, strValue);

    p += strValue.size();
  }

  return true;
}

bool MoveDicomFile(const std::string &strFileName, const std::string &strPattern, bool bCopy) {
  typedef itk::GDCMImageIO ImageIOType;

  ImageIOType::Pointer p_clImageIO = ImageIOType::New();

  if (!p_clImageIO->CanReadFile(strFileName.c_str()))
    return false; // Not a DICOM file

  p_clImageIO->SetFileName(strFileName.c_str());
  p_clImageIO->LoadPrivateTagsOn();

  try {
    p_clImageIO->ReadImageInformation();
  }
  catch (itk::ExceptionObject &e) {
    std::cerr << "Error: Failed to read DICOM file '" << strFileName << "': " << e << std::endl;
    return false;
  }

  itk::MetaDataDictionary clDicomTags = p_clImageIO->GetMetaDataDictionary();

  std::string strDestPath = strPattern;

  if (!ProcessString(strDestPath, clDicomTags, strFileName)) {
    std::cerr << "Error: Failed to form destination path." << std::endl;
    return false;
  }

  MakeFoldersForFile(strDestPath);

  if (bCopy) {
    if (!Copy(strFileName, strDestPath, false)) {
      std::cerr << "Error: Failed to copy file '" << strFileName << "' to '" << strDestPath << "'." << std::endl;
      return false;
    }
  }
  else {
    if (!Rename(strFileName, strDestPath, false)) {
      std::cerr << "Error: Failed to move file '" << strFileName << "' to '" << strDestPath << "'." << std::endl;
      return false;
    }
  }

  std::cout << strFileName << " --> " << strDestPath << std::endl;

  return true;
}

itk::ImageBase<3>::Pointer LoadImageBase(const itk::MetaDataDictionary &clDicomTags, const std::string &strPath) {
  typedef itk::ImageBase<3> ImageType;

  std::string strSeriesUID;

  if (!itk::ExposeMetaData(clDicomTags, "0020|000e", strSeriesUID))
    return ImageType::Pointer();

  Trim(strSeriesUID);

  ImageType::Pointer p_clImageBase;

  auto itr = g_mSeriesUIDToImageBase.find(strSeriesUID);

  if (itr == g_mSeriesUIDToImageBase.end()) {
    itk::Image<short, 3>::Pointer p_clImage = LoadDicomImage<short, 3>(strPath, strSeriesUID);

    if (!p_clImage) {
      // Cache the failure so we don't try this again

      g_mSeriesUIDToImageBase[strSeriesUID] = ImageType::Pointer();

      return ImageType::Pointer();
    }

    p_clImageBase = ImageType::New();

    p_clImageBase->SetRegions(p_clImage->GetLargestPossibleRegion());
    p_clImageBase->SetSpacing(p_clImage->GetSpacing());
    p_clImageBase->SetDirection(p_clImage->GetDirection());
    p_clImageBase->SetOrigin(p_clImage->GetOrigin());

    g_mSeriesUIDToImageBase[strSeriesUID] = p_clImageBase;
  }
  else {
    p_clImageBase = itr->second;
  }

  return p_clImageBase;
}

std::string ComputeZSpacing(const itk::MetaDataDictionary &clDicomTags, const std::string &strPath) {
  typedef itk::ImageBase<3> ImageType;

  ImageType::Pointer p_clImageBase = LoadImageBase(clDicomTags, strPath);

  if (!p_clImageBase) 
    return std::string();

  const ImageType::SpacingType &clSpacing = p_clImageBase->GetSpacing();

  std::stringstream spacingStream;

  spacingStream << std::setprecision(3) << clSpacing[2];

  return spacingStream.str();
}

std::string ComputeDimension(const itk::MetaDataDictionary &clDicomTags, const std::string &strPath, int iDim) {

  if (iDim < 0 || iDim > 2)
    return std::string();

  if (iDim == 2) {
    typedef itk::ImageBase<3> ImageType;

    ImageType::Pointer p_clImageBase = LoadImageBase(clDicomTags, strPath);

    if (!p_clImageBase) 
      return std::string();

    const ImageType::SizeType &clSize = p_clImageBase->GetLargestPossibleRegion().GetSize();

    return std::to_string((long long)clSize[iDim]); // long long for VS 2010
  }

  // Calculate X and Y based on DICOM since some images could be inconsistent and share the same series UID

  const char * const a_cSizeTags[2] = {
    "0028|0011",
    "0028|0010"
  };

  std::string strValue;

  if (!itk::ExposeMetaData(clDicomTags, a_cSizeTags[iDim], strValue))
    return std::string();

  Trim(strValue);
  
  return strValue;
}

std::string ComputeDiffusionBValue(const itk::MetaDataDictionary &clDicomTags) {
  std::string strBValue;

  if (itk::ExposeMetaData(clDicomTags, "0018|9087", strBValue)) {
    Trim(strBValue);
    return strBValue;
  }

  std::string strPatientName;
  std::string strPatientId;
  std::string strManufacturer;

  itk::ExposeMetaData(clDicomTags, "0010|0010", strPatientName);
  itk::ExposeMetaData(clDicomTags, "0010|0020", strPatientId);

  if (strcasestr(strPatientName.c_str(), "prostatex") != nullptr || strcasestr(strPatientId.c_str(), "prostatex") != nullptr)
    return ComputeDiffusionBValueProstateX(clDicomTags);

  if (!itk::ExposeMetaData(clDicomTags, "0008|0070", strManufacturer)) {
    std::cerr << "Error: Could not determine manufacturer." << std::endl;
    return std::string();
  }

  if (strcasestr(strManufacturer.c_str(), "siemens") != nullptr)
    return ComputeDiffusionBValueSiemens(clDicomTags);
  else if (strcasestr(strManufacturer.c_str(), "ge") != nullptr)
    return ComputeDiffusionBValueGE(clDicomTags);
  else if (strcasestr(strManufacturer.c_str(), "philips") != nullptr)
    return ComputeDiffusionBValuePhilips(clDicomTags);

  return std::string();
}

std::string ComputeZCoordinate(const itk::MetaDataDictionary &clDicomTags, const std::string &strPath) {
  typedef itk::ImageBase<3> ImageType;

  std::string strImagePosition; // Try to grab this first before doing expensive stuff
  if (!itk::ExposeMetaData(clDicomTags, "0020|0032", strImagePosition))
    return std::string();

  Trim(strImagePosition);

  ImageType::Pointer p_clImageBase = LoadImageBase(clDicomTags, strPath);

  if (!p_clImageBase) 
    return std::string();

  ImageType::PointType clOrigin;

  {
    float a_fTmp[3] = { 0.0f, 0.0f, 0.0f };

    if (sscanf(strImagePosition.c_str(), "%f\\%f\\%f", a_fTmp+0, a_fTmp+1, a_fTmp+2) != 3)
      return std::string();

    clOrigin[0] = itk::SpacePrecisionType(a_fTmp[0]);
    clOrigin[1] = itk::SpacePrecisionType(a_fTmp[1]);
    clOrigin[2] = itk::SpacePrecisionType(a_fTmp[2]);
  }

  // Now transform the physical origin to an index (fail if not in the volume)

  ImageType::IndexType clIndex;
  if (!p_clImageBase->TransformPhysicalPointToIndex(clOrigin, clIndex))
    return std::string();

  return std::to_string((long long)clIndex[2]);
}

std::string ComputeZOrigin(const itk::MetaDataDictionary &clDicomTags, const std::string &strPath) {
  std::string strImagePosition; // Try to grab this first before doing expensive stuff
  if (!itk::ExposeMetaData(clDicomTags, "0020|0032", strImagePosition))
    return std::string();

  float a_fTmp[3] = { 0.0f, 0.0f, 0.0f };

  if (sscanf(strImagePosition.c_str(), "%f\\%f\\%f", a_fTmp+0, a_fTmp+1, a_fTmp+2) != 3)
    return std::string();

  return std::to_string(a_fTmp[2]);
}

std::string ComputeDiffusionBValueSiemens(const itk::MetaDataDictionary &clDicomTags) {
  std::string strModel;

  if (itk::ExposeMetaData(clDicomTags, "0008|1090", strModel)) {
    if ((strcasestr(strModel.c_str(), "skyra") != nullptr || strcasestr(strModel.c_str(), "verio") != nullptr)) {
      std::string strTmp = ComputeDiffusionBValueProstateX(clDicomTags);

      if (strTmp.size() > 0)
        return strTmp;
    }
  }

  gdcm::CSAHeader clCSAHeader;

  if (!GetCSAHeaderFromElement(clDicomTags, "0029|1010", clCSAHeader)) // Nothing to do
    return std::string();

  std::string strTmp;

  if (ExposeCSAMetaData(clCSAHeader, "B_value", strTmp))
    return strTmp;

  return std::string();
}

std::string ComputeDiffusionBValueGE(const itk::MetaDataDictionary &clDicomTags) {
  std::string strValue;
  if (!itk::ExposeMetaData(clDicomTags, "0043|1039", strValue))
    return std::string(); // Nothing to do

  size_t p = strValue.find('\\');
  if (p == std::string::npos)
    return std::string(); // Not sure what to do

  strValue.erase(p);

  std::stringstream valueStream;
  valueStream.str(strValue);

  double dValue = 0.0;

  if (!(valueStream >> dValue) || dValue < 0.0) // Bogus value
    return std::string();

  // Something is screwed up here ... let's try to remove the largest significant digit
  if (dValue > 3000.0) {
    p = strValue.find_first_not_of(" \t0");

    strValue.erase(strValue.begin(), strValue.begin()+p+1);

    valueStream.clear();
    valueStream.str(strValue);

    if (!(valueStream >> dValue) || dValue < 0.0 || dValue > 3000.0)
      return std::string();
  }

  return std::to_string((long long)dValue);
}

std::string ComputeDiffusionBValueProstateX(const itk::MetaDataDictionary &clDicomTags) {
  std::string strSequenceName;
  if (!itk::ExposeMetaData(clDicomTags, "0018|0024", strSequenceName)) {
    std::cerr << "Error: Could not extract sequence name (0018,0024)." << std::endl;
    return std::string();
  }

  Trim(strSequenceName);

  if (strSequenceName.empty()) {
    std::cerr << "Error: Empty sequence name (0018,0024)." << std::endl;
    return std::string();
  }

  std::stringstream valueStream;

  unsigned int uiBValue = 0;

  size_t i = 0, j = 0;
  while (i < strSequenceName.size()) {
    i = strSequenceName.find('b', i); 

    if (i == std::string::npos || ++i >= strSequenceName.size())
      break;

    j = strSequenceName.find_first_not_of("0123456789", i); 

    // Should end with a 't' or a '\0'
    if (j == std::string::npos)
      j = strSequenceName.size();
    else if (strSequenceName[j] != 't')
      break;

    if (j > i) {
      std::string strBValue = strSequenceName.substr(i, j-i);
      valueStream.clear();
      valueStream.str(strBValue);

      uiBValue = 0;

      if (valueStream >> uiBValue) {
        if (uiBValue < 3000)
          return strBValue;
        else
          std::cerr << "Error: B-value of " << uiBValue << " seems bogus. Continuing to parse." << std::endl;
      }   
    }   

    i = j;
  }

  std::cerr << "Error: Could not parse sequence name '" << strSequenceName << "'." << std::endl;

  return std::string();
}

std::string ComputeDiffusionBValuePhilips(const itk::MetaDataDictionary &clDicomTags) {
  std::string strBValue;

  if (!itk::ExposeMetaData(clDicomTags, "2001|1003", strBValue))
    return std::string();

  return strBValue;
}

bool IsHexDigit(char c) {
  if (std::isdigit(c))
    return true;

  switch (std::tolower(c)) {
  case 'a':
  case 'b':
  case 'c':
  case 'd':
  case 'e':
  case 'f':
    return true;
  }

  return false;
}

bool ParseITKTag(const std::string &strKey, uint16_t &ui16Group, uint16_t &ui16Element) {
  if (strKey.empty() || !IsHexDigit(strKey[0]))
    return false;

  ui16Group = ui16Element = 0;

  char *p = nullptr;
  unsigned long ulTmp = strtoul(strKey.c_str(), &p, 16);

  if (*p != '|' || *(p+1) == '\0' || ulTmp > std::numeric_limits<uint16_t>::max())
    return false;

  ui16Group = (uint16_t)ulTmp;

  ulTmp = strtoul(p+1, &p, 16);

  if (*p != '\0' || ulTmp > std::numeric_limits<uint16_t>::max())
    return false;

  ui16Element = (uint16_t)ulTmp;

  return true;
}


template<typename ValueType>
bool ExposeCSAMetaData(gdcm::CSAHeader &clHeader, const char *p_cKey, ValueType &value) {
  if (!clHeader.FindCSAElementByName(p_cKey))
    return false;

  const gdcm::CSAElement &clElement = clHeader.GetCSAElementByName(p_cKey);
  const gdcm::ByteValue * const p_clByteValue = clElement.GetByteValue();

  if (p_clByteValue == nullptr || p_clByteValue->GetLength() != sizeof(ValueType))
    return false;

  return p_clByteValue->GetBuffer((char *)&value, sizeof(ValueType));
}

template<>
bool ExposeCSAMetaData<std::string>(gdcm::CSAHeader &clHeader, const char *p_cKey, std::string &strValue) {
  if (!clHeader.FindCSAElementByName(p_cKey))
    return false;

  const gdcm::CSAElement &clElement = clHeader.GetCSAElementByName(p_cKey);
  const gdcm::ByteValue * const p_clByteValue = clElement.GetByteValue();

  if (p_clByteValue == nullptr)
    return false;

  std::vector<char> vBuffer(p_clByteValue->GetLength());
  if (!p_clByteValue->GetBuffer(&vBuffer[0], vBuffer.size()))
    return false;

  strValue.assign(vBuffer.begin(), vBuffer.end());

  return true;
}


bool GetCSAHeaderFromElement(const itk::MetaDataDictionary &clDicomTags, const std::string &strKey, gdcm::CSAHeader &clCSAHeader) {
  uint16_t ui16Group = 0, ui16Element = 0;

  clCSAHeader = gdcm::CSAHeader();

  if (!ParseITKTag(strKey, ui16Group, ui16Element))
    return false;

  std::string strValue;

  if (!itk::ExposeMetaData<std::string>(clDicomTags, strKey, strValue))
    return false;

  const int iDecodeLength = gdcm::Base64::GetDecodeLength(strValue.c_str(), (int)strValue.size());

  if (iDecodeLength <= 0)
    return false;

  std::vector<char> vBuffer(iDecodeLength);

  if (gdcm::Base64::Decode(&vBuffer[0], vBuffer.size(), strValue.c_str(), strValue.size()) == 0)
    return false;

  gdcm::DataElement clDataElement;
  clDataElement.SetTag(gdcm::Tag(ui16Group, ui16Element));
  clDataElement.SetByteValue(&vBuffer[0], vBuffer.size());

  return clCSAHeader.LoadFromDataElement(clDataElement);
}

#ifdef USE_MD5
template<typename PixelType, unsigned int Dimension>
std::string ComputeMD5HashHelper(const std::string &strFilePath) {
  typedef itk::Image<PixelType, Dimension> ImageType;  

  typename ImageType::Pointer p_clImage = LoadDicomImage<PixelType, Dimension>(strFilePath);

  if (!p_clImage)
    return std::string();

  typedef itk::Testing::HashImageFilter<ImageType> HashFilterType;

  typename HashFilterType::Pointer p_clHasher = HashFilterType::New();

  p_clHasher->SetHashFunction(HashFilterType::MD5);
  p_clHasher->SetInput(p_clImage);

  p_clHasher->Update();

  return p_clHasher->GetHash();
}

template<unsigned int Dimension>
std::string ComputeMD5Hash(const itk::MetaDataDictionary &clDicomTags, const std::string &strFilePath) {
  typedef itk::GDCMImageIO ImageIOType;

  std::string strSeriesUID;

  if (Dimension == 3) {
    if (!itk::ExposeMetaData(clDicomTags, "0020|000e", strSeriesUID))
      return std::string();

    Trim(strSeriesUID);

    auto itr = g_mSeriesUIDToMD5.find(strSeriesUID);

    if (itr != g_mSeriesUIDToMD5.end())
      return itr->second;
  }

  // This is really lame since we already read this information
  ImageIOType::Pointer p_clImageIO = ImageIOType::New();

  p_clImageIO->SetFileName(strFilePath);

  try {
    p_clImageIO->ReadImageInformation();
  }
  catch (itk::ExceptionObject &) {
    return std::string();
  }

  std::string strHash;

  switch (p_clImageIO->GetPixelType()) {
  case ImageIOType::SCALAR:
    switch (p_clImageIO->GetInternalComponentType()) {
    case ImageIOType::UCHAR:
      strHash = ComputeMD5HashHelper<unsigned char, Dimension>(strFilePath);
      break;
    case ImageIOType::CHAR:
      strHash = ComputeMD5HashHelper<char, Dimension>(strFilePath);
      break;
    case ImageIOType::USHORT:
      strHash = ComputeMD5HashHelper<unsigned short, Dimension>(strFilePath);
      break;
    case ImageIOType::SHORT:
      strHash = ComputeMD5HashHelper<short, Dimension>(strFilePath);
      break;
    case ImageIOType::UINT:
      strHash = ComputeMD5HashHelper<unsigned int, Dimension>(strFilePath);
      break;
    case ImageIOType::INT:
      strHash = ComputeMD5HashHelper<int, Dimension>(strFilePath);
      break;
    case ImageIOType::ULONG:
      strHash = ComputeMD5HashHelper<unsigned long, Dimension>(strFilePath);
      break;
    case ImageIOType::LONG:
      strHash = ComputeMD5HashHelper<long, Dimension>(strFilePath);
      break;
    case ImageIOType::FLOAT:
      strHash = ComputeMD5HashHelper<float, Dimension>(strFilePath);
      break;
    case ImageIOType::DOUBLE:
      strHash = ComputeMD5HashHelper<double, Dimension>(strFilePath);
      break;
    default:
      break; // Not supported component type?
    }
    break;
  case ImageIOType::RGB:
    switch (p_clImageIO->GetInternalComponentType()) {
    case ImageIOType::UCHAR:
      strHash = ComputeMD5HashHelper<itk::RGBPixel<unsigned char>, Dimension>(strFilePath);
      break;
    default:
      break; // Not supported component type?
    }
    break;
  case ImageIOType::RGBA:
    switch (p_clImageIO->GetInternalComponentType()) {
    case ImageIOType::UCHAR:
      strHash = ComputeMD5HashHelper<itk::RGBAPixel<unsigned char>, Dimension>(strFilePath);
      break;
    default:
      break; // Not supported component type?
    }
    break;
  default:
    break; // Not supported pixel type?
  }

  if (Dimension == 3) {
    // Cache the result
    g_mSeriesUIDToMD5[strSeriesUID] = strHash;
  }

  return strHash;
}

#endif // USE_MD5

