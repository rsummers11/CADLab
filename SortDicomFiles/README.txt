#
# Nathan Lay
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# National Institutes of Health
# March 2017
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

#######################################################################
# Introduction                                                        #
#######################################################################
SortDicomFiles is a utility that can process DICOM files stored in one
folder hierarchy and move/rename them into a folder hierarchy specified
by the user using information stored in the DICOM header.

Aside of data organization, SortDicomFiles be used in a number of other
ways to identity or aid in processing DICOM images/volumes. 

Examples include:
- Quickly identify DICOM volumes with 3mm slice spacing.
- Separate a diffusion MR series into b-value-specific folders.
- Name DICOM slices after their z voxel coordinate.
- Move DICOM slices into a folder named after their series description.

#######################################################################
# Installing                                                          #
#######################################################################
If a precompiled version is available for your operating system, either
extract the archive where it best suits you, or copy the executable to
the desired location.

Once installed, the path to SortDicomFiles should be added to PATH.

Windows: Right-click on "Computer", select "Properties", then select
"Advanced system settings." In the "System Properties" window, look
toward the bottom and click the "Environment Variables" button. Under
the "System variables" list, look for "Path" and select "Edit." Append
the ";C:\Path\To\Folder" where "C:\Path\To\Folder\SortDicomFiles.exe"
is the path to the executable. Click "OK" and you are done.

Linux: Use a text editor to open the file ~/.profile or ~/.bashrc
Add the line export PATH="${PATH}:/path/to/folder" where
/path/to/folder/SortDicomFiles is the path to the executable. Save
the file and you are done.

SortDicomFiles can also be compiled from source. Instructions are
given in the "Building from Source" section.

#######################################################################
# Usage                                                               #
#######################################################################
Once installed, SortDicomFiles must be run from the command line. In
Windows this is accomplished using Command Prompt or PowerShell.
Unix-like environments include terminals where commands may be issued.

WINDOWS TIP: Command Prompt can be launched conveniently in a folder
holding shift and right clicking in the folder's window and selecting
"Open command window here."

SortDicomFiles accepts one or more given folders, files or DOS-wildcard
patterns to process and moves them to a folder hierarchy specifed in
the last argument. For example:

SortDicomFiles . '<patient id>/<study date>/<series description>/<series number>/<instance number>.dcm'

NOTE: May need double quote (") instead of single quote (') on Windows.

This command takes all DICOM files in the current directory and moves
them to the folder hierarchy specified above. These values are
computed from information stored in each file's DICOM header. You will
see output that resembles the following:

Example output:
Moving DICOM files to '<patient id>/<study date>/<series description>/<series number>/<instance number>.dcm' ...
Created folder 'ProstateX-0000/'.
Created folder 'ProstateX-0000/20110707/'.
Created folder 'ProstateX-0000/20110707/tfl_3d PD ref_tra_1.5x1.5_t3/'.
Created folder 'ProstateX-0000/20110707/tfl_3d PD ref_tra_1.5x1.5_t3/9/'.
./000000.dcm --> ProstateX-0000/20110707/tfl_3d PD ref_tra_1.5x1.5_t3/9/8.dcm
...

This works for DICOMs stored in one folder, but DICOMs are often
nested in some pre-existing file hierarchy. For example, this
ProstateX case has the following hierarchy:

Raw DICOM hierarchy from the ProstateX challenge:
ProstateX-0000/
+-- 1.3.6.1.4.1.14519.5.2.1.7311.5101.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    +-- 1.3.6.1.4.1.14519.5.2.1.7311.5101.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    +-- 1.3.6.1.4.1.14519.5.2.1.7311.5101.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    +-- 1.3.6.1.4.1.14519.5.2.1.7311.5101.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    +-- 1.3.6.1.4.1.14519.5.2.1.7311.5101.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    +-- 1.3.6.1.4.1.14519.5.2.1.7311.5101.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    +-- 1.3.6.1.4.1.14519.5.2.1.7311.5101.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    +-- 1.3.6.1.4.1.14519.5.2.1.7311.5101.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

Instead one can issue the -r flag to instruct SortDicomFiles to
search recursively. If the command were issued like

SortDicomFiles -e -r ProstateX-0000 '<patient id>/<study date>/<series description>/<series number>/<instance number>.dcm'

NOTE: The -e flag causes SortDicomFiles to try to delete
now-empty folders processed by SortDicomFiles. Only empty
folders will ever be deleted.

This produces the following human-readable directory hierarchy:
ProstateX-0000/
+-- 20110707
    +-- ep2d_diff_tra_DYNDIST
    ¦   +-- 6
    +-- ep2d_diff_tra_DYNDIST_ADC
    ¦   +-- 7
    +-- ep2d_diff_tra_DYNDISTCALC_BVAL
    ¦   +-- 8
    +-- t2_tse_cor
    ¦   +-- 5
    +-- t2_tse_sag
    ¦   +-- 3
    +-- t2_tse_tra
    ¦   +-- 4
    +-- tfl_3d PD ref_tra_1.5x1.5_t3
        +-- 9

Lastly, SortDicomFiles provides the below usage message when
provided with the -h flag or no arguments. It's useful if you
forget.

Usage: ./SortDicomFiles [-cehlr] folder|filePattern [folder2|filePattern2 ...] destinationPattern

Options:
-c -- Copy instead of move.
-e -- Try to remove empty folders.
-l -- List supported patterns.
-h -- This help message.
-r -- Search for DICOMs recursively.

This version supports the following destination patterns:

Supported patterns:
<accession number> (0008|0050)
<body part examined> (0018|0015)
<instance number> (0020|0013)
<instance uid> (0008|0018)
<patient id> (0010|0020)
<patient name> (0010|0010)
<sequence name> (0018|0024)
<series description> (0008|103e)
<series number> (0020|0011)
<series uid> (0020|000e)
<study date> (0008|0020)
<diffusion b-value> (0018|9087 or vendor-specific)
<z spacing> (z voxel spacing)
<z coordinate> (z voxel coordinate)
<z origin> (z patient origin)
<x dim> (x voxel dimension)
<y dim> (y voxel dimension)
<z dim> (z voxel dimension)
<slice md5> (MD5 hash of pixel data)
<volume md5> (MD5 hash of voxel data)
<file> (file's basename)
<folder> (file's dirname)

See the "Modifying" section to learn how to add new patterns.

NOTE: You must enable the USE_MD5 option in CMake to use <slice md5> 
      and <volume md5>.

#######################################################################
# Building from Source                                                #
#######################################################################
To build SortDicomFiles from source, you will need a recent version of
CMake, a C++11 compiler, and InsightToolkit version 4 or later.

First extract the source code somewhere. Next create a separate
directory elsewhere. This will serve as the build directory. Run CMake
and configure the source and build directories as chosen. More
specifically

On Windows:
- Run cmake-gui (Look in Start Menu) and proceed from there.

On Unix-like systems:
- From a terminal, change directory to the build directory and then
run:

ccmake /path/to/source/directory

In both cases, "Configure." If you encounter an error, set ITK_DIR
and then run "Configure" again. Then select "Generate." On Unix-like
systems, you may additionally want to set CMAKE_BUILD_TYPE to "Release"

NOTE: ITK_DIR should be set to the cmake folder in the ITK lib
folder. For example: /path/to/ITK/lib/cmake/ITK-4.11/

Visual Studio:
- Open the solution in the build directory and build SortDicomFiles.
Make sure you select "Release" mode.

Unix-like systems:
- Run the "make" command.

SortDicomFiles has been successfully built and tested with:
GCC 4.8.5 on Ubuntu 14.04
Microsoft Visual Studio 2010 on Windows 7 and Windows Server 2012
Microsoft Visual Studio 2013 on Windows 7

using ITK versions:
ITK 4.11
ITK 4.7

#######################################################################
# Modifying                                                           #
#######################################################################
Additional DICOM tags may be added to SortDicomFiles by mapping a 
description string (e.g. "instance number") to a DICOM tag. The mapping
is constructed in the main() function. ITK expects the format of the
tag to be given as:

group|element

Where group and element are hexadecimal numbers with lower-case digits.

If more specialized processing is needed, you may need to modify the
MakeValue() function. Specialized processing examples include:
file
folder
diffusion b-value
z spacing
z coordinate

#######################################################################
# Caveats                                                             #
#######################################################################
DOS-wildcard patterns may not work properly when matching subfolders.
For example: /path/to/*/folder

This mostly affects Windows users since Unix shells already expand 
these.

Using absolute paths to Windows shares (i.e. \\name\folder) could cause
problems since BaseName() and DirName() have not yet implemented
parsing these kinds of paths.

