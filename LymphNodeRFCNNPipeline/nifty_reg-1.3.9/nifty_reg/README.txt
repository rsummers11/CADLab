#####################
# NIFTY_REG PACKAGE #
#####################

##############################################################################

--------------------------------
1 WHAT DOES THE PACKAGE CONTAIN?
--------------------------------
The code contains programs to perform rigid, affine and non-linearregistration
of 3D images.


The rigid and afiine registration are performed using an algorithm presented by
Ourselin et al.[1, 2]; whereas the non-rigid registration is based on the work
of Modat et al.[3].

Ourselin et al.[1, 2] presented an algorithm called Aladin, which is based on
a block-matching approach and a Trimmed Least Square (TLS) scheme. Firstly,
the block matching provides a set of corresponding points between a target and
a source image. Secondly, using this set of corresponding points, the best
rigid or affine transformation is evaluated. This two-step loop is repeated
until convergence to the best transformation.
In our implementation, we used the normalised cross-correlation between the
target and source blocks to extract the best correspondence. The block width
is constant and has been set to 4 voxels. A coarse-to-ﬁne approach is used,
where the registration is ﬁrst performed on down-sampled images (using a
Gaussian ﬁlter to resample images) and finally performed on full resolution
images.
reg aladin is the name of the command to perform rigid or affine registration.

The non-rigid algorithm implementation is based on the Free-From Deformation
presented by Rueckert et al.[4]. However, the algorithm has been re-factored
in order to speed-up registration. The deformation of the source image is
performed using cubic B-splines to generate the deformation ﬁeld. Concretely,
a lattice of equally spaced control points is defined over the target image
and moving each point allows to locally modify the mapping to the source image.
In order to assess the quality of the warping between both input images, an
objective function composed from the Normalised Mutual Information (NMI) and
the Bending-Energy (BE) is used. The ob jective function value is optimised
using the analytical derivative of both, the NMI and the BE within a conjugate
gradient scheme.
reg f3d is the command to perform non-linear registration.

A third program, called reg resample, is been embedded in the package. It
uses the output of reg aladin and reg f3d to apply transformation, generate
deformation ﬁelds or Jacobian map images for example.

The code has been implemented for CPU and GPU architecture. The former
code is based on the C/C++ language, whereas the later is based on CUDA
(http://www.nvidia.com).

The nifti library (http://nifti.nimh.nih.gov/) is used to read and write
images. The code is thus dealing with nifti and analyse formats.

If you are planning to use any of our research, we would be grateful if you
would be kind enough to cite reference(s) 1, 2 (rigid or affine) and/or
3 (non-rigid).

##############################################################################

-----------------------
2 HOW TO BUILD THE CODE
-----------------------
The code can be easily build using cmake (http://www.cmake.org/). The latest 
version can be downloaded from http://www.cmake.org/cmake/resources/software.html
Assuming that the code source are in the source path folder, you will have 
to ﬁrst create a new folder, i.e. build path (#1) and then to change 
directory to move into that folder (#2).
#1 >> mkdir build path 
#2 >> cd build path 

There you will need to call ccmake (#3a) in order to ﬁll in the 
build options. If you don’t want to specify options, we could just use cmake 
(#3b) and the default build values will be used.
#3a >> ccmake source path
#3b >> cmake source path

The main option in the ccmake gui are deﬁned bellow:
>BUILD ALADIN if this ﬂag is set to ON, the reg aladin command will be created
>BUILD F3D if this ﬂag is set to ON, the reg f3d command will be created 
>BUILD RESAMPLE if this ﬂag is set to ON, the reg resample command will be
created 
>CMAKE BUILD INSTALL options are Release, RelWithDebInfo or Debug 
>USE CUDA if the ﬂag is set to ON, both version CPU and GPU version can be
used; otherwise, only the CPU version is compiled 
>USE DEBUG if the ﬂag is set to ON, the program will print out some information
which are only used to debug 
>USE SSE if the ﬂag is set to ON, the spline computation will be perform using
sse in order to speed-up to processing

If the USE CUDA ﬂag is ON, some other informations will be require to use the
Cuda compiler. The ﬂag CUDA TOOL ROOT DIR expect the path to the 
cuda installation (/usr/local/cuda for example) and the CUDA TOOLKIT ROOT DIR 
expected the path the cuda SDK installation (/usr/local/cuda-sdk/C for example). 
Once all the ﬂags are properly ﬁlled in, just press the ”c” to conﬁgure the Make- 
ﬁle and then the ”g” key to generate them. In the prompt, you just have to 
make (#4) ﬁrst and then make install (#5).
#4 >> make 
#5 >> make install 

##############################################################################

---------
3 EXAMPLE
---------
In this example, we will register two nifti images called target img.nii and 
source img.nii. 
Firstly, we perform an affine registration using reg aladin using the following
command line: 
# reg aladin -target target img.nii -source source img.nii ...
 -aff source-to-target_affine.txt 
More options can be speciﬁed, for more details, use the command: 
# reg aladin -help 
Secondly, we will perform a non-rigid registration between the two previous 
images and the warping will be initialised by the previously found affine
transformation: 
# reg f3d -target target img.nii -source source img.nii ...
 -aff source-to-target_affine.txt -cpp source-to-target cpp.nii ...
 -result source-to-target warped.nii 
As previously, more details can be found using the command: 
# reg f3d -help 
Lastly, the control point position image will be used to generate the Jacobian
map of the transformation. This will be done using: 
# reg resample -target target img.nii -source source img.nii ...
 -cpp source-to-target cpp.nii -jac source-to-target jac.nii 
Once again, the ”-help” argument gives more details about the options. Please
note that the affine matrix does not need to be speciﬁed. We choose to include
the affine transformation in the control point position image. 

##############################################################################

---------
4 LICENSE
---------
Copyright (c) 2009, University College London, United-Kingdom
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

Neither the name of the University College London nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.

##############################################################################

---------
5 CONTACT
---------
For any comment, please, feel free to contact Marc Modat (m.modat@ucl.ac.uk).

##############################################################################

------------
6 REFERENCES
------------
[1] Sebastien Ourselin, A Roche, G Subsol, Xavier Pennec, and Nicholas Ayache.
Reconstructing a 3d structure from serial histological sections. Image 
and Vision Computing, 19(1-2):25–31, 2001. 
[2] Robust registration of multi-modal images: Towards real-time clinical
applications, 2002. 
[3] Marc Modat, Gerard G Ridgway, Zeike A Taylor, Manja Lehmann, 
Josephine Barnes, Nick C Fox, David J Hawkes, and S´ebastien Ourselin. 
Fast free-form deformation using graphics processing units. Comput Meth 
Prog Bio, accepted. 
[4] D. Rueckert, L.I. Sonoda, C. Hayes, D.L.G. Hill, M.O. Leach, and D.J. 
Hawkes. Nonrigid Registration Using Free-Form Deformations: Application 
to Breast MR Images. IEEE Trans. Med. Imag., 18:712–721, 1999.

##############################################################################
##############################################################################
##############################################################################
##############################################################################

