/**
 * @file reg_jacobian.h
 * @author Marc Modat
 * @date 26/06/2012
 * @brief Header file that contains the string to be returned
 * for the slicer extension of reg_jacobian
 *
 * Created by Marc Modat on 26/06/2012.
 * Copyright (c) 2012, University College London. All rights reserved.
 * Centre for Medical Image Computing (CMIC)
 * See the LICENSE.txt file in the nifty_reg root folder
 *
 */

char xml_jacobian[] =
   "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
   "<executable>\n"
   "  <category>Registration.NiftyReg</category>\n"
   "  <title>RegJacobian (NiftyReg)</title>\n"
   "  <description><![CDATA[NiftyReg module to create Jacobian-based images]]></description>\n"
   "  <version>0.0.1</version>\n"
   "  <documentation-url>http://cmic.cs.ucl.ac.uk/home/software/</documentation-url>\n"
   "  <license>BSD</license>\n"
   "  <contributor>Marc Modat (UCL)</contributor>\n"
   "  <parameters advanced=\"false\">\n"
   "    <label>Input reference image</label>\n"
   "    <description>Input images</description>\n"
   "    <file fileExtensions=\".nii,.nii.gz,.nrrd,.txt,.mat\">\n"
   "      <name>InTrans</name>\n"
   "      <longflag>trans</longflag>\n"
   "      <description>Input transformation</description>\n"
   "      <label>Input Trans.</label>\n"
   "      <default>required</default>\n"
   "      <channel>input</channel>\n"
   "    </file>\n"
   "    <image fileExtensions=\".nii,.nii..gz,.nrrd,.png\">"
   "      <name>referenceImageName</name>\n"
   "      <longflag>ref</longflag>\n"
   "      <description>Reference image filename, required if the transformation is a spline parametrisation</description>\n"
   "      <label>Reference image</label>\n"
   "      <default>required</default>\n"
   "      <channel>input</channel>\n"
   "    </image>\n"
   "  </parameters>"
   "  <parameters advanced=\"false\">\n"
   "    <label>Output image</label>\n"
   "    <description>Jacobian determinants or matrices images</description>\n"
   "    <image fileExtensions=\".nii,.nii.gz,.nrrd,.png\">"
   "      <name>jacDetImage</name>\n"
   "      <longflag>jac</longflag>\n"
   "      <description>Jacobian determinant image</description>\n"
   "      <label>Jac. det. image</label>\n"
#ifdef GIMIAS_CLI
   "      <default>None</default>\n"
#else
   "      <default>jacDetImage.nii</default>\n"
#endif
   "      <channel>output</channel>\n"
   "    </image>"
   "    <image fileExtensions=\".nii,.nii.gz,.nrrd,.png\">"
   "      <name>logJacDetImage</name>\n"
   "      <longflag>jacL</longflag>\n"
   "      <description>Log of the Jacobian determinant image</description>\n"
   "      <label>Log. Jac. det. image</label>\n"
#ifdef GIMIAS_CLI
   "      <default>None</default>\n"
#else
   "      <default>logJacDetImage.nii</default>\n"
#endif
   "      <channel>output</channel>\n"
   "    </image>"
   "    <image fileExtensions=\".nii,.nii.gz,.nrrd,.png\">"
   "      <name>JacMatImage</name>\n"
   "      <longflag>jacM</longflag>\n"
   "      <description>Jacobian matrix image</description>\n"
   "      <label>Jac. mat. image</label>\n"
#ifdef GIMIAS_CLI
   "      <default>None</default>\n"
#else
   "      <default>JacMatImage.nii</default>\n"
#endif
   "      <channel>output</channel>\n"
   "    </image>"
   "  </parameters>\n"
   "</executable>"
   ;
