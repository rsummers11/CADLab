# command filelist outputdir
import sys
import argparse
import os
import shutil
import numpy as np
import SimpleITK as sitk
import numpy as np
#import torch
#import nibabel as nib
#import nilearn
#from nilearn.image import resample_img
#import scipy
import SimpleITK as sitk

def read_filelist(filename):
    filelist = [line.rstrip('\n') for line in open(filename)]
    return filelist

def sitk_resample_to_spacing(image, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear, default_value=0.):
    zoom_factor = np.divide(image.GetSpacing(), new_spacing)
    new_size = np.asarray(np.ceil(np.round(np.multiply(zoom_factor, image.GetSize()), decimals=5)), dtype=np.int16)
    offset = calculate_origin_offset(new_spacing, image.GetSpacing())
    reference_image = sitk_new_blank_image(size=new_size, spacing=new_spacing, direction=image.GetDirection(),
                                           origin=image.GetOrigin() + offset, default_value=default_value)
    return sitk_resample_to_image(image, reference_image, interpolator=interpolator, default_value=default_value)


def calculate_origin_offset(new_spacing, old_spacing):
    return np.subtract(new_spacing, old_spacing)/2


def sitk_new_blank_image(size, spacing, direction, origin, default_value=0.):
    image = sitk.GetImageFromArray(np.ones(size, dtype=np.float).T * default_value)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image


def sitk_resample_to_image(image, reference_image, default_value=0., interpolator=sitk.sitkLinear, transform=None,
                           output_pixel_type=None):
    if transform is None:
        transform = sitk.Transform()
        transform.SetIdentity()
    if output_pixel_type is None:
        output_pixel_type = image.GetPixelID()
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetTransform(transform)
    resample_filter.SetOutputPixelType(output_pixel_type)
    resample_filter.SetDefaultPixelValue(default_value)
    resample_filter.SetReferenceImage(reference_image)
    return resample_filter.Execute(image)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist',  type=str,      help='list of nii files')
    parser.add_argument('--file',  type=str,      help='single nii file, if given, filelist will be ignored')
    parser.add_argument('--outputdir',  type=str,      help='folder to store result')
    parser.add_argument('--zres' ,   type=float, help='target voxel size in mm')
    args = parser.parse_args()

    if not args.file == None:
        files = list()
        files.append(args.file)
    else:
        files = read_filelist(args.filelist)

    for f in files:
        image = sitk.ReadImage(f)
        zooms = image.GetSpacing()
        newimage = sitk_resample_to_spacing(image, new_spacing= (zooms[0],zooms[1], args.zres) ,interpolator =  sitk.sitkGaussian)
        outfilename = os.path.join(args.outputdir, os.path.basename(f).split('.')[0] + '__resized.nii.gz' )
        print(outfilename)
        sitk.WriteImage(newimage, outfilename) 


if __name__ == '__main__':
    main()
