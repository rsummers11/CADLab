#import SimpleITK as sitk
import sys, os, glob
import numpy as np
from skimage.measure import label as label_connected_components
import nibabel as nib
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import SimpleITK as sitk
#from joblib import Parallel, delayed
from inference_single_box import InferenceSingleBox
from model.utils import avg_x_seg, avg_z_seg, measure_z_bounds

#-------------------------------------------------------------------------------
def denoise_ct(ct, weight=2, NUM_BLUR_ITERATIONS=1):
    ## options for denoise_tv_chambolle:
    # weight : float, optional
    # Denoising weight. The greater weight, the more denoising (at the expense of fidelity to input).
    # eps : float, optional
    # Relative difference of the value of the cost function that determines the stop criterion. The algorithm stops when:
    # (E_(n-1) - E_n) < eps * E_0
    # n_iter_max : int, optional
    # Maximal number of iterations used for the optimization.
    # multichannel : bool, optional
    # Apply total-variation denoising separately for each channel. This option should be true for color images,
    #  otherwise the denoising is also applied in the 3rd dimension.
    #temporary rescale
    #min_before = -200# np.min(ct)
    #max_before = 1000#np.max(ct)
    #ct = np.clip(ct, min_before, max_before)
    #avg_before = np.average(ct)
    #print("avg before", avg_before)
    #ct = (ct - min_before)*(256/(max_before - min_before))
    #ct_denoised = (max_before - min_before)*ct_denoised/256 + min_before
    #avg_after = np.average(ct_denoised)
    #print("avg after", avg_after)
    #ct_denoised = ct_denoised + (avg_after - avg_before)
    #max_after = np.max(ct_denoised)
    #ct_denoised = ct_denoised*max_before/max_after
    ##
    #ct_denoised = denoise_tv_chambolle(ct, weight=weight)
    #max_before = np.max(ct)
    #print("max before", max_before)
    ct_denoised = ct
    image = sitk.GetImageFromArray(ct_denoised)
    blurFilter = sitk.CurvatureAnisotropicDiffusionImageFilter()
    blurFilter.SetNumberOfIterations(NUM_BLUR_ITERATIONS)
    ##blurFilter.SetTimeStep(0.125)
    image = blurFilter.Execute(image)
    ct_denoised = sitk.GetArrayFromImage(image)
    #max_after = np.max(ct_denoised)
    #print("max after", max_after)
    ##ct_denoised = np.swapaxes(ct_denoised, 0, 2) #zxy -> yxz
    ##ct_denoised = np.swapaxes(ct_denoised, 0, 1).astype(np.float32) #yxz -> xyz
    ##ct_denoised = np.ascontiguousarray(ct_denoised)
    #_, _, sz = ct.shape
    #num_cores = 32
    #results = Parallel(n_jobs=num_cores)(delayed(denoise_tv_chambolle)(ct[:,:,iz], weight=weight, multichannel=False) for iz in range(sz))
    #ct_denoised = np.stack(results, axis=2)
    return ct_denoised

#-------------------------------------------------------------------------------
def threshold_and_label(ct_kidneys_denoised, THRESHOLD_CALCULI):
    ct_kidneys_boolean = np.where(ct_kidneys_denoised > THRESHOLD_CALCULI, 1, 0)
    labeled_components, num_components = label_connected_components(ct_kidneys_boolean, return_num=True)
    return labeled_components, num_components

#-------------------------------------------------------------------------------
def region_growing(labeled_components, ct, ir, lower_threshold=130):

    this_region_seg = (labeled_components == ir)
    this_region_ct = np.where(this_region_seg, ct, 0)  # look at original CT
    this_max = np.max(this_region_ct)
    point = np.argwhere(this_region_ct == this_max)[0]
    point_int = (int(point[0]), int(point[1]), int(point[2]))
    #lower_threshold = np.max([this_max/2, 130)
    sitk_img = sitk.GetImageFromArray(ct)
    point_int_sitk = (point_int[2], point_int[1], point_int[0])
    grown_region = sitk.ConnectedThreshold(sitk_img,
                                seedList = [point_int_sitk],
                                lower = lower_threshold,
                                upper = 2000)
    grown_np = sitk.GetArrayFromImage(grown_region)
    return grown_np

#-------------------------------------------------------------------------------
def find_calculi(ct_orig, seg_kidney, px=1, py=1, pz=1,
                 HUOffset = 0, filename="",
                 THRESHOLD_CALCULI = 150,
                 MIN_CALCULI_VOLUME = 0.5,
                 MAX_CALCULI_VOLUME = 10*10*10,
                 verbose=True,
                 NUM_BLUR_ITERATIONS = 1,
                 MAX_COMPONENTS = 200,
                 RUN_CNN_CLASSIFIER=True,
                 affine=np.eye(4),
                 subbox_inference_object=None):

    sx, sy, sz = ct_orig.shape

    ## dilate kidney segmentation a bit
    #if (verbose): print("...performing small erosion ", end='', flush=True)
    #binarylabel = (seg_kidney > 0.5)
    #zerosion = 2 #int(ez / pz)
    #xerosion = 2 #int(ex / px)
    #yerosion = 2 #int(ey / py)
    #struct = np.ones((xerosion,yerosion,zerosion))
    #seg_kidney = binary_erosion(binarylabel, structure=struct, iterations=10) #3->10

    ## crop in z direction for speed
    minn, maxx = measure_z_bounds(seg_kidney)
    ct = ct_orig[:, :, minn:maxx].copy()
    seg_kidney = seg_kidney[:, :, minn:maxx]
    seg_kidney = np.flip(seg_kidney, axis=0)

    #nib.Nifti1Image(ct.astype(np.float32), affine).to_filename(filename+"ct_before.nii.gz")
    #nib.Nifti1Image(seg_kidney.astype(np.int16), np.eye(4)).to_filename(filename+"seg_kidney_before.nii.gz")


    labeled_components, num_components = label_connected_components(seg_kidney, return_num=True)

    if num_components > 2:
        print("ERROR :!!!!! more than 2 objects in segmentation !!!!! for ", filename)
        print("trying to continue anyway...")

    if num_components > 1:
        k1 = labeled_components == 1
        k2 = labeled_components == 2

        cx1 = avg_x_seg(k1)
        cx2 = avg_x_seg(k2)

        #cz1 = avg_z_seg(k1)
        #cz2 = avg_z_seg(k2)

        minz1, maxz1 = measure_z_bounds(k1)
        cz1 = (maxz1 + minz1)//2

        minz2, maxz2 = measure_z_bounds(k2)
        cz2 = (maxz2 + minz1)//2

        #if cx1 < cx2 then k1 is the left kidney, otherwise swap
        if (cx1 < cx2):
            k1_l_or_r = 'l'
            k2_l_or_r = 'r'
        else:
            k1_l_or_r = 'r'
            k2_l_or_r = 'l'

    if num_components == 1:
        k1 = labeled_components == 1

        cx1 = avg_x_seg(k1)
        cx2 = 10000000 #will be used later

        #cz1 = avg_z_seg(k1)

        minz1, maxz1 = measure_z_bounds(k1)
        cz1 = (maxz1 + minz1)//2

        #if cx1 is on left half of image than is left
        if (cx1 < sx//2):
            k1_l_or_r = 'l'
        else:
            k1_l_or_r = 'r'


    num_components = 100
    iter = 0
    ct_denoised = np.clip(ct.copy(), -200, 1000)
    while iter < 10:
        ct_denoised = denoise_ct(ct_denoised, NUM_BLUR_ITERATIONS=NUM_BLUR_ITERATIONS)
        ct_denoised_kidneys = np.where(seg_kidney == True, ct_denoised, 0) ## isolate kidneys region
        labeled_components, num_components = threshold_and_label(ct_denoised_kidneys, THRESHOLD_CALCULI)
        if (verbose): print("... denoised and found", num_components, "components", end='', flush=True)
        if (num_components < MAX_COMPONENTS):
            break
        iter +=1

    #nib.Nifti1Image(ct_denoised.astype(np.float32), affine).to_filename(filename+"ct_after_denoising.nii.gz")
    #nib.Nifti1Image(labeled_components.astype(np.int16), affine).to_filename(filename+"cropped_calculi_seg_before_region_growing.nii.gz")

    ## ---- region growing -----------------------------------------------------
    grown_region_binary = np.zeros(labeled_components.shape)

    for ir in range(1, num_components+1):
        this_region_seg = (labeled_components == ir)
        this_region_ct = np.where(this_region_seg, ct, 0)  # look at original CT
        this_max = np.max(this_region_ct)
        point = np.argwhere(this_region_ct == this_max)[0]
        point_int = (int(point[0]), int(point[1]), int(point[2]))
        lower_threshold = THRESHOLD_CALCULI #np.max([this_max/2, 130)
        sitk_img = sitk.GetImageFromArray(ct)
        point_int_sitk = (point_int[2], point_int[1], point_int[0])
        grown_region = sitk.ConnectedThreshold(sitk_img,
                                    seedList = [point_int_sitk],
                                    lower = lower_threshold,
                                    upper = 2000)
        grown_np = sitk.GetArrayFromImage(grown_region)
        grown_region_binary += grown_np

    grown_region_binary = grown_region_binary > 0

    #----------- repeat connected components analysis and find sizes ----------
    calculi_median_HU = []
    calculi_avg_HU = []
    calculi_std_HU = []
    calculi_max_HU = []
    calculi_volumes = []
    poles = []
    l_or_rs = []
    labeled_components, num_components = label_connected_components(grown_region_binary, return_num=True)
    if (verbose): print("..found", num_components, " after region growing and joining")
    final_seg = np.zeros(labeled_components.shape)
    num_calculi_found = 0
    calculi_volumes = []
    voxel_volume = px*py*pz
    min_voxels = np.ceil(MIN_CALCULI_VOLUME/voxel_volume)
    max_voxels = np.ceil(MAX_CALCULI_VOLUME/voxel_volume)
    tops = [] #top most kidney coord
    bottoms = [] #bottom most kidney coord

    for ir in range(1, num_components+1):
        this_region_seg = (labeled_components == ir)
        region_size = np.sum(this_region_seg)

        if (region_size < max_voxels) and (region_size > min_voxels):
            num_calculi_found += 1
            final_seg = np.where(this_region_seg > 0, num_calculi_found, final_seg)
            calculi_volumes += [region_size*voxel_volume]

            #figure out if in left or right half of image
            xks = avg_x_seg(this_region_seg)
            zks = avg_z_seg(this_region_seg)

            if (np.abs(xks - cx1)  <  np.abs(xks - cx2)):
                #is in k1
                tops += [minz1]
                bottoms += [maxz2]

                l_or_r = k1_l_or_r
                if (zks < cz1):
                    pole = 0
                else:
                    pole = 1
            else:
                #is in k2
                tops += [minz2]
                bottoms += [maxz2]

                l_or_r = k2_l_or_r
                if (zks < cz2):
                    pole = 0
                else:
                    pole = 1

            nanseg = np.where(this_region_seg == False, np.nan, ct)
            max_HU =np.nanmax(nanseg)
            median_HU = np.nanmedian(nanseg)
            avg_HU = np.nanmean(nanseg)
            std_HU = np.nanstd(nanseg)

            calculi_max_HU += [max_HU]
            calculi_median_HU += [median_HU]
            calculi_avg_HU += [avg_HU]
            calculi_std_HU += [std_HU]
            poles += [pole]
            l_or_rs += [l_or_r]

    if (verbose): print("..found ", num_calculi_found, " within size limits", end='')
    #nib.Nifti1Image(ct_denoised_kidneys.astype(np.float32), affine).to_filename(filename+"cropped__ct_denoised_dilated.nii.gz")
    #nib.Nifti1Image(final_seg.astype(np.int16), affine).to_filename(filename+"cropped_calculi_seg.nii.gz")

    ## insert subvolume back in
    final_seg_full = np.zeros((sx,sy,sz))
    final_seg_full[:,:, minn:maxx] = final_seg

    #---  get scores for each detection --------
    hb = 12
    scores = []
    coords = []

    for i in range(1, num_calculi_found+1):
        xx, yy, zz = find_label_com(final_seg_full, i)

        coords += [[xx, yy, zz]]

        score = 0

        if (RUN_CNN_CLASSIFIER):
            if (xx > hb) and (yy > hb) and (zz > hb):
                try:
                    cx = xx - hb
                    cy = yy - hb
                    cz = zz - hb

                    #check bounds
                    diffx = cx + 2*hb - sx
                    diffy = cy + 2*hb - sy
                    diffz = cz + 2*hb - sz

                    #correct if out of bounds
                    if (diffx > 0):
                        cx =  cx - diffx
                    if (diffy > 0):
                        cy =  cy - diffy
                    if (diffz > 0):
                        cz =  cz - diffz

                    box = ct_orig[cx:cx+2*hb, cy:cy+2*hb, cz:cz+2*hb]
                    assert box.shape == (2*hb, 2*hb, 2*hb)
                    score = subbox_inference_object.inference_single_box(box)
                except:
                    print("Warning - stone detection too close to scan edge to make centered cropped box")
                    score = 0
            else:
                score = 0
        else:
            score = 1

        scores += [score]


    return final_seg_full, num_calculi_found, np.array(calculi_volumes), np.array(calculi_median_HU), np.array(calculi_avg_HU), \
    np.array(calculi_std_HU), np.array(calculi_max_HU), np.array(poles), np.array(l_or_rs), np.array(scores), np.array(coords), \
    np.array(tops), np.array(bottoms)


#----------------------------------------------------------------------------
def find_label_com(labels, i_label):

    this_labels = np.where(labels == i_label, 1, 0)

    px = labels.shape[0]
    py = labels.shape[1]
    pz = labels.shape[2]

    tot_pixel_sum = 0
    avg_x = 0
    for i in range(px):
        this_pixel_sum = np.sum(this_labels[i,:,:])
        tot_pixel_sum += this_pixel_sum
        avg_x += this_pixel_sum*i

    avg_x = int(avg_x//(tot_pixel_sum+.000001))

    tot_pixel_sum = 0
    avg_y = 0
    for i in range(py):
        this_pixel_sum = np.sum(this_labels[:,i,:])
        tot_pixel_sum += this_pixel_sum
        avg_y += this_pixel_sum*i

    avg_y = int(avg_y//(tot_pixel_sum+.00001))

    tot_pixel_sum = 0
    avg_z = 0
    for i in range(pz):
        this_pixel_sum = np.sum(this_labels[:,:,i])
        tot_pixel_sum += this_pixel_sum
        avg_z += this_pixel_sum*i

    avg_z = int(avg_z//(tot_pixel_sum+.00001))

    return avg_x, avg_y, avg_z

#------------------------------------------------------------------------
def measure_z_bounds(seg):
    zsum = np.sum(seg, axis=(0,1))
    minn = 1000
    maxx = 0
    for i in range(seg.shape[2]):
        if (zsum[i] > 0):
            if i < minn:
                minn = i
            elif i > maxx:
                maxx = i

    return minn, maxx
