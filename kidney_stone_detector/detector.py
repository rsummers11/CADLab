#!/usr/bin/env python3
import sys, os, argparse, json, time, random
import nibabel as nib
import numpy as np
import pandas as pd
from inference_kidney_only import InferenceKidney
from find_calculi import find_calculi, measure_z_bounds, find_label_com
from inference_single_box import InferenceSingleBox
from matplotlib import pyplot as plt
from collections import defaultdict

#-------------------------------------------------------------------------------
def main():
    RUN_CNN_CLASSIFIER = True
    SAVE_MEASUREMENTS = True
    MAX_COMPONENTS = 200
    default_operating_point = 0.5
    NUM_BLUR_ITERATIONS = 1
    MIN_CALCULI_VOLUME = 0.25
    ok_when_volume_above = 10000

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_list',  help='list of test images')
    parser.add_argument('--result_root', default="results/", help='folder to store result')
    #parser.add_argument('--config_file_name', help='Name of the model file (and the config file)')
    #parser.add_argument('--model_file_name', help='Name of the model filepath (if not included then assumes name is same as config_file_name)')
    parser.add_argument('--single_file', action='store', help='path to single file to run on')
    parser.add_argument('--offset', type=float, default=0, action='store', help='HU offset to apply (default = 0 )')
    parser.add_argument('--compute_test_metrics', action='store_true', help='compute test metrics using ground truth .json')
    parser.add_argument('--filename_tag', default="", help='filename tag for testing results')
    parser.add_argument('--save_all_calculi_seg', default=False, action='store_true', help='save all calculi prior to passing to CNN')
    parser.add_argument('--save_seg', default=False, action='store_true', help='save calculi seg after passing through CNN to remove false positives')
    parser.add_argument('--save_kidney_seg', default=True, action='store_true', help='save kidney seg')
    parser.add_argument('--threshold', type=float, default=130, action='store', help='threshold (default is 130 HU)')
    parser.add_argument('--max_components', type=int, default=MAX_COMPONENTS, action='store', help='denoising is done interatively until the number of detections (connected components) is less than this number. (Default 200)')
    parser.add_argument('--run_CNN', default=RUN_CNN_CLASSIFIER, action='store_true', help='Run CNN for false positive removal?')
    parser.add_argument('--operating_point', type=float, default=default_operating_point, action='store', help='Operating point for the CNN (default is 0.5)')
    parser.add_argument('--min_calculi_volume', type=float, default=MIN_CALCULI_VOLUME, action='store', help='minimum volume for calculi in mm^3 (default is 0.25)')
    parser.add_argument('--ok_when_volume_above', type=float, default=ok_when_volume_above, action='store', help='automatically accept kidney stone as true positive when volume is above this in mm^3 (default 10,000)')

    verbose = True
    args = parser.parse_args()
    RUN_CNN_CLASSIFIER = args.run_CNN
    MAX_COMPONENTS = args.max_components
    default_operating_point =  args.operating_point
    THRESHOLD_CALCULI = args.threshold
    MAX_COMPONENTS = args.max_components
    SAVE_SEG = args.save_seg
    SAVE_ALL_CALCULI_SEG = args.save_all_calculi_seg
    MIN_CALCULI_VOLUME = args.min_calculi_volume
    ok_when_volume_above = args.ok_when_volume_above

    #iterative denosing is performed until this number of components is found
    ## you can specify the number of thresholds here and it will generate FP/FN results for each threshold - this is to facilitate making an AUC curve
    #thresholds = np.array([0.01, .01, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75,  .8,.85,  .9, .95,  .99])
    thresholds = np.array([default_operating_point])
    #thresholds = np.linspace(0,1,99)[1:-1]
    #thresholds = np.array([0.5306])
    num_thresholds = len(thresholds)
    operating_point_index = len(thresholds)//2
    operating_point = thresholds[operating_point_index]
    print("using threshold of ", operating_point)


    if (args.test_list):
        filelist = read_filelist(args.test_list)
        print("processing ", len(filelist), "files")
    else:
        filelist = [args.single_file.rstrip('\n').strip()]

    HUOffset = args.offset
    SAVE_NII = args.save_kidney_seg

    COMPUTE_TEST_METRICS = args.compute_test_metrics
    result_root = args.result_root
    if not (os.path.exists(result_root)):
        os.mkdir(result_root)
    KidneySegmentor = None

    precisions = []
    recalls = []
    sizes_pred = []
    sizes_true = []
    tot_tps = defaultdict(lambda: np.zeros(num_thresholds))
    tot_fps = defaultdict(lambda: np.zeros(num_thresholds))
    tot_fns = defaultdict(lambda: np.zeros(num_thresholds))

    tot_gt_num_stones = defaultdict(lambda: 0)
    size_errors = []
    start_time_total = time.time()

    if RUN_CNN_CLASSIFIER:
        subbox_inference_object = InferenceSingleBox()
    else:
        subbox_inference_object = None

    for filepath_ct in filelist:
        start_time = time.time()
        filename = os.path.basename(filepath_ct)
        print("------ running on ", filename, "-----")

        need_flipx, need_flipy, need_flipz = False, False, False
        ctnib = nib.load(filepath_ct)
        a1 = ctnib.affine
        ctnib = nib.as_closest_canonical(ctnib)
        a2 = ctnib.affine
        if (np.sign(a1[0,0]) != np.sign(a2[0,0])):
            need_flipx = True
        if (np.sign(a1[1,1]) != np.sign(a2[1,1])):
            need_flipy = True
        if (np.sign(a1[2,2]) != np.sign(a2[2,2])):
            need_flipz = True

        affine = ctnib.affine
        zooms = ctnib.header.get_zooms()
        px = zooms[0]
        py = zooms[1]
        pz = zooms[2]
        voxel_volume = px*py*pz
        ct = ctnib.get_fdata().astype(np.float32) + HUOffset

        sx, sy, sz = ct.shape

        filepath_seg = os.path.join("./kidney_seg_output/", filename)
        if os.path.exists(filepath_seg):
            ctnib = nib.load(filepath_seg)
            seg_kidney = ctnib.get_fdata().astype(np.int16)
        else:
            if (KidneySegmentor==None):
                KidneySegmentor = InferenceKidney()

            if not(SAVE_NII):
                filepath_seg = ""

            seg_kidney = KidneySegmentor.inference_kidney_only(ct, filepath_seg,
                        need_flipx=need_flipx, need_flipy=need_flipy, need_flipz=need_flipz)

        final_seg, num_calculi_found, calculi_volumes, calculi_median_HU, calculi_avg_HU, calculi_std_HU, calculi_max_HU, poles, l_or_rs, scores, coords, tops, bottoms = find_calculi(ct, seg_kidney, verbose=True,
                                        px=px, py=py, pz=pz, filename=filename,
                                        affine=a2, RUN_CNN_CLASSIFIER=RUN_CNN_CLASSIFIER,
                                        THRESHOLD_CALCULI=THRESHOLD_CALCULI,
                                        MAX_COMPONENTS=MAX_COMPONENTS,
                                        NUM_BLUR_ITERATIONS=NUM_BLUR_ITERATIONS,
                                        MIN_CALCULI_VOLUME = MIN_CALCULI_VOLUME,
                                        subbox_inference_object=subbox_inference_object)
        if (SAVE_ALL_CALCULI_SEG):
            numpy_seg_out = final_seg
            if (need_flipx): numpy_seg_out = np.ascontiguousarray(np.flip(numpy_seg_out, axis = 0))
            if (need_flipy): numpy_seg_out = np.ascontiguousarray(np.flip(numpy_seg_out, axis = 1))
            if (need_flipz): numpy_seg_out = np.ascontiguousarray(np.flip(numpy_seg_out, axis = 2))
            nib.Nifti1Image(numpy_seg_out.astype(np.int16), affine).to_filename(os.path.join("calculi_seg_output/", filename+"_all_calculi_seg.nii.gz"))


        if (SAVE_SEG):
            final_seg_to_use = np.zeros(final_seg.shape)
            if len(scores) > 0:
                idx = 0
                indices = np.argwhere(scores > operating_point)[:,0]
                for index in indices:
                    idx += 1
                    final_seg_to_use = np.where(final_seg == index+1, idx, final_seg_to_use)

            numpy_seg_out = final_seg_to_use
            if (need_flipx): numpy_seg_out = np.ascontiguousarray(np.flip(numpy_seg_out, axis = 0))
            if (need_flipy): numpy_seg_out = np.ascontiguousarray(np.flip(numpy_seg_out, axis = 1))
            if (need_flipz): numpy_seg_out = np.ascontiguousarray(np.flip(numpy_seg_out, axis = 2))
            nib.Nifti1Image(numpy_seg_out.astype(np.int16), affine).to_filename(os.path.join("calculi_seg_output/", filename+"_calculi_seg.nii.gz"))

        #-----------------------------------------------------------------------
        if COMPUTE_TEST_METRICS:
            filepath_json = filepath_ct.replace(".nii.gz", "_stone.json")
            json_dict = json.load(open(filepath_json, 'r'))
            has_stone = json_dict['has_stone']
            gt_num_stones = json_dict['num_stones']
            gt_stone_coords = json_dict['stone_positions']
            gt_stone_sizes = json_dict['stone_sizes']
            print('..gt # of stones: ', gt_num_stones)

            fps = np.zeros(num_thresholds)
            fns = np.zeros(num_thresholds)
            tps = np.zeros(num_thresholds)

            for it, threshold in enumerate(thresholds):

                indices = np.argwhere(scores > float(threshold))[:,0]

                if len(coords) > 0:
                    coords_to_use = coords[indices, :]
                    volumes_to_use = list(calculi_volumes[indices])

                #print(volumes_to_use , ":SHAPE")
                #final_seg_to_use = np.zeros(final_seg.shape)
                #idx = 0
                #for index in indices:
                #    idx += 1
                #    final_seg_to_use = np.where(final_seg == index, idx, final_seg_to_use)

                indices_used = []

                for i in range(gt_num_stones):
                    stx, sty, stz = gt_stone_coords[3*i], gt_stone_coords[3*i+1], gt_stone_coords[3*i+2]
                    stx = stx - 1 #correction for zero indexing
                    if not ("nn1" in filename):
                        sty = sy - sty #flip y
                    stz = sz - stz #flip z

                    pred_index = search_ball(coords_to_use, stx, sty, stz)
                    #pred_index = search_small_ball(final_seg_to_use, stx, sty, stz)
                    #save_box(ct, stx, sty, stz, filename, label=1, num=i)

                    if pred_index == None:
                        if (it == operating_point_index):  print("FN at", stx, sty, stz)
                        fns[it] += 1
                        for size_limit in ["0", "8", "27", "64", "125"]:
                            if (size_limit == "125" ):  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!------------------ > 125mm FN at", stx, sty, stz)

                            if (gt_stone_sizes[i] >= float(size_limit)):
                                tot_fns[size_limit][it] += 1

                    elif not(pred_index in indices_used):
                        tps[it] += 1
                        #if (it == operating_point_index):  print("TP at ", stx, sty, stz)
                        for size_limit in ["0", "8", "27", "64", "125"]:
                            if (gt_stone_sizes[i] >= float(size_limit)):
                                tot_tps[size_limit][it] += 1

                        if (it == operating_point_index):
                            size_errors += [volumes_to_use[pred_index] - gt_stone_sizes[i]]
                            #sizes_pred += [volumes_to_use[pred_index]]
                            sizes_true += [gt_stone_sizes[i]]
                        indices_used += [pred_index]

                ## false positives
                for i in range(len(volumes_to_use)):
                    if not(i in indices_used):
                        fpx, fpy, fpz = coords_to_use[i]
                        if (it == operating_point_index):  print("FP at ", fpx, fpy, fpz)
                        #save_box(ct, fpx, fpy, fpz, filename, label=0, num=i)
                        fps[it] += 1
                        for size_limit in ["0", "8", "27", "64", "125"]:
                            if len(indices) > 0:
                                if not( pred_index == None):
                                    if (float(volumes_to_use[pred_index])  >= float(size_limit)):
                                        tot_fps[size_limit][it] += 1

                if (it == operating_point_index):
                    print("TPs:", tps[it], "FPs:", fps[it], "FNs:", fns[it])

            for size_limit in ["0", "8", "27", "64", "125"]:
                if not(gt_stone_sizes == None):
                    if (len(gt_stone_sizes)>0):
                        for stone in gt_stone_sizes:
                            if (stone >= float(size_limit)):
                                tot_gt_num_stones[size_limit] += 1

        if (SAVE_MEASUREMENTS):

            if num_calculi_found > 0:
                if (need_flipx):
                    coords[:,0] = sx - coords[:,0]
                if (need_flipy):
                    coords[:,1] = sy - coords[:,1]
                if (need_flipz):
                    coords[:,2] = sz - coords[:,2]
                    tops = sz - tops
                    bottoms = sz - bottoms

            indices = []
            for idx in range(len(scores)):
                if (scores[idx] > operating_point) or (calculi_volumes[idx] > ok_when_volume_above):
                    indices += [idx]
            #indices = np.argwhere(scores > operating_point)

            if len(scores) > 0:
                scores = np.round(scores, 4)
                calculi_volumes = np.round(calculi_volumes, 4)
                coords_to_use = coords[indices, :]
                volumes_to_use = calculi_volumes[indices]
                scores_to_use = scores[indices]
                num_stones_found = len(scores_to_use)
                calculi_median_HU_to_use =  calculi_median_HU[indices]
                calculi_avg_HU_to_use = calculi_avg_HU[indices]
                calculi_std_HU_to_use = calculi_std_HU[indices]
                calculi_max_HU_to_use = calculi_max_HU[indices]
                pole_to_use = poles[indices]
                l_or_rs_to_use = l_or_rs[indices]
                tops_to_use = tops[indices]
                bottoms_to_use = bottoms[indices]
            else:
                coords_to_use = []
                volumes_to_use = []
                num_stones_found = 0
                calculi_median_HU_to_use = []
                calculi_avg_HU_to_use = []
                calculi_std_HU_to_use = []
                calculi_max_HU_to_use = []
                pole_to_use = []
                l_or_rs = []
                tops_to_use = []
                bottoms_to_use = []


            print("found", num_stones_found, "stones with score > 0.5")
            if (num_stones_found > 0):
                print("stone coords:")
                for c in coords_to_use:
                    print(c+1)

            with open(os.path.join(result_root,"stone_summary_"+filename+".csv"), 'w') as fo2:
                if len(scores) > 0:
                    fo2.write(filename+","+str(np.max(scores))+","+str(int(np.round(np.max(scores))))+"\n")
                else:
                    fo2.write(filename+",0,0\n")

            with open(os.path.join(result_root,"stone_list_"+filename+".csv"), 'w') as fo:
                if len(scores) > 0:
                    for i, score in enumerate(scores_to_use):
                        coords_string = ","+str(coords_to_use[i][0]+1)+","+str(coords_to_use[i][1]+1)+","+str(coords_to_use[i][2]+1)+","
                        #binary = str(int(np.round(score)))
                        fo.write(filename+","+str(score)+coords_string+str(volumes_to_use[i])+","+str(np.round(calculi_median_HU_to_use[i], 4))+
                        ","+str(np.round(calculi_avg_HU_to_use[i], 4))+","+str(np.round(calculi_std_HU_to_use[i], 4))+","+str(np.round(calculi_max_HU_to_use[i], 4))+","+str(pole_to_use[i])+","+l_or_rs_to_use[i]+","+str(tops_to_use[i])+","+str(bottoms_to_use[i])+"\n")
                else:
                    fo.write("")

        print("time = ", time.time() - start_time, " sec")


#-------------------------------------------------------------------------------
    if (COMPUTE_TEST_METRICS):
        num_cases = len(filelist)

        avg_size_err = np.round(np.average(size_errors), 3)
        std_size_err = np.round(np.std(size_errors), 3)

        print("Avg # false positives per patient:", tot_fps[operating_point_index]/num_cases)
        print("Avg volume error = ", avg_size_err, " +/- ", std_size_err, " mm^3")

        for size_limit in ["0", "8", "27", "64", "125"]:
            precisions = tot_tps[size_limit]/(tot_tps[size_limit] + tot_fps[size_limit] + 0.0000001)
            sensitivities = tot_tps[size_limit]/(tot_gt_num_stones[size_limit] + 0.0000001)
            recalls = sensitivities
            avg_fps_per_case = tot_fps[size_limit]/num_cases

            tag = args.filename_tag
            #plt.figure()
            #plt.plot(avg_fps_per_case[size_limit], sensitivities[size_limit], color='darkorange', lw=2)
            #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel(' avg numbeter false positives per patient')
            #plt.ylabel('sensitivity')
            #for i, x in enumerate(thresholds):
            #    plt.annotate(str(x), # this is the text
            #             (avg_fps_per_case[i], sensitivities[i]), # this is the point to label
            #             textcoords="offset points", # how to position the text
            #             xytext=(0.05,.05), # distance from text to points (x,y)
            #             ha='center')
            #plt.title('Receiver operating characteristic example')
            #plt.legend(loc="lower right")
            #plt.savefig("FROC_"+tag+".png")

            #plt.figure()
            #plt.plot(recalls, precisions, color='darkorange', lw=2)
            #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            #plt.xlim([0.0, 1.05])
            #plt.ylim([0.0, 1.05])
            #plt.xlabel('recall')
            #plt.ylabel('precision')
            #for i, x in enumerate(thresholds):
            #    plt.annotate(str(x), # this is the text
            #             (recalls[i], precisions[i]), # this is the point to label
            #             textcoords="offset points", # how to position the text
            #             xytext=(0.05,.05), # distance from text to points (x,y)
            #             ha='center')
            #plt.title('Receiver operating characteristic example')
            #plt.legend(loc="lower right")
            #plt.savefig("PR_curve_"+str(tag)+".png")

            import pickle as pkl
            datadump = [avg_fps_per_case, sensitivities, recalls, precisions]
            pkl.dump(datadump, open(tag+"_"+size_limit+"_data.pkl", 'wb'))
            #pkl.dump([sizes_pred, sizes_true], open(tag+"_sizes_data.pkl", 'wb'))

        precision = np.round(tot_tps["0"][operating_point_index]/(tot_tps["0"][operating_point_index] + tot_fps["0"][operating_point_index] + .0000000001), 3)

        if tot_gt_num_stones["0"] > 0:
            recall = np.round(tot_tps["0"][operating_point_index]/(tot_gt_num_stones["0"]), 3)
        else:
            recall = 1

        #print("precision: ", precision, "recall:", recall)
        F1 = np.round(2*precision*recall/(precision + recall + .000000001), 3)
        print("F1: ", F1)

        print("total run time = ", (time.time() - start_time_total)/60, "minutes")
        #df = pd.DataFrame({'a':range(len(avg_fps_per_case))})
        #df['avg_fps_per_case'] = pd.Series(avg_fps_per_case, index = df.index[:len(avg_fps_per_case)])
        #df['sensitivities'] = pd.Series(sensitivities, index = df.index[:len(sensitivities)])
        #df['recalls'] = pd.Series(recalls, index = df.index[:len(recalls)])
        #df['precisions'] = pd.Series(sensitivities, index = df.index[:len(precisions)])
        #df.to_csv(tag+"_data.csv")

#-------------------------------------------------------------------------------
def search_ball(coords, stx, sty, stz):

    if len(coords) > 0:
        for i in range(coords.shape[0]):
            xx, yy, zz = coords[i, 0], coords[i, 1], coords[i, 2]
            dist = np.sqrt((stx - xx)**2 + (sty - yy)**2 + (stz - zz)**2)
            if (dist < 6):
                return i

    return None

#-------------------------------------------------------------------------------
def search_small_ball(final_seg, stx, sty, stz):
    idxs = []
    for ix in range(stx-5, stx+6):
        for iy in range(sty-5, sty+6):
            for iz in range(stz-3, stz+4):
                idx = final_seg[ix, iy, iz]
                if (idx > 0):
                    idxs += [idx]
    if len(idxs) > 0:
        return int(np.median(idxs)) - 1
    else:
        return None

#-------------------------------------------------------------------------------
# read config and save
def read_filelist(filepath):
    filelist = list()
    with open(filepath) as f:
      for line in f:
        arg1 = line.rstrip('\n')
        arg1 = arg1.strip()
        if len(arg1)<2:
            continue
        if arg1[0]=='#':
            continue
        if arg1=="-1":
            break
        filelist.append(arg1)
    return filelist

#-------------------------------------------------------------------------------
def save_box(ct, xx, yy, zz, filename, label=0, hb=12, num=1):

    r = random.random()

    if (r < 0.1):
        out_root = "/home/delton/data/kidney_stone_boxes_4/validation"
        box = ct[xx-hb:xx+hb, yy-hb:yy+hb, zz-hb:zz+hb ]
        out_filename = filename+str(num)+"__"+str(label)+"__.nii.gz"
        out_filepath = os.path.join(out_root, out_filename)
        nib.Nifti1Image(box.astype(np.float32), np.eye(4)).to_filename(out_filepath)
    else:
        out_root = "/home/delton/data/kidney_stone_boxes_4/"
        for i in range(4):
            jx = np.random.randint(-4,4)
            jy = np.random.randint(-4,4)
            jz = np.random.randint(-4,4)
            xx = xx + jx
            yy = yy + jy
            zz = zz + jz
            box = ct[xx-hb:xx+hb, yy-hb:yy+hb, zz-hb:zz+hb ]
            out_filename = filename+str(num)+str(jx)+str(jy)+str(jz)+"__"+str(label)+"__.nii.gz"
            out_filepath = os.path.join(out_root, out_filename)
            nib.Nifti1Image(box.astype(np.float32), np.eye(4)).to_filename(out_filepath)

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
