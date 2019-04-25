# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file visualizes the lesion classification and retrieval
# results of the hand-labeled test set in DeepLesion.
# --------------------------------------------------------

import cPickle
import os
import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
import shutil
import time
from load_save_utils import load_DL_info, load_ontology_from_xlsfile, cellname
import json
from scipy.io import loadmat, savemat
from scipy.spatial.distance import cdist, squareform


def draw_html_words(words, colors, atts=None):
    clr_fmt = '<font color="%s">%s</font>'
    att_fmt = '<%s>%s</%s>'
    if atts is None:
        atts = ['' for _ in words]

    html = ""
    for q in range(len(words)):
        if colors[q] != '':
            html1 = clr_fmt % (colors[q], words[q])
        else:
            html1 = words[q] + " "
        if atts[q] != '':
            for att in ['u', 'b', 'i', 's']:
                if att in atts[q]:
                    html1 = att_fmt % (att, html1, att)
        html = html + html1 + " "
    return html


def pad_box(boxes0, pad, imsz):
    boxes = boxes0.copy()
    if len(boxes.shape) == 1:
        boxes = boxes[np.newaxis, :]
    boxes[:, 0] = np.maximum(0, boxes[:, 0] - pad)
    boxes[:, 1] = np.maximum(0, boxes[:, 1] - pad)
    boxes[:, 2] = np.minimum(imsz[1]-1, boxes[:, 2] + pad)
    boxes[:, 3] = np.minimum(imsz[0]-1, boxes[:, 3] + pad)
    return boxes


def crop_patch(im, boxes, pad, spacing=None, spacing_after=None):
    if spacing is not None:
        im_scale = spacing / spacing_after
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        boxes *= im_scale
    boxes_pad = pad_box(boxes, pad, im.shape).astype(int)
    patches = []
    for box in boxes_pad:
        patch1 = im[box[1]:box[3], box[0]:box[2]]
        patches.append(patch1)
    return patches


def visualize_clsf_res(score_file, ft_mat_fn, test_annot_fn, split_fn,
                       ontology_fn, info_fn,
                       res_fd, html_fn, imfdin, imfdout,
                       pad, show_neighbor):
    mat = loadmat(score_file)
    pred_scores = mat['scores']
    pred_labels = mat['pred_labels']
    if pred_scores.min() < 0:
        sigmoid = lambda x: 1 / (1. + np.exp(-x))
        pred_scores = sigmoid(pred_scores)
    lb_te_all = mat['gt_labels']
    term_list = [t.strip() for t in mat['term_list']]
    info_dict = load_DL_info(info_fn)
    with open(test_annot_fn, "rt") as f:
        val_dicts = json.load(f)
    ontology = load_ontology_from_xlsfile(ontology_fn)
    parents_map = {term_dict['term']: term_dict['parents'] for term_dict in ontology}
    parent_list = []
    for t in term_list:
        ps = parents_map[t]
        parent_list.append([term_list.index(p) for p in ps if p in term_list])
    all_synonyms = [s for t in ontology for s in t['synonyms'] if t['term'] in term_list]

    fmt = '<p>%d.  %s</p>'
    htmls = []
    all_lesion_idxs = [d['lesion_idx'] for d in val_dicts]
    if 'lesion_idx' not in mat.keys():
        assert len(val_dicts) == pred_scores.shape[0]
        select = all_lesion_idxs
    else:
        select = mat['lesion_idx'][0]

    if ft_mat_fn is not None:
        ft_all = loadmat(ft_mat_fn)['embedding']
        with open(split_fn, "rt") as f:
            split_data = json.load(f)
        tr_lesion_idx = split_data['train_lesion_idxs']
        ft_cand = ft_all[tr_lesion_idx]

    if not os.path.exists(res_fd+imfdout):
        os.mkdir(res_fd+imfdout)

    clr_fmt = '<font color="%s">%s</font>'
    att_fmt = '<%s>%s</%s>'

    for score_idx, lesion_idx in enumerate(select):
        print score_idx
        pat_idx = [info_dict['patient_idx'][lesion_idx]]
        dict_idx = all_lesion_idxs.index(lesion_idx)
        val_dict = val_dicts[dict_idx]
        imfn = info_dict['filenames'][lesion_idx].replace(os.sep, '_')
        spacing = info_dict['spacing'][lesion_idx]
        fnout = os.path.join(imfdout, '%d_%s' % (lesion_idx, imfn))
        if not os.path.exists(fnout):
            box1 = info_dict['boxes'][lesion_idx]
            im = cv2.imread(os.path.join(imfdin, imfn))
            patch = crop_patch(im, box1, pad, spacing, spacing_after=1)[0]
            cv2.imwrite(res_fd+fnout, patch)

        html1 = "<b>" + imfn +'</b></br>'
        html1 += '<img src="%s"></br>' % fnout
        words = val_dict['text'].encode('ascii', 'ignore')
        html1 = html1 + '<b>Manual description</b>: ' + words + '<br/>'

        pred_idx = np.where(pred_labels[score_idx])[0]
        cls_score_ord = np.argsort(pred_scores[score_idx, pred_idx])[::-1]
        pred_idx = pred_idx[cls_score_ord]
        tps = []
        for cls in pred_idx:
            term1 = term_list[cls]
            sc = pred_scores[score_idx, cls]
            tc = '%s : %.4f' % (term1, sc)
            if lb_te_all[score_idx, cls] == 1:
                tc = clr_fmt % ('green', 'TP: ' + tc)
            else:
                tc = clr_fmt % ('red', 'FP: ' + tc)
            html1 = html1 + '<br/>' + tc
            tps.append(cls)

        for cls in np.where(lb_te_all[score_idx] > 0)[0]:
            if cls in tps:
                continue
            term1 = term_list[cls]
            sc = pred_scores[score_idx, cls]
            tc = '%s : %.4f' % (term1, sc)
            tc = clr_fmt % ('blue', 'FN: ' + tc)
            html1 = html1 + '<br/>' + tc

        htmls.append(fmt % (lesion_idx, html1))

        if ft_mat_fn is not None:
            ft_query = ft_all[lesion_idx]
            dist = cdist(ft_query[np.newaxis, :], ft_cand)[0]
            ord = np.argsort(dist)
            nb = 0
            idx = 1
            while nb < show_neighbor:
                lidx_retr = tr_lesion_idx[ord[idx]]
                pat1 = info_dict['patient_idx'][lidx_retr]
                idx = idx + 1
                if pat1 in pat_idx:  # retrieved lesions must come from different patients
                    continue
                else:
                    pat_idx.append(pat1)
                    nb += 1
                imfn = info_dict['filenames'][lidx_retr].replace(os.sep, '_')
                spacing = info_dict['spacing'][lidx_retr]
                fnout = os.path.join(imfdout, '%d_%s' % (lidx_retr, imfn))
                if not os.path.exists(fnout):
                    box1 = info_dict['boxes'][lidx_retr]
                    im = cv2.imread(os.path.join(imfdin, imfn))
                    patch = crop_patch(im, box1, pad, spacing, spacing_after=1)[0]
                    cv2.imwrite(fnout, patch)

                html1 = "Retrieved #%d: %d. <b>" % (nb, lidx_retr) + imfn + '</b> dist=%.4f </br>' % dist[ord[idx+1]]
                html1 += '<img src="%s"></br>' % fnout
                labels1 = '<b>Relevant labels</b>: '
                tr_dict = split_data['train_relevant_labels'][tr_lesion_idx.index(lidx_retr)]
                labels1 += ', '.join([term_list[p] for p in tr_dict])
                labels1 += '<br/><b>Uncertain labels</b>: '
                tr_dict = split_data['train_uncertain_labels'][tr_lesion_idx.index(lidx_retr)]
                labels1 += ', '.join([term_list[p] for p in tr_dict])

                html1 = html1 + labels1 + '<br/><br/>'
                htmls.append(html1)

    with open(res_fd+html_fn, 'w') as f:
        f.writelines(htmls)


def main():
    info_fn = 'YOURPATH/DeepLesion/DL_info.csv'  # downloadable in the DeepLesion dataset
    im_fd_in = 'YOURPATH/DeepLesion/Key_slices/'  # downloadable in the DeepLesion dataset

    res_mat_fn = 'results/LesaNet_output.mat'
    ft_mat_fn = 'results/LesaNet_embedding.mat'
    # ft_mat_fn = None  # if you don't want to show lesion retrieval results

    test_annot_fn = 'program_data/hand_labeled_test_set.json'
    ontoloty_fn = 'program_data/lesion_ontology_181022.xlsx'
    split_file = 'program_data/text_mined_labels_171_and_split.json'

    res_fd = 'results/'
    html_name = 'show_results.html'
    im_fd_out = 'visualize_images/'

    pad = 100
    show_neighbor = 2

    visualize_clsf_res(score_file=res_mat_fn, ft_mat_fn=ft_mat_fn, test_annot_fn=test_annot_fn,
                       split_fn=split_file, ontology_fn=ontoloty_fn, info_fn=info_fn,
                       res_fd=res_fd, html_fn=html_name, imfdin=im_fd_in, imfdout=im_fd_out,
                       pad=pad, show_neighbor=show_neighbor)


if __name__ == '__main__':
    main()
