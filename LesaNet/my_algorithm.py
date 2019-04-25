# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains codes of certain algorithmic functions.
# --------------------------------------------------------

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

from config import config, default


def select_triplets_multilabel(emb, target):
    target = target.detach().cpu().numpy()
    num_smp = len(target)
    inters = np.matmul(target, target.T)
    iou = 1 - squareform(pdist(target, 'jaccard'))
    sim = inters * iou + np.diag(np.nan*np.ones((num_smp,)))
    # sim = inters + np.diag(np.nan*np.ones((num_smp,)))

    triplets = []
    for p in range(config.TRAIN.NUM_TRIPLET):
        a = np.random.choice(num_smp)
        p_candidates = np.where(sim[a] >= config.TRAIN.SIMILAR_LABEL_THRESHOLD)[0]
        # p = np.random.choice(np.setdiff1d(range(num_smp), [a]))
        if len(p_candidates) == 0:
            p = np.nanargmax(sim[a])
        else:
            p = np.random.choice(p_candidates)
        if sim[a, p] - np.nanmin(sim[a]) <= config.TRAIN.DISSIMILAR_LABEL_THRESHOLD:
            n = np.nanargmin(sim[a])
            # if sim[a,p] == sim[a,n]:
            #     continue
        else:
            n_candidates = np.where(sim[a] < sim[a, p] - config.TRAIN.DISSIMILAR_LABEL_THRESHOLD)[0]
            n = np.random.choice(n_candidates)
        triplets.append([a, p, n])

    A = emb[[triplet[0] for triplet in triplets]]
    P = emb[[triplet[1] for triplet in triplets]]
    N = emb[[triplet[2] for triplet in triplets]]
    return A, P, N
