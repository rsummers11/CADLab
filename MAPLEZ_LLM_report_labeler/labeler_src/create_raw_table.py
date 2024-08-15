# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-07-12
# Use `python <python_file> --help` to check inputs and outputs
# Description:
# Files that converts outputs from labelers and classifiers into
# scores (precision, recall, f1, mae or auc) with confidence intervals
# and p-values comparing the scores for the proposed labeler/classifier 
# MAPLEZ against the baselines using bootstrapping

from calculate_auc_v2 import main as get_labels
from types import SimpleNamespace
import pandas as pd
import numpy as np
from collections import defaultdict
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score,f1_score, confusion_matrix, accuracy_score, precision_recall_curve
from pathlib import Path
from scipy.interpolate import interp1d
from sortedcollections import OrderedSet
import random
import sys
import copy

from scipy.stats._resampling import _bootstrap_iv
from scipy.stats._resampling import _bootstrap_resample
from scipy.stats._resampling import _bca_interval
from scipy.stats._resampling import _percentile_along_axis
from scipy.stats._resampling import BootstrapResult
from scipy.stats._resampling import ConfidenceInterval

def bootstrap(data, statistic, *, n_resamples=9999, batch=None,
              vectorized=None, paired=False, axis=0, confidence_level=0.95,
              alternative='two-sided', method='BCa', bootstrap_result=None,
              random_state=None):
    # Input validation
    args = _bootstrap_iv(data, statistic, vectorized, paired, axis,
                         confidence_level, alternative, n_resamples, batch,
                         method, bootstrap_result, random_state)
    (data2, statistic2, vectorized, paired, axis, confidence_level,
     alternative, n_resamples, batch, method, bootstrap_result,
     random_state) = args

    theta_hat_b = ([] if bootstrap_result is None
                   else [bootstrap_result.bootstrap_distribution])

    batch_nominal = batch or n_resamples or 1
    
    for k in range(0, n_resamples, batch_nominal):
        batch_actual = min(batch_nominal, n_resamples-k)
        # Compute bootstrap distribution of statistic
        for batch_index in range(batch_actual):
            rng = np.random.RandomState()
            theta_hat_b.append([statistic(*data, rng = rng)])
    theta_hat_b = np.concatenate(theta_hat_b, axis=-1)

    # Calculate percentile interval
    alpha = ((1 - confidence_level)/2 if alternative == 'two-sided'
             else (1 - confidence_level))
    if method == 'bca':
        interval = _bca_interval(data2, statistic2, axis=-1, alpha=alpha,
                                 theta_hat_b=theta_hat_b, batch=batch)[:2]
        percentile_fun = _percentile_along_axis
    else:
        interval = alpha, 1-alpha

        def percentile_fun(a, q):
            return np.percentile(a=a, q=q, axis=-1)
    # Calculate confidence interval of statistic
    ci_l = percentile_fun(theta_hat_b, interval[0]*100)
    ci_u = percentile_fun(theta_hat_b, interval[1]*100)
    if method == 'basic':  # see [3]
        theta_hat = statistic2(*data2, axis=-1)
        ci_l, ci_u = 2*theta_hat - ci_u, 2*theta_hat - ci_l

    if alternative == 'less':
        ci_l = np.full_like(ci_l, -np.inf)
    elif alternative == 'greater':
        ci_u = np.full_like(ci_u, np.inf)

    return BootstrapResult(confidence_interval=ConfidenceInterval(ci_l, ci_u),
                           bootstrap_distribution=theta_hat_b,
                           standard_error=np.std(theta_hat_b, ddof=1, axis=-1))

n_iterations = 2000
tolerance = sys.float_info.epsilon * 10
def auc_metric(target_column, output_column):
    if len(target_column)==0:
        return float('inf')
    if target_column.std()==0:
        return float('inf')
    return roc_auc_score(target_column, output_column, average=None)
    
def find_full_folder_name(base_folder, partial_name):
    # List all entries in the base folder
    for entry in os.listdir(base_folder):
        # Construct the full path
        full_path = os.path.join(base_folder, entry)
        # Check if this entry is a directory and if the partial name is in this entry's name
        if os.path.isdir(full_path) and partial_name in entry:
            return entry
    return None

def get_if_several_predictions(vector):
    return isinstance(vector,list) or len(vector.shape)>=2

from joblib import Parallel, delayed
import joblib

import hashlib
def get_hashed_seed(global_seed, index):
    hash_input = f"{global_seed}_{index}".encode()
    hash_output = hashlib.sha256(hash_input).hexdigest()
    # Convert the hexadecimal hash to an integer
    return int(hash_output, 16) % (2**32 - 1)

def average_seeds_metric(scores_fn, rng, gt_vector, pred_vector, ids_vector = None):
    several_predictions = get_if_several_predictions(pred_vector)
    score = []
    if several_predictions:
        for index_seed in range(len(pred_vector)):
            sampled_pred = pred_vector[index_seed]
            if ids_vector is None:
                if rng is None:
                    score.append(scores_fn(gt_vector, sampled_pred))
                else:
                    score.append(scores_fn(gt_vector, sampled_pred, rng))
            else:
                if rng is None:
                    score.append(scores_fn(gt_vector, sampled_pred, ids_vector))
                else:
                    score.append(scores_fn(gt_vector, sampled_pred, ids_vector, rng))
        score = sum(score)/len(score)
    else:
        sampled_pred = pred_vector
        if ids_vector is None:
            if rng is None:
                score = scores_fn(gt_vector, sampled_pred)
            else:
                score = scores_fn(gt_vector, sampled_pred, rng)
        else:
            if rng is None:
                score = scores_fn(gt_vector, sampled_pred, ids_vector)
            else:
                score = scores_fn(gt_vector, sampled_pred, ids_vector, rng)
    return score
def calculate_micro_score(gt_vector, pred_vector, score_fn, rng = None):
    if rng is not None:
        sampled_indices = rng.choice(len(gt_vector), size=len(gt_vector), replace=True)
        gt_vector = gt_vector[sampled_indices]
        pred_vector = pred_vector[sampled_indices]
    score = score_fn(gt_vector,pred_vector)
    return score

def calculate_macro_score(dataset_sizes, gt_vector, pred_vector, ids_vector, score_fn, row_weights = None, rng = None):
    if row_weights is None:
        row_weights = np.array([1] * len(dataset_sizes))
    sum_weights = row_weights.sum()
    score = 0.
    for index_dataset, dataset_size in enumerate(dataset_sizes):
        g1 = gt_vector[ids_vector==index_dataset]
        p1 = pred_vector[ids_vector==index_dataset]
        assert(len(g1)==len(p1))
        assert(len(g1)==dataset_size or len(g1)==dataset_size-1)
        if rng is not None:
            sampled_indices = rng.choice(len(g1), size=len(g1), replace=True)
            g1 = g1[sampled_indices]
            p1 = p1[sampled_indices]
        score+=score_fn(g1,p1)*row_weights[index_dataset]/sum_weights
    return score
def flexible_concatenate(target_array, new_array):
    # If the target array is "empty" (in this case, we define "empty" as None)
    if target_array is None:
        # Return the new array as is
        return new_array
    else:
        # Concatenate along a chosen axis if the target array is not None
        # This example uses axis 0, but this can be modified
        return np.concatenate((target_array, new_array), axis=0)

def custom_binary_operation(x, y):
    # interval = {0:[-5,5],10:[0,15],25:[10,40],50:[35,65],75:[60,90],90:[85,100]}[x]
    f_linear_low = interp1d([0,10,25,50,75,90,100], [-5,0,10,35,60,85,95], kind='linear')
    f_linear_high = interp1d([0,10,25,50,75,90,100], [5,15,40,65,90,100,105], kind='linear')
    
    interval = [f_linear_low(x),f_linear_high(x)]
    # 0-5: 0
    # 0-15: 10
    # 10-40: 25
    # 35-65: 50
    # 60-90: 75
    # 85-100: 90
    if y<interval[0]:
        return interval[0]-y
    if y>interval[1]:
        return y-interval[1]
    return 0
def mae_score(x,y):
    return np.vectorize(custom_binary_operation)(x,y).mean()

def get_items_from_argparse(args, argparse_string):
    # Create an empty dictionary to store the entries
    result_dict = {}

    # Iterate through the arguments
    for arg_name, arg_value in vars(args).items():
        # Check if the argument starts with 'prediction_file_'
        if arg_name.startswith(argparse_string):
            # Split the argument name by '_' and extract the keys
            keys = arg_name.split('_')[2:]
            # Get the parent dictionary for the keys
            parent_dict = result_dict
            for key in keys[:-1]:
                # If the key is not in the parent dictionary, create an empty dictionary for it
                if key not in parent_dict:
                    parent_dict[key] = {}
                # Move to the next level in the dictionary hierarchy
                parent_dict = parent_dict[key]
            # Assign the argument value to the dictionary with the extracted keys
            parent_dict[keys[-1]] = arg_value

    return result_dict

def score_statistic(data1,data2,score_fn):
    data2 = np.array(data2)
    if len(data2.shape)>2:
        data2 =  np.transpose(data2, (1, 0, 2))
    
    if len(data1.shape)>1 and (data1.shape[0]==n_iterations or data1.shape[0]-1==data1.shape[-1]) :
        scores = []
        for index_perm in range(data1.shape[0]):
            d1 = data1[index_perm]
            d2 = data2[index_perm]
            scores.append(score_fn(d1,d2))
        return np.array(scores)
    else:
        d1 = data1
        d2 = data2
        return score_fn(d1, d2)


def score_statistic_aggregate(data1,data2,data3,rng,score_fn):
    data1 = np.array(data1)
    data2 = np.array(data2)
    data3 = np.array(data3)
    if len(data1.shape)>2:
        data1 =  np.transpose(data1, (1, 0, 2))
    if len(data2.shape)>2:
        data2 =  np.transpose(data2, (1, 0, 2))
    if len(data3.shape)>2:
        data3 =  np.transpose(data3, (1, 0, 2))
    if len(data1.shape)>1 and (data1.shape[0]==n_iterations or data1.shape[0]-1==data1.shape[-1]):
        scores = []
        for index_perm in range(data1.shape[0]):
            d1 = data1[index_perm]
            d2 = data2[index_perm]
            d3 = data3[index_perm]
            scores.append(score_fn(d1,d2,d3,rng))
        return np.array(scores)
    else:
        d1 = data1
        d2 = data2
        d3 = data3
        return score_fn(d1, d2, d3,rng)

def score_statistic_difference(data1,data2, axis = None, score_fn = None):
    if len(data1.shape)>1 and (data1.shape[0]==n_iterations or data1.shape[0]-1==data1.shape[-1]):
        scores = []
        for index_perm in range(data1.shape[0]):
            d1 = data1[index_perm]
            d2 = data2[index_perm]
            scores.append(score_fn(d1) - score_fn(d2))
        return np.array(scores)
    else:
        d1 = data1
        d2 = data2
        return score_fn(d1) - score_fn(d2)
                                    
def main(args):
    group_by_abnormality = True
    location_labels_allowed = {}   

    prediction_files_dict = get_items_from_argparse(args, 'prediction_file_')
    groundtruth_csv_dict = get_items_from_argparse(args, 'groundtruth_csv_')

    if ('other_modalities' in args.type_annotation):
        from convert_annotations_ct_mri import main as build_from_annotation_files_
        
        _labelers = ['llm']
        if 'location' in args.type_annotation:
            datasets = ['ct','mri']
        else:
            datasets = ['ct','mri','pet']
        
        abnormalities =  ['lung lesion', 'liver lesion', 'kidney lesion', 'adrenal gland abnormality','pleural effusion', 'hypermetabolic abnormality in the thorax', 'hypermetabolic abnormality in the abdomen', 'hypermetabolic abnormality in the pelvis']
        model_to_compare_against = []
        aggregation_datasets = ['ct','mri','pet','all']
        if args.do_macro_scores:
            aggregation_datasets += ['all_macro']
        vqa_forbidden_datasets = ['']
        output_filename = f'{args.output_folder}/{args.type_annotation}_table_raw_v2.csv'
        score_type_dict = {'f1':f1_score, 'precision': precision_score, 'recall': recall_score}
        extra_name = ''
        main_score = 'f1'
        if 'location' in args.type_annotation:
            word_type = 'loc'
            datasets = ['ct','mri']
        else:
            word_type = 'labeling'
            datasets = ['ct','mri','pet']
        def build_from_annotation_files(labeler, dataset):
            assert(labeler=='llm')
            if 'location' in args.type_annotation:
                type_annotation_string = 'location'
            else:
                type_annotation_string = 'labels'
            return build_from_annotation_files_(type_annotation_string, args.output_folder, dataset, prediction_files_dict[dataset][labeler], groundtruth_csv_dict[dataset])
        location_labels_allowed = {
            'lung lesion': ['right', 'left', 'upper','lower','middle'],
            'liver lesion':['right', 'left'],
            'kidney lesion':['right', 'left'],
            'adrenal gland abnormality':['right', 'left'],
            'pleural effusion':['right', 'left'],
        }
        str_labels_location = list(OrderedSet([item for sublist in location_labels_allowed.values() for item in sublist]))

    if (args.type_annotation=='classifier'):
        _labelers = {'llm_model':['test_llm_68295',
            'test_llm_71272',
            'test_llm_24023',],
            'no_loc': ['test_llm_no_loc_73290',
            'test_llm_no_loc_42251',
            'test_llm_no_loc_81034',],
            'generic':['test_llm_generic_76772',
            'test_llm_generic_57922',
            'test_llm_generic_13962',],
            'vqa_model':['test_vqa_73780',
            'test_vqa_64412',
            'test_vqa_58139',],
            '3notignore':['test_llm_3notignore_62604',
            'test_llm_3notignore_82272',
            'test_llm_3notignore_35673',],
            'labels':['test_llm_labels_40256',
            'test_llm_labels_16701',
            'test_llm_labels_28757',],
            'chexpert_model':['test_chexpert_90577',
            'test_chexpert_65897',
            'test_chexpert_34197'],
            'chexbert_model':['test_chexbert_71774',
                              'test_chexbert_26693',
                              'test_chexbert_14852'],
            'all_changes':['test_llm_all_changes_54847',
            'test_llm_all_changes_61681',
            'test_llm_all_changes_61785']}
        model_to_compare_against = ['llm_model']
        str_labels_mimic = [
            'cardiomegaly',
            'lung opacity',
            'lung edema',
            'consolidation',
            'atelectasis',
            'pneumothorax',
            'pleural effusion',
            'fracture']
        datasets = ['reflacx','chexpert','pneumonia','pneumothorax']
        score_type_dict = {'auc':auc_metric}
        main_score = 'auc'
        aggregation_datasets = ['reflacx','chexpert','all']
        if args.do_macro_scores:
            aggregation_datasets += ['all_macro']
        vqa_forbidden_datasets = ['']
        output_filename = f'{args.output_folder}/models_table_raw_v2.csv'
        abnormalities = ['atelectasis', 'cardiomegaly', 'consolidation', 'lung edema', 'fracture',\
                        'lung opacity', 'pleural effusion', 'pneumothorax']
        def build_from_annotation_files(labeler, dataset):
            folder_test_files = './classifier_tests_outputs/'
            to_return = {}
            for folder_name in _labelers[labeler]:
                full_folder_name = find_full_folder_name(folder_test_files, folder_name)
                if dataset in ['pneumonia','pneumothorax']:
                    dataset_in_filename = 'nih' + dataset +'_val'
                    if dataset=='pneumonia':
                        abnormalities_in_this_case = ['consolidation']
                    else:
                        abnormalities_in_this_case = [dataset]
                else:
                    dataset_in_filename = 'val_' + dataset 
                    abnormalities_in_this_case = abnormalities
                if dataset=='chexpert':
                    dataset_in_filename = dataset_in_filename + 'dataset'
                for abnormality in abnormalities_in_this_case:
                    pred_annot_filename = folder_test_files + full_folder_name + '/' + dataset_in_filename + ('' if dataset in ['pneumonia','pneumothorax'] else ('_' + str(str_labels_mimic.index(abnormality))))+ '_model_outputs.csv'
                    loaded_predictions_annots = pd.read_csv(pred_annot_filename)
                    this_annot = (loaded_predictions_annots['annot'].values != 0)*1
                    this_pred = loaded_predictions_annots['pred'].values
                    if abnormality in to_return:
                        assert((this_annot==to_return[abnormality][0]).all())
                        to_return[abnormality] = (this_annot, \
                             to_return[abnormality][1] + [this_pred])
                    else:
                        to_return[abnormality] = (this_annot, [this_pred])
            return to_return

    if ('human' in args.type_annotation):
        datasets = ['reflacx_phase12']
        abnormalities = ['cardiomegaly', 'consolidation', 'lung edema', 
                        'lung opacity', 'pneumothorax']
        _labelers = {'llm':[''], 'human':['r1','r2']}
        output_filename = f'{args.output_folder}/{args.type_annotation}_table_raw_human_v2.csv'
        if 'probability' in args.type_annotation:
            score_type_dict = {'mae':mae_score}
            main_score = 'mae'
            word_type = 'probs'
        else:
            score_type_dict = {'f1':f1_score, 'precision': precision_score, 'recall': recall_score}
            main_score = 'f1'
            word_type = 'labeling'
        limit_to_label_agreement = False
        
        extra_name = ''
        
        aggregation_datasets = ['reflacx_phase12']
        if args.do_macro_scores:
            aggregation_datasets += ['all_macro']
        vqa_forbidden_datasets = ['human', 'pneumothorax', 'pneumonia','nih','all']
        if args.do_macro_scores:
            vqa_forbidden_datasets += ['all_macro']
        model_to_compare_against = ['llm']
        def build_from_annotation_files(labeler, dataset):
            assert(dataset=='reflacx_phase12')
            folder_csv_files = args.output_folder
            to_return = {}
            for folder_name_suffix in _labelers[labeler]:
                # reflacx_phase12_dataset_converted_r1
                loaded_predictions = pd.read_csv(folder_csv_files + 'reflacx_phase12_dataset_converted_' + folder_name_suffix + '.csv')
                # reflacx_phase12_dataset_converted.csv
                loaded_annotations = pd.read_csv(folder_csv_files + 'reflacx_phase12_dataset_converted.csv')
                if 'probability' in args.type_annotation:
                    loaded_predictions = loaded_predictions[loaded_predictions['type_annotation']=='probability']
                    loaded_annotations = loaded_annotations[loaded_annotations['type_annotation']=='probability']
                    loaded_predictions = pd.merge(loaded_annotations['subjectid_studyid'], loaded_predictions, on='subjectid_studyid', how='inner') 
                else:
                    loaded_predictions = loaded_predictions[loaded_predictions['type_annotation']=='labels']
                    loaded_annotations = loaded_annotations[loaded_annotations['type_annotation']=='labels']
                    loaded_predictions = pd.merge(loaded_annotations['subjectid_studyid'], loaded_predictions, on='subjectid_studyid', how='inner') 
                for abnormality in abnormalities:
                    if 'probability' not in args.type_annotation:
                        this_pred = (loaded_predictions[abnormality].values!=0)*1
                        this_annot = (loaded_annotations[abnormality].values!=0)*1
                    else:
                        this_pred = loaded_predictions[abnormality].values
                        this_annot = loaded_annotations[abnormality].values
                    if abnormality in to_return:
                        assert((this_annot==to_return[abnormality][0]).all())
                        to_return[abnormality] = (this_annot,  \
                                                  [to_return[abnormality][1]] + [this_pred])
                    else:
                        to_return[abnormality] = (this_annot, this_pred)
                
            return to_return
    
    if (args.type_annotation=='location'):
        _labelers = ['llm', 'vqa', 'llm_generic']
        datasets = ['mimic']
        abnormalities = ['atelectasis', 'consolidation', 'lung edema', 'fracture',\
                        'lung opacity', 'pleural effusion', 'pneumothorax']
        extra_name = f'{"_agree" if args.limit_to_label_agreement else ""}{"_limited" if args.limit_location_vocabulary else ""}'
        output_filename = f'{args.output_folder}/location_table_raw{extra_name}_v2.csv'
        score_type_dict = {'f1':f1_score, 'precision': precision_score, 'recall': recall_score}
        limit_to_label_agreement = args.limit_to_label_agreement
        main_score = 'f1'
        word_type = 'loc'
        aggregation_datasets = ['all']
        if args.do_macro_scores:
            aggregation_datasets += ['all_macro']
        vqa_forbidden_datasets = ['human', 'pneumothorax', 'pneumonia','nih']
        model_to_compare_against = ['llm']
        location_labels_allowed = {
            'enlarged cardiomediastinum':['right', 'left', 'upper', 'lower', 'base', 'ventricle', 'atrium'],
            'cardiomegaly':['right', 'left', 'upper', 'lower', 'base', 'ventricle', 'atrium'],
            'lung lesion':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
            'lung opacity':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
            'lung edema':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
            'consolidation':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
            'atelectasis':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
            'pneumothorax':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
            'pleural effusion':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac', 'fissure'],
            'pleural other':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac', 'fissure'],
            'fracture':['middle', 'right', 'left', 'upper', 'lower', 'lateral', 'posterior', 'anterior', 'rib', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'clavicular', 'spine'],
            'support devices':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'pleural', 'chest wall', 'ventricle', 'atrium', 'svc', 'below the diaphragm', 'jugular', 'above the carina', 'cavoatrial', 'stomach','endotracheal','pic','subclavian', 'nasogastric', 'enteric', 'duodenum']
        }
        str_labels_location = list(OrderedSet([item for sublist in location_labels_allowed.values() for item in sublist]))
        allowed_locations = ['right', 'left','upper', 'lower', 'base', 'apical', 'retrocardiac', 'rib','middle']
        if args.limit_location_vocabulary:
            str_labels_location = allowed_locations

            for abnormality in location_labels_allowed:
                location_labels_allowed[abnormality] = [element for element in allowed_locations if element in location_labels_allowed[abnormality]]

    # severity
    if (args.type_annotation=='severity'):
        _labelers = ['llm', 'vqa', 'llm_generic']
        datasets = ['mimic']
        extra_name = f'{"_agree" if args.limit_to_label_agreement else ""}'
        output_filename = f'{args.output_folder}/severity_table_raw_all_cases{extra_name}_v2.csv'
        abnormalities = ['atelectasis', 'cardiomegaly', 'consolidation', 'lung edema', 'fracture',\
                        'lung opacity', 'pleural effusion', 'pneumothorax']
        score_type_dict = {'f1':f1_score, 'precision': precision_score, 'recall': recall_score}
        
        limit_to_label_agreement = args.limit_to_label_agreement
        main_score = 'f1'
        word_type = 'sev'
        aggregation_datasets = ['all']
        if args.do_macro_scores:
            aggregation_datasets += ['all_macro']
        vqa_forbidden_datasets = ['human', 'pneumothorax', 'pneumonia','nih']
        model_to_compare_against = ['llm']

    #labels
    if (args.type_annotation=='labels'):
        if args.table_origin_weights is not None:
            _labelers = ['llm', 'vicuna', 'llm_generic']
        else:
            _labelers = ['llm', 'vicuna', 'chexbert', 'chexpert', 'vqa', 'llm_generic',  'template']
        if args.include_human_dataset:
            datasets = ['nih', 'mimic', 'reflacx', 'pneumonia', 'pneumothorax']

        else:
            datasets = ['nih', 'mimic']
        abnormalities = ['atelectasis', 'cardiomegaly', 'consolidation', 'lung edema', 'fracture',\
                        'lung opacity', 'pleural effusion', 'pneumothorax']
        if args.table_origin_weights is not None:
            output_filename = f'{args.output_folder}/label_table_raw_newllms_v2.csv'
        else:
            output_filename = f'{args.output_folder}/label_table_raw_v2.csv'
        score_type_dict = {'f1':f1_score, 'precision': precision_score, 'recall': recall_score}
        limit_to_label_agreement = False
        main_score = 'f1'
        extra_name = ''
        word_type = 'labeling'
        if args.include_human_dataset:
            aggregation_datasets = ['nih','mimic','reflacx','human','all']
        else:
            aggregation_datasets = ['nih','mimic','all']
        if args.do_macro_scores:
            aggregation_datasets += ['all_macro']
        vqa_forbidden_datasets = ['human', 'pneumothorax', 'pneumonia','nih','all',\
            'label_atelectasis','label_cardiomegaly','label_consolidation', \
                'label_lung edema', 'label_lung opacity', 'label_fracture', 'label_pleural effusion', 'label_pneumothorax']
        if args.do_macro_scores:
            vqa_forbidden_datasets += ['all_macro']
        model_to_compare_against = ['llm']
    #probability
    if (args.type_annotation=='probability'):
        aggregation_datasets = ['all']
        if args.do_macro_scores:
            aggregation_datasets += ['all_macro']
        _labelers = ['llm', 'vqa', 'llm_generic']
        datasets = ['reflacx', 'reflacx_phase12']
        abnormalities = ['cardiomegaly', 'consolidation', 'lung edema',\
                        'lung opacity', 'pneumothorax']
        output_filename = f'{args.output_folder}/probability_table_raw_v2.csv'
        model_to_compare_against = ['llm']     
        score_type_dict = {'mae':mae_score}
        limit_to_label_agreement = False
        main_score = 'mae'
        extra_name = ''
        word_type = 'probs'
        vqa_forbidden_datasets = ['human', 'pneumothorax', 'pneumonia','nih']
    def labelers(dataset):
        if args.only_llm:
            return ['llm']
        to_return = []
        for labeler in _labelers:
            if labeler=='vqa' and dataset in vqa_forbidden_datasets:
                continue
            if labeler=='vicuna' and dataset in ['reflacx_phase12']:
                continue
            to_return.append(labeler)
        return to_return
    
    def get_row_var(row, dataset):
        count = 0.
        total = 0.
        for labeler in labelers(dataset):
            if labeler != 'vqa' or row[f'vqa_{main_score}_var'] == row[f'vqa_{main_score}_var']:
                total += row[f'{labeler}_{main_score}_var_normalized']
                count += 1
        return total/count

    results = {}
    bootstrapped_scores = {}
    bootstrapped_scores_concatenated = {}
    macro_gts = {}
    macro_weights = {}
    macro_preds = {}
    for dataset in datasets:
        results[dataset] = {}
        if dataset=='nih':
            groundtruth = groundtruth_csv_dict['nih']
            dataset_arg = 'nih'
            use_relabeled = None
        if dataset=='mimic':
            groundtruth = groundtruth_csv_dict['mimic']
            dataset_arg = 'mimic'
            use_relabeled = None
        if dataset=='reflacx':
            groundtruth = groundtruth_csv_dict['reflacx']
            dataset_arg = 'mimic'
            use_relabeled = dataset
        if dataset=='reflacx_phase12':
            groundtruth = groundtruth_csv_dict['reflacx12']
            dataset_arg = 'mimic'
            use_relabeled = dataset
        if dataset=='pneumonia':
            groundtruth = groundtruth_csv_dict['pneumonia']
            dataset_arg = 'nih'
            use_relabeled = dataset
        if dataset=='pneumothorax':
            groundtruth = groundtruth_csv_dict['pneumothorax']
            dataset_arg = 'nih'
            use_relabeled = dataset
        list_of_labelers = labelers(dataset)
        
        for labeler in list_of_labelers:
            if labeler in ['chexpert', 'vicuna', 'vqa', 'llm', 'llm_generic', 'chexbert', 'template'] and 'dataset_arg' in locals():
                results[dataset][labeler] = {}
                done_before = False
                for abnormality in abnormalities:
                    file_path = Path(f'{args.output_folder}/{abnormality}_{labeler}_{dataset}_{word_type}_outputs{extra_name}.csv')
                    done_before = done_before or file_path.exists()
                if not done_before:
                    if dataset=='pet':
                        continue
                    args2 = SimpleNamespace(type_annotation=("probability" if (args.type_annotation=='probability') else "labels"), \
                                        run_labels = (args.type_annotation=='labels' or args.type_annotation=='probability' or args.type_annotation=='human_label'), \
                                            run_location=(args.type_annotation=='location'),\
                                            run_severity= (args.type_annotation=='severity'),\
                        groundtruth = groundtruth, labeler = labeler, dataset = dataset_arg, use_relabeled = use_relabeled, 
                        limit_to_label_agreement = limit_to_label_agreement, \
                        limit_location_vocabulary = args.limit_location_vocabulary, \
                        allow_location_replacements = not args.limit_location_vocabulary,
                        single_file = prediction_files_dict[dataset_arg]['llmgeneric' if labeler == 'llm_generic' else labeler],
                        prediction_file_mimic_vqa = args.prediction_file_mimic_vqa,
                        prediction_file_mimic_llm = args.prediction_file_mimic_llm,
                        prediction_file_mimic_llmgeneric = args.prediction_file_mimic_llm,
                        groundtruth_csv_pneumothorax = args.groundtruth_csv_pneumothorax,
                        groundtruth_csv_pneumonia = args.groundtruth_csv_pneumonia,
                        groundtruth_csv_reflacx = args.groundtruth_csv_reflacx,
                        groundtruth_csv_reflacx12 = args.groundtruth_csv_reflacx12,
                        prediction_file_nih_vicuna = args.prediction_file_nih_vicuna,
                        prediction_file_mimic_vicuna = args.prediction_file_mimic_vicuna,
                        prediction_file_nih_chexbert = args.prediction_file_nih_chexbert,
                        prediction_file_mimic_chexbert = args.prediction_file_mimic_chexbert,
                        prediction_file_nih_template = args.prediction_file_nih_template,
                        prediction_file_mimic_template = args.prediction_file_mimic_template
                        )
                    results[dataset][labeler] = get_labels(args2)
                    for abnormality in abnormalities:
                        if (abnormality+'_annotpred') in results[dataset][labeler]:
                            list1, list2= results[dataset][labeler][abnormality+'_annotpred']
                            df = pd.DataFrame({
                                'annot': list1,
                                'pred': list2
                            })

                            csv_file_path = f'{args.output_folder}/{abnormality}_{labeler}_{dataset}_{word_type}_outputs{extra_name}.csv' 
                            df.to_csv(csv_file_path, index=False)  
                else:
                    for abnormality in abnormalities:
                        file_path = Path(f'{args.output_folder}/{abnormality}_{labeler}_{dataset}_{word_type}_outputs{extra_name}.csv')
                        if file_path.exists():
                            csv_file_path = f'{args.output_folder}/{abnormality}_{labeler}_{dataset}_{word_type}_outputs{extra_name}.csv' 
                            df = pd.read_csv(csv_file_path)
                            # Convert DataFrame columns to numpy arrays
                            array1 = df['annot'].to_numpy()
                            array2 = df['pred'].to_numpy()
                            results[dataset][labeler][abnormality] = array1, array2
            else:
                results[dataset][labeler] = build_from_annotation_files(labeler, dataset)
    
    file_path_table = Path(output_filename)
    if file_path_table.exists():
        df_table = pd.read_csv(output_filename)
    else:
        df_table = pd.DataFrame()
    for abnormality in abnormalities:
        print(abnormality)
        for dataset in datasets:
            row = {'abnormality': abnormality, 'dataset':dataset, 'n':None, 'n_pos':None, 'human': 1 if dataset in ['reflacx','pneumonia','pneumothorax'] else 0}
            values_to_check = {'abnormality': [abnormality], 'dataset':[dataset]}
            if 'abnormality' in df_table.columns and 'dataset' in df_table.columns and df_table.loc[:, list(values_to_check.keys())].isin(values_to_check).all(axis=1).any():
                continue
            print(dataset)
            for labeler in labelers(dataset):
                print(labeler)
                if abnormality in results[dataset][labeler]:
                    gt_vector,pred_vector = results[dataset][labeler][abnormality] 

                    if 'location' in args.type_annotation and dataset!='mri':

                        indices_to_keep = np.array([location in location_labels_allowed[abnormality] for location in str_labels_location])
                        repeat_factor = len(gt_vector) // len(indices_to_keep)
                        repeated_indices = np.tile(indices_to_keep, repeat_factor)
                        print(len(indices_to_keep), repeat_factor, len(repeated_indices), len(gt_vector))
                        gt_vector = gt_vector[repeated_indices]
                        pred_vector = pred_vector[repeated_indices]
                        assert(len(pred_vector)==len(gt_vector))

                    n_pos = (gt_vector>0).sum()
                    if n_pos==0:
                        continue
                    if row['n_pos'] is None:
                        row['n_pos'] = n_pos
                    else:
                        assert(row['n_pos'] == n_pos)
                    if row['n'] is None:
                        row['n'] = len(gt_vector)
                    else:
                        assert(row['n'] == len(gt_vector))
                    for score in score_type_dict:
                        import scipy
                        bootstrap_fn_1 = lambda g1, p1: average_seeds_metric(score_type_dict[score], None, g1, p1)
                        bootstrap_fn = lambda data1, data2, axis: score_statistic(data1, data2, bootstrap_fn_1)
                        row[f'{labeler}_{score}_value'] =  bootstrap_fn(gt_vector,pred_vector, None)
                        res = scipy.stats.bootstrap((gt_vector,pred_vector), bootstrap_fn, n_resamples=n_iterations, paired = True, axis = -1, vectorized = True)
                        row[f'{labeler}_{score}_low'] = res.confidence_interval.low
                        row[f'{labeler}_{score}_high'] = res.confidence_interval.high
                        row[f'{labeler}_{score}_var'] = res.standard_error**2
                        if not row[f'{labeler}_{score}_low']==row[f'{labeler}_{score}_low']:
                            row[f'{labeler}_{score}_low']=row[f'{labeler}_{score}_value']
                        if not row[f'{labeler}_{score}_high']==row[f'{labeler}_{score}_high']:
                            row[f'{labeler}_{score}_high']=row[f'{labeler}_{score}_value']
                        row[f'{labeler}_{score}_median'] = (row[f'{labeler}_{score}_low']+row[f'{labeler}_{score}_high'])


                    for labeler2 in model_to_compare_against:
                        if labeler == labeler2:
                            continue
                        if labeler2 in results[dataset]:
                            gt_vector2,pred_vector2 = results[dataset][labeler2][abnormality] 
                            if 'location' in args.type_annotation and dataset!='mri':
                                gt_vector2 = gt_vector2[repeated_indices]
                                pred_vector2 = pred_vector2[repeated_indices]
                                assert(len(pred_vector2)==len(gt_vector2))

                            for score in score_type_dict:
                                import scipy
                                bootstrap_fn_1 = lambda g1, p1: average_seeds_metric(score_type_dict[score], None, g1, p1)
                                bootstrap_fn = lambda data1, data2, axis: score_statistic_difference(data1, data2, axis, lambda d1: bootstrap_fn_1(gt_vector, d1))
                                print('oi9', np.array(pred_vector).shape,np.array(pred_vector2).shape)
                                res = scipy.stats.permutation_test((np.array(pred_vector),np.array(pred_vector2)),bootstrap_fn , permutation_type='samples',
                                    vectorized=True, n_resamples=n_iterations, axis = -1)
                                row[f'{labeler}_{score}_p'] = res.pvalue

            if row['n_pos'] is not None:
                for labeler in labelers(dataset):
                    if labeler=='vqa' and dataset not in ['mimic', 'reflacx', 'reflacx_phase12']:
                        continue
                    if labeler=='vicuna' and dataset =='reflacx_phase12':
                        continue
                    row[f'{labeler}_{main_score}_var_normalized'] = row[f'{labeler}_{main_score}_var']/((row[f'{labeler}_{main_score}_median']**2)+1e-20)
                row['row_var'] = get_row_var(row, dataset)
                row['row_unnormalized_weight'] = 1/(row['row_var']+1e-20)
                df_table = pd.concat([df_table, pd.DataFrame(row, index=[0])], ignore_index=True)
                df_table.to_csv(output_filename, index=False)

    
    if group_by_abnormality:
        aggregation_datasets = ['label_' + abnormality for abnormality in abnormalities] + aggregation_datasets
    
    #grouped by label or dataset
    for dataset in aggregation_datasets:
        print(dataset)
        if dataset in ['pneumonia', 'pneumothorax']:
            continue
        if dataset[:6]=='label_':
            this_df_table = df_table[df_table['abnormality']==dataset[6:]]
        elif dataset=='human':
            this_df_table = df_table[df_table['human']==1]
        elif dataset=='all' or dataset=='all_macro':
            this_df_table = df_table
        else:
            this_df_table = df_table[df_table['dataset']==dataset]
        if dataset=='all' or dataset=='all_macro' or dataset[:6]=='label_':
            if 'human' not in args.type_annotation:
                this_df_table = this_df_table[this_df_table['dataset']!='reflacx_phase12']
        this_df_table = this_df_table[this_df_table['abnormality']!='-']
        this_df_table = this_df_table[this_df_table['n_pos']>10]
        if args.table_origin_weights is not None:
            weights_table = pd.read_csv(args.table_origin_weights)
            weights_table = pd.merge(this_df_table[['abnormality', 'dataset']], weights_table, on = ['abnormality', 'dataset'])
            this_df_table = this_df_table.reset_index(drop=True)
            weights_table = weights_table.reset_index(drop=True)

            assert(len(weights_table)==len(this_df_table))
            total_weights = weights_table['row_unnormalized_weight'].values.sum()
            this_df_table['this_result_row_weight'] = weights_table['row_unnormalized_weight']/total_weights
        else:
            total_weights = this_df_table['row_unnormalized_weight'].values.sum()
            this_df_table['this_result_row_weight'] = this_df_table['row_unnormalized_weight']/total_weights
        row = {'dataset': dataset, 'abnormality':'-'}
        values_to_check = {'abnormality': [row['abnormality']], 'dataset':[row['dataset']]}
        if df_table.loc[:, list(values_to_check.keys())].isin(values_to_check).all(axis=1).any():
            continue
        
        
        if len(this_df_table)>1:
            rng_seed = random.randint(0, 2**32 - 1)
            for labeler in labelers(dataset):
                print(labeler)
                for score in score_type_dict:
                    weights = np.array([])
                    gt_vector = np.array([])
                    dataset_ids = np.array([])
                    dataset_sizes = np.array([])
                    row_weights = np.array([])
                    pred_vector = None
                    for index_row, old_row in this_df_table.reset_index().iterrows():
                        this_gt_vector,this_pred_vector = results[old_row['dataset']][labeler][old_row['abnormality']] 
                        
                        row_weights = np.hstack((row_weights, [old_row['this_result_row_weight']]))
                        
                        if 'location' in args.type_annotation and old_row['dataset']!='mri':
                            indices_to_keep = np.array([location in location_labels_allowed[old_row['abnormality']] for location in str_labels_location])
                            repeat_factor = len(this_gt_vector) // len(indices_to_keep)
                            repeated_indices = np.tile(indices_to_keep, repeat_factor)
                            this_gt_vector = this_gt_vector[repeated_indices]
                            this_pred_vector = this_pred_vector[repeated_indices]
                            assert(len(this_gt_vector)==len(this_pred_vector))
                        if not args.do_micro_scores:
                            assert(old_row['n']==len(this_gt_vector))
                            weights = np.hstack((weights, [old_row['this_result_row_weight']/old_row['n']] * len(this_gt_vector)))
                        dataset_size = len(this_gt_vector)
                        dataset_sizes = np.hstack((dataset_sizes, [dataset_size]))
                        dataset_ids = np.hstack((dataset_ids, [index_row] * len(this_gt_vector)))     
                        gt_vector = np.hstack((gt_vector,this_gt_vector))
                        if isinstance(this_pred_vector,list):
                            for index_preds in range(len(this_pred_vector)):
                                if pred_vector is None:
                                    pred_vector = []
                                if pred_vector is not None and len(pred_vector)<index_preds+1:
                                    pred_vector.append(None)
                                pred_vector[index_preds] = flexible_concatenate(pred_vector[index_preds],this_pred_vector[index_preds])
                        else:
                            pred_vector = flexible_concatenate(pred_vector,this_pred_vector)
                    
                    if args.do_micro_scores and not dataset=='all_macro':
                        bootstrap_fn_1 = lambda g1, p1, i1, rng = None: calculate_micro_score(g1, p1, i1, score_type_dict[score], rng = rng)
                    else:
                        bootstrap_fn_1 = lambda g1, p1, i1, rng = None: calculate_macro_score(dataset_sizes, g1, p1, i1, score_type_dict[score], row_weights if dataset!='all_macro' else None, rng = rng)
                    bootstrap_fn_2 = lambda g1, p1, i1, rng=None: average_seeds_metric(bootstrap_fn_1, rng, g1, p1, i1)
                    bootstrap_fn = lambda data1, data2, data3, axis=-1, rng=None: score_statistic_aggregate(data1, data2, data3, rng, bootstrap_fn_2)
                    import scipy
                    row[f'{labeler}_{score}_value'] =  bootstrap_fn(gt_vector,pred_vector,dataset_ids)
                    res = bootstrap((gt_vector,pred_vector,dataset_ids), bootstrap_fn, n_resamples=n_iterations, paired = True, vectorized = True, axis = -1)
                    row[f'{labeler}_{score}_low'] = res.confidence_interval.low
                    
                    row[f'{labeler}_{score}_high'] = res.confidence_interval.high
                    row[f'{labeler}_{score}_var'] = res.standard_error**2
                    if not row[f'{labeler}_{score}_low']==row[f'{labeler}_{score}_low']:
                        row[f'{labeler}_{score}_low']=row[f'{labeler}_{score}_value']
                    if not row[f'{labeler}_{score}_high']==row[f'{labeler}_{score}_high']:
                        row[f'{labeler}_{score}_high']=row[f'{labeler}_{score}_value']
                    row[f'{labeler}_{score}_median'] = (row[f'{labeler}_{score}_low']+row[f'{labeler}_{score}_high'])/2
                        
                    for labeler2 in model_to_compare_against:
                        if labeler == labeler2:
                            continue
                        weights2 = np.array([])
                        gt_vector2 = np.array([])
                        pred_vector2 = None
                        for _, old_row in this_df_table.iterrows():
                            this_gt_vector,this_pred_vector = results[old_row['dataset']][labeler2][old_row['abnormality']] 
                            if 'location' in args.type_annotation and old_row['dataset']!='mri':
                                indices_to_keep = np.array([location in location_labels_allowed[old_row['abnormality']] for location in str_labels_location])
                                repeat_factor = len(this_gt_vector) // len(indices_to_keep)
                                repeated_indices = np.tile(indices_to_keep, repeat_factor)
                                this_gt_vector = this_gt_vector[repeated_indices]
                                this_pred_vector = this_pred_vector[repeated_indices]
                                assert(len(this_gt_vector)==len(this_pred_vector))
                            if not args.do_micro_scores:
                                assert(old_row['n']==len(this_gt_vector))
                                weights2 = np.hstack((weights2, [old_row['this_result_row_weight']/old_row['n']] * len(this_gt_vector)))
                            gt_vector2 = np.hstack((gt_vector2,this_gt_vector))
                            if isinstance(this_pred_vector,list):
                                for index_preds in range(len(this_pred_vector)):
                                    if pred_vector2 is None:
                                        pred_vector2 = []
                                    if pred_vector2 is not None and len(pred_vector2)<index_preds+1:
                                        pred_vector2.append(None)
                                    pred_vector2[index_preds] = flexible_concatenate(pred_vector2[index_preds],this_pred_vector[index_preds])
                            else:
                                pred_vector2 = flexible_concatenate(pred_vector2,this_pred_vector)
                        assert((weights2==weights).all())
                        if df_table.loc[:, list(values_to_check.keys())].isin(values_to_check).all(axis=1).any() or len(this_df_table)==1:
                            continue
                        import scipy
                        score_fn = lambda data1, data2, axis: score_statistic_difference(data1, data2, axis, lambda d1: bootstrap_fn_2(gt_vector, d1, dataset_ids))
                        test_pred_vector = np.array(pred_vector)
                        test_pred_vector2 = np.array(pred_vector2)
                        len1 = len(test_pred_vector.shape)
                        len2 = len(test_pred_vector2.shape)

                        # Determine which array is smaller and which is larger
                        if len1 < len2:
                            test_pred_vector = test_pred_vector[None]
                            test_pred_vector = np.tile(test_pred_vector, ((len2 // len1),1))
                        elif len2 < len1:
                            test_pred_vector2 = test_pred_vector2[None]
                            test_pred_vector2 = np.tile(test_pred_vector2, ((len1 // len2), 1))
                        print(test_pred_vector.shape, test_pred_vector2.shape)
                        if len(test_pred_vector.shape)>1:
                            test_pred_vector = test_pred_vector.transpose((1,0))
                        if len(test_pred_vector2.shape)>1:
                            test_pred_vector2 = test_pred_vector2.transpose((1,0))
                            
                        res = scipy.stats.permutation_test((test_pred_vector,test_pred_vector2), score_fn, permutation_type='samples',
                            vectorized=True, n_resamples=n_iterations)
                        row[f'{labeler}_{score}_p'] = res.pvalue

                        print(labeler, score, row[f'{labeler}_{score}_p'])
        if len(this_df_table)==1:
            row = this_df_table.iloc[0].to_dict()
            keys_to_remove = [labeler_3 + '_f1_var_normalized' for labeler_3 in labelers(this_df_table['dataset'].values[0])] + ['row_var', 'row_unnormalized_weight', 'this_result_row_weight', 'n', 'n_pos', 'human']

            row = {k: v for k, v in row.items() if k not in keys_to_remove}

            row['dataset'] = dataset
            row['abnormality'] = '-'

        if not (df_table.loc[:, list(values_to_check.keys())].isin(values_to_check).all(axis=1).any()) and not ((not group_by_abnormality) and dataset[:6]=='label_') and not (len(this_df_table)<=0):

            df_table = pd.concat([df_table, pd.DataFrame(row, index=[0])], ignore_index=True)
            df_table.to_csv(output_filename, index=False)

import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_to_label_agreement", type=str2bool, default="true", help='''If True, only calculated the table for cases that all labelers and ground truths agree. Only used if type_annotation is 'location'.''')
    parser.add_argument("--limit_location_vocabulary", type=str2bool, default="false", help='''If True,  limits the keywords used to evaluate location annotations to this subset: ['right', 'left','upper', 'lower', 'base', 'apical', 'retrocardiac', 'rib','middle']''')
    parser.add_argument("--type_annotation", type=str, choices = ['other_modalities','other_modalities_location', 'classifier',
        'location','human_probability','human_label','severity','probability','labels'], default="labels",\
        help='''The type of annotation/prediction to create a table for.''')
    parser.add_argument("--output_folder", type=str, default="./", help='''The folder where the results are going to be written to.''')

    parser.add_argument("--groundtruth_csv_nih", type=str, default='experiment_test_annotations/nih_human_annotations.csv', help='''''')
    parser.add_argument("--groundtruth_csv_mimic", type=str, default='experiment_test_annotations/mimic_human_annotations.csv', help='''''')
    parser.add_argument("--groundtruth_csv_ct", type=str, default='experiment_test_annotations/ct_human_annotations.csv', help='''''')
    parser.add_argument("--groundtruth_csv_mri", type=str, default='experiment_test_annotations/mri_human_annotations.csv', help='''''')
    parser.add_argument("--groundtruth_csv_pet", type=str, default='experiment_test_annotations/pet_human_annotations.csv', help='''''')
    parser.add_argument("--groundtruth_csv_reflacx", type=str, default='reflacx_dataset_converted.csv', help='''''')
    parser.add_argument("--groundtruth_csv_reflacx12", type=str, default='reflacx_phase12_dataset_converted.csv', help='''''')
    parser.add_argument("--groundtruth_csv_pneumonia", type=str, default='pneumonia_relabeled_dataset_converted.csv', help='''''')
    parser.add_argument("--groundtruth_csv_pneumothorax", type=str, default='pneumothorax_relabeled_dataset_converted.csv', help='''''')
    
    parser.add_argument("--prediction_file_nih_llm", type=str, default='./new_dataset_annotations/nih_llm_annotations_test.csv', help='''''')
    parser.add_argument("--prediction_file_nih_chexbert", type=str, default='./test_nih_chexbert_labels_converted_id_converted.csv', help='''''')
    parser.add_argument("--prediction_file_mimic_chexbert", type=str, default='./mimic_chexbert_labels_converted.csv', help='''''')
    parser.add_argument("--prediction_file_mimic_vqa", type=str, default='./vqa_dataset_converted.csv', help='''''')
    parser.add_argument("--prediction_file_mimic_llm", type=str, default='./new_dataset_annotations/mimic_llm_annotations.csv', help='''''')
    parser.add_argument("--prediction_file_nih_chexpert", type=str, default='./chexpert_nih_dataset_converted.csv', help='''''')
    parser.add_argument("--prediction_file_mimic_chexpert", type=str, default='./chexpert_mimic_dataset_converted.csv', help='''''')
    parser.add_argument("--prediction_file_nih_llmgeneric", type=str, default='./nih_llm_annotations_test_generic.csv', help='''''')
    parser.add_argument("--prediction_file_mimic_llmgeneric", type=str, default='./mimic_llm_annotations_generic.csv', help='''''')
    parser.add_argument("--prediction_file_nih_vicuna", type=str, default='./llama2_vicuna_nih_converted.csv', help='''''')
    parser.add_argument("--prediction_file_mimic_vicuna", type=str, default='./llama2_vicuna_mimic_converted.csv', help='''''')
    parser.add_argument("--prediction_file_nih_template", type=str, default='./template_with_vllm_nih_v2_converted.csv', help='''''')
    parser.add_argument("--prediction_file_mimic_template", type=str, default='./template_with_vllm_mimic_v2_converted.csv', help='''''')
    parser.add_argument("--prediction_file_ct_llm", type=str, default='./parsing_results_llm_ct.csv', help='''''')
    parser.add_argument("--prediction_file_pet_llm", type=str, default='./parsing_results_llm_pet.csv', help='''''')
    parser.add_argument("--prediction_file_mri_llm", type=str, default='./parsing_results_llm_mri.csv', help='''''')

    parser.add_argument("--do_macro_scores", type=str2bool, default='true', help='''''')
    parser.add_argument("--do_micro_scores", type=str2bool, default='false', help='''''')
    parser.add_argument("--table_origin_weights", type=str, default=None, help='''''')

    parser.add_argument("--include_human_dataset", type=str2bool, default="true", help='''''')
    parser.add_argument("--only_llm", type=str2bool, default="false", help='''''')
    args = parser.parse_args()
    main(args)
