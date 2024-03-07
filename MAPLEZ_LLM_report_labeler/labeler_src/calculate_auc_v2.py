# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Description:
# Auxiliary file called by create_raw_table.py.

import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score,f1_score, confusion_matrix, accuracy_score, precision_recall_curve
import argparse
import os
from sortedcollections import OrderedSet
import numpy as np
import re

def main(args):
    if args.use_relabeled == 'pneumothorax':
        columns_to_use = ['pneumothorax']
    elif args.use_relabeled == 'pneumonia':
        columns_to_use = ['consolidation']
    elif args.use_relabeled is not None and 'reflacx' in args.use_relabeled:
        columns_to_use = ['pneumothorax', 'cardiomegaly','lung opacity','consolidation','lung edema']
    else:
        columns_to_use = ['pneumothorax','pleural effusion','fracture', 'cardiomegaly','lung opacity','consolidation','lung edema','atelectasis']
    # missing Enlarged Cardiomediastinum, Lung Lesion, Pleural Other, Support Devices

    columns_to_join_annot = {}

    allowable_indices_loc_sev = {}
    if args.limit_to_label_agreement:
        #label agreement
        #load labels from VQA, LLM, LLMGeneric, Annotation
        vqa_labels = pd.read_csv(args.prediction_file_mimic_vqa)
        vqa_labels = vqa_labels[vqa_labels['type_annotation']=='labels']
        llm_labels = pd.read_csv(args.prediction_file_mimic_llm)
        llm_labels = llm_labels[llm_labels['type_annotation']=='labels']
        llm_generic_labels =pd.read_csv(args.prediction_file_mimic_llmgeneric)
        llm_generic_labels = llm_generic_labels[llm_generic_labels['type_annotation']=='labels']
        if args.use_relabeled is not None and 'reflacx' in args.use_relabeled:
            if args.use_relabeled=='reflacx':
                annot_labels  = pd.read_csv(args.groundtruth_csv_reflacx)
            elif args.use_relabeled=='reflacx_phase12':
                annot_labels  = pd.read_csv(args.groundtruth_csv_reflacx12)
        else:
            annot_labels = pd.read_csv(args.groundtruth)
        annot_labels = annot_labels[annot_labels['type_annotation']=='labels']
        annot_labels= annot_labels.fillna(-2)
        vqa_labels['subjectid_studyid'] = vqa_labels['subjectid_studyid'].astype(str).str.lstrip('_')
        llm_labels['subjectid_studyid'] = llm_labels['subjectid_studyid'].astype(str).str.lstrip('_')
        llm_generic_labels['subjectid_studyid'] = llm_generic_labels['subjectid_studyid'].astype(str).str.lstrip('_')
        annot_labels[['subject_id', 'study_id']] = annot_labels['subjectid_studyid'].str.split('_', expand=True)
        vqa_labels= vqa_labels.fillna(-2)
        for col in vqa_labels.columns[3:]:
            vqa_labels[col] = pd.to_numeric(vqa_labels[col], errors='coerce').astype('float')
        llm_labels= llm_labels.fillna(-2)
        for col in llm_labels.columns[3:]:
            llm_labels[col] = pd.to_numeric(llm_labels[col], errors='coerce').astype('float')
        llm_generic_labels= llm_generic_labels.fillna(-2)
        for col in llm_generic_labels.columns[3:]:
            llm_generic_labels[col] = pd.to_numeric(llm_generic_labels[col], errors='coerce').astype('float')
        vqa_labels[['subject_id', 'study_id']] = vqa_labels['subjectid_studyid'].str.split('_', expand=True)
        llm_labels[['subject_id', 'study_id']] = llm_labels['subjectid_studyid'].str.split('_', expand=True)
        llm_generic_labels[['subject_id', 'study_id']] = llm_generic_labels['subjectid_studyid'].str.split('_', expand=True)
        study_ids = pd.merge(llm_labels['study_id'], vqa_labels['study_id'], on='study_id', how='inner') 
        study_ids = pd.merge(study_ids['study_id'], llm_generic_labels['study_id'], on='study_id', how='inner') 
        study_ids = pd.merge(study_ids['study_id'], annot_labels['study_id'], on='study_id', how='inner') 
        vqa_labels = pd.merge(study_ids['study_id'], vqa_labels, on='study_id', how='inner') 
        llm_labels = pd.merge(study_ids['study_id'], llm_labels, on='study_id', how='inner') 
        llm_generic_labels = pd.merge(study_ids['study_id'], llm_generic_labels, on='study_id', how='inner')
        annot_labels = pd.merge(study_ids['study_id'], annot_labels, on='study_id', how='inner')
        for column_name in annot_labels.columns:
            if column_name in columns_to_use:
                annot_labels[column_name] = pd.to_numeric(annot_labels[column_name])
                vqa_labels[column_name] = pd.to_numeric(vqa_labels[column_name])
                llm_labels[column_name] = pd.to_numeric(llm_labels[column_name])
                llm_generic_labels[column_name] = pd.to_numeric(llm_generic_labels[column_name])
        if args.use_relabeled is not None and args.use_relabeled=='reflacx':
            for column_name in columns_to_use:
                annot_labels.loc[annot_labels[column_name] == -1, column_name] = 1
                vqa_labels.loc[vqa_labels[column_name] == -1, column_name] = 1
                llm_labels.loc[llm_labels[column_name] == -1, column_name] = 1
                llm_generic_labels.loc[vqa_labels[column_name] == -1, column_name] = 1

        for column_name in annot_labels.columns:
            if column_name in columns_to_use:
                # print(len(annot_labels[column_name].values), len(llm_generic_labels[column_name].values), len(llm_labels[column_name].values), len(vqa_labels[column_name].values))
                allowable_indices_loc_sev[column_name] = ((annot_labels[column_name].values==1)*(vqa_labels[column_name].values==1)*(llm_labels[column_name].values==1)*(llm_generic_labels[column_name].values==1)).astype(bool)
                print(column_name,allowable_indices_loc_sev[column_name].sum()) 
    else:
        #all ones because we want to calculate the scores for all cases,
        # not only the ones with agreements
        annot_labels = pd.read_csv(args.groundtruth)
        for column_name in annot_labels.columns:
            if column_name in columns_to_use:
                if args.use_relabeled=='pneumonia':
                    length_indices = 7186
                elif args.use_relabeled=='pneumothorax':
                    length_indices = 24709
                elif args.use_relabeled=='reflacx':
                    length_indices = 506
                elif args.use_relabeled=='reflacx_phase12':
                    length_indices = 109
                elif args.dataset=='mimic':
                    length_indices = 350
                else:
                    length_indices = 200
                allowable_indices_loc_sev[column_name] = np.ones(length_indices).astype(bool)
    
    dict_aucs = {}

    if args.run_labels:
        print(args.single_file, args.groundtruth, args.use_relabeled, args.groundtruth_csv_reflacx)
        preds = pd.read_csv(args.single_file)
        annot = None
        
        if args.use_relabeled=='pneumothorax':
            annot  = pd.read_csv(args.groundtruth_csv_pneumothorax)
        elif args.use_relabeled=='pneumonia':
            annot  = pd.read_csv(args.groundtruth_csv_pneumonia)
        elif args.use_relabeled=='reflacx':
            annot  = pd.read_csv(args.groundtruth_csv_reflacx)
        elif args.use_relabeled=='reflacx_phase12':
            annot  = pd.read_csv(args.groundtruth_csv_reflacx12)
        if args.use_relabeled is not None and 'reflacx' in args.use_relabeled:
            annot = annot[annot['type_annotation']==args.type_annotation]
        if annot is None:
            annot = pd.read_csv(args.groundtruth)
            annot = annot[annot['type_annotation']=='labels']
        annot= annot.fillna(-2)
        
        preds['subjectid_studyid'] = preds['subjectid_studyid'].astype(str).str.lstrip('_')

        if args.dataset=='mimic' and args.use_relabeled!='pneumothorax' and args.use_relabeled!='pneumonia':
            annot[['subject_id', 'study_id']] = annot['subjectid_studyid'].str.split('_', expand=True)
        else:
            annot['study_id'] = annot['subjectid_studyid']
        if args.labeler[:3]=='llm' or args.labeler=='vqa':
            preds = preds[preds['type_annotation']==args.type_annotation]
        preds= preds.fillna(-2)
        for col in preds.columns[3:]:
            preds[col] = pd.to_numeric(preds[col], errors='coerce').astype('float')

        if args.dataset=='mimic' and args.labeler!='vicuna':
            preds[['subject_id', 'study_id']] = preds['subjectid_studyid'].str.split('_', expand=True)
        else:
            preds['study_id'] = preds['subjectid_studyid']
        if args.use_relabeled=='reflacx':
            if args.dataset=='nih':
                vicuna_set = pd.read_csv(args.prediction_file_nih_vicuna)
            elif args.dataset=='mimic':
                vicuna_set = pd.read_csv(args.prediction_file_mimic_vicuna)
            vicuna_set['subjectid_studyid'] = vicuna_set['subjectid_studyid'].astype(str).str.lstrip('_')
            vicuna_set['study_id'] = vicuna_set['subjectid_studyid']
            vicuna_set = vicuna_set['study_id']
            annot = pd.merge(annot, vicuna_set, on='study_id', how='inner')
        annot['study_id'] = annot['study_id'].str.lstrip('./images/').str.rstrip('.png')
        preds['study_id'] = preds['study_id'].str.lstrip('./images/').str.rstrip('.png')
        preds = pd.merge(annot['study_id'], preds, on='study_id', how='inner') 
        annot = pd.merge(preds['study_id'], annot, on='study_id', how='inner') 
        for column_name in annot.columns:
            if column_name in columns_to_use:
                annot[column_name] = pd.to_numeric(annot[column_name])
                preds[column_name] = pd.to_numeric(preds[column_name])

        if args.type_annotation=='labels':
            annot = annot.replace(1, 4)
            annot = annot.replace(-1, 3)
            annot = annot.replace(-3, 2)
            annot = annot.replace(-2, 1)

        if args.type_annotation=='probability':
            if args.labeler[:3]=='llm':
                preds = preds.replace(101, 0)
        else:
            preds = preds.replace(1, 4)
            preds = preds.replace(-1, 3)
            preds = preds.replace(-3, 2)
            preds = preds.replace(-2, 1)

        
        for column_name in annot.columns:
            if column_name in columns_to_use:
                print(preds)
                print(column_name, len(preds[column_name].values), len(allowable_indices_loc_sev[column_name]))
                preds_to_use =  preds[column_name].values[allowable_indices_loc_sev[column_name]]
                annots_to_use = annot[column_name].values[allowable_indices_loc_sev[column_name]]
                if args.use_relabeled is not None and args.type_annotation=='probability' and 'reflacx' in args.use_relabeled:
                    preds_to_use[preds_to_use==-2] = 0
                    def custom_binary_operation(x, y):
                        interval = {0:[-5,5],10:[0,15],25:[10,40],50:[35,65],75:[60,90],90:[85,100]}[y]
                        # 0-5: 0
                        # 0-15: 10
                        # 10-40: 25
                        # 35-65: 50
                        # 60-90: 75
                        # 85-100: 90
                        if x<interval[0]:
                            return interval[0]-x
                        if x>interval[1]:
                            return x-interval[1]
                        return 0
                    # Apply the custom function element-wise
                    dict_aucs[column_name + '_annotpred'] = (annots_to_use, preds_to_use)
                if (not (args.use_relabeled is not None and 'reflacx' in args.use_relabeled)) or args.type_annotation=='labels':

                    annots_to_use[annots_to_use==1] = 0
                    annots_to_use[annots_to_use==3] = 1
                    annots_to_use[annots_to_use==2] = 0
                    annots_to_use[annots_to_use==4] = 1		

                    if args.type_annotation=='labels':		
                        preds_to_use[preds_to_use==1] = 0
                        preds_to_use[preds_to_use==3] = 1
                        preds_to_use[preds_to_use==2] = 0
                        preds_to_use[preds_to_use==4] = 1
                        
                    dict_aucs[column_name + '_annotpred'] = (annots_to_use, preds_to_use)
                    
        return dict_aucs

    if args.run_location:
        list_of_location_labels_per_abnormality = {
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

        list_of_replacements_labels_location = {'leftward': ['left'],
                                                'left-sided':['left'],
                                                'left-side':['left'],
                                                'right-sided':['right'],
                                                'right-side':['right'],
                                                'rightward':['right'],
                                                'infrahilar':['lower'],
                                                'suprahilar':['upper'],
                                                'fissural':['fissure'],
                                                'fissures':['fissure'],
                                'lingula':['left', 'lower'],
                                'lingular':['left', 'lower'],
                                'bilateral': ['right', 'left'], 
                                'central':['perihilar'],
                                'medial':['perihilar'],
                                'medially':['perihilar'],
                                'costrophenic':['base'],
                                'bilaterally': ['right', 'left'], 
                                'lungs': ['right', 'left'], 
                                'biapical': ['right', 'left', 'apical'], 
                                'apices': ['right', 'left', 'apical'], 
                                'apexes': ['right', 'left', 'apical'], 
                                'apex': ['apical'], 
                                'apically': ['apical'], 
                                'retrocardiac': ['left','lower', 'retrocardiac'],
                                'mid': ['middle'], 
                                'basilar':['base'], 
                                'bases': ['base', 'right', 'left'], 
                                'bibasilar': ['base', 'right', 'left'], 
                                'chest walls': ['right', 'left', 'chest wall'], 
                                'ventricular': ['ventricle'],
                                'ventricles': ['right', 'left', 'ventricle'], 
                                'atriums': ['right', 'left', 'atrium'], 
                                'superior vena cava': ['svc'], 
                                'below the diaphragms': ['below the diaphragm'], 
                                'above the carinal':['above the carina'],
                                'posterolateral': ['posterior', 'lateral'], 
                                'apicolateral': ['apical', 'lateral'], 
                                'ribs': ['rib'], 
                                '3rd': ['third'], 
                                '3': ['third'], 
                                '4th': ['fourth'], 
                                '4': ['fourth'], 
                                'four': ['fourth'], 
                                '5th': ['fifth'], 
                                'five': ['fifth'], 
                                '5': ['fifth'], 
                                '6th': ['sixth'], 
                                '6': ['sixth'], 
                                'six': ['sixth'], 
                                '7th': ['seventh'],
                                '7': ['seventh'],  
                                'seven': ['seventh'], 
                                '8th': ['eighth'], 
                                '8': ['eighth'],  
                                'eight': ['eighth'], 
                                '9th': ['ninth'], 
                                '9': ['ninth'], 
                                'nine': ['ninth'], 
                                'clavicle': ['clavicular'], 
                                'clavicles': ['clavicular', 'left', 'right'], 
                                'vertebrae':['spine'], 
                                'vertebra':['spine'], 
                                'vertebral':['spine'],
                                'spinal':['spine'],
                                'ij':['jugular'],
                                'caval atrial': ['cavoatrial'],
                                'et': ['endotracheal'],
                                'picc': ['pic'],
                                'ng':['nasogastric']
                                }

        str_labels_location = list(OrderedSet([item for sublist in list_of_location_labels_per_abnormality.values() for item in sublist]))
        if args.limit_location_vocabulary:
            allowed_locations = ['right', 'left','upper', 'lower', 'base', 'apical', 'retrocardiac', 'rib','middle']
        else:
            allowed_locations = str_labels_location

        def get_location_labels_from_list_of_str(label, location):
            location_labels = np.zeros([len(allowed_locations)], dtype=np.int8)
            if location!='-1':
                for location_label_str in set(str_labels_location + list(list_of_replacements_labels_location.keys())):                    
                    regexp = re.compile(fr'\b{location_label_str}\b', flags=re.IGNORECASE)
                    if regexp.search(location):
                        if args.allow_location_replacements and location_label_str in list_of_replacements_labels_location.keys():
                            words_found = list_of_replacements_labels_location[location_label_str]
                        else:
                            words_found = [location_label_str]
                        for replacement_word in words_found:
                            if replacement_word in allowed_locations:
                                location_labels[allowed_locations.index(replacement_word)] = 1
            return location_labels

        def get_location_labels_per_row(row,columns_to_join_preds):
            for column_name in columns_to_use:
                all_locations_collected = []
                if column_name in columns_to_join_preds:
                    for column_name_to_join in columns_to_join_preds[column_name]:
                        if row[column_name_to_join]!='-1':
                            all_locations_collected+=row[column_name_to_join].split(';')
                else:
                    if row[column_name]!='-1' and row[column_name]!=-1:
                        all_locations_collected+= row[column_name].split(';')
                if len(all_locations_collected)==0:
                    all_locations_collected = '-1'
                else:
                    all_locations_collected = ';'.join(all_locations_collected)
                row[column_name] = get_location_labels_from_list_of_str(column_name, all_locations_collected)
            return row


        preds = pd.read_csv(args.single_file)

        annot = pd.read_csv(args.groundtruth)
        annot = annot[annot['type_annotation']=='location']
        annot= annot.fillna('-1')

        preds =preds[preds['type_annotation']=='location']
        preds= preds.fillna('-1')
        preds['subjectid_studyid'] = preds['subjectid_studyid'].str.lstrip('_')
        preds = pd.merge(annot['subjectid_studyid'], preds, on='subjectid_studyid', how='inner') 

        annot = annot.apply(lambda row: get_location_labels_per_row(row,columns_to_join_annot), axis=1)
        preds = preds.apply(lambda row: get_location_labels_per_row(row,{}), axis=1)
        for column_name in annot.columns:
            if column_name in columns_to_use:
                preds_to_use =  np.stack(preds[column_name].values[allowable_indices_loc_sev[column_name]]).flatten()
                annots_to_use = np.stack(annot[column_name].values[allowable_indices_loc_sev[column_name]]).flatten()
                dict_aucs[column_name + '_annotpred'] = (annots_to_use, preds_to_use)
        return dict_aucs

    if args.run_severity:
        preds = pd.read_csv(args.single_file)

        annot = pd.read_csv(args.groundtruth)
        annot = annot[annot['type_annotation']=='severity']
        annot= annot.fillna('-1')

        preds =preds[preds['type_annotation']=='severity']
        preds= preds.fillna('-1')
        preds['subjectid_studyid'] = preds['subjectid_studyid'].str.lstrip('_')
        preds = pd.merge(annot['subjectid_studyid'], preds, on='subjectid_studyid', how='inner') 


        for column_name in annot.columns:
            if column_name in columns_to_use:
                annot[column_name] = pd.to_numeric(annot[column_name])
                preds[column_name] = pd.to_numeric(preds[column_name])

        for column_name in annot.columns:
            if column_name in columns_to_use:
                preds_to_use =  preds[column_name].values[allowable_indices_loc_sev[column_name]]
                annots_to_use = annot[column_name].values[allowable_indices_loc_sev[column_name]]
                annots_to_use[annots_to_use==-1] = 0
                preds_to_use[preds_to_use==-1] = 0
                annots_to_use = (annots_to_use!=0)*1
                preds_to_use = (preds_to_use!=0)*1
                dict_aucs[column_name + '_annotpred'] = annots_to_use, preds_to_use

        return dict_aucs

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
    parser.add_argument("--type_annotation", type=str, choices = ["probability", "labels"], default="labels", help='''''')
    parser.add_argument("--allow_location_replacements", type=str2bool, default='true', help='''''')
    parser.add_argument("--limit_location_vocabulary", type=str2bool, default='false', help='''''')
    parser.add_argument("--run_severity", type=str2bool, default='false', help='''''')
    parser.add_argument("--run_location", type=str2bool, default='false', help='''''')
    parser.add_argument("--run_labels", type=str2bool, default='false', help='''''')
    parser.add_argument("--limit_to_label_agreement", type=str2bool, default='false', help='''''')
    parser.add_argument('--single_file', type=str, default = './parsing_results_llm.csv', help='''Path to file to run the script for. For example, 
    use "--single_file=./parsing_results.csv" to run the script for the ./parsing_results.csv file.
    . Default: ./parsing_results_llm.csv''')
    parser.add_argument('--groundtruth', type=str, default = './human_annotation.csv', help='''Path to file containing the groundtruth for all 
    reports that you are testing for. The ID (filename) of a report should be in the image2 column. The abnormality labels should be in columns named
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",  
    "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices". Default: ./human_annotation.csv''')
    parser.add_argument("--labeler", type=str, choices = ["chexpert", "vqa", 'llm', 'vicuna', 'llm_generic'], default=None, help='''''')
    parser.add_argument("--dataset", type=str, choices = ['nih', 'mimic'], default=None, help='''''')
    parser.add_argument("--use_relabeled", type=str, choices = ['reflacx', 'pneumothorax','pneumonia'], default=None, help='''''')
    parser.add_argument("--prediction_file_mimic_vqa", type=str, default='./vqa_dataset_converted.csv', help='''''')
    parser.add_argument("--prediction_file_mimic_llm", type=str, default='./new_dataset_annotations/mimic_llm_annotations.csv', help='''''')
    parser.add_argument("--prediction_file_mimic_llmgeneric", type=str, default='./mimic_llm_annotations_generic.csv', help='''''')
    parser.add_argument("--groundtruth_csv_pneumothorax", type=str, default='pneumothorax_relabeled_dataset_converted.csv', help='''''')
    parser.add_argument("--groundtruth_csv_pneumonia", type=str, default='pneumonia_relabeled_dataset_converted.csv', help='''''')
    parser.add_argument("--groundtruth_csv_reflacx", type=str, default='reflacx_dataset_converted.csv', help='''''')
    parser.add_argument("--groundtruth_csv_reflacx12", type=str, default='reflacx_phase12_dataset_converted.csv', help='''''')
    parser.add_argument("--prediction_file_nih_vicuna", type=str, default='./vicuna_nih_dataset_converted.csv', help='''''')
    parser.add_argument("--prediction_file_mimic_vicuna", type=str, default='./vicuna_mimic_dataset_converted.csv', help='''''')
    args = parser.parse_args()
    main(args)