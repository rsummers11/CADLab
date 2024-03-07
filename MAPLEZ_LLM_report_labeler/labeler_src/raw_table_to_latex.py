# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-07-12
# Use `python <python_file> --help` to check inputs.
# A Latex formatted table is outputed to stdout.
# Description:
# This file converts the table outputs from the create_raw_table.py
# script into the formatting required by Latex files. It also
# filters out the contaent to present only what is presented in the paper

import pandas as pd
from scipy import stats
from collections import defaultdict
import argparse

def main(args):
    results = pd.read_csv(args.input_raw_table)
    results = results[(results['n_pos'] > 10) | (results['n_pos'] != results['n_pos'])]
    results = results[(results['dataset']!='reflacx_phase12')]
    total_unnormalized_weights = results[results['row_unnormalized_weight']==results['row_unnormalized_weight']]['row_unnormalized_weight'].values.sum()

    do_large_table = args.do_large_table
    type_annotation = args.type_annotation

    precision_weight = 3

    p_value_dict = {0.001:['\makebox[\widthof{\(^{\scriptscriptstyle{*}\scriptscriptstyle{*}\scriptscriptstyle{*}}\)}][c]{}',\
                            '\(^{\scriptscriptstyle{*}\scriptscriptstyle{*}\scriptscriptstyle{*}}\)'], 
                            0.01:['\makebox[\widthof{\(^{\scriptscriptstyle{*}\scriptscriptstyle{*}\scriptscriptstyle{*}}\)}][c]{}',\
                            '\makebox[\widthof{\(^{\scriptscriptstyle{*}\scriptscriptstyle{*}\scriptscriptstyle{*}}\)}][l]{\(^{\scriptscriptstyle{*}\scriptscriptstyle{*}}\)}'],
                            0.05:['\makebox[\widthof{\(^{\scriptscriptstyle{*}\scriptscriptstyle{*}\scriptscriptstyle{*}}\)}][c]{}',\
                            '\makebox[\widthof{\(^{\scriptscriptstyle{*}\scriptscriptstyle{*}\scriptscriptstyle{*}}\)}][l]{\(^{\scriptscriptstyle{*}}\)}'],
                            1:['\makebox[\widthof{\(^{\scriptscriptstyle{*}\scriptscriptstyle{*}\scriptscriptstyle{*}}\)}][c]{}',\
                            '\makebox[\widthof{\(^{\scriptscriptstyle{*}\scriptscriptstyle{*}\scriptscriptstyle{*}}\)}][l]{\(^{ns}\)}']}
    if type_annotation=='other_modalities':
        precision=3
        precision_confidence = 3
        label_dict = {'lung lesion': 'Lung lesion', 
        'liver lesion': 'Liver lesion',  
        'kidney lesion': 'Kidney lesion', 
        'adrenal gland abnormality':'Adrenal gland abnormality',
        'pleural effusion':'Pleural effusion',
        'hypermetabolic abnormality in the thorax':'Hypermetab. thorax', 
        'hypermetabolic abnormality in the abdomen':'Hypermetab. abdomen', 
        'hypermetabolic abnormality in the pelvis':'Hypermetab. pelvis','-':'-'
    }
        if do_large_table:
            dataset_dict = {
                'ct':'CT',
                'pet':'PET',
                'mri':'MRI',
                    'label_liver lesion': 'Liver lesion',  
        'label_kidney lesion': 'Kidney lesion', 
        'label_adrenal gland abnormality':'Adrenal gland abnormality',
        'all':'All',
            }
            column_dict = {'dataset':'Data',
                'abnormality':'Abn.',
                'n':'$N$',
            'n_pos':'$N^+$',
            'row_unnormalized_weight':'W',
            'median':''
            }

        else:
            label_dict = {
            '-':'-'}
            dataset_dict = {
                'ct':'CT',
                'pet':'PET',
                'mri':'MRI',
            'all':'All',
            }
            column_dict = {
                'dataset':'Data',
            'median':''
            }

        start_best_with = 0
        best_comparison = lambda x,y: x>y

        main_score = 'f1'

        labeler_dict = {'llm':'MAPLEZ'}

        if do_large_table:
            score_type_dict = {'precision':'Precision', 'recall':'Recall', 'f1':'F1'}
        else:
            score_type_dict = {'f1':'F1'}
    if type_annotation=='classifier':
        precision=3
        precision_confidence = 3
        label_dict = {'atelectasis':'Atel.',
        'cardiomegaly':'Card.',
        'consolidation':'Cons.',
        'lung edema':'Edema',
        'fracture':'Fract.',
        'lung opacity':'Opac.',
        'pleural effusion':'Effus.',
        'pneumothorax':'PTX',
        '-':'-'}
        if do_large_table:
            dataset_dict = {'nih':'NIH',
            'mimic':'MIMIC',
            'reflacx':'\\textit{RFL-3}',
            'pneumonia':'\\textit{PNA}',
            'pneumothorax':'\\textit{PTX}',
            'human':'\\textit{Human}',
            'all':'All',
            'reflacx_phase12':'\\textit{RFL-12}',
            'chexpert':'\\textit{CheXpert}',
                'label_atelectasis':'Atel.',
        'label_cardiomegaly':'Card.',
        'label_consolidation':'Cons.',
        'label_lung edema':'Edema',
        'label_fracture':'Fract.',
        'label_lung opacity':'Opac.',
        'label_pleural effusion':'Effus.',
        'label_pneumothorax':'PTX',
            }
            column_dict = {'dataset':'Data',
                'abnormality':'Abn.',
                'n':'$N$',
            'n_pos':'$N^+$',
            'row_unnormalized_weight':'W',
            'median':''
            }

        else:
            label_dict = {
            '-':'-'}
            dataset_dict = {
            'reflacx':'\\textit{RFL-3}',
            'pneumonia':'\\textit{PNA}',
            'pneumothorax':'\\textit{PTX}',
            'chexpert':'\\textit{CheXpert}',
            'all':'All',
            }
            column_dict = {
                'dataset':'Data',
            'median':''
            }

        start_best_with = 0
        best_comparison = lambda x,y: x>y

        main_score = 'auc'

        labeler_dict = {'chexpert_model':'CheXpert','vqa_model':'VQA','llm_model':'LLM',\
                        'no_loc': '$\lambda_{loc}=0$', 'labels':'Cat. Labels', '3notignore':'Use "Stable"',\
                            'generic':'MAPLEZ-G', 'all_changes':'All Changes'}



        score_type_dict = {'auc':'AUC'}
    if type_annotation=='human_label':
        precision=3
        precision_confidence = 3
        precision_weight = 2
        label_dict = {'atelectasis':'Atel.',
        'cardiomegaly':'Card.',
        'consolidation':'Cons.',
        'lung edema':'Edema',
        'fracture':'Fract.',
        'lung opacity':'Opac.',
        'pleural effusion':'Effus.',
        'pneumothorax':'PTX',
        '-':'-'}

        dataset_dict = {'nih':'NIH',
        'mimic':'MIMIC',
        'reflacx':'\\textit{RFL-3}',
        'pneumonia':'\\textit{PNA}',
        'pneumothorax':'\\textit{PTX}',
        'human':'\\textit{Human}',
        'all':'All',
        'reflacx_phase12':'\\textit{RFL-12}',
        'chexpert':'\\textit{CheXpert}'
        }

        column_dict = {'dataset':'Data',
            'abnormality':'Abn.',
        'n_pos':'$N^+$',
        'row_unnormalized_weight':'W',
        'median':''
        }

        start_best_with = 0
        best_comparison = lambda x,y: x>y

        main_score = 'f1'

        labeler_dict = {'human': 'Rad.','llm':'MAPLEZ'}

        if do_large_table:
            score_type_dict = {'precision':'Precision', 'recall':'Recall', 'f1':'F1'}
        else:
            score_type_dict = {'f1':'F1'}
    if type_annotation=='human_probability':
        precision=3
        precision_confidence = 3
        precision_weight = 2
        label_dict = {'atelectasis':'Atel.',
        'cardiomegaly':'Card.',
        'consolidation':'Cons.',
        'lung edema':'Edema',
        'fracture':'Fract.',
        'lung opacity':'Opac.',
        'pleural effusion':'Effus.',
        'pneumothorax':'PTX',
        '-':'-'}

        dataset_dict = {
        'all':'All',
        'reflacx_phase12':'\\textit{RFL-12}'
        }

        column_dict = {'dataset':'Data',
            'abnormality':'Abn.',
        'n_pos':'$N^+$',
        'row_unnormalized_weight':'W',
        'median':''
        }

        start_best_with = 100
        best_comparison = lambda x,y: x<y

        main_score = 'mae'

        labeler_dict = {'human': 'Rad.', 'llm':'MAPLEZ'}

        score_type_dict = {'mae':'MAE'}
    if type_annotation=='labels':
        precision=3
        precision_confidence = 3
        label_dict = {'atelectasis':'Atel.',
        'cardiomegaly':'Card.',
        'consolidation':'Cons.',
        'lung edema':'Edema',
        'fracture':'Fract.',
        'lung opacity':'Opac.',
        'pleural effusion':'Effus.',
        'pneumothorax':'PTX',
        '-':'-'}

        dataset_dict = {'nih':'NIH',
        'mimic':'MIMIC',
        'reflacx':'\\textit{RFL-3}',
        'pneumonia':'\\textit{PNA}',
        'pneumothorax':'\\textit{PTX}',
        'human':'\\textit{Human}',
        'all':'All',
        'reflacx_phase12':'\\textit{RFL-12}',
        'label_atelectasis':'Atel.',
        'label_cardiomegaly':'Card.',
        'label_consolidation':'Cons.',
        'label_lung edema':'Edema',
        'label_fracture':'Fract.',
        'label_lung opacity':'Opac.',
        'label_pleural effusion':'Effus.',
        'label_pneumothorax':'PTX',
        }

        column_dict = {'dataset':'Data',
            'abnormality':'Abn.',
        'n':'$N$',
        'n_pos':'$N^+$',
        'row_unnormalized_weight':'W',
        'median':''
        }

        start_best_with = 0
        best_comparison = lambda x,y: x>y

        main_score = 'f1'

        labeler_dict = {'chexpert':'CheXpert', 'vicuna':'Vicuna', 'vqa': 'VQA', 'llm_generic': 'MAPLEZ-G', 'llm':'MAPLEZ'}

        if do_large_table:
            score_type_dict = {'precision':'Precision', 'recall':'Recall', 'f1':'F1'}
        else:
            score_type_dict = {'f1':'F1'}
    if type_annotation=='location':
        precision=3
        precision_confidence = 3
        label_dict = {'atelectasis':'Atel.',
        'consolidation':'Cons.',
        'lung edema':'Edema',
        'fracture':'Fract.',
        'lung opacity':'Opac.',
        'pleural effusion':'Effus.',
        'pneumothorax':'PTX',
        '-':'-'}

        dataset_dict = {
        'mimic':'MIMIC',
        'all':'All'
        }

        column_dict = {
            'abnormality':'Abn.',
        'n':'$N$',
        'n_pos':'$N^+$',
        'row_unnormalized_weight':'W',
        'median':''
        }

        start_best_with = 0
        best_comparison = lambda x,y: x>y

        labeler_dict = {'vqa': 'VQA', 'llm_generic': 'MAPLEZ-G', 'llm':'MAPLEZ'}

        if do_large_table:
            score_type_dict = {'precision':'Precision', 'recall':'Recall', 'f1':'F1'}
        else:
            score_type_dict = {'f1':'F1'}
    if type_annotation=='severity':
        precision=3
        precision_confidence = 3
        label_dict = {'atelectasis':'Atel.',
                    'cardiomegaly':'Card.',
        'consolidation':'Cons.',
        'lung edema':'Edema',
        'fracture':'Fract.',
        'lung opacity':'Opac.',
        'pleural effusion':'Effus.',
        'pneumothorax':'PTX',
        '-':'-'}

        dataset_dict = {
        'mimic':'MIMIC'
        }

        column_dict = {
            'abnormality':'Abn.',
        'n':'$N$',
        'n_pos':'$N^+$',
        'row_unnormalized_weight':'W',
        'median':''
        }

        start_best_with = 0
        best_comparison = lambda x,y: x>y

        labeler_dict = {'vqa': 'VQA', 'llm_generic': 'MAPLEZ-G', 'llm':'MAPLEZ'}

        if do_large_table:
            score_type_dict = {'precision':'Precision', 'recall':'Recall', 'f1':'F1'}
        else:
            score_type_dict = {'f1':'F1'}
    if type_annotation=='probability':
        precision=1
        precision_confidence = 1

        label_dict = {'cardiomegaly':'Card.',
        'consolidation':'Cons.',
        'lung edema':'Edema',
        'lung opacity':'Opac.',
        'pneumothorax':'PTX',
        '-':'-'}

        dataset_dict = {
        'all':'All'
        }

        column_dict = {
            'abnormality':'Abn.',
        'row_unnormalized_weight':'W',
        'median':''
        }

        start_best_with = 100
        best_comparison = lambda x,y: x<y

        labeler_dict = {'vqa': 'VQA', 'llm_generic': 'MAPLEZ-G', 'llm':'MAPLEZ'}

        score_type_dict = {'mae':'MAE'}

    table_string = ''

    for column in column_dict:
        if column =='median':
            if do_large_table and 'other_modalities' not in type_annotation:
                table_string += 'Labeler & '
                for_loop_list = [0]
            else:
                for_loop_list = labeler_dict
            for labeler in for_loop_list:
                for score_type in score_type_dict:
                    if do_large_table:
                        table_string += score_type_dict[score_type]
                    else:
                        table_string += labeler_dict[labeler]
                    
                    table_string += ' & '
        else:
            table_string += column_dict[column]
            table_string += ' & '
    table_string+='\\\\\n'
    table_string+='\\midrule'
    table_string+='\n'
    average_midrule_added = False
    for row_index, row in results.iterrows():
        if row['dataset'] not in dataset_dict:
            continue
        
        if row['abnormality'] not in label_dict:
            continue
        if row['abnormality']=='-' and not average_midrule_added:
            table_string+='\\midrule'
            table_string+='\n'
            average_midrule_added = True
        best_score = defaultdict(lambda: start_best_with)
        best_var = {}
        for column in column_dict:
            if column =='median':
                for labeler in labeler_dict:
                    for score_type in score_type_dict:
                        if row[f'{labeler}_{score_type}_median']==row[f'{labeler}_{score_type}_median']:
                            if best_comparison(row[f'{labeler}_{score_type}_median'],best_score[score_type]):
                                best_score[score_type] = row[f'{labeler}_{score_type}_median']
                                best_var[score_type] = row[f'{labeler}_{score_type}_var']

        for column in column_dict:
            
            if column =='median':
                first_labeler = True
                for labeler in labeler_dict:
                    if do_large_table and 'other_modalities' in type_annotation and not first_labeler:
                        table_string+='\\\\\n'
                        for column in column_dict:
                            if column !='median':
                                table_string+=' & '
                    first_labeler = False
                    if do_large_table and 'other_modalities' not in type_annotation:
                        table_string += f'{labeler_dict[labeler]} & '
                    for score_type in score_type_dict:
                        if not row[f'{labeler}_{score_type}_median']==row[f'{labeler}_{score_type}_median']:
                            table_string += '-'
                        else:
                            if labeler not in ['llm','llm_model']:
                                p_value = 2*(0.5-abs(row[f'{labeler}_{score_type}_p']-0.5))
                                for p_value_limit in p_value_dict:
                                    if p_value<=p_value_limit:
                                        p_value_string = p_value_dict[p_value_limit]
                                        break

                                table_string += p_value_string[0]
                            if best_score[score_type]==row[f'{labeler}_{score_type}_median'] and 'other_modalities' not in type_annotation:
                                table_string += '\\textbf{'
                            if do_large_table or 'probability' in type_annotation:
                                table_string += f"{row[f'{labeler}_{score_type}_median']:.{precision}f} [{row[f'{labeler}_{score_type}_low']:.{precision_confidence}f},{row[f'{labeler}_{score_type}_high']:.{precision_confidence}f}]"
                            else:
                                table_string += f"{row[f'{labeler}_{score_type}_median']:.{precision}f}"
                            if best_score[score_type]==row[f'{labeler}_{score_type}_median'] and 'other_modalities' not in type_annotation:
                                table_string += '}'
                            if labeler  not in ['llm','llm_model']:
                                table_string += p_value_string[1]
                        table_string += ' & '
                    if do_large_table and 'other_modalities' not in type_annotation:
                        table_string+='\\\\\n'
                        for column in column_dict:
                            if column !='median':
                                table_string+=' & '
            else:
                if not row[column]==row[column]:
                    value = '-'
                elif column == 'dataset':
                    if 'label_' in row['dataset']:
                        value = '-'
                    else:
                        value = dataset_dict[row[column]]
                elif column=='abnormality':
                    if 'label_' in row['dataset']:
                        value = dataset_dict[row['dataset']]
                    elif 'abnormality' in row:
                        value = label_dict[row[column]]
                    else:
                        value = ''
                elif column=='row_unnormalized_weight':
                    value = f"{row[column]/total_unnormalized_weights:.{precision_weight}f}"
                else:
                    value = str(row[column])
                table_string+=value
                table_string += ' & '
        table_string+='\\\\\n'
    table_string = table_string.replace('& \\\\\n', '\\\\\n')
    table_string = table_string.replace('.0 ', ' ')
    print(table_string)

if __name__=='__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_raw_table", type=str, default="./location_table_raw_agree_v2.csv", help='''
The raw table file to transform into a latex table, generated by create_raw_table.py
    ''')
    parser.add_argument("--do_large_table", type=str2bool, default="false", help='''
If True, the script will generate the full version of the table. OTherwise, it will be a smaller version to put in the paper.
    ''')
    parser.add_argument("--type_annotation", type=str, choices = ['probability', 'severity', 'location',
    'labels', 'classifier', 'human_label', 'human_probability', 'other_modalities'], default='location', help='''
The type of data that is contained in the input_raw_table.
    ''')
    args = parser.parse_args()
    main(args)