# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-07-12
# The input file is dataset_statistics.csv, which is the output file for the 
# ../src_classifier/get_data_statistics.py script.
# Two Latex formatted tables are outputed to stdout.
# Description:
# File used to generate Tables 6 and 7 in "Enhancing Chest X-ray
# Datasets with Privacy-Preserving LLMs and Multi-Type Annotations:
# A Data-Driven Approach for Improved Classification", containing
# the statistics for the annotations of the MIMIC-CXR dataset used to train the classifier

import pandas as pd

statistics_df = pd.read_csv('./dataset_statistics.csv')
labels_dict = {'stable':-3,'not mentioned':-2,'uncertain':-1,'absent':0,'present':1}
severities_dict = {'mild':1,'moderate':2,'severe':3}
str_labels_mimic = {
    'Atelectasis':'Atel.',
    'Cardiomegaly':'Card.',
    'Consolidation':'Cons.',
    'Enlarged Cardiomediastinum':'Enl.CM',
    'Edema':'Edema',
    'Fracture':'Fract.',
    'Lung Lesion':'Les.',
    'Lung Opacity':'Opac.',
    'Pleural Effusion':'Effus.',
    'Pleural Other':'P.Ot.',
    'Pneumothorax':'PTX',
    'Support Devices':'Sup.D.'}
datasets = {'llm': 'MAPLEZ', 'vqa': 'VQA', 'chexpert':'CheXpert'}


row_names_list = [{'gt_present': 'Pr.',
'gt_absent': 'Ab.',
'gt_uncertain': 'Unc.',
'gt_not mentioned': 'NM',
'gt_stable': 'St.',
'probabilities_mean': 'P.$\mu$',
'probabilities_stable': 'P.St.',
'probabilities_low': 'P.$\leq$10\%',
'probabilities_high': 'P.$\geq$90\%',
'stable':'C.St.',

},
 {
'severities_present': 'Sevs.',
'severities_mild': 'Mild',
'severities_moderate': 'Moderate',
'severities_severe': 'Severe',
'location_present': 'Locs.',
'location_positive': 'Loc.+',
'location_negative': 'Loc.-',

}]

table = []
for abnormality in str_labels_mimic:
    row = {}
    if abnormality=='all':
        this_df = statistics_df
    else:
        this_df = statistics_df[statistics_df['abnormality']==abnormality]
    row['abnormality'] = abnormality
    row['n'] = len(this_df)
    for category in labels_dict:
        row['chexpert_gt_'+category] = this_df['mimic_gt_'+category].sum()
        row['llm_gt_'+category]  = this_df['new_gt_'+category].sum()
        row['vqa_gt_'+category] = this_df['vqa_new_gt_'+category].sum()
        row['reflacx_gt_'+category] = this_df['reflacx_new_gt_'+category][this_df['reflacx_present']==1].sum()
    row['llm_severities_present'] = this_df['severities_present'].sum()
    row['vqa_severities_present'] = (this_df['vqa_severities_present']>0).sum()
    
    for category in severities_dict:
        row['llm_severities_'+category] = this_df['severities_'+category].sum()
        row['vqa_severities_'+category] = this_df['vqa_severities_'+category].sum()
    row['llm_probabilities_mean'] = this_df['probabilities'][this_df['probabilities']!=101].mean()
    row['llm_probabilities_stable'] = (this_df['probabilities'][this_df['probabilities']==101]).sum()/101
    row['vqa_probabilities_mean'] = this_df['vqa_probabilities'].mean()
    row['reflacx_probabilities_mean'] = this_df['reflacx_probabilities'][this_df['reflacx_present']==1].mean()
    row['llm_probabilities_low'] = (this_df['probabilities'][this_df['probabilities']!=101]<=10).sum()
    row['llm_probabilities_high'] = (this_df['probabilities'][this_df['probabilities']!=101]>=90).sum()
    row['vqa_probabilities_low'] = (this_df['vqa_probabilities']<=10).sum()
    row['vqa_probabilities_high'] = (this_df['vqa_probabilities']>=90).sum()
    row['reflacx_probabilities_low'] = (this_df['reflacx_probabilities'][this_df['reflacx_present']==1]<=10).sum()
    row['reflacx_probabilities_high'] = (this_df['reflacx_probabilities'][this_df['reflacx_present']==1]>=90).sum()
    row['llm_location_present'] = (this_df['location_labels_positive']>0).sum()
    row['vqa_location_present'] = (this_df['vqa_location_labels_positive']>0).sum()
    row['llm_location_positive'] = this_df['location_labels_positive'].sum()
    row['vqa_location_positive'] = this_df['vqa_location_labels_positive'].sum()
    row['llm_location_negative'] = this_df['location_labels_negative'].sum()
    row['vqa_location_negative'] = this_df['vqa_location_labels_negative'].sum()
    row['llm_location_total'] = this_df['location_labels'].sum()
    row['vqa_location_total'] = this_df['vqa_location_labels'].sum()
    row['llm_stable'] = this_df['unchanged_uncertainties'].sum()
    row['n_reflacx'] = this_df['reflacx_present'].sum()
    table.append(row)
summarized_statistics = pd.DataFrame(table)

precision = 0
precision_percentage = 2
table_string = ''

def format_with_two_significant_digits(number):
    # Check if the number is a whole number and has less than two significant digits
    if number <10:
        # Add one decimal place to make it two significant digits
        return f"{number:.1f}"
    else:
        # Else, format the number to have two significant digits
        return f"{number:.2g}"

for row_names in row_names_list:
    

    table_string+= 'Abn.'
    table_string += ' & '
    table_string+= 'Labeler'
    table_string += ' & '

    for row_name in row_names:
        table_string+= row_names[row_name]
        table_string += ' & '

    table_string+='\\\\\n'
    table_string+='\\midrule'
    table_string+='\n'


    for abnormality in str_labels_mimic:
        for dataset in datasets:
            row_string = ''
            for row_name in row_names:
                if 'stable' in row_name and dataset!='llm':
                    string_to_add = '-'
                elif row_name=='n':
                    if dataset=='reflacx':
                        value = summarized_statistics[summarized_statistics['abnormality']==abnormality][f'n_reflacx'].values[0]
                        string_to_add = f'{value:.{precision}f}'
                    else:
                        value = summarized_statistics[summarized_statistics['abnormality']==abnormality][f'n'].values[0]
                    string_to_add = f'{value:.{precision}f}'
                    if value==0:
                        string_to_add = '-'
                elif f'{dataset}_{row_name}' in summarized_statistics:
                    value = summarized_statistics[summarized_statistics['abnormality']==abnormality][f'{dataset}_{row_name}'].values[0]
                    if 'location' in row_name and 'tive' in row_name:
                        total_location_present = summarized_statistics[summarized_statistics['abnormality']==abnormality][f'{dataset}_location_present'].values[0]
                        total_locations = summarized_statistics[summarized_statistics['abnormality']==abnormality][f'{dataset}_location_total'].values[0]
                        total_cases = summarized_statistics[summarized_statistics['abnormality']==abnormality][f'n'].values[0]
                        if total_cases==0:
                            total = 0
                        else:
                            total = total_locations/total_cases*total_location_present
                        
                    elif dataset=='reflacx':
                        total = summarized_statistics[summarized_statistics['abnormality']==abnormality][f'n_reflacx'].values[0]
                    else:
                        total = summarized_statistics[summarized_statistics['abnormality']==abnormality][f'n'].values[0]
                    if value==value and total>0:
                        percentage = value/total*100
                        if 'n'!=row_name and 'n_reflacx'!=row_name and 'total' not in row_name and 'probabilities_mean'!=row_name:
                            string_to_add = f'{format_with_two_significant_digits(percentage)}\\%'
                        elif 'probabilities_mean'!=row_name:
                            string_to_add = f'{value:.{precision}f}'
                        else:
                            string_to_add = f'{format_with_two_significant_digits(value)}\\%'
                    else:
                        string_to_add = '-'
                else:
                    string_to_add = '-'
                row_string+= string_to_add
                row_string += ' & '
            if len(row_string.replace('&','').replace('-','').replace(' ',''))>0:
                table_string+=str_labels_mimic[abnormality]
                table_string += ' & '
                table_string+=datasets[dataset]
                table_string += ' & '
                table_string+=row_string
                table_string+='\\\\\n'

table_string = table_string.replace('& \\\\\n', '\\\\\n')
table_string = table_string.replace('.0 ', ' ')
print(table_string)

