# script used to calculate the aucs for the paper
# table_summary_results.csv should have been generated with table_summary_results.py 
import pandas as pd
from list_labels import list_labels

def get_auc(runs_folder):
    results_list = []
    full_df = pd.read_csv(f'./table_summary_results_{runs_folder.replace("/", "|")}.csv')
    for folder in full_df['folder'].unique():
        folder_df = full_df[full_df['folder'] == folder]
        with open(f'{runs_folder}/{folder}/log.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line.split(':')[2]=='load_checkpoint_d':
                    original_folder = '/'.join(line.split(':')[3][1:].split('/')[:-1])
        
        #using the name of the training folder as a method for separating each of the methods reported in the paper
        setting = '_'.join(original_folder.split('/')[-1].split('_')[0:-1])
        for label in list_labels:
            assert(len(folder_df)==1)
            results_list.append({'folder':folder,
                                'auc':folder_df[f"score_{label.replace(' ','_')}_val_mimic_all"].values[0],
                                'setting':setting,
                                'label': label})
    
    pd.DataFrame(results_list).to_csv(f'auc_{runs_folder.replace("/", "|")}.csv', index=False)
