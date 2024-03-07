#script used to join the logged metrics for a set of folders (all folders located inside folder_runs)
# results are saved to table_summary_results.csv
from collections import defaultdict
import pandas as pd
import os

def get_table(folder_runs):
    folders = []
    for _, dirnames, _ in os.walk(folder_runs):
        folders += dirnames
        break
    
    table = pd.DataFrame()
    for folder in folders:
        current_epoch = -1
        lines = open(folder_runs+folder+"/log.txt", "r")
        values_gathered = defaultdict(list)
        for line in lines:
            if len(line.split(':'))>=3:
                if line.split(':')[2]=='Metrics for epoch':
                    if current_epoch>-1:
                        values_gathered['folder'] = folder
                        table = table.append(pd.DataFrame( values_gathered, index = [0]))
                        values_gathered = defaultdict(list)
                    current_epoch = int(line.split(':')[3][1:])
                    values_gathered['epoch'] = current_epoch
                elif line.split(':')[1]=='root' and current_epoch>-1 and len(line.split(':'))>3: 
                    metric = line.split(':')[2]
                    if line.split(':')[3][:7]==' tensor':
                        value = float(line.split(':')[3][8:line.split(':')[3].index(')')])
                    else:
                        value = float(line.split(':')[3])
                    values_gathered[metric] = value
        values_gathered['folder'] = folder
        table = table.append(pd.DataFrame( values_gathered, index = [0]))
        lines.close()
    table.to_csv(f'./table_summary_results_{folder_runs.replace("/", "|")}.csv')