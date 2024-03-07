# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Description:
# class that manages the saving of logs to the output folder.
# logs are in the form of a txt file, a csv file, a tnsorboard log.
# this class also saves the current source code from the src folder,
# and the configurations used to run the specific training script

import logging
import os
import torch
import glob
import shutil
import sys
import csv
from utils import SmoothedValue
import numpy as np
from PIL import Image
import pandas as pd

def save_image(filepath, numpy_array):
    im = Image.fromarray(((numpy_array*0.5 + 0.5)*255))

    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(filepath)

class Outputs():
    def __init__(self, opt, output_folder):
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        for handler in logging.root.handlers:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename = output_folder +'/log.txt' ,level = logging.INFO)
        self.log_configs(opt)
        self.csv_file =  output_folder +'/log.csv' 
        with open(self.csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['key','value','epoch'])
    
    # Write all the configurations from the opt variable to the log.txt file
    def log_configs(self, opt):
        logging.info('-------------------------------used configs-------------------------------')
        for key, value in sorted(vars(opt).items()):
            logging.info(key + ': ' + str(value).replace('\n', ' ').replace('\r', ''))
        logging.info('-----------------------------end used configs-----------------------------')
    
    # activate the average calculation for all metric values and save them to log and tensorboard
    def log_added_values(self, epoch, metrics):
        averages = metrics.get_average()
        logging.info('Metrics for epoch: ' + str(epoch))
        for key, average in averages.items():
            logging.info(key + ': ' + str(average))
            with open(self.csv_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([key, str(average), str(epoch)])
        return averages
    
    def log_added_values_pytorch(self, epoch, metrics):
        logging.info('Metrics for epoch: ' + str(epoch))
        for meter in metrics.meters:
            if isinstance(metrics.meters[meter], SmoothedValue):
                average_ = metrics.meters[meter].avg
                value = metrics.meters[meter].value
                average_names = {'average':average_, 'value':value}
            else:
                average_names = {'value':metrics.meters[meter]}
            for average_name in average_names:
                average = average_names[average_name]
                logging.info(meter +f'_{average_name}' + ': ' + str(average))
                with open(self.csv_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([meter, str(average), str(epoch)])
    
    #save the source files used to run this experiment
    def save_run_state(self, py_folder_to_save):
        if not os.path.exists('{:}/src/'.format(self.output_folder)):
            os.mkdir('{:}/src/'.format(self.output_folder))
        [shutil.copy(filename, ('{:}/src/').format(self.output_folder)) for filename in glob.glob(py_folder_to_save + '/*.py')]
        self.save_command()
    
    #saves the command line command used to run these experiments
    def save_command(self, command = None):
        if command is None:
            command = ' '.join(sys.argv)
        with open("{:}/command.txt".format(self.output_folder), "w") as text_file:
            text_file.write(command)
    
    # save the weights of a model
    def save_models(self, net_d, suffix):
        torch.save(net_d.state_dict(), '{:}/state_dict_d_'.format(self.output_folder) + str(suffix)) 

    def save_model_outputs(self, annot, pred, name):
        if len(annot.shape)>1:
            annot = annot[:,0]
        if len(pred.shape)>1:
            pred = pred[:,0]
        df = pd.DataFrame({
            'annot': annot,
            'pred': pred
        })

        csv_file_path = f'{self.output_folder}/{name}_model_outputs.csv' 
        
        df.to_csv(csv_file_path, index=False)  
    
    def save_image(self, image, title, epoch):
        if image.shape[1]<=3:
            image = image[:,0]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if len(image.shape)<4:
            if len(image.shape) == 3:
                image = np.vstack(np.hsplit(np.hstack(image), 4))
            path = f'{self.output_folder}/{title}-{epoch}.png'
            save_image(path, image)
