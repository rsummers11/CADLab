# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Description:
# Auxiliary file containing clip a class used to manage the accumulation and calculation of metrics

import collections
from sklearn.metrics import roc_auc_score
import numpy as np
from list_labels import str_labels_mimic as label_names
import time

class Metrics():
    def __init__(self):
        self.values = collections.defaultdict(list)
        self.registered_score = {}
        self.registered_iou = {}
        self.start_times = {}
    
    #methods used to measure how much time differentaprts of the code take
    def start_time(self, key):
        self.start_times[key] = time.time()
    def end_time(self, key):
        self.values['time_' + key] += [time.time()- self.start_times[key]]
    
    def add_list(self, key, value):
        value = value.detach().cpu().tolist()
        self.values[key] += value
    
    # add metrics that have its average calculate by simply averaging all the values given during one epoch
    # loss is an example of a value that can use this function
    # DO NOT use this function for models predictions (used for calculating AUC) and for iou values
    def add_value(self, key, value):
        value = value.detach().cpu()
        self.values[key].append( value)
    
    def calculate_average(self):
        self.average = {}
        for key, element in self.values.items():
            #do not calculate the average for these keys since they have a special aggregating method defined below
            if key[:7]=='y_true_' or key[:12]=='y_predicted_':
                continue
            
            #calculate the avearage for all other keys
            n_values = len(element)
            if n_values == 0:
                self.average[key] = 0
                continue
            sum_values = sum(element)
            self.average[key] = sum_values/float(n_values)
        

        #calculate the AUC using the keys 'y_true_' + suffix and 'y_predicted_' + suffix
        for suffix in self.registered_score.keys():
            y_true = np.array(self.values['y_true_' + suffix])
            y_predicted = np.array( self.values['y_predicted_' + suffix])
            auc_average = []
            for i in range(y_true.shape[1]):
                if (y_true[:,i]>0).any():
                    # calculate the auc by inputting all predicted and groundtruth values of each label into the roc_auc_score function
                    self.average['score_' + label_names[i].replace(' ','_') + '_' + suffix] = roc_auc_score(y_true[:,i], y_predicted[:,i])
                    auc_average.append(self.average['score_' + label_names[i].replace(' ','_') + '_' + suffix])
            # calculate teh average AUC by averaging the AUC of all labels
            self.average['auc_average_' + suffix] = sum(auc_average)/len(auc_average)
        self.values = collections.defaultdict(list)
    
    def get_average(self):
        self.calculate_average()
        return self.average
    
    # the variable consider is true for the values that should be considered during the calculation of the iou.
    # metric_value and consider should have the same shape.
    # iou should only be calculated for cases/labels that had at least one ellipse drawn
    def add_iou(self, name, metric_value, consider):
        self.registered_iou[name] = 0
        metric_value = metric_value.detach()
        consider = consider.detach()
        self.add_list('iou_metric_' + name, metric_value)
        self.add_list('iou_consider_' + name, consider)
    
    # for adding predictions and groundtruths, used later to calculate auc.
    def add_score(self, y_true, y_predicted, suffix):
        self.registered_score[suffix] = 0
        y_predicted = y_predicted.detach().squeeze(1)
        if len(y_true.size())>1:
            y_true = y_true.detach().squeeze(1)
        self.add_list('y_true_' + suffix, y_true)
        self.add_list('y_predicted_' + suffix, y_predicted)