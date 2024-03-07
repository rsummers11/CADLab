# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Description:
# File containing the label classes used to train the models,
# translation between how they are called in varied documents, 
# location keywords, replacement location words, adjacent location words, 
# opposite location words

from sortedcollections import OrderedSet

# a list of the labels in the mimic-cxr dataset, used for selecting the label columns
# in the mimic-cxr label table
# str_labels_mimic = [
#     'Enlarged Cardiomediastinum',
#     'Cardiomegaly',
#     'Lung Lesion',
#     'Lung Opacity',
#     'Edema',
#     'Consolidation',
#     'Atelectasis',
#     'Pneumothorax',
#     'Pleural Effusion',
#     'Pleural Other',
#     'Fracture',
#     'Support Devices']

str_labels_mimic = [
    'Cardiomegaly',
    'Lung Opacity',
    'Edema',
    'Consolidation',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Fracture']

translation_new_labels_to_mimic = {
'enlarged cardiomediastinum':'Enlarged Cardiomediastinum',
'consolidation':'Consolidation',
'pleural other':'Pleural Other',
'lung lesion':'Lung Lesion',
'cardiomegaly':'Cardiomegaly',
'lung opacity':'Lung Opacity',
'atelectasis':'Atelectasis',
'lung edema':'Edema',
'fracture':'Fracture',
'pleural effusion':'Pleural Effusion',
'pneumothorax':'Pneumothorax',
'support device':'Support Devices'
}

translation_mimic_to_new_labels = {v: k for k, v in translation_new_labels_to_mimic.items()}

list_of_location_labels_per_abnormality = {
    'Enlarged Cardiomediastinum':['right', 'left', 'upper', 'lower', 'base', 'ventricle', 'atrium'],
    'Cardiomegaly':['right', 'left', 'upper', 'lower', 'base', 'ventricle', 'atrium'],
    'Lung Lesion':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
    'Lung Opacity':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
    'Edema':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
    'Consolidation':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
    'Atelectasis':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
    'Pneumothorax':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
    'Pleural Effusion':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
    'Pleural Other':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'lateral', 'perihilar', 'retrocardiac'],
    'Fracture':['middle', 'right', 'left', 'upper', 'lower', 'lateral', 'posterior', 'anterior', 'rib', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'clavicular', 'spine'],
    'Support Devices':['apical', 'middle', 'right', 'left', 'upper', 'lower', 'base', 'pleural', 'chest wall', 'ventricle', 'atrium', 'svc', 'below the diaphragm', 'jugular', 'above the carina', 'cavoatrial', 'stomach']
}

list_of_replacements_labels_location = {'bilateral': ['right', 'left'], 
                        'bilaterally': ['right', 'left'], 
                        'lungs': ['right', 'left'], 
                        'biapical': ['right', 'left', 'apical'], 
                        'apices': ['right', 'left', 'apical'], 
                        'apexes': ['right', 'left', 'apical'], 
                        'apex': ['apical'], 
                        'retrocardiac': ['left','lower', 'retrocardiac'],
                        'mid': ['middle'], 
                        'basilar':['base'], 
                        'bases': ['base', 'right', 'left'], 
                        'bibasilar': ['base', 'right', 'left'], 
                        'chest walls': ['right', 'left', 'chest wall'], 
                        'ventricles': ['right', 'left', 'ventricle'], 
                        'atriums': ['right', 'left', 'atrium'], 
                        'superior vena cava': ['svc'], 
                        'below the diaphragms': ['below the diaphragm'], 
                        'above the carinal':['above the carina'],
                        'posterolateral': ['posterior', 'lateral'], 
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
                        'ij':['jugular'],
                        'caval atrial': ['cavoatrial'],
                        }

str_labels_location = list(OrderedSet([item for sublist in list_of_location_labels_per_abnormality.values() for item in sublist]))

list_of_uncertain_labels_location = {'apical':['upper'], 'middle':['upper', 'lower', 'cavoatrial', 'svc'], 
                                     'right':[], 'left':[], 'upper':['apical', 'middle', 'above the carina', 'jugular'], 
                                     'lower':['base', 'retrocardiac', 'middle'], 'base':['lower', 'pleural', 'retrocardiac'], 
                                     'lateral':[], 'perihilar':['atrium', 'ventricle', 'cavoatrial', 'svc'],
                'pleural':['base'], 'chest wall':['rib'], 'ventricle':['retrocardiac', 'perihilar', 'lower', 'left'], 
                'atrium':['retrocardiac', 'cavoatrial', 'perihilar', 'lower'], 'svc':['perihilar', 'middle', 'lower', 'upper'],
                'below the diaphragm':['stomach'], 'jugular':[],
                'above the carina':['perihilar', 'upper'], 'cavoatrial':['retrocardiac', 'svc', 'perihilar', 'atrium', 'middle', 'lower'], 
                'stomach':['below the diaphragm'], 'posterior':['rib'], 'anterior':['rib'], 'rib':['chest wall'],
                  'third':['rib'], 'fourth':['rib'], 'fifth':['rib'], 'sixth':['rib'], 'seventh':['rib'], 
                'eighth':['rib'], 'ninth':['rib'],
                'clavicular':['upper', 'apical'], 'spine':['perihilar'], 
                'retrocardiac':['left', 'lower', 'perihilar', 'base', 'atrium', 'ventricle', 'cavoatrial'] }

list_of_negative_labels_location = {'apical':['lower', 'base', 'retrocardiac', 'below the diaphragm', 'cavoatrial', 'svc', 'stomach', 'ventricle', 'atrium', 'above the carina'], 
                                    'middle':['apical', 'below the diaphragm', 'cavoatrial', 'base', 'ventricle', 'atrium', 'jugular', 'stomach'], 
                                     'right':['left', 'retrocardiac', 'cavoatrial', 'svc', 'atrium', 'ventricle', 'above the carina'], 
                                     'left':['right', 'cavoatrial', 'svc', 'atrium', 'above the carina'], 
                                     'upper':['lower', 'base', 'retrocardiac', 'below the diaphragm', 'cavoatrial', 'stomach', 'ventricle', 'atrium'], 
                                     'lower':['upper', 'apical', 'jugular', 'above the carina','clavicular'], 
                                     'base':['upper', 'apical', 'jugular', 'above the carina', 'middle', 'cavoatrial', 'svc'], 
                                     'lateral':['perihilar', 'retrocardiac', 'cavoatrial', 'svc', 'atrium', 'above the carina', 'spine'], 
                                     'perihilar':['lateral', 'below the diaphragm', 'stomach'],
                'pleural':['perihilar', 'atrium', 'ventricle', 'cavoatrial', 'svc', 'above the carina'], 
                'chest wall':['perihilar', 'atrium', 'ventricle', 'cavoatrial', 'svc', 'above the carina'], 
                'ventricle':['right', 'atrium', 'upper', 'apical', 'jugular', 'above the carina', 'pleural'], 
                'atrium':['ventricle', 'upper', 'apical', 'jugular', 'above the carina', 'pleural', 'lateral', 'chest wall'], 
                'svc':['below the diaphragm', 'stomach', 'atrium', 'jugular', 'lateral', 'retrocardiac', 'base', 'lower', 'upper', 'apical', 'ventricle', 'pleural', 'chest wall'],
                'below the diaphragm':['atrium', 'middle', 'upper', 'apical', 'jugular', 'perihilar', 'above the carina'], 
                'jugular':['below the diaphragm', 'stomach', 'atrium', 'ventricle', 'jugular', 'base', 'lower', 'middle', 'above the carina'],
                'above the carina':['below the diaphragm', 'stomach', 'atrium', 'ventricle', 'base', 'lower', 'left', 'right'], 
                'cavoatrial':['below the diaphragm', 'stomach', 'atrium', 'jugular', 'lateral', 'base', 'lower', 'upper', 'apical', 'above the carina', 'pleural', 'chest wall', 'right', 'left'], 
                'stomach':['atrium', 'middle', 'upper', 'apical', 'jugular', 'perihilar', 'above the carina'], 
                'posterior':['clavicular', 'anterior', 'spine'], 
                'anterior':['clavicular', 'posterior', 'spine'], 
                'rib':['clavicular', 'spine'],
                  'third':['fourth','fifth','sixth','seventh','eighth', 'ninth', 'clavicular', 'spine'], 
                  'fourth':['third','fifth','sixth','seventh','eighth', 'ninth', 'clavicular', 'spine'], 
                  'fifth':['third','fourth','sixth','seventh','eighth', 'ninth', 'clavicular', 'spine'], 
                  'sixth':['third','fourth','fifth','seventh','eighth', 'ninth', 'clavicular', 'spine'], 
                  'seventh':['third','fourth','fifth','sixth','eighth', 'ninth', 'clavicular', 'spine'], 
                'eighth':['third','fourth','fifth','sixth','seventh', 'ninth', 'clavicular', 'spine'], 
                'ninth':['third','fourth','fifth','sixth','seventh','eighth', 'clavicular', 'spine'],
                'clavicular':['third','fourth','fifth','sixth','seventh','eighth', 'ninth','spine', 'rib', 'lower'],
                'spine':['third','fourth','fifth','sixth','seventh','eighth', 'ninth','clavicular', 'rib', 'lateral'], 
                'retrocardiac': ['right', 'lateral', 'pleural', 'chest wall', 'upper', 'apical', 'jugular', 'above the carina']}