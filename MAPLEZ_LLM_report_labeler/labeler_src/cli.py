# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Description:
# Auxiliary file called by one_load_model.py.
# Substantially modified from https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/cli.py
# Changes included the use of automatic prompts and prompt output processing

import torch

from conversation import conv_templates, SeparatorStyle
from conversation import answer_template
import numpy as np
from collections import defaultdict, OrderedDict
from filelock import FileLock
import pandas as pd
from tqdm import tqdm
import os
from load import section_text
import re
from joblib import Parallel, delayed

class ProgressParallel(Parallel):
    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

from retry import retry

class Open(object):
    @retry((FileNotFoundError, IOError, OSError), delay=1, backoff=2, max_delay=10, tries=100)
    def __init__(self, file_name, method):
        self.file_obj = open(file_name, method)
    def __enter__(self):
        return self.file_obj
    def __exit__(self, type, value, traceback):
        self.file_obj.close()

@retry((FileNotFoundError, IOError, OSError), delay=1, backoff=2, max_delay=10, tries=100)
def get_lock(lock):
    lock.acquire()

@torch.inference_mode()
def generate_stream(tokenizer, model, params, device,
                    context_len=30000, stream_interval=2):

    prompt = params["prompt"]

    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 0))
    max_new_tokens = int(params.get("max_new_tokens", 512))
    stop_str = params.get("stop", None)

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    past_key_values = ''
    for i in range(max_new_tokens):
        
        if i == 0:
            out = model(
                    torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device)
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                            use_cache=True,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]
        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)
        
        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False
        
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            pos = output.rfind(stop_str, l_prompt)
            if pos != -1:
                output = output[:pos]
                stopped = True
            yield output

        if stopped:
            break

    del past_key_values

def parse_yes_no(sentence, fn_inputs):
    sentence = sentence.lower()
    if sentence[:3]=='yes':
        return 1
    else:
        return 0

def parse_severity(sentence, fn_inputs):
    sentence = sentence.lower().replace('"','').strip()
    if sentence not in ['mild','moderate','severe']:
        return -1
    return sentence

def parse_percentage(sentence, fn_inputs):
    sentence = int(re.sub(r'\D', '', sentence))
    if sentence>=90:
        return 1
    else:
        return 0

def parse_percentage_number(sentence, fn_inputs):
    sentence = re.sub(r'\D', '', sentence)
    if len(sentence)==0:
        return 101
    sentence = int(sentence)
    return sentence

def parse_location_filter(sentence, fn_inputs):
    if "1" in sentence:
        return 1
    return 0

class Node:
    def __init__(self, data, subdata, max_new_tokens=1, parse_sentence=parse_yes_no):
        self.subdata = subdata
        self.data = data
        self.max_new_tokens = max_new_tokens
        self.parse_sentence = parse_sentence
    def __len__(self):
        return len(self.subdata)
    def __getitem__(self, index):
        if self.subdata is None:
            return index
        return self.subdata[index]

def get_node_output(current_node, fn_inputs, tokenizer, model, model_name, do_tasks, args):
    if type(current_node)==list:
        return [get_node_output(current_node[node_index], fn_inputs, tokenizer, model, model_name, do_tasks, args) if do_tasks[node_index] else -1 for node_index in range(len(current_node)) ]
    if type(current_node)==Node:
        conv = conv_templates["v1"].copy()
        for prompt_index in range(len(current_node.data)):

            inp =f"{current_node.data[prompt_index](**fn_inputs)}"
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
        
            prompt = conv.get_prompt()
            
            params = {
                "model": model_name,
                "prompt": prompt,
                "temperature": args.temperature,
                "max_new_tokens": current_node.max_new_tokens,
                "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
            }
            pre = 0
            for outputs in generate_stream(tokenizer, model, params, args.device):
                outputs = outputs[len(prompt) + 1:].strip()
                outputs = outputs.split(" ")
                now = len(outputs)
                if now - 1 > pre:
                    pre = now - 1

            conv.messages[-1][-1] = " ".join(outputs)
        answer = current_node.parse_sentence(conv.messages[-1][-1], fn_inputs)
        current_node = current_node[answer]
        return get_node_output(current_node, fn_inputs, tokenizer, model, model_name, do_tasks, args)
    else:
        return current_node
        
def main(args, tokenizer, model):
    if os.path.isdir(args.result_root):
        args.result_root = args.result_root + "/parsing_results_llm.csv"
    report_label_file = args.result_root
    model_name = args.model

    labels_to_join = OrderedDict()

    if not args.use_generic_labels:
        label_set_mimic_labels =  ['enlarged cardiomediastinum (enlarged heart silhouette or large heart vascularity or cardiomegaly or abnormal mediastinal contour)',
            'cardiomegaly (enlarged cardiac/heart contours)', 'atelectasis (collapse of the lung)',
            'consolidation or infiltrate', 'lung edema (congestive heart failure)',
            'fracture (bone)', 'lung lesion (mass or nodule)',
            'pleural effusion (pleural fluid) or hydrothorax/hydropneumothorax', 'pneumonia (infection)', 'pneumothorax (or pneumomediastinum or hydropneumothorax)', 
            'medical equipment or medical support devices (lines or tubes or pacers or apparatus)',
            'lung opacity (or decreased lucency or lung scarring or bronchial thickening or infiltration or reticulation or interstitial lung)',
            'pleural abnormalities other than pleural effusion (pleural thickening, fibrosis, fibrothorax, pleural scaring)']

        label_set_mimic_probability =  ['enlarged cardiomediastinum (enlarged heart silhouette or large heart vascularity or cardiomegaly or abnormal mediastinal contour)',
            'cardiomegaly (enlarged cardiac/heart contours)', 'atelectasis (collapse of the lung)',
            'consolidation or infiltrate', 'lung edema (congestive heart failure)',
            'fracture (bone)', 'lung lesion (mass or nodule)', 
            'pleural effusion (pleural fluid) or hydrothorax/hydropneumothorax', 'pneumonia (infection)', 'pneumothorax (or pneumomediastinum or hydropneumothorax)', 
            'medical equipment or support device (line or tube or pacer or apparatus or valve or catheter)',
            'lung opacity (or decreased lucency or lung scarring or bronchial thickening or infiltration or reticulation or interstitial lung)',
            'fibrothorax (not lung fibrosis) or pleural thickening or abnormalities in the pleura (not pleural effusion)']

    label_set_mimic_generic =  ['enlarged cardiomediastinum',
            'cardiomegaly', 'atelectasis',
            'consolidation', 'lung edema',
            'fracture', 'lung lesion', 
            'pleural effusion', 'pneumonia', 'pneumothorax', 
            'medical equipment or support device',
            'lung opacity',
            'abnormalities in the pleura (not pleural effusion)']
    if args.use_generic_labels:
        label_set_mimic_labels = label_set_mimic_generic
        label_set_mimic_probability = label_set_mimic_generic

    label_set_mimic_location = label_set_mimic_probability
    label_set_mimic_severity = label_set_mimic_probability

    primary_labels = {}

    primary_labels['consolidation'] = [3]
    primary_labels['lung opacity'] = [11]

    labels_to_join['enlarged cardiomediastinum'] = [0]
    labels_to_join['cardiomegaly'] = [1]
    labels_to_join['atelectasis'] = [2]
    labels_to_join['consolidation'] = [3,8]
    labels_to_join['lung edema'] = [4]
    labels_to_join['fracture'] = [5]
    labels_to_join['lung lesion'] = [6]
    labels_to_join['pleural effusion'] = [7]
    labels_to_join['pneumonia'] = [8]
    labels_to_join['pneumothorax'] = [9]
    labels_to_join['support device'] = [10]
    labels_to_join['lung opacity'] = [11,2, 3,
                                4, 6, 8]
    labels_to_join['pleural other'] = [12]

    tasks = ['labels','probability','location','severity']
    do_tasks = args.do_tasks
    heart_labels = [0,1]
    do_labels = args.do_labels

    def parse_location(sentence_answer, fn_inputs):
        report = fn_inputs['sentence_']
        label = fn_inputs['label_'] 
        sentence_answer = sentence_answer.lower().replace(',',';').replace('"','').replace('[','').replace(']','').replace('; ',';').strip()
        location_expressions = []
        sentence_split = sentence_answer.split(';')
        res = []
        [res.append(x) for x in sentence_split if x not in res]
        sentence_split = res

        location_filter_prompt = Node([lambda sentence_, label_, expression_: f'''Consider in your answer: 1) medical wording synonyms, subtypes of abnormalities 2) abreviations of the medical vocabulary. Given the complete report "{sentence_}", is the isolated adjective "{expression_}", on its own, characterizing a medical finding in what way? Respond only with "_" where _ is the number corresponding to the correct answer.
(1) Anatomical location of "{label_set_mimic_location[label_]}"
(2) Comparison with a previous report for "{label_set_mimic_location[label_]}"
(3) Severity of "{label_set_mimic_location[label_]}"
(4) Size of "{label_set_mimic_location[label_]}"
(5) Probability of presence of "{label_set_mimic_location[label_]}"
(6) Visual texture description of "{label_set_mimic_location[label_]}"
(7) It is not characterizing the "{label_set_mimic_location[label_]}" mention noun
(8) A type of support device
Answer:"'''],[False,True], 3 , parse_location_filter)
        for expression in sentence_split:
            if len(expression.strip())>0:
                sentence_outputs = get_node_output(location_filter_prompt, {'sentence_':report, 'label_':label, 'expression_':expression}, tokenizer, model, model_name, None, args)
                if sentence_outputs:
                    location_expressions.append(expression)
        return ";".join(location_expressions)

    if args.dataset=='mimic':
        mimic_root = args.mimic_root
        mimic_report_root = f'{mimic_root}/files/mimic-cxr-reports'

        chexpert_df = pd.read_csv(os.path.join(mimic_root, 'mimic-cxr-2.0.0-chexpert.csv.gz'), compression='gzip').astype('Int64')
    elif args.dataset=='nih':
        chexpert_df = pd.read_csv(args.nih_reports_csv)
        # chexpert_df['subject_id'] = chexpert_df['image1'].apply(lambda row: row.split('/')[-1].split('.')[0])
        # chexpert_df['subject_id'] = chexpert_df['image2'].apply(lambda row: '')
        chexpert_df['subject_id'] = ''

        # chexpert_df['study_id'] = chexpert_df['image2'].apply(lambda row: row.split('/')[-1].split('.')[0])
        chexpert_df['study_id'] = chexpert_df['image1']

    elif args.single_file is not None or args.test_list is not None:
        if args.single_file is not None:
            file_paths = args.single_file
        else:
            file_paths = []
            with open(args.test_list, 'r') as input_file:
                for line in input_file:
                    # Remove leading and trailing whitespace and newline characters
                    file_paths.append(line.strip())
        data = {'filepath': [], 'report': [], 'study_id': [], 'subject_id': []}

        # Loop through the file paths and extract the data
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                file_content = file.read()
            
            # Extract study_id from the filename without extension
            study_id = os.path.splitext(os.path.basename(file_path))[0]
            
            # Append data to lists
            data['filepath'].append(file_path)
            data['report'].append(file_content)
            data['study_id'].append(study_id)
            data['subject_id'].append('')

        # Create a DataFrame from the data
        chexpert_df = pd.DataFrame(data)
    if args.end_index is None:
        chexpert_df = chexpert_df.iloc[args.start_index:,:]
    else:
        chexpert_df = chexpert_df.iloc[args.start_index:args.end_index,:]

    report_label_file = report_label_file
    report_label_lockfile = f"{report_label_file}.lock"
    report_label_lock = FileLock(report_label_lockfile, timeout = 300)

    get_lock(report_label_lock)
    values_to_add = []
    for destination_label in labels_to_join:
        values_to_add.append(destination_label)

    final_label_set = values_to_add
    header_text_labels = '","'.join(final_label_set)
    try:
        if not os.path.isfile(report_label_file):
            with Open(report_label_file, "w") as f:
                f.write(f'subjectid_studyid,report,type_annotation,"{header_text_labels}"\n')
    finally:
        report_label_lock.release()
    
    get_lock(report_label_lock)
    try:
        starting_state = pd.read_csv(report_label_file, sep = ',')['subjectid_studyid'].values
    finally:
        report_label_lock.release()
    
    location_node =  Node([lambda sentence_, label_: f'Given the complete report "{sentence_}", does it mention a location for specifically "{label_set_mimic_location[label_]}"? Respond only with "Yes" or "No".'],
    ['',Node([lambda sentence_, label_: f'Given the report "{sentence_}", list the localizing expressions characterizing specifically the "{label_set_mimic_location[label_]}" finding. Each adjective expression should be between quotes, broken down into each and every one of the localizing adjectives and each independent localiziation prepositional phrase, and separated by comma. Output an empty list ("[]" is an empty list) if there are 0 locations mentioned for "{label_set_mimic_location[label_]}". Do not mention the central nouns identified as "{label_set_mimic_location[label_]}". Do not mention any nouns that are not part of an adjective. Only include in your answer location adjectives adjacent to the mention of the "{label_set_mimic_location[label_]}" finding. Exclude from your answer adjectives for other findings. Use very short answers without complete sentences. Start the list (0+ elements) of only localizing adjectives or localizing expressions (preposition + noun) right here: ['],None, 200, parse_location)
    ])
    
    severity_node =  Node([lambda sentence_, label_: f'Given the complete report "{sentence_}", would you be able to characterize the severity of "{label_set_mimic_severity[label_]}", as either "Mild", "Moderate" or "Severe" only from the words of the report? Respond only with "Yes" or "No".'],
        [-1,Node([lambda sentence_, label_: f'Given the complete report "{sentence_}", characterize the severity of "{label_set_mimic_severity[label_]}" as either "Mild", "Moderate" or "Severe" or "Undefined" only from the words of the report, and not from comparisons or changes. Do not add extra words to your answer and exclusively use the words from one of those four options.'],None, 5, parse_severity)
        ])

    # severity_node =  Node([lambda sentence_, label_: f'Given the complete report "{sentence_}", would you be able to characterize the severity of "{label_set_mimic_severity[label_]}", as either "Mild/Small", "Moderate" or "Severe/Large" only from the words of the report? Respond only with "Yes" or "No".'],
    #     [-1,Node([lambda sentence_, label_: f'Given the complete report "{sentence_}", characterize the severity of "{label_set_mimic_severity[label_]}" as either "Mild/Small", "Moderate" or "Severe/Large" or "Undefined" only from the words of the report, and not from comparisons or changes. Do not add extra words to your answer and exclusively use the words from one of those four options.'],None, 5, parse_severity)
    #     ])
    
    first_part = 'Consider in your answer: 1) radiologists might skip some findings because of their low priority 2) explore all range of probabilities, giving preference to non-round probabilities 3) medical wording synonyms, subtypes of abnormalities 4) radiologists might express their uncertainty using words such as "or", "possibly", "can\'t exclude", etc..' 
    number_prompt =  Node([lambda sentence_, label_: f'{first_part}. Given the complete report "{sentence_}", estimate from the report wording how likely another radiologist is to observe the presence of any type of "{label_set_mimic_probability[label_]}" in the same imaging. Respond with the template "___% likely." and no other words.'],
        list(range(102)), 3, parse_percentage_number)
    
    number_prompt_present =  Node([lambda sentence_, label_: f'{first_part} Given the complete report "{sentence_}", consistent with the radiologist observing "{label_set_mimic_probability[label_]}", estimate from the report wording how likely another radiologist is to observe the presence of any type of "{label_set_mimic_probability[label_]}" in the same imaging. Respond with the template "___% likely." and no other words.'],
        list(range(102)), 4, parse_percentage_number)
    
    number_prompt_absent =  Node([lambda sentence_, label_: f'{first_part} Given the complete report "{sentence_}", consistent with the radiologist stating the absence of evidence "{label_set_mimic_probability[label_]}", estimate from the report wording how likely another radiologist is to observe the presence of any type of "{label_set_mimic_probability[label_]}" in the same imaging. Respond with the template "___% likely." and no other words.'],
        list(range(102)), 4, parse_percentage_number)

    inner_prompts = Node([lambda sentence_, label_:f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if the radiologist characterized {"specifically" if label_ not in (heart_labels) else ""} "{label_set_mimic_labels[label_]}" as stable or unchanged. Respond only with "Yes" or "No".'],
                [Node([lambda sentence_, label_: f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if the radiologist stated the absence of evidence of "{label_set_mimic_labels[label_]}". Respond only with "Yes" or "No".'],
                    [[-2, number_prompt, '', -1],
                    [0, number_prompt, '', -1]]),
                Node([lambda sentence_, label_: 'Say "Yes".' if label_ in labels_to_join['support device'] else  f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if "{label_set_mimic_labels[label_]}" might be present. Respond only with "Yes" or "No".'],
                    [Node([lambda sentence_, label_: f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if the radiologist characterized {"specifically" if label_ not in (heart_labels) else ""} "{label_set_mimic_labels[label_]}" as normal. Respond only with "Yes" or "No".'],
                        [[-3,101,'',-1],
                        [0, number_prompt_absent, '', -1]]
                        ),
                    [1, number_prompt_present, location_node, severity_node]]
                )])
    number_inner_prompts_solo = Node([lambda sentence_, label_:f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if the radiologist characterized {"specifically" if label_ not in (heart_labels) else ""} "{label_set_mimic_probability[label_]}" as stable or unchanged. Respond only with "Yes" or "No".'],
            [number_prompt,
            Node([lambda sentence_, label_: 'Say "Yes".' if label_ in labels_to_join['support device'] else  f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if "{label_set_mimic_probability[label_]}" might be present. Respond only with "Yes" or "No".'],
                [Node([lambda sentence_, label_: f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if the radiologist characterized {"specifically" if label_ not in (heart_labels) else ""} "{label_set_mimic_probability[label_]}" as normal. Respond only with "Yes" or "No".'],
                    [101,
                    number_prompt_absent]
                    ),
                number_prompt_present]
            )])
    prompts = Node([lambda sentence_, label_: f'Given the full report "{sentence_}", use a one sentence logical deductive reasoning to infer if the radiologist observed possible presence of evidence of "{label_set_mimic_labels[label_]}". Respond only with "Yes" or "No".'],
            [inner_prompts,
            [Node([lambda sentence_, label_: f'Given the complete report "{sentence_}", consistent with the radiologist observing "{label_set_mimic_labels[label_]}", estimate from the report wording how likely another radiologist is to observe "{label_set_mimic_labels[label_]}" in the same imaging. Respond with the template "___% likely." and no other words.'],
                [-1,
                1], 4, parse_percentage),
            number_inner_prompts_solo,
            location_node,
            severity_node
            ]])

    def run_one_report(idx, row, model_list):
        
        subject_id = row['subject_id']
        study_id = row['study_id']
        if f"{row['subject_id']}_{row['study_id']}" in starting_state:
            return
        model = model_list[0]
        if args.dataset == 'mimic':
            report_path = os.path.join(
                mimic_report_root,
                f'files/p{str(subject_id)[:2]}/p{str(subject_id)}/s{str(study_id)}.txt')
            with Open(report_path, 'r') as f:
                report = f.read()
            extracted_report = section_text(report, study_id)
            if len(extracted_report)>0:
                report = extracted_report
                
        elif args.dataset == 'nih':
            report = row['anonymized_report']
        else:
            report = row['report']
        
        report = report.replace('\n', '').replace('\t', '').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')

        sentences = [sentence.replace('\n','').strip().replace('___','thing').replace('XXXX','thing').replace('  ',' ') for sentence in [report] if len(sentence.strip())>0]
        
        all_report_outputs = []
        for sentence in sentences:
            
            for label_index in range(len(label_set_mimic_generic)):
                if not do_labels[label_index]:
                    all_report_outputs.append([-1,-1,-1,-1])
                    continue
                sentence_outputs = []
                current_node = prompts
                sentence_outputs = get_node_output(current_node, {'sentence_':sentence, 'label_':label_index}, tokenizer, model, model_name, do_tasks, args)
                all_report_outputs.append(sentence_outputs)
        all_report_outputs = [list(row) for row in zip(*all_report_outputs)]
        if len(all_report_outputs)>0:
            for index_task, task in enumerate(tasks):
                if not do_tasks[index_task]:
                    continue
                report_outputs = all_report_outputs[index_task]
                if task=='probability' or task=='labels':
                    report_outputs = np.array(report_outputs)
                    labels_of_joined_labels = defaultdict(lambda: -2)
                elif task=='location' or task=='severity':
                    if task=='severity':
                        labels_of_joined_labels = defaultdict(lambda: -1)
                    if task=='location':
                        labels_of_joined_labels = defaultdict(lambda: '')

                values_to_add = []

                for destination_label in labels_to_join:
                    for index_origin_label in labels_to_join[destination_label]:
                        
                        if task=='probability':
                            report_outputs[report_outputs == 101] = 51
                        elif task=='labels':
                            report_outputs[report_outputs == 1] = 10
                            report_outputs[report_outputs == -1] = 7
                            report_outputs[report_outputs == -3] = 3
                            report_outputs[report_outputs == -2] = 2
                            # if the primary label (consolidation, lung opacity) was 
                            # said to be absent, the the aggregated label should be absent.
                            # Making the 0 be a higher number (5) than the -2 (2) for the primary label
                            # will make it be kept through the maximum function
                            if not (destination_label not in primary_labels or index_origin_label not in primary_labels[destination_label]):
                                report_outputs[report_outputs == 0] = 5
                        elif task=='severity':
                            report_outputs = [-1 if x == "undefined" else x for x in report_outputs]
                            report_outputs = [1 if x == "mild" else x for x in report_outputs]
                            report_outputs = [2 if x == "moderate" else x for x in report_outputs]
                            report_outputs = [3 if x == "severe" else x for x in report_outputs]
                            report_outputs = np.array(report_outputs)
                        elif task=='location':
                            report_outputs = ['' if x == -1 else x for x in report_outputs]
                        if task=='location':
                            labels_of_joined_labels[destination_label] = (labels_of_joined_labels[destination_label].split(';') if (len(labels_of_joined_labels[destination_label])>0) else []) + (report_outputs[index_origin_label].split(';') if (len(report_outputs[index_origin_label])>0) else [])
                            res = []
                            [res.append(x) for x in labels_of_joined_labels[destination_label] if x not in res]
                            labels_of_joined_labels[destination_label] = ';'.join(res)
                        else:
                            labels_of_joined_labels[destination_label] = max(labels_of_joined_labels[destination_label], report_outputs[index_origin_label])
                        if task=='labels':
                            report_outputs[report_outputs == 10] = 1
                            report_outputs[report_outputs == 7] = -1
                            report_outputs[report_outputs == 3] = -3
                            report_outputs[report_outputs == 2] = -2
                            report_outputs[report_outputs == 5] = 0
                        elif task=='probability':
                            report_outputs[report_outputs == 51] = 101
                    values_to_add.append(labels_of_joined_labels[destination_label])
                report_outputs = values_to_add
                if task=='probability' or task=='labels':
                    report_outputs = np.array(report_outputs)
                if task=='probability':
                    report_outputs[report_outputs == 51] = 101
                elif task=='labels':
                    report_outputs[report_outputs == 10] = 1
                    report_outputs[report_outputs == 7] = -1
                    report_outputs[report_outputs == 5] = 0
                    report_outputs[report_outputs == 2] = -2
                    report_outputs[report_outputs == 3] = -3
                elif task=='location':
                    report_outputs = [x[1:] if (len(x)>0 and x[0]==';') else x for x in report_outputs]
                    report_outputs = [-1 if x == '' else x for x in report_outputs]
                if task=='probability' or task=='labels':
                    report_outputs = report_outputs.tolist()

                line_text = ','.join([str(value) for value in report_outputs])
                get_lock(report_label_lock)
                try:
                    with Open(report_label_file, "a") as f:
                        skipped_report = report.replace('"','""').replace('\n',' ')
                        skipped_line_text = line_text.replace('"','""').replace('\n',' ')
                        f.write(f'{subject_id}_{study_id},"{skipped_report}",{task},{skipped_line_text}\n')
                finally:
                    report_label_lock.release()
        return
    import joblib
    import contextlib
    @contextlib.contextmanager
    def tqdm_joblib(tqdm_object):
        """Context manager to patch joblib to report into tqdm progress bar given as argument"""
        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()
    with tqdm_joblib(tqdm(desc="Parsing reports", total=len(chexpert_df))) as progress_bar:
        Parallel(n_jobs=args.n_jobs, batch_size = 1, require='sharedmem')(delayed(run_one_report)(idx,row, [model]) for idx, row in chexpert_df.reset_index().iterrows())