# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-07-12
# Use `python <python_file> --help` to check inputs and outputs
# Description:
# This file runs the LLM labeler for all of the CT, MRI, or PET reports in a csv file or dataset

import sys
import cli_other_modalities as cli
import torch
import argparse
import importlib
import traceback
import math

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="upstage/Llama-2-70b-instruct-v2", help='''The name of a huggingface LLM model 
to use or the path to where a LLM is locally saved. Default: upstage/Llama-2-70b-instruct-v2''')
    parser.add_argument("--num-gpus", type=str, default="5", help='The number of gpus used to run the model. For A100 80GB GPUs, 2 are needed. For V100X 32GB GPUS, 5 are needed. Default: 2.')
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help='Use "--device=cuda" to run on GPUs and "--device=cpu" to run on cpus. Default: cuda')
    parser.add_argument("--temperature", type=float, default=0.0, help='''The temperature with which to run the model. A temperature of 0 will run the model determinitically and always have the same outputs. 
The higher the temperature, the more "creative" the model gets. The temperature that the model was validated with was 0. Default: 0''')
    parser.add_argument("--start_index", type=int, default = 0, help="The index of the first report to run the script for. Useful for dividing runs for full datasets into several clusters. Default: 0")
    parser.add_argument("--end_index", type=int, default = None, help='''The index of the last report to run the script for plus one. For example, use 
"--start_index=20000 --end_index=40000" to run the script for the 20000 reports from report 20000 to report 39999. Default: Run until the last of the reports in the list/dataset.''')
    parser.add_argument("--download_location", type=str, default = "./scratch/", help='''The folder where to download the model to,
or load the model from it it has already been downloaded, in case a huggingface model name is provided in the --model argument. Default: ./scratch/''')
    parser.add_argument("--result_root", type=str, default = './', help='''Where to save the results. If only a folder is provided, 
the results will be saved to the "parsing_results_llm.csv" file. You may also provide a csv file name with your path to change the filename where it is saved. If 
the csv file already exists, results will be appended to that file. If any results were already present in that csv for file with the same ID (report filename),
that report file will be skipped. The script can be run in parallel in several servers with no problems to writing the outputs, since access to the file is locked
when writing each row. Default: ./''')
    parser.add_argument("--keep_model_loaded_in_memory", type=str2bool, default = "false", help='''Turning this flag to True will allow rerun of the script with 
modifications of the cli.py file without reloading the model from disk to memory (~3 minutes). This flag was mainly used for development. 
Default: False''')
    parser.add_argument('--single_file', type=str, default = None, help='''The path to a csv file containing reports in a column named "report" and ids in a column named "subjectid_studyid".''')
    parser.add_argument('--do_tasks', type=int, default = [1,1,1,1], nargs=4, help='''List of 4 0s or 1s used to indicate which type of labels to run. The indices of the tasks are:
0-> categorical labels of presence, 1-> probability soft labels of presence, 2-> location expressions, 3-> severity categorical labels. For example
 use "--do_tasks 0 0 1 0" to run only the location task. Running all of them in one script is faster than running each of them in 4 separate scripts, since they share a few common branches in the
decision tree of prompts to provide to the LLM. Default: run it for all tasks.''')
    parser.add_argument('--do_labels', type=int, default = None, nargs='+', help='''List of 13 0s or 1s used to indicate which labels to run the script for. 
The indices of the labels are: 0-> enlarged cardiomediastinum, 1 -> cardiomegaly, 2 -> atelectasis, 3 -> consolidation,
4 -> lung edema, 5 -> fracture, 6 -> lung lesion, 7 -> pleural effusion, 8 -> pneumonia, 9 -> pneumothorax, 10 -> support device, 
11 -> lung opacity, 12 -> pleural other. Be advised that for consolidation and lung opacity, the output of the model might be incomplete if you do not run some
of the other labels whose results are concatenated to those. For consolidation, run pneumonia and consolidation to get complete results. For 
lung opacity, run atelectasis, consolidation, lung edema, lung lesion, pneumonia and lung opacity to get complete results. For example, use "--do_labels 0 0 1 1 1 0 1 0 1 0 0 1 0" to get a complete
output for lung opacity. Default: run it for all labels''')
    parser.add_argument('--n_jobs', type=int, default = 8, help='''Number of parallel jobs to run in this script. For 5 V100X GPU, running it with 2 parallel jobs made a fuller use of the GPU than with only 1 job, 
running it at almost double the speed. No speedup was seen for running it with more than 2 jobs. Default: 8 jobs.''')
    parser.add_argument('--modality', type=str, choices = ['pet', 'mri', 'ct'], default = None, help='''The modality of the reprots to run this script for.''')

    args = parser.parse_args()
    download_location = args.download_location
    model_name = args.model
    num_gpus = args.num_gpus
    if args.do_labels is None:
        if args.modality == 'ct':
            args.do_labels = [1,1,1,1,1]
        if args.modality == 'mri':
            args.do_labels = [1,1,1]
        if args.modality == 'pet':
            args.do_labels = [1,1,1]
    reloaded = False
    try:
        importlib.reload(cli)
        reloaded = True
    except Exception as e: 
            traceback.print_exception(*sys.exc_info())
    # Model
    if args.device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: f"{math.ceil(155/num_gpus)}GiB" for i in range(num_gpus)},
                })
    elif args.device == "cpu":
        kwargs = {}
    else:
        raise ValueError(f"Invalid device: {args.device}")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, cache_dir=download_location)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True,
        low_cpu_mem_usage=True, cache_dir=download_location, **kwargs)

    if args.device == "cuda" and num_gpus == 1:
        model.cuda()
        
    try:
        cli.main(args, tokenizer, model)
    except Exception as e: 
        traceback.print_exception(*sys.exc_info())

    if args.keep_model_loaded_in_memory:
        while True:
            if reloaded:
                try:
                    cli.main(args, tokenizer, model)
                except Exception as e: 
                    traceback.print_exception(*sys.exc_info())
            reloaded = False
            print("Press enter to re-run the script, CTRL-C to exit")
            
            sys.stdin.readline()
            try:
                importlib.reload(cli)
                reloaded = True
            except Exception as e: 
                traceback.print_exception(*sys.exc_info())
        