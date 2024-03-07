# Privacy-Preserving LLM CXR report labeler

by [Ricardo Bigolin Lanfredi](https://github.com/ricbl)

Repository for data and code for the paper [Enhancing Chest X-ray Datasets with Privacy-Preserving LLMs and Multi-Type Annotations: A Data-Driven Approach for Improved Classification](). If you are here for the extended annotations of the ChestXray14 and MIMIC-CXR datasets, check the "new_dataset_annotations" folder in [this link](https://nihcc.box.com/s/ctvezyhzxitamo05gmlumuu8m8h999la).

## Datasets

We annotated reports for a few of the experiments in the paper and made them available to the public for reproducibility. Download the NIH ChestXray14 dataset annotations and reports, the MIMIC-CXR annotations, the CT dataset annotations and reports, the MRI dataset annotations and reports, and the PET dataset annotations and reports [here](https://nihcc.box.com/s/ctvezyhzxitamo05gmlumuu8m8h999la). 

To share the reports with the public, they underwent an additional round of anonymization after the paper's experiments. This anonymization focused on removing any missed dates and names from the previous anonymization, in addition to removing "Indication" sections that had been wrongly left in a few reports and removing any mention of a very rare diagnosis that could be used to identify the patient.

### Results for provided annotations

The two tables that contained labeler evaluations for the NIH dataset were Tables 1 and 13. Here, we present the impact on the median F1 scores for those evaluations:

| Abnormality      | Dataset | MAPLEZ (paper) | MAPLEZ (after second anonymization) |
| ---------------- | ------- | -------------- | ------------------- |
| Atelectasis      | NIH     | 0.820          | 0.806               |
| Cardiomegaly     | NIH     | 0.955          | 0.952               |
| Consolidation    | NIH     | 0.952          | 0.952               |
| Edema            | NIH     | 0.688          | 0.690               |
| Lung opacity     | NIH     | 0.923          | 0.924               |
| Pleural effusion | NIH     | 0.961          | 0.961               |
| Pneumothorax     | NIH     | 0.882          | 0.884               |

The tables containing labeler evaluations for the NIH dataset were 4, 17, and 18. Here, we present the impact on the median F1 scores for those evaluations:

Categorical label:

| Abnormality                               | Dataset | MAPLEZ (paper) | MAPLEZ (after second anonymization) |
| ----------------------------------------- | ------- | -------------- | ------------------- |
| Lung lesion                               | CT      | 0.902          | 0.881               |
| Liver lesion                              | CT      | 0.865          | 0.867               |
| Liver lesion                              | MRI     | 0.889          | 0.889               |
| Kidney lesion                             | MRI     | 0.882          | 0.884               |
| Hypermetabolic abnormality in the thorax  | PET     | 0.840          | 0.840               |
| Hypermetabolic abnormality in the abdomen | PET     | 0.850          | 0.850               |
| Hypermetabolic abnormality in the pelvis  | PET     | 0.552          | 0.615               |

Location labels:

| Abnormality      | Dataset | MAPLEZ (paper) | MAPLEZ (anonymized) |
| ---------------- | ------- | -------------- | ------------------- |
| Lung lesion      | CT      | 0.786          | 0.787               |
| Liver lesion     | CT      | 0.800          | 0.800               |
| Liver lesion     | MRI     | 0.848          | 0.857               |
| kidney lesion    | mri     | 0.793          | 0.793               |
| pleural effusion | ct      | 0.960          | 0.960               |

The results can be considered identical to the paper, and the differences were well within the confidence intervals. We also highlight that the provided annotations for the entire ChestXray 14 dataset and the provided annotations for the entire MIMIC-CXR dataset were the same as mentioned in the paper, so there is no change to classifier results (Tables 5, 6, 7, 8, 9, and 21).

## Using the MAPLEZ labeler

To learn about all the arguments to run the labeler, run `python labeler_src/one_load_model.py --help`.

Example for running the script with slurm:

```
sinteractive --gres=gpu:a100:2 --cpus-per-task=8  --mem=60g

python3 one_load_model.py --num-gpus=2 --download_location=./scratch/ --result_root=./test_run_results.csv --single_file ./mimic/files/mimic-cxr-reports/files/p10/p10010150/s50799795.txt
```

To run one_load_model.py for a batch of reports stored in a single CSV file, set `--dataset=nih --nih_reports_csv=<csv file>`, and the CSV file should have a column `image1` containing an ID for the report, and a row `anonymized_report` containing the report. To run one_load_model.py for a batch of reports where each report is stored in a separate txt file, set `--test_list=<list file>`, where the list file is a text file containing a list of paths to all report txt files, one per line.

The outputs of the script will be stored in a CSV file. 

For rows where "type_annotation" is "labels":
- -3 means uncertainty because stability was mentioned.
- -2 means not mentioned
- -1 means uncertainty because of other reasons
- 0 means absence mentioned
- 1 means presence mentioned

For rows where "type_annotation" is "probability":
- 101 means uncertainty because stability was mentioned.
- any other value represents a percentage probability of presence (0-100).

For rows where "type_annotation" is "severity":
- 1 means "mild"
- 2 means "moderate"
- 3 means "severe"
- -1 for when the label is absent, not mentioned, or severity is not mentioned.

For rows where "type_annotation" is "location", the output is a list with mentioned locations for each label, in lists separated by ";".

## Paper experiments

We used the following commands to generate the labeled outputs for each of the CXR datasets:

`python labeler_src/one_load_model.py --result_root <filename> --dataset mimic --use_generic_labels false`
`python labeler_src/one_load_model.py --result_root <filename> --dataset nih --use_generic_labels false`
`python labeler_src/one_load_model.py --result_root <filename> --dataset mimic --use_generic_labels true`
`python labeler_src/one_load_model.py --result_root <filename> --dataset nih --use_generic_labels true`

and the following commands to generate labeled outputs for the limited set of examples from other modalities:

`python labeler_src/one_load_model_other_modalities.py --result_root <filename> --modality ct --single_file ../../annotations/CT_annotations.csv`
`python labeler_src/one_load_model_other_modalities.py --result_root <filename> --modality mri --single_file ../../annotations/MRI_annotations.csv`
`python labeler_src/one_load_model_other_modalities.py --result_root <filename> --modality pet --single_file ../../annotations/PET_annotations.csv`

The create_raw_table.py file was used to create aggregated results and calculate p-values and confidence intervals.

For classifier experiments, check the [README file in the classifier folder](https://github.com/rsummers11/CADLab/tree/master/MAPLEZ_LLM_report_labeler/classifier_src/).

## Requirements

It was tested with

- Python 3.11
- torchvision                        0.12.0
- tqdm                               4.65.0
- torch                              1.11.0
- tokenizers                         0.13.3
- scikit-image                       0.21.0
- scikit-learn                       1.3.0
- scipy                              1.11.1
- numpy                              1.24.4
- accelerate                         0.22.0
- transformers                       4.31.0
- joblib                             1.3.0
- pandas                             2.0.3

## Licenses

All files in this repository should be used under the terms described in the Health_Data_LICENSE file. In addition to that, we release the code we produced and the files in the experiment_test_annotations folder under the terms in the LICENSE file. Code excerpts derived from other works are released under the original license. Files containing these excerpts with an MIT license retained the notice inside the respective file. These are the other licenses: 

Apache_2.0_LICENSE file:
- labeler_src/cli.py
- labeler_src/cli_other_modalities.py
- labeler_src/conversation.py

BSD_3-Clause_LICENSE file:
- classifier_src/utils.py
- classifier_src/transforms.py
- classifier_src/train_pytorch.py
- classifier_src/sampler.py
- classifier_src/presets.py

[CC BY-NC 4.0 Deed](https://creativecommons.org/licenses/by-nc/4.0/), since it is derived work from https://huggingface.co/upstage/SOLAR-0-70b-16bit/tree/main:
- new_dataset_annotations/mimic_llm_annotations.csv
- new_dataset_annotations/nih_llm_annotations_train.csv
- new_dataset_annotations/nih_llm_annotations_val.csv
- new_dataset_annotations/nih_llm_annotations_test.csv