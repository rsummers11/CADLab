# Automated abnormality classification of chest X-rays (CXR-Binary-Classifier) 

This project contains the code and labels of our **npj Digital Medicine** paper: “Automated abnormality classification of chest radiographs using deep convolutional neural networks” (accepted, to appear online soon).

    @article{tang2020npj,
        title={Automated abnormality classification of chest radiographs using deep convolutional neural networks},
        author={Tang, Yu-Xing and Tang, You-Bao and Peng, Yifan and Yan, Ke and Bagheri, Mohammadhadi and Redd, Bernadette A and Brandon, Catherine J and Lu, Zhiyong and Han, Mei and Xiao, Jing and Summers, Ronald M},
        journal= {npj Digital Medicine},
        volume={},
        number={},
        pages={},
        year={2020},
        publisher={Nature Publishing Group}
    }

Developed by Yuxing Tang (yuxing.tang@nih.gov), Imaging Biomarkers and Computer-Aided Diagnosis Laboratory, National Institutes of Health (NIH) Clinical Center.

## Introduction

CXR-Binary-Classifier is a **binary CNN classifier** that can predict if a chest radiograph (chest X-ray, or CXR) is normal or abnormal. The abnormalities include 14 abnormal findings that were defined in the [NIH-CXR paper](https://nihcc.app.box.com/v/ChestXray-NIHCC/file/256057377774).

    @inproceedings{wang2017chestx,
        title={Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases},
        author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
        booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
        pages={2097--2106},
        year={2017}
    }
Please refer to Section B of the Supplementary Materials of the [NIH-CXR paper](https://nihcc.app.box.com/v/ChestXray-NIHCC/file/256057377774).


### Labels

Binary labels (normal or abnormal) were extracted from the text reports associated with the chest X-ray image. Please note that only part of the 'no findings' CXRs in the NIH CXR data set are actually normal CXRs (some 'no findings' CXRs have other abnormalities outside the 14 abnormal findings). Please refer to our **npj Digital Medicine** paper to find the details of the label definition.

We provide binary labels at the `./dataset_split` folder.  4 text files are listed under this directory. For each line in the text files, the first column shows the name of the CXR image and the second column shows the label (**0 indicates a normal CXR and 1 indicates an abnormal CXR**).
 1. `train.txt` Training split. Labels were extracted from the text reports using NLP.
 2. `val.txt` Validation split. Labels were extracted from the text reports using NLP.
 3. `test_attending_rad.txt`Testing split. Labels were first extracted from the text reports using NLP and then verified and corrected by a human reader by manually reading the text reports compiled by the attending radiologists.
 4. `test_rad_consensus_voted3.txt`Testing split. Labels were constructed by the majority vote of 3 U.S. board-certified radiologists who read the CXR images independently.

## Running the code

### Requirements

Linux Ubuntu 16.04 (or later versions)
A GPU with 12G memories (_e.g._ NVIDIA TITAN XP)  
Anaconda 3 [https://www.anaconda.com/](https://www.anaconda.com/) (The below operations are based on Anaconda. But you can also use other virtual environment like *virtualenv*.)
CUDA (10.0 or 9.0 or 8.0)
PyTorch (0.4.0 to 1.4.0)
Python 3.7 (old versions should work too)  

### Configure the environment

If you use **Anaconda**:

 1. Create a conda environment `cxr-bin-cls` by using my `environment.yml` file:
	

    `conda env create -f environment.yml`

2.  Activate the conda virtual environment:
	

    `conda activate cxr-bin-cls`

If you do not use **Anaconda**, you have to alternatively install the necessary libraries listed in the `environment.yml`.

### Chest X-ray image data
Please download the CXR images from [here](https://nihcc.app.box.com/v/ChestXray-NIHCC/file/371647823217) and put them in `./images-nih/`.

### Run the testing code
If you just want to test our model, a DenseNet-121 CNN model trained on the NIH CXR data set for normal vs. abnormal classification is provided [here](https://nihcc.box.com/s/tiniov0agwsewzd243dxrt9mqa271pat).  You should put the trained model in `./trained_models_nih/`.

Run the following bash file to test the model:

    bash run_test_CXR.sh

You can modify `--test_labels 'att'`(to evaluate on the attending radiologist labels) to `--test_labels 'con'`(to evaluate on the consensus labels of 3 radiologists) in `run_test_CXR.sh`.

### Run the training code
We have provided an example to train a DenseNet-121 CNN model for normal vs. abnormal classification on the NIH CXR data set. You can modify the `train_densenet.py` file accordingly to train other CNN models or with other parameters.

    bash run_train_CXR.sh
After training is done, run `bash run_test_CXR.sh`to test the model.

### Results
Running the testing code will generate a receiver operating characteristic (ROC) curve in a PDF file. The quantitative results (_e.g._, accuracy, sensitivity, specificity, F1, PPV, NPV, etc) will be shown in the terminal.

## Software terms of use
Please refer to [Software Terms of Use-CXR Binary Classifier.pdf](https://github.com/rsummers11/CADLab/blob/master/CXR-Binary-Classifier/Software%20Terms%20of%20Use-CXR%20Binary%20Classifier.pdf).

Please cite our papers if you use our source code/data/labels/models.

    @article{tang2020npj,
        title={Automated abnormality classification of chest radiographs using deep convolutional neural networks},
        author={Tang, Yu-Xing and Tang, You-Bao and Peng, Yifan and Yan, Ke and Bagheri, Mohammadhadi and Redd, Bernadette A and Brandon, Catherine J and Lu, Zhiyong and Han, Mei and Xiao, Jing and Summers, Ronald M},
        journal= {npj Digital Medicine},
        volume={},
        number={},
        pages={},
        year={2020},
        publisher={Nature Publishing Group}
    }

    @inproceedings{wang2017chestx,
        title={Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases},
        author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
        booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
        pages={2097--2106},
        year={2017}
    }
