# Automated abnormality classification of chest X-rays (CXR-Binary-Classifier) 

This project contains the code and labels of our **npj Digital Medicine** paper: “Automated abnormality classification of chest radiographs using deep convolutional neural networks” (accepted, to appear online soon).

    @article{tang2020npj,
        title={Automated abnormality classification of chest radiographs using deep convolutional neural networks},
        author={Tang, Yu-Xing and Tang, You-Bao and Peng, Yifan and Yan, Ke and Bagheri, Mohammadhadi and Redd, Bernadette A. and Brandon, Catherine J. and Lu, Zhiyong and Han, Mei and Xiao, Jing and Summers, Ronald M},
        journal= {npj Digital Medicine},
        volume={},
        number={},
        pages={},
        year={2020},
        publisher={Nature Publishing Group}
    }

Developed by Yuxing Tang (yuxing.tang@nih.gov), Imaging Biomarkers and Computer-Aided Diagnosis Laboratory, National Institutes of Health (NIH) Clinical Center.

# Introduction

CXR-Binary-Classifier is a **binary CNN classifier** than can predict if a chest radiograph (chest X-ray, or CXR) is normal or abnormal. The abnormalities include 14 abnormal findings that were defined in the [NIH-CXR paper](https://nihcc.app.box.com/v/ChestXray-NIHCC/file/256057377774).

    @inproceedings{wang2017chestx,
        title={Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases},
        author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
        booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
        pages={2097--2106},
        year={2017}
    }
Please refer to Section B of the Supplementary Materials of the [NIH-CXR paper](https://nihcc.app.box.com/v/ChestXray-NIHCC/file/256057377774).


## Labels

Binary labels (normal or abnormal) were extracted from the text reports associated with the chest X-ray image. Please note that only part of the 'no findings' CXRs in the NIH CXR data set are actually normal CXRs (some 'no findings' CXRs have other abnormalities outside the 14 abnormal findings). Please refer to our **npj Digital Medicine** paper to find the details of the label definition.

We provide the binary labels at the `./dataset_split` folder.  4 text files are listed under this directory. For each line in the text files, the first column shows the name of the CXR image and the second column shows the label (**0 indicates a normal CXR and 1 indicates an abnormal CXR**).
 1. `train.txt` Training split. Labels were extracted from the text reports using NLP.
 2. `val.txt` Validation split. Labels were extracted from the text reports using NLP.
 3. `test_attending_rad.txt`Testing split. Labels were first extracted from the text reports using NLP and then verified and corrected by a human reader by manually reading the text reports.
 4. `test_rad_consensus_voted3.txt`Testing split. Labels were constructed by the majority vote of 3 US board certified radiologist.
