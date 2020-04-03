# Introduction

CXR-Binary-Classifier is a **binary CNN classifier** than can predict if a chest radiograph (chest X-ray, or CXR) is normal or abnormal. The abnormalities include 14 abnormal findings that were defined in the [NIH-CXR paper](https://nihcc.app.box.com/v/ChestXray-NIHCC/file/256057377774) 
  @inproceedings{wang2017chestx,
    title={Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases},
    author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    pages={2097--2106},
    year={2017}
  }
Please refer to Section B of the Supplementary Materials.


## Labels

Binary labels (normal or abnormal) were extracted from the text reports associated with the chest X-ray image. Please not that only part of the 'no findings' CXRs in the NIH CXR data set are actually normal CXRs (some 'no findings' CXRs have other abnormalities outside the 14 abnormal findings). Please refer to our **npj Digital Medicine** paper to find the details of the label definition.
