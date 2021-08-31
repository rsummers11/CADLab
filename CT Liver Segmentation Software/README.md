# Liver Segmentation tool

This software segments the liver on either contrast-enhanced or noncontrast CT.

## Installation
Copy the code folder to target system.

Install requirements:

pytorch 0.4 or higher
nibabel
simpleitk
numpy
importlib
pprint
argparse
multiprocessing

Hardware requirements:

GPU with >= 6 GB memory recommended.

Download the model file "LiverNC" from here:
https://nihcc.box.com/s/3mkx3whudn8peylw6apsvik5cjc4n6cy
Put it in the configs folder.


## Example Usage

python  Inference.py --test_list fn_list_example.txt --result_root output_dir --model LiverNC


```
usage: Inference.py [-h] [--test_list TEST_LIST] [--result_root RESULT_ROOT]
                      [--model MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --test_list TEST_LIST
                        text file containing a list of test images in NIFTI format, one filename per line
  --result_root RESULT_ROOT
                        folder to store result
  --model MODEL         Name of the model file (and the config file)

```
If you find this code useful, please cite this paper:
Sandfort V, Yan K, Pickhardt PJ, Summers RM. Data augmentation using generative adversarial networks (CycleGAN) to improve generalizability in CT segmentation tasks. Sci Rep 2019;9(1):16884. doi: 10.1038/s41598-019-52737-x
