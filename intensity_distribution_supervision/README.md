# Improving segmentation and detection of lesions in CT scans using intensity distribution supervision

This is the code for ["Improving segmentation and detection of lesions in CT scans using intensity distribution supervision"](https://www.sciencedirect.com/science/article/abs/pii/S0895611123000770).

## Dependency
* Python 3.6.9
* numpy 1.19.2
* scipy 1.5.2
* nibabel 3.0.1
* MedPy 0.4.0
* scikit-image 0.17.2
* matplotlib 3.3.2
* torch 1.8.2
* torchvision 0.9.2

## Scripts
We provide key scripts (functions) for incorporating the proposed intensity distribution supervision in training a segmentation (detection) network.
1. dataset_stat.py
- To compute an intensity histogram and an intensity-based lesion probability (ILP) function of target lesions (e.g. Fig.1 in our paper).
2. dataset.py (and data_augmentation.py)
- Custom dataset class and data-augmentation functions including 'ComputeILP' for computing the corresponding ILP volume from each input volume.
- It requires a precomputed histogram file (.npy) from 'dataset_stat.py'.
3. loss.py
- Cross-entropy loss that we use for the ILP loss.

## Usage example (code)
This code should be modified according to one's purpose.
```
import torch
from torchvision import transforms
from torch.nn import functional as F
    
from dataset import CustomDataset
from dataset import Scale, Rotation, ElasticTransform
from dataset import PatchSampling, Normalization, ZeroCentering
from dataset import ComputeILP
from dataset import ToTensor

from loss import CrossEntropy 


device = torch.device("cuda")
aug_type = 'r'
weight_loss_ilp = 1


""" prepare a training set """
train_tsfm_list = []
if 's' in aug_type:
    train_tsfm_list.append(Scale())
if 'r' in aug_type:
    train_tsfm_list.append(Rotation())
if 'e' in aug_type:
    train_tsfm_list.append(ElasticTransform(use_segm=False))

train_tsfm_list.append(PatchSampling(size=224, lesion_segm_name='tumor_segm'))

# Note that 'ComputeILP' is executed before 'Normalization' since it requires Hounsfield unit values
# 'tumor_int_hist_list_sbct.npy' is a precomputed histogram file from 'dataset_stat.py'
train_tsfm_list.append(ComputeILP(target_int_file='lesion_stat/tumor_int_hist_list_sbct.npy'))  

train_tsfm_list.append(Normalization())
train_tsfm_list.append(ZeroCentering())
train_tsfm_list.append(ToTensor())

composed_transform_train = transforms.Compose(train_tsfm_list)

train_set = CustomDataset(set_txt = '../../dataset/carcinoid/carcinoid_det_pos.txt', \
                          root_dir = '../../dataset', \
                          data_type_list = ['img', 'tumor_segm'], \
                          transform = composed_transform_train)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                           shuffle=True, num_workers=15, pin_memory=True)


""" define a model """
model = ... # +1 output channel to predict the ILP
model = model.to(device)  


""" define an optimizer """
opt = ...


""" define losses """
# segm. loss
loss_func_lesion_segm = ... # the generalized Dice loss for example 

# ILP loss
loss_func_ilp = CrossEntropy()
    
    
""" train the model """
model.train()
iter_cnt = 0
for data in train_loader:
        
    img = data['img'].float().to(device)
    label_lesion_segm = data['tumor_segm'].bool().float().to(device)
    label_ilp = data['ilp'].float().to(device)

    opt.zero_grad()
    
    output = model(img)
    output_lesion_segm = F.sigmoid(output[:, [0], ...])
    output_ilp = F.sigmoid(output[:, [1], ...])

    loss_lesion_segm = loss_func_lesion_segm(output_lesion_segm, label_lesion_segm)
    loss_ilp = loss_func_ilp(output_ilp, label_ilp)

    loss = loss_lesion_segm + weight_loss_ilp * loss_ilp

    loss.backward()
    opt.step() 
    iter_cnt += 1
    
    ...
```

## Citation
```
@article{shin_cmig23,
    title = {Improving segmentation and detection of lesions in CT scans using intensity distribution supervision},
    journal = {Computerized Medical Imaging and Graphics},
    volume = {108},
    pages = {102259},
    year = {2023},
    issn = {0895-6111},
    doi = {https://doi.org/10.1016/j.compmedimag.2023.102259},
    url = {https://www.sciencedirect.com/science/article/pii/S0895611123000770},
    author = {Seung Yeon Shin and Thomas C. Shen and Ronald M. Summers},
}
```
