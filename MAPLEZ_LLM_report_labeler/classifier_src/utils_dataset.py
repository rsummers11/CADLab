# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Description:
# file containing auxiliary classes or functions for pytorch datasets

from torch.utils.data import Dataset
import numpy as np
import torch
import multiprocessing
from joblib import Parallel, delayed
import torchvision.transforms as transforms
import math
import random
from transforms import get_mixup_cutmix
from sampler import RASampler

#Transforms

# Convert numpy array to float tensor while adding a channel dimension
class ToTensor1(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return torch.tensor(tensor).float().unsqueeze(0)

# Convert numpy array to tensor, keeping type
class ToTensorMine(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return torch.tensor(tensor)

# Convert pytorch tensor to numpy array, keeping type
class ToNumpy(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return np.array(tensor)

# Triple the channel dimension of grayscale images, for use with traditional CNN that process color images
class GrayToThree(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return np.tile(tensor, [3,1,1])

class Times255(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        return 255*(tensor-tensor.min())/(tensor.max()-tensor.min())

# resize the image such that one of the dimensions is equal to size
# if fn is max, the longest dimention will be equal to size
# if fn is min, the shortest dimention will be equal to size
class XRayResizerAR(object):
    def __init__(self, size, fn):
        self.size = size
        self.fn = fn
    
    def __call__(self, img):
        old_size = img.size()[-2:]
        ratio = float(self.size)/self.fn(old_size)
        new_size = tuple([round(x*ratio) for x in old_size])
        img = transforms.Resize(new_size, antialias = True)(img)
        return img

# get the size that makes the longest dimension a multiple of size_last_layer while making the shortest dimension
# as close to size as possible
def get_32_size(shape, size, size_last_layer = 16):
    projected_max_size = size/min(np.array(shape))*max(np.array(shape))
    return round(projected_max_size/size_last_layer)*size_last_layer

# resizes an image such that its longest dimension is resized following the rules given by the get_32_size
# The shortest dimension is padded to have the same size as the longest dimension
class XRayResizerPadRound32(object):
    def __init__(self, size, size_last_layer = 16):
        self.size = size
        self.size_last_layer = size_last_layer
    
    def __call__(self, img):
        self.resizer = XRayResizerAR(get_32_size(img.size()[-2:], self.size, self.size_last_layer), max)
        img = self.resizer(img)
        pad_width = (-np.array(img.size()[-2:])+max(np.array(img.size()[-2:])))/2
        return torch.nn.functional.pad(img, (math.floor(pad_width[1]),math.ceil(pad_width[1]),math.floor(pad_width[0]),math.ceil(pad_width[0])))

# Dataset/dataloader Wrappers

#dataset wrapper to load a dataset to memory for faster batch loading.
# The loading can be done using several threads by setting positive n_processes
class LoadToMemory(Dataset):
    def __init__(self, original_dataset, n_processes = 0):
        super().__init__()
        indices_iterations = np.arange(len(original_dataset))
        if n_processes>0:
            manager = multiprocessing.Manager()
            numpys = manager.list([original_dataset])
            self.list_elements = Parallel(n_jobs=n_processes, batch_size = 1)(delayed(self.get_one_sample)(list_index,element_index, numpys) for list_index, element_index in enumerate(indices_iterations))
        else:
            self.list_elements = [original_dataset[0]]*len(original_dataset)
            for list_index, element_index in enumerate(indices_iterations): 
                print(f'{element_index}+/{len(original_dataset)}')
                self.list_elements[list_index] = original_dataset[element_index]
                self.ait(element_index, list_index, original_dataset, self.list_elements)
    
    def __len__(self):
        return len(self.list_elements)
    
    def __getitem__(self, index):
        return self.list_elements[index]
    
    def get_one_sample(self, list_index,element_index, original_dataset_):
        print(f'{element_index}-/{len(original_dataset_[0])}')
        return original_dataset_[0][element_index]

#dataset wrapper to apply transformations to a pytorch dataset. indices_to_transform defines the indices of the elements
# of the tuple returned by original_dataset to which the transformation should be applied
class TransformsDataset(Dataset):
    def __init__(self, original_dataset, original_transform, indices_to_transform):
        if isinstance(indices_to_transform, int):
            indices_to_transform = [indices_to_transform]
        self.original_transform = original_transform
        if type(self.original_transform)==type([]):
            self.original_transform  = transforms.Compose(self.original_transform)
        self.indices_to_transform = indices_to_transform
        self.original_dataset = original_dataset
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        to_return = self.original_dataset[index]
        fn_run_transform = (lambda s,a: s.original_transform(a))
        for i in self.indices_to_transform:            
            to_return = [*to_return[:i], fn_run_transform(self, to_return[i]), *to_return[(i+1):]]
        return to_return

# Dataset wrapper that applies a sequence of transformations original_transform to elements indexed by 
# indices_to_transform while giving the same random seed to all transformations for the same sample
class SeedingPytorchTransformSeveralElements(TransformsDataset):
    def __init__(self, original_dataset, original_transform, indices_to_transform):
        super().__init__(original_dataset, original_transform, indices_to_transform)
        self.seed = None
        self.previouspythonstate = None
        self.previoustorchstate = None
        self.previousnumpystate = None
    
    def __getitem__(self, index):
        x = self.original_dataset[index]
        
        fn_run_transform = (lambda s,a: s.original_transform(a))
        
        #saving the random state from outside to restore it afterward
        outsidepythonstate = random.getstate()
        outsidetorchstate = torch.random.get_rng_state()
        outsidenumpystate = np.random.get_state()
        
        to_return = x
        for i in self.indices_to_transform:
            if self.previouspythonstate is None:
                if self.seed is not None:
                    # for the first sample in a training, set a seed
                    random.seed(self.seed)
                    torch.manual_seed(self.seed)
                    np.random.seed(self.seed)
            else:
                # restores the state from either last sample or last element of this sample
                random.setstate(self.previouspythonstate)
                torch.random.set_rng_state(self.previoustorchstate)
                np.random.set_state(self.previousnumpystate)
            
            # saves data augmentation random state to use the same state for all elements of this sample
            self.previouspythonstate = random.getstate() 
            self.previoustorchstate = torch.random.get_rng_state()
            self.previousnumpystate = np.random.get_state()
            
            # apply transform to element i
            to_return = [*to_return[:i], fn_run_transform(self, to_return[i]), *to_return[(i+1):]]
        
        # saves data augmentation random state to continue from same state when next sample is sampled
        self.previouspythonstate = random.getstate() 
        self.previoustorchstate = torch.random.get_rng_state()
        self.previousnumpystate = np.random.get_state()
        
        #restoring external random state
        random.setstate(outsidepythonstate)
        torch.random.set_rng_state(outsidetorchstate)
        np.random.set_state(outsidenumpystate)
        
        return to_return

from torch.utils.data.dataloader import default_collate

#generic function to get dataloaders from datasets
def return_dataloaders(instantiate_dataset, split, args):
    instantiated_dataset = instantiate_dataset()
    if args.distributed:
        if split=='train':
            if hasattr(args, "ra_sampler") and args.ra_sampler:
                sampler = RASampler(instantiated_dataset, shuffle=True, repetitions=args.ra_reps)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(instantiated_dataset)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(instantiated_dataset, shuffle=False)
    else:
        if split=='train':
            sampler = torch.utils.data.RandomSampler(instantiated_dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(instantiated_dataset) 

    batch_size = args.batch_size
    num_workers = args.num_workers
    if split=='train':
        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_categories=args.num_classes, use_v2=args.use_v2
        )
        if mixup_cutmix is not None:
            def collate_fn(batch):
                # return mixup_cutmix(*default_collate(batch))
                batch = default_collate(batch)
                # img, mimic_gt, new_gt, severities, location_labels, location_vector_index, probabilities, unchanged_uncertainties, vqa_new_gt, vqa_severities, vqa_location_labels, vqa_location_vector_index, vqa_probabilities, reflacx_new_gt, reflacx_probabilities, reflacx_present
                labels_to_use = batch[1 if args.labeler=='chexpert' else 2 if args.labeler=='llm' else 8]
                labels_to_use[labels_to_use==-3] = args.uncertainty_label
                labels_to_use[labels_to_use==-2] = 0
                labels_to_use[labels_to_use==-1] = args.uncertainty_label
                probabilities_to_use = batch[12 if args.labeler=='vqa' else 6]
                severity_to_use = batch[9 if args.labeler=='vqa' else 3]
                location_to_use = batch[10 if args.labeler=='vqa' else 4]
                mixed_up_batch = mixup_cutmix(batch[0], [labels_to_use, severity_to_use, location_to_use, probabilities_to_use])

                if args.labeler=='chexpert':
                    return mixed_up_batch[0], mixed_up_batch[1][0], batch[2], mixed_up_batch[1][1], mixed_up_batch[1][2], \
                        batch[5], mixed_up_batch[1][3], batch[7], batch[8],\
                        batch[9], batch[10], batch[11], batch[12], batch[13], batch[14], batch[15]
                if args.labeler=='llm':
                    return mixed_up_batch[0], batch[1], mixed_up_batch[1][0], mixed_up_batch[1][1], mixed_up_batch[1][2], batch[5], \
                        mixed_up_batch[1][3], batch[7], batch[8],\
                        batch[9], batch[10], batch[11], batch[12], batch[13], batch[14], batch[15]
                if args.labeler=='vqa':
                    return mixed_up_batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], mixed_up_batch[1][0],\
                        mixed_up_batch[1][1], mixed_up_batch[1][2], batch[11], mixed_up_batch[1][3], batch[13], batch[14], batch[15]
                
                
        else:
            collate_fn = default_collate
    else:
        collate_fn = default_collate
    return torch.utils.data.DataLoader(dataset=instantiated_dataset, batch_size=batch_size,
                        sampler = sampler, num_workers=num_workers, pin_memory=True, drop_last = (split=='train'), collate_fn = collate_fn), sampler