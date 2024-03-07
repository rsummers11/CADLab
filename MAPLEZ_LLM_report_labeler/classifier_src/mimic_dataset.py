# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Description:
# Auxiliary file containing functions handling the use of datasets

# file defining how the mimic dataset is preprocessed
import numpy as np
from utils_dataset import TransformsDataset, SeedingPytorchTransformSeveralElements, GrayToThree, ToNumpy, LoadToMemory
from utils_dataset import XRayResizerPadRound32, ToTensorMine, ToTensor1, Times255
from h5_dataset import H5Dataset,change_np_type_fn
import pandas as pd
import torchvision.transforms as transforms
from mimic_object import MIMICCXRDataset
from chexpert_object import ChexpertCXRDataset
from nih_object import NIHCXRDataset
from global_paths import h5_path, nih_dataset_location, chexpert_dataset_location
from utils_dataset import return_dataloaders
import torchvision
import presets
from torchvision.transforms.functional import InterpolationMode

def get_train_val_dfs(args):
    train_df = pd.read_csv(f'./train_df{"_all" if args.include_ap else ""}.csv')
    val_df = pd.read_csv(f'./val_df{"_all" if args.include_ap else ""}.csv')
    test_df = pd.read_csv(f'./test_df{"_all" if args.include_ap else ""}.csv')
    return train_df, val_df, test_df

def get_mimic_dataset_by_split(args, split, fn_create_dataset, post_transform, h5_filename, pad):
    if pad:
        pre_transform_train = [ToTensor1(), XRayResizerPadRound32(608), transforms.Resize(608, antialias=True), ToNumpy()]
    else:
        pre_transform_train = [ToTensor1(), transforms.Resize(608, antialias=True), transforms.CenterCrop(608), ToNumpy()]
        h5_filename = h5_filename + '_centercrop'

    return TransformsDataset(SeedingPytorchTransformSeveralElements(H5Dataset(path = h5_path, 
        filename = f'{split}_{h5_filename}',
        fn_create_dataset = lambda: 
            TransformsDataset(fn_create_dataset(split, args), pre_transform_train, 0),
         preprocessing_functions = [change_np_type_fn(np.ubyte, 1), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], # chest x-rays are converted to 1 byte of precision to save space and disk IO when saving as hdf5 file
         postprocessing_functions = [change_np_type_fn(np.float32, 1./255.), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
         n_processes = 16, load_to_memory = [False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]),
            post_transform, [0]  ),[ToTensorMine()], 1  )

def get_mimic_by_split(split,args):
    train_df, val_df, test_df = get_train_val_dfs(args)
    if args.do_generic:
        new_labels_df_llm = pd.read_csv('./mimic_llm_annotations_generic.csv')
    else:
        new_labels_df_llm = pd.read_csv('./new_dataset_annotations/mimic_llm_annotations.csv')
    new_labels_df_vqa = pd.read_csv('vqa_dataset_converted.csv')
    df_labels_reflacx = pd.read_csv('reflacx_dataset_converted.csv')
    return MIMICCXRDataset({'train': train_df, 'val': val_df, 'test': test_df}[split], new_labels_df_llm, new_labels_df_vqa, df_labels_reflacx)

def get_chexpert_by_split(split, args):
    test_df = pd.read_csv(chexpert_dataset_location + '/groundtruth.csv')
    val_df = pd.read_csv(chexpert_dataset_location + '/CheXpert/val_labels.csv')
    return ChexpertCXRDataset({'val': val_df, 'test': test_df}[split])

def get_nih_by_split(split, args):
    df = pd.read_csv(nih_dataset_location + 'Data_Entry_2017_v2020.csv')
    df_labels_pneumothorax = pd.read_csv('pneumothorax_relabeled_dataset_converted.csv')
    df_labels_pneumonia = pd.read_csv('pneumonia_relabeled_dataset_converted.csv')
    df_new_labels_llm_train = pd.read_csv('./new_dataset_annotations/nih_llm_annotations_train.csv')
    df_new_labels_llm_val = pd.read_csv('./new_dataset_annotations/nih_llm_annotations_val.csv')
    df_new_labels_llm_test = pd.read_csv('./new_dataset_annotations/nih_llm_annotations_test.csv') 
    return NIHCXRDataset(df, \
                        {'train':df_new_labels_llm_train, 'val': df_new_labels_llm_val, 'test': df_new_labels_llm_test}[split], \
                            df_labels_pneumothorax, df_labels_pneumonia)

def get_dataset(split, args, h5_filename = 'mimic_8_labels', fn_create_dataset = get_mimic_by_split):
    load_to_memory = args.load_to_memory
    use_data_aug = args.use_data_aug
    pretrained_weights_preprocessing = None
    if args.weights:
        weights = args.weights

        if not (args.model=='chexzero' or args.model=='xrv'):
            pretrained_weights_preprocessing = weights.transforms(antialias=True)
            preprocessing_step_1 = pretrained_weights_preprocessing
        else:
            pretrained_weights_preprocessing = weights.transforms
            preprocessing_step_1 = torchvision.transforms.Compose([Times255(), pretrained_weights_preprocessing])
            
    post_transform_val = preprocessing_step_1
    post_transform_train_with_data_aug = [GrayToThree(), ToTensorMine()]
    if not (args.model=='chexzero' or args.model=='xrv'):
        crop_size = pretrained_weights_preprocessing.crop_size[0]
    else:
        crop_size = args.resolution
    interpolation = InterpolationMode(args.interpolation)
    if split =='train' and use_data_aug:
        if args.use_old_aug:
            post_transform_train_with_data_aug += [transforms.RandomAffine(degrees=45, translate=(0.15, 0.15),
                                scale=(0.85, 1.15), fill=0)
                                ]
        else:   
            
            auto_augment_policy = getattr(args, "auto_augment", None)
            random_erase_prob = getattr(args, "random_erase", 0.0)
            ra_magnitude = getattr(args, "ra_magnitude", None)
            augmix_severity = getattr(args, "augmix_severity", None)
            # print(auto_augment_policy, random_erase_prob, ra_magnitude, augmix_severity)
            # 1/0

            post_transform_train_with_data_aug += [
            presets.ClassificationPresetTrain(
                        crop_size=crop_size,
                        interpolation=interpolation,
                        auto_augment_policy=auto_augment_policy,
                        random_erase_prob=random_erase_prob,
                        ra_magnitude=ra_magnitude,
                        augmix_severity=augmix_severity,
                        backend=args.backend,
                        use_v2=args.use_v2,
                    )]
        
    else:
        if use_data_aug and not args.use_old_aug:
            post_transform_train_with_data_aug += [presets.ClassificationPresetEval(
                        crop_size=crop_size,
                        interpolation=interpolation,
                        resize_size=crop_size,
                        backend=args.backend,
                        use_v2=args.use_v2,
                    )]
    post_transform_train_with_data_aug = torchvision.transforms.Compose(post_transform_train_with_data_aug)
    all_transforms = torchvision.transforms.Compose([post_transform_train_with_data_aug, post_transform_val])
    print(post_transform_val, post_transform_train_with_data_aug)
    imageset = get_mimic_dataset_by_split(args, split, fn_create_dataset, all_transforms, h5_filename, pad = args.pad)
    if load_to_memory:
        imageset = LoadToMemory(imageset)
    return return_dataloaders(lambda: imageset, split, args)

if __name__=='__main__':
    # code used to print the size and number of positive labels in each split of the filtered mimic-cxr dataset
    dataset = get_dataset('train', 1, use_data_aug = False, crop = True)
    print('Train')
    print('Dataset size:')
    print(len(dataset))
    print('Count positive labels:')
    dataset = get_dataset('val', 1, use_data_aug = False, crop = True)
    print('Val')
    print('Dataset size:')
    print(len(dataset))
    print('Count positive labels:')
    dataset = get_dataset('test', 1, use_data_aug = False, crop = True)
    print('Test')
    print('Dataset size:')
    print(len(dataset))
    print('Count positive labels:')
