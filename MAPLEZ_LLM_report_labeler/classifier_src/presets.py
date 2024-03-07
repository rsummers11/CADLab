# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Description:
# File containing data augmentation presets.
# File modified from https://github.com/pytorch/vision/blob/main/references/classification/presets.py

import torch
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms

def get_module(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2

        return torchvision.transforms.v2
    else:
        import torchvision.transforms

        return torchvision.transforms


    
class ClassificationPresetTrain:
    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter. We may change that in the
    # future though, if we change the output type from the dataset.
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        backend="pil",
        use_v2=False,
    ):
        T = get_module(use_v2)
        class TrivialAugmentWideRedefined(T.TrivialAugmentWide):
            def _augmentation_space(self, num_bins):
                return {
                    # op_name: (magnitudes, signed)
                    "Identity": (torch.tensor(0.0), False),
                    "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
                    "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
                    "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
                    "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
                    "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
                    "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
                    "Color": (torch.linspace(0.0, 0.99, num_bins), True),
                    "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
                    "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
                    "Solarize": (torch.linspace(1.0, 0.0, num_bins), False),
                    "AutoContrast": (torch.tensor(0.0), False),
                    
                    #removed Equalize, Posterize
                }
        transforms = []
        backend = backend.lower()

        # transforms.append(T.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))
        # if hflip_prob > 0:
        #     transforms.append(T.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                class RandAugmentModified(T.RandAugment):
                    _AUGMENTATION_SPACE = {
                        "Identity": (lambda num_bins, height, width: None, False),
                        "ShearX": (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True),
                        "ShearY": (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True),
                        "TranslateX": (
                            lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * width, num_bins),
                            True,
                        ),
                        "TranslateY": (
                            lambda num_bins, height, width: torch.linspace(0.0, 150.0 / 331.0 * height, num_bins),
                            True,
                        ),
                        "Rotate": (lambda num_bins, height, width: torch.linspace(0.0, 30.0, num_bins), True),
                        "Brightness": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
                        "Color": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
                        "Contrast": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
                        "Sharpness": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
                        "Solarize": (lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins), False),
                        "AutoContrast": (lambda num_bins, height, width: None, False),
                    }
                    def _augmentation_space(self, num_bins, image_size):
                        return {
                            # op_name: (magnitudes, signed)
                            "Identity": (torch.tensor(0.0), False),
                            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
                            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
                            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
                            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
                            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
                            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
                            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
                            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
                            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
                            "Solarize": (torch.linspace(1.0, 0.0, num_bins), False),
                            "AutoContrast": (torch.tensor(0.0), False),
                        }
                transforms.append(RandAugmentModified(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                transforms.append(TrivialAugmentWideRedefined(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                class AugMixModified(T.AugMix):
                    _PARTIAL_AUGMENTATION_SPACE = {
                        "ShearX": (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True),
                        "ShearY": (lambda num_bins, height, width: torch.linspace(0.0, 0.3, num_bins), True),
                        "TranslateX": (lambda num_bins, height, width: torch.linspace(0.0, width / 3.0, num_bins), True),
                        "TranslateY": (lambda num_bins, height, width: torch.linspace(0.0, height / 3.0, num_bins), True),
                        "Rotate": (lambda num_bins, height, width: torch.linspace(0.0, 30.0, num_bins), True),
                        "Solarize": (lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins), False),
                        "AutoContrast": (lambda num_bins, height, width: None, False),
                    }
                    _AUGMENTATION_SPACE = {
                        **_PARTIAL_AUGMENTATION_SPACE,
                        "Brightness": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
                        "Color": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
                        "Contrast": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
                        "Sharpness": (lambda num_bins, height, width: torch.linspace(0.0, 0.9, num_bins), True),
                    }
                    def _augmentation_space(self, num_bins, image_size):
                        return {
                            # op_name: (magnitudes, signed)
                            "Identity": (torch.tensor(0.0), False),
                            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
                            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
                            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
                            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
                            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
                            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
                            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
                            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
                            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
                            "Solarize": (torch.linspace(1.0, 0.0, num_bins), False),
                            "AutoContrast": (torch.tensor(0.0), False),
                        }
                transforms.append(AugMixModified(interpolation=interpolation, severity=augmix_severity))
            elif auto_augment_policy == "old":
                transforms.append(torchvision.transforms.RandomAffine(degrees=45, translate=(0.15, 0.15),
                                scale=(0.85, 1.15), fill=0))
            else:
                aa_policy = T.AutoAugmentPolicy(auto_augment_policy)
                transforms.append(T.AutoAugment(policy=aa_policy, interpolation=interpolation))

        if backend == "pil":
            transforms.append(T.PILToTensor())

        transforms.extend(
            [
                T.ToDtype(torch.float, scale=True) if use_v2 else T.ConvertImageDtype(torch.float),
                # T.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            transforms.append(T.RandomErasing(p=random_erase_prob))

        if use_v2:
            transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        backend="pil",
        use_v2=False,
    ):
        T = get_module(use_v2)
        transforms = []
        backend = backend.lower()

        # transforms += [
        #     T.Resize(resize_size, interpolation=interpolation, antialias=True),
        #     T.CenterCrop(crop_size),
        # ]

        if backend == "pil":
            transforms.append(T.PILToTensor())

        transforms += [
            T.ToDtype(torch.float, scale=True) if use_v2 else T.ConvertImageDtype(torch.float),
            # T.Normalize(mean=mean, std=std),
        ]

        if use_v2:
            transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)
