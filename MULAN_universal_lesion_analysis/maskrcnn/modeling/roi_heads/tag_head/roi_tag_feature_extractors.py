# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
from torch import nn
from torch.nn import functional as F

# from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn.modeling.poolers import Pooler
from maskrcnn.layers import Conv2d
from maskrcnn.config import cfg


class TagFeatureExtractor(nn.Module):

    def __init__(self):
        super(TagFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS * resolution ** 2
        representation_size = cfg.MODEL.ROI_TAG_HEAD.MLP_HEAD_DIM
        self.fc8 = nn.Linear(self.input_size, representation_size)
        self.fc9 = nn.Linear(representation_size, representation_size)

        self.pooler = pooler
        for l in [self.fc8, self.fc9]:
            # Caffe2 implementation uses XavierFill, which in fact
            # corresponds to kaiming_uniform_ in PyTorch
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), self.input_size)
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))

        return x


_ROI_TAG_FEATURE_EXTRACTORS = {
    "TagFeatureExtractor": TagFeatureExtractor,
}


def make_roi_tag_feature_extractor():
    func = _ROI_TAG_FEATURE_EXTRACTORS[cfg.MODEL.ROI_TAG_HEAD.FEATURE_EXTRACTOR]
    return func()
