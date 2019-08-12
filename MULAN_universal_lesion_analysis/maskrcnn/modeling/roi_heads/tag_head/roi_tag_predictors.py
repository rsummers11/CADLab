# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
from torch import nn
from torch.nn import functional as F

from maskrcnn.config import cfg


class MultiLabelPredictor(nn.Module):
    def __init__(self):
        super(MultiLabelPredictor, self).__init__()
        num_classes = cfg.runtime_info.num_tags
        representation_size = cfg.MODEL.ROI_TAG_HEAD.MLP_HEAD_DIM

        self.cls_score = nn.Linear(representation_size, num_classes)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        for l in [self.cls_score,]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        scores = self.cls_score(x)

        return scores


_ROI_TAG_PREDICTOR = {'MultiLabelPredictor': MultiLabelPredictor, }


def make_roi_tag_predictor():
    func = _ROI_TAG_PREDICTOR[cfg.MODEL.ROI_TAG_HEAD.PREDICTOR]
    return func()
