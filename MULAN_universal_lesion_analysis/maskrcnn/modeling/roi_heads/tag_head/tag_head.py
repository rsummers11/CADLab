# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
import torch
from torch import nn

from maskrcnn.structures.bounding_box import BoxList
from .roi_tag_predictors import make_roi_tag_predictor
from .roi_tag_feature_extractors import make_roi_tag_feature_extractor
from .loss import WeightedCeLoss, WeightedCeLossBatchOhem
from maskrcnn.modeling.utils import cat

from maskrcnn.config import cfg


class ROITagHead(torch.nn.Module):
    def __init__(self):
        super(ROITagHead, self).__init__()
        self.feature_extractor = make_roi_tag_feature_extractor()
        self.predictor = make_roi_tag_predictor()
        # self.post_processor = make_roi_mask_post_processor()
        self.loss_evaluator = WeightedCeLoss(cfg.runtime_info.cls_pos_wts, cfg.runtime_info.cls_neg_wts)
        self.loss_evaluator2 = WeightedCeLossBatchOhem(cfg.runtime_info.cls_pos_wts, cfg.runtime_info.cls_neg_wts)
        self.num_tag = cfg.runtime_info.num_tags

    def forward(self, features, proposals, targets=None):
        x = self.feature_extractor(features, proposals)
        tag_logits = self.predictor(x)
        tag_prob = torch.sigmoid(tag_logits)

        if not self.training:
            box_num_per_image = [prop.bbox.shape[0] for prop in proposals]
            tag_prob = tag_prob.split(box_num_per_image, dim=0)

            for i in range(len(proposals)):
                proposals[i].add_field("tag_scores", tag_prob[i])
            return tag_logits, proposals, {}

        # sort the labels
        labels = [proposal.get_field("labels") for proposal in proposals]  # lesion/nonlesion
        matched_idxs = [proposal.get_field("matched_idxs") for proposal in proposals]
        tags_target = [target.get_field("tags") for target in targets]
        reliable_neg_tags = [target.get_field("reliable_neg_tags") for target in targets]
        tags = []
        rhem_wts = []
        for img_idx in range(len(labels)):
            tags_per_image = - torch.ones(labels[img_idx].shape[0], self.num_tag).to(torch.int).cuda()
            TP_lesion_idxs = labels[img_idx] > 0
            tags_per_image[TP_lesion_idxs] = tags_target[img_idx][matched_idxs[img_idx][TP_lesion_idxs]]

            wts_per_image = torch.zeros(labels[img_idx].shape[0], self.num_tag).to(torch.int).cuda()
            wts_per_image[TP_lesion_idxs] = \
                torch.clamp(reliable_neg_tags[img_idx][matched_idxs[img_idx][TP_lesion_idxs]], min=0)
            wts_per_image[TP_lesion_idxs] += torch.clamp(tags_per_image[TP_lesion_idxs], min=0)

            tags.append(tags_per_image)
            rhem_wts.append(wts_per_image)

        tag_target = cat(tags, dim=0)
        rhem_wts = cat(rhem_wts, dim=0)
        loss_tag = self.loss_evaluator(tag_prob, tag_target)
        loss_tag_ohem = self.loss_evaluator2(tag_prob, tag_target, rhem_wts)  # RHEM
        # loss_tag_ohem = self.loss_evaluator2(tag_prob, tag_target)  # OHEM
        losses = dict(loss_tag=loss_tag, loss_tag_ohem=loss_tag_ohem)

        return tag_logits, proposals, losses


def build_roi_tag_head():
    return ROITagHead()
