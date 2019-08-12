# Ke Yan, Imaging Biomarkers and Computer-Aided Diagnosis Laboratory,
# National Institutes of Health Clinical Center, July 2019
"""The score refine layer in MULAN, which fuses detection and
tagging scores and predict them again with an FC layer"""
import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn.structures.bounding_box import BoxList
from maskrcnn.modeling.roi_heads.tag_head.loss import WeightedCeLoss, WeightedCeLossBatchOhem
from maskrcnn.modeling.utils import cat

from maskrcnn.config import cfg


class ROIRefineHead(torch.nn.Module):
    def __init__(self):
        super(ROIRefineHead, self).__init__()
        self.loss_evaluator_tag = WeightedCeLoss(cfg.runtime_info.cls_pos_wts, cfg.runtime_info.cls_neg_wts)
        self.num_tag = cfg.runtime_info.num_tags
        self.num_new_ft = 0
        if cfg.MODEL.ROI_REFINE_HEAD.BOX_FEATURE:
            self.num_new_ft += 4
        if cfg.MODEL.ROI_REFINE_HEAD.Z_FEATURE:
            self.num_new_ft += 1
        if cfg.MODEL.ROI_REFINE_HEAD.DEMOGRAPHIC_FEATURE:
            self.num_new_ft += 2
        self.score_propa_tag = nn.Linear(self.num_tag+2 + self.num_new_ft, self.num_tag)
        self.score_propa_clsf = nn.Linear(self.num_tag+2 + self.num_new_ft, 2)
        nn.init.zeros_(self.score_propa_tag.weight)
        nn.init.eye_(self.score_propa_tag.weight[:, 2:self.num_tag+2])
        nn.init.constant_(self.score_propa_tag.bias, 0)
        nn.init.zeros_(self.score_propa_clsf.weight)
        nn.init.eye_(self.score_propa_clsf.weight[:, :2])
        nn.init.constant_(self.score_propa_clsf.bias, 0)

    def forward(self, features, proposals, targets=None, infos=None):
        if cfg.MODEL.ROI_REFINE_HEAD.BOX_FEATURE:
            box_ft = self.get_box_ft(proposals, infos)
            features = torch.cat([features, box_ft], dim=1)
        if cfg.MODEL.ROI_REFINE_HEAD.Z_FEATURE:
            z_ft = self.get_z_ft(proposals, infos)
            features = torch.cat([features, z_ft], dim=1)
        if cfg.MODEL.ROI_REFINE_HEAD.DEMOGRAPHIC_FEATURE:
            demog_fts = self.get_demog_ft(proposals, infos)
            features = torch.cat([features, demog_fts], dim=1)
        tag_logits = self.score_propa_tag(features)
        clsf_logits = self.score_propa_clsf(features)
        tag_prob = torch.sigmoid(tag_logits)
        clsf_prob = F.softmax(clsf_logits, -1)

        if not self.training:
            box_num_per_image = [prop.bbox.shape[0] for prop in proposals]
            tag_prob = tag_prob.split(box_num_per_image, dim=0)
            clsf_prob = clsf_prob.split(box_num_per_image, dim=0)

            for i in range(len(proposals)):
                proposals[i].add_field("tag_scores", tag_prob[i])
                proposals[i].add_field("scores", clsf_prob[i][:, 1])
            return tag_logits, proposals, {}

        labels = [proposal.get_field("labels") for proposal in proposals]  # lesion/nonlesion
        matched_idxs = [proposal.get_field("matched_idxs") for proposal in proposals]
        tags_target = [target.get_field("tags") for target in targets]
        # reliable_neg_tags = [target.get_field("reliable_neg_tags") for target in targets]
        tags = []
        # rhem_wts = []
        for img_idx in range(len(labels)):
            tags_per_image = - torch.ones(labels[img_idx].shape[0], self.num_tag).to(torch.int).cuda()
            TP_lesion_idxs = labels[img_idx] > 0
            tags_per_image[TP_lesion_idxs] = tags_target[img_idx][matched_idxs[img_idx][TP_lesion_idxs]]

            # wts_per_image = torch.zeros(labels[img_idx].shape[0], self.num_tag).to(torch.int).cuda()
            # wts_per_image[TP_lesion_idxs] = \
            #     torch.clamp(reliable_neg_tags[img_idx][matched_idxs[img_idx][TP_lesion_idxs]], min=0)
            # wts_per_image[TP_lesion_idxs] += torch.clamp(tags_per_image[TP_lesion_idxs], min=0)

            tags.append(tags_per_image)
            # rhem_wts.append(wts_per_image)

        tag_target = cat(tags, dim=0)
        labels = cat(labels, dim=0)
        # rhem_wts = cat(rhem_wts, dim=0)
        loss_tag = self.loss_evaluator_tag(tag_prob, tag_target)

        classification_loss = F.cross_entropy(clsf_logits, labels)

        losses = dict(loss_tag2=loss_tag, loss_clsf2=classification_loss)

        return tag_logits, proposals, losses

    # extra input features
    def get_box_ft(self, proposals, infos):
        """x, y, w, h"""
        box_fts = []
        for prop, info in zip(proposals, infos):
            imsz = torch.tensor(prop.size, dtype=torch.float).cuda()
            xy = (prop.bbox[:, :2] + prop.bbox[:, 2:])/2 / imsz
            wh = (prop.bbox[:, 2:] - prop.bbox[:, :2]) / imsz
            box_fts.append(torch.cat([xy, wh], dim=1))
        return torch.cat(box_fts).requires_grad_()

    def get_demog_ft(self, proposals, infos):
        """age and sex"""
        demog_fts = []
        for prop, info in zip(proposals, infos):
            demographic = torch.FloatTensor([info['gender'], info['age']])
            demographic = demographic.repeat(len(prop), 1).cuda()
            demog_fts.append(demographic)
        return torch.cat(demog_fts).requires_grad_()

    def get_z_ft(self, proposals, infos):
        """normalized z coordinate in DeepLesion"""
        z_fts = []
        for prop, info in zip(proposals, infos):
            z = torch.FloatTensor([info['z_coord']])
            z = z.repeat(len(prop), 1).cuda()
            z_fts.append(z)
        return torch.cat(z_fts).requires_grad_()


def build_roi_refine_head():
    return ROIRefineHead()
