# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn.structures.bounding_box import BoxList
from maskrcnn.structures.boxlist_ops import boxlist_nms
from maskrcnn.structures.boxlist_ops import cat_boxlist
from maskrcnn.modeling.box_coder import BoxCoder
from maskrcnn.config import cfg


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self, score_thresh=0.05, nms=0.5, detections_per_img=100, box_coder=None
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )

        num_classes = class_prob.shape[1]
        assert num_classes == 2, 'currently only support one fg class: lesion. Otherwise code should be revised'

        proposals = list(proposals.split(boxes_per_image, dim=0))
        class_prob = list(class_prob.split(boxes_per_image, dim=0))
        class_logits = list(class_logits.split(boxes_per_image, dim=0))

        results = []
        for i in range(len(boxes)):

            if cfg.TEST.EVAL_SEG_TAG_ON_GT:
                # box results should not include gt
                gt_idxs = [box.get_field('is_gt') for box in boxes]
                boxlist = self.prepare_boxlist(
                    proposals[i], class_prob[i], image_shapes[i], class_logits[i], gt_idxs[i])
                boxlist = boxlist[~boxlist.get_field('is_gt')]
            else:
                boxlist = self.prepare_boxlist(
                    proposals[i], class_prob[i], image_shapes[i], class_logits[i], None)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_results(boxlist, num_classes)
            if cfg.TEST.EVAL_SEG_TAG_ON_GT:
                # If gt boxes are needed for evaluating tagging and segmentation,
                # concat the gt boxes. They will not be used when evaluating detection
                gt_boxlist = boxes[i][gt_idxs[i]]
                gt_boxlist.add_field(
                    "labels", torch.full((len(gt_boxlist),), 1, dtype=torch.int64, device=boxlist.bbox.device)
                )
                gt_boxlist.add_field("scores", class_prob[i][gt_idxs[i]][:, 1])
                gt_boxlist.add_field("class_logits", class_logits[i][gt_idxs[i]])
                del gt_boxlist.extra_fields["objectness"]
                boxlist = cat_boxlist([boxlist, gt_boxlist])
            results.append(boxlist)

        return results

    def prepare_boxlist(self, boxes, scores, image_shape, class_logits, gt_idxs):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        ncls = scores.shape[1]
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)  # cls0, cls1, cls0, cls1, ... ! not cls0,...cls0,cls1,...,cls1
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        boxlist.add_field("class_logits", class_logits.repeat(1,ncls).reshape(-1,ncls))
        if gt_idxs is not None:
            boxlist.add_field("is_gt", gt_idxs.reshape(-1,1).repeat(1,ncls).reshape(-1))
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)
        class_logits = boxlist.get_field("class_logits").reshape(-1, num_classes * num_classes)
        if cfg.TEST.EVAL_SEG_TAG_ON_GT:
            is_gt = boxlist.get_field("is_gt").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            ord = torch.argsort(scores[inds, j], descending=True)
            scores_j = scores[inds[ord], j]
            class_logits_j = class_logits[inds[ord], j*num_classes:(j+1)*num_classes]
            boxes_j = boxes[inds[ord], j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            if cfg.TEST.EVAL_SEG_TAG_ON_GT:
                is_gt_j = is_gt[inds[ord], j]
                boxlist_for_class.add_field("is_gt", is_gt_j)
            boxlist_for_class.add_field("class_logits", class_logits_j)
            boxlist_for_class, keep = boxlist_nms(
                boxlist_for_class, self.nms, score_field="scores"
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            # boxlist_for_class.add_field("box_index_after_filter", inds[ord][keep])  # for tag head
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        # result.add_field("box_num_before_filter", scores.shape[0])
        return result


def make_roi_box_post_processor():
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    if cfg.MODE in ('vis', 'demo', 'batch'):
        score_thresh = cfg.TEST.VISUALIZE.SCORE_THRESH
        detections_per_img = cfg.TEST.VISUALIZE.DETECTIONS_PER_IMG
        nms_thresh = cfg.TEST.VISUALIZE.NMS
    else:
        score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
        detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
        nms_thresh = cfg.MODEL.ROI_HEADS.NMS

    postprocessor = PostProcessor(
        score_thresh, nms_thresh, detections_per_img, box_coder
    )
    return postprocessor
