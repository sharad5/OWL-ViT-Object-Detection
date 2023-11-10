import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_area
import numpy as np

from src.matcher import HungarianMatcher, box_iou, generalized_box_iou


class PushPullLoss(torch.nn.Module):
    def __init__(self, n_classes, scales):
        super().__init__()
        self.matcher = HungarianMatcher(n_classes)
        self.class_criterion = torch.nn.BCELoss(reduction="none", weight=scales)
        self.background_label = n_classes

    def class_loss(self, outputs, target_classes):
        """
        Custom loss that works off of similarities
        """

        src_logits = torch.abs(outputs["pred_logits"])
        src_logits = src_logits.transpose(1, 2)
        target_classes.squeeze_(0)
        src_logits.squeeze_(0)

        pred_logits = src_logits[:, target_classes != self.background_label].t()
        bg_logits = src_logits[:, target_classes == self.background_label].t()
        target_classes = target_classes[target_classes != self.background_label]

        # Positive loss
        pos_targets = torch.nn.functional.one_hot(target_classes, self.background_label)
        neg_targets = torch.zeros(bg_logits.shape).to(bg_logits.device)

        pos_loss = self.class_criterion(pred_logits, pos_targets.float())
        neg_loss = self.class_criterion(bg_logits, neg_targets)

        pos_loss = (torch.pow(1 - torch.exp(-pos_loss), 2) * pos_loss).sum(dim=1).mean()
        neg_loss = (torch.pow(1 - torch.exp(-neg_loss), 2) * neg_loss).sum(dim=1).mean()

        return pos_loss, neg_loss

    def loss_boxes(self, outputs, targets, indices, idx, num_boxes):
        """
        (DETR box loss)

        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        assert "pred_boxes" in outputs
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = torch.nn.functional.l1_loss(
            src_boxes, target_boxes, reduction="none"
        )

        metadata = {}

        loss_bbox = loss_bbox.sum() / num_boxes
        metadata["loss_bbox"] = loss_bbox.tolist()

        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        loss_giou = loss_giou.sum() / num_boxes

        return loss_bbox, loss_giou

    def forward(
        self,
        predicted_classes,
        target_classes,
        predicted_boxes,
        target_boxes,
    ):
        # Format to detr style
        in_preds = {
            "pred_logits": predicted_classes,
            "pred_boxes": predicted_boxes,
        }

        in_targets = [
            {"labels": _labels, "boxes": _boxes}
            for _boxes, _labels in zip(target_boxes, target_classes)
        ]

        target_classes, indices, idx = self.matcher(in_preds, in_targets)

        loss_bbox, loss_giou = self.loss_boxes(
            in_preds,
            in_targets,
            indices,
            idx,
            num_boxes=sum(len(t["labels"]) for t in in_targets),
        )

        # TODO: Optimize this part
        for box, label in zip(predicted_boxes[0], target_classes[0]):
            if label == self.background_label:
                continue

            iou, _ = box_iou(box.unsqueeze(0), predicted_boxes.squeeze(0))
            idx = iou > 0.85
            target_classes[idx] = label.item()

        loss_class, loss_background = self.class_loss(in_preds, target_classes)

        losses = {
            "loss_ce": loss_class,
            "loss_bg": loss_background,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }
        return losses
