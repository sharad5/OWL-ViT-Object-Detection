import torch
from torch import nn
from torchvision.ops import box_convert
from utils import BoxUtil, paco_to_owl_box
from DETR.matcher import HungarianMatcher

class ContrastiveDetectionLoss(torch.nn.Module):
    def __init__(self, focal_alpha=0.5, focal_gamma=2, focal_loss_coef=1.0/3, bbox_loss_coef=1.0/3, giou_loss_coef=1.0/3):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_loss_coef = focal_loss_coef
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.matcher = HungarianMatcher()
    
    def pos_neg_focal_loss(self, logits):
        p = nn.functional.sigmoid(logits) # P(y=1)
        pos_class_losses = -self.focal_alpha * torch.pow(1-p, self.focal_gamma) * torch.log(p) # Loss if y=1
        neg_class_losses = -(1-self.focal_alpha) * torch.pow(p, self.focal_gamma) * torch.log(1-p) # Loss if y=0
        return pos_class_losses, neg_class_losses
    
    def get_contrastive_focal_loss(self, logits, target_labels):
        pos_class_losses, neg_class_losses = self.pos_neg_focal_loss(logits)
        pos_class_loss = torch.einsum('bnc,bmc->bnm',
                                      pos_class_losses,
                                      target_labels.to(torch.float32)) # Sum of losses for pos queries
        pos_class_loss = pos_class_loss/target_labels.sum(dim=-1).unsqueeze(-1) # Scale by num pos queries
        neg_class_loss = torch.einsum('bnc,bmc->bnm',
                                      neg_class_losses,
                                      (1-target_labels.to(torch.float32))) # Sum of losses for neg queries
        neg_class_loss = neg_class_loss/(1-target_labels).sum(dim=-1).unsqueeze(-1) # Scale by num neg queries
        contrastive_focal_loss = pos_class_loss + neg_class_loss
        return contrastive_focal_loss
    
    def get_bbox_loss(self, pred_boxes, target_boxes):
        coord_dists = torch.abs(pred_boxes[:, :, None] - target_boxes[:, None, :])  # [B,N,M,4]
        bbox_loss = torch.sum(coord_dists, axis=-1)  # [B,N,M]
        return bbox_loss

    def get_giou_loss(self, pred_boxes, target_boxes):
        giou_loss = BoxUtil.generalized_box_iou(pred_boxes, target_boxes)
        return giou_loss
    
    def get_hungarian_matches(self, logits, pred_boxes, target_boxes, metadata):
        outputs_for_matcher = {"pred_logits": logits, "pred_boxes": pred_boxes}
        targets_for_matcher = [{"labels": torch.tensor([0]).to(logits.device), \
                                "boxes":box.to(logits.device)} \
                                 for box in paco_to_owl_box(target_boxes[:, None, :], metadata)]
        matches = self.matcher(outputs_for_matcher, targets_for_matcher)
        matches = torch.tensor(matches)
        return matches

    def forward(self, logits, pred_boxes, target_boxes, target_labels, metadata):
        batch_size = logits.shape[0]
        focal_loss = self.get_contrastive_focal_loss(logits, target_labels)
        
        pred_boxes = box_convert(pred_boxes, "cxcywh", "xyxy")
        target_boxes = paco_to_owl_box(target_boxes, metadata)
        target_boxes = target_boxes[:,None,:] # (B,4) -> (B,M=1,4)
         
        bbox_loss = self.get_bbox_loss(pred_boxes, target_boxes)
        giou_loss = self.get_giou_loss(pred_boxes, target_boxes)
            
        total_loss = self.focal_loss_coef * focal_loss \
                     + self.bbox_loss_coef * bbox_loss \
                     + self.giou_loss_coef * giou_loss
        
        matches = self.get_hungarian_matches(logits, pred_boxes, target_boxes, metadata)
        total_matched_loss = total_loss[torch.arange(batch_size), matches[:,0], matches[:,1]]
        mean_loss = total_matched_loss.mean()
        return mean_loss