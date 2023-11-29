import torch
from torch import nn
from torchvision.ops import box_convert
from src.utils import BoxUtil, paco_to_owl_box
from src.DETR.matcher import HungarianMatcher
import wandb

class ContrastiveDetectionLoss(torch.nn.Module):
    def __init__(self, focal_alpha=0.5, focal_gamma=2, focal_loss_coef=0.5, bbox_loss_coef=1.0, giou_loss_coef=1.0, non_overlap_loss_coef=1.0, non_overlap_giou_thresh=0.2):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_loss_coef = focal_loss_coef
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.non_overlap_loss_coef = non_overlap_loss_coef
        self.non_overlap_giou_thresh = non_overlap_giou_thresh
        self.matcher = HungarianMatcher()
    
    def pos_neg_focal_loss(self, logits, step, eps=1e-6):
        p = nn.functional.sigmoid(logits) # P(y=1)
        #torch.save(p, f'logs/run3/probabilitae_{step}.pt')
#         pos_class_losses = -self.focal_alpha * torch.pow(1-p, self.focal_gamma) * torch.log(p) # Loss if y=1
#         neg_class_losses = -(1-self.focal_alpha) * torch.pow(p, self.focal_gamma) * torch.log(1-p) # Loss if y=0
        pos_class_losses = - torch.log(p +eps) # Loss if y=1
        neg_class_losses = - torch.log(1-p +eps) # Loss if y=0
        # torch.save(pos_class_losses, f'logs/run3/pos_class_losses_{step}.pt')
        # torch.save(neg_class_losses, f'logs/run3/neg_class_losses_{step}.pt')
        return pos_class_losses, neg_class_losses
    
    def get_contrastive_focal_loss(self, pos_class_losses, neg_class_losses, target_labels, step):
        #pos_class_losses, neg_class_losses = self.pos_neg_focal_loss(logits, step)
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
        coord_dists = torch.abs(pred_boxes[:, :, None, :] - target_boxes[:, None, :, :])  # [B,N,M,4]
        bbox_loss = torch.sum(coord_dists, axis=-1)  # [B,N,M]
        return bbox_loss

    def get_giou_loss(self, pred_boxes, target_boxes, step):
        giou_loss = -1.0 * BoxUtil.generalized_box_iou(pred_boxes, target_boxes, step)
        return giou_loss
    
    def get_mean_non_overlap_loss(self, giou_loss, matches, neg_class_losses, metadata):
        ''' Penalize boxes not overlapping with the GT & having high prob for +ve query '''
        batch_size = matches.shape[0]
        non_overlapping_mask = giou_loss > self.non_overlap_giou_thresh # Boxes having low overlap with ground truth
        non_overlapping_mask[torch.arange(batch_size), matches[:,0], matches[:,1]] = False # Remove matched boxes
        mean_non_overlap_loss = neg_class_losses[:,:,:metadata["num_pos_queries"][0]][non_overlapping_mask].mean()
        return mean_non_overlap_loss
    
    def get_hungarian_matches(self, logits, pred_boxes, target_boxes, metadata):
        outputs_for_matcher = {"pred_logits": logits, "pred_boxes": pred_boxes}
        targets_for_matcher = [{"labels": torch.arange(metadata["num_pos_queries"][idx]).to(logits.device), \
                                "boxes":box.to(logits.device)} \
                                 for idx, box in enumerate(target_boxes)]
        matches = self.matcher(outputs_for_matcher, targets_for_matcher)
        matches = torch.tensor(matches)
        return matches

    def forward(self, logits, pred_boxes, target_boxes, target_labels, metadata, step):
        batch_size = logits.shape[0]
        pos_class_losses, neg_class_losses = self.pos_neg_focal_loss(logits, step)
        focal_loss = self.get_contrastive_focal_loss(pos_class_losses, neg_class_losses, target_labels, step)
        
        pred_boxes = box_convert(pred_boxes, "cxcywh", "xyxy")
        target_boxes = box_convert(target_boxes[:, None, :], "xywh", "xyxy")#paco_to_owl_box(target_boxes[:, None, :], metadata)
#         target_boxes = target_boxes[:,None,:] # (B,4) -> (B,M=1,4)
         
        bbox_loss = self.get_bbox_loss(pred_boxes, target_boxes)#/2000
        giou_loss = self.get_giou_loss(pred_boxes, target_boxes, step)
            
        total_loss = self.focal_loss_coef * focal_loss \
                     + self.bbox_loss_coef * bbox_loss \
                     + self.giou_loss_coef * giou_loss
        
        matches = self.get_hungarian_matches(logits, pred_boxes, target_boxes, metadata)
        
        mean_non_overlap_loss = self.get_mean_non_overlap_loss(giou_loss, matches, neg_class_losses, metadata)

        # torch.save(matches, f'logs/run3/matches_{step}.pt')
        total_matched_loss = total_loss[torch.arange(batch_size), matches[:,0], matches[:,1]]
        
        matched_focal_loss_mean = focal_loss[torch.arange(batch_size), matches[:,0], matches[:,1]].mean()
        mean_loss = total_matched_loss.mean() #+ self.non_overlap_loss_coef * mean_non_overlap_loss
#         mean_focal_loss = focal_loss.mean()
        mean_bbox_loss = bbox_loss[torch.arange(batch_size), matches[:,0], matches[:,1]].mean()
        mean_giou_loss = giou_loss[torch.arange(batch_size), matches[:,0], matches[:,1]].mean()
#         wandb.log(
#             {
#                 "matched_focal_loss": matched_focal_loss_mean, 
#                 "mean_bbox_loss": mean_bbox_loss,
#                 "mean_giou_loss": mean_giou_loss
#             }
#         )
        return mean_loss, matched_focal_loss_mean, mean_bbox_loss, mean_giou_loss, mean_non_overlap_loss