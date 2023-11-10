import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torchvision.ops import box_convert as _box_convert
from torchvision.utils import draw_bounding_boxes

class BoxUtil:
    @classmethod
    def scale_bounding_box(
        cls,
        boxes_batch: torch.tensor,  # [M, N, 4, 4]
        imwidth: int or torch.tensor,
        imheight: int or torch.tensor,
        owl_image_dim: (int, int),
        mode: str,  # up | down
    ):
        imwidth = imwidth[:, None, None]
        imheight = imheight[:, None, None]
        if mode == "down":
            boxes_batch[:, :, (0, 2)] *= (owl_image_dim[0]/imwidth)
            boxes_batch[:, :, (1, 3)] *= (owl_image_dim[1]/imheight)
            return boxes_batch
        elif mode == "up":
            boxes_batch[:, :, (0, 2)] *= (imwidth/owl_image_dim[0])
            boxes_batch[:, :, (1, 3)] *= (imheight/owl_image_dim[1])
            return boxes_batch

    @classmethod
    def draw_box_on_image(
        cls,
        image: str or torch.tensor,  # cv2 image
        boxes_batch: torch.tensor,
        labels_batch: list = None,
        color=(0, 255, 0),
    ):
        if isinstance(image, str):
            image = read_image(image)
        if labels_batch is None:
            for _boxes in boxes_batch:
                if not len(_boxes):
                    continue
                image = draw_bounding_boxes(image, _boxes, width=2)
        else:
            for _boxes, _labels in zip(boxes_batch, labels_batch):
                if not len(_boxes):
                    continue
                image = draw_bounding_boxes(image, _boxes, _labels, width=2)
        return image

    # see https://pytorch.org/vision/main/generated/torchvision.ops.box_convert.html
    @classmethod
    def box_convert(
        cls,
        boxes_batch: torch.tensor,  # [M, N, 4, 4]
        in_format: str,  # [‘xyxy’, ‘xywh’, ‘cxcywh’]
        out_format: str,  # [‘xyxy’, ‘xywh’, ‘cxcywh’]
    ):
        return _box_convert(boxes_batch, in_format, out_format)
    
    @classmethod
    def box_iou(boxes1, boxes2, eps = 1e-6):
        """Computes IoU between two sets of boxes (Works on batched inputs).
            Adopted from: https://github.com/google-research/scenic/blob/main/scenic/model_lib/base_models/box_utils.py

        Boxes are in [x, y, x', y'] format [x, y] is top-left, [x', y'] is bottom right.

        Args:
            boxes1: Predicted bounding-boxes in shape [bs, n, 4].
            boxes2: Target bounding-boxes in shape [bs, m, 4].
            eps: Epsilon for numerical stability.

        Returns:
            Pairwise IoU cost matrix of shape [bs, n, m].
        """
        # First, compute box areas. These will be used later for computing the union.
        wh1 = boxes1[..., 2:] - boxes1[..., :2] # W & H of box1
        area1 = wh1[..., 0] * wh1[..., 1]  # [bs, n]

        wh2 = boxes2[..., 2:] - boxes2[..., :2]
        area2 = wh2[..., 0] * wh2[..., 1]  # [bs, m]

        # Compute pairwise top-left and bottom-right corners of the intersection of the boxes.
        lt = torch.maximum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])  # [bs, n, m, 2].
        rb = torch.minimum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])  # [bs, n, m, 2].

        # intersection = area of the box defined by [lt, rb]
        wh = (rb - lt).clip(0.0)  # [bs, n, m, 2]
        intersection = wh[..., 0] * wh[..., 1]  # [bs, n, m]

        # union = sum of areas - intersection
        union = area1[..., :, None] + area2[..., None, :] - intersection

        iou = intersection / (union + eps)
        return iou, union  # pytype: disable=bad-return-type 
    
    @classmethod
    def generalized_box_iou(boxes1, boxes2, eps = 1e-6):
        """Generalized IoU from https://giou.stanford.edu/.

        The boxes should be in [x, y, x', y'] format specifying top-left and bottom-right corners.

        Args:
            boxes1: Predicted bounding-boxes in shape [..., N, 4].
            boxes2: Target bounding-boxes in shape [..., M, 4].
            eps: Epsilon for numerical stability.

        Returns:
            A [bs, n, m] pairwise matrix, of generalized ious.
        """
        # Degenerate boxes gives inf / nan results, so do an early check.
        assert (boxes1[:, :, 2:] >= boxes1[:, :, :2]).all()
        assert (boxes2[:, :, 2:] >= boxes2[:, :, :2]).all()

        iou, union = box_iou(boxes1, boxes2, eps=eps)

        # Generalized IoU has an extra term which takes into account the area of
        # the box containing both of these boxes. The following code is very similar
        # to that for computing intersection but the min and max are flipped.
        lt = torch.minimum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])  # [bs, n, m, 2]
        rb = torch.maximum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])  # [bs, n, m, 2]

        # Now, compute the covering box's area.
        wh = (rb - lt).clip(0.0)  # Either [bs, n, 2] or [bs, n, m, 2].
        area = wh[..., 0] * wh[..., 1]  # Either [bs, n] or [bs, n, m].

        # Finally, compute generalized IoU from IoU, union, and area.
        # Somehow the PyTorch implementation does not use eps to avoid 1/0 cases.
        return iou - (area - union) / (area + eps)
    

def paco_to_owl_box(boxes, metadata, owl_image_dim=(768, 768)):
    """
    absolute xywh -> relative xyxy
    """
    boxes = BoxUtil.box_convert(boxes, "xywh", "xyxy")
    boxes = BoxUtil.scale_bounding_box(
        boxes, metadata["width"], metadata["height"], owl_image_dim, mode="down"
    )

    return boxes