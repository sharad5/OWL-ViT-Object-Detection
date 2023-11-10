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


def paco_to_owl_box(boxes, metadata, owl_image_dim=(768, 768)):
    """
    absolute xywh -> relative xyxy
    """
    boxes = BoxUtil.box_convert(boxes, "xywh", "xyxy")
    boxes = BoxUtil.scale_bounding_box(
        boxes, metadata["width"], metadata["height"], owl_image_dim, mode="down"
    )

    return boxes