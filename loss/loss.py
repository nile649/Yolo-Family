import torch
import torch.nn as nn
from utils import utils#intersection_over_union
import pdb

def bbox_iou(box1, box2, mode='iou'):
    """Compute iou(variant) between two bbox sets
    Arguments:
        box1 (tensor): shape (N, 4)
        box2 (tensor): shape (N, 4)
    Returns:
        tensor of shape (N, 4) containing iou metrics
    NOTE:
        The format of bounding box is (x_offset, y_offset, w_cell, h_cell)
    """
    mode = mode.lower()
    eplison = 1e-9
    # Pred boxes
    box1_x1 = box1[..., 0:1]-(box1[..., 2:3]/2)
    box1_y1 = box1[..., 1:2]-(box1[..., 3:4]/2)
    box1_x2 = box1[..., 0:1]+(box1[..., 2:3]/2)
    box1_y2 = box1[..., 1:2]+(box1[..., 3:4]/2)
    box1_w = (box1_x2-box1_x1).clamp(0)
    box1_h = (box1_y2-box1_y1).clamp(0)
    box1_area = (box1_w*box1_h) + eplison
    # True boxes
    box2_x1 = box2[..., 0:1]-(box2[..., 2:3]/2)
    box2_y1 = box2[..., 1:2]-(box2[..., 3:4]/2)
    box2_x2 = box2[..., 0:1]+(box2[..., 2:3]/2)
    box2_y2 = box2[..., 1:2]+(box2[..., 3:4]/2)
    box2_w = (box2_x2-box2_x1).clamp(0)
    box2_h = (box2_y2-box2_y1).clamp(0)
    box2_area = (box2_w*box2_h) + eplison
    # Intersection boxes
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)
    inter_w = (inter_x2-inter_x1).clamp(0)
    inter_h = (inter_y2-inter_y1).clamp(0)
    inter_area = (inter_w*inter_h) + eplison
    union = (box1_area+box2_area-inter_area+eplison)
    # Computer IoU
    iou = inter_area / union

    if mode == 'iou':
        return iou

    if mode == 'giou':
        # Convex diagnal length
        convex_w = torch.max(box1_x2, box2_x2)-torch.min(box1_x1, box2_x1)
        convex_h = torch.max(box1_y2, box2_y2)-torch.min(box1_y1, box2_y1)
        convex_area = convex_w*convex_h + eplison
        giou = iou - ((convex_area-union)/convex_area)
        return giou

    raise RuntimeError(f"Cannot compute '{mode}' metric")


class YoloLoss(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        """
        MSE : Box prediciton
        BCEwithlogits : 
        BCEwithlogits : classes / here CE
        sigmoid : boxoffset

        """
        self.config = config
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        #consatants
        self.lambda_class = 1 # 
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10


    def forward(self,predicition,target, anchors):
        """
        loss is for a single stride.
        We will call this loss 3 times for 3 different scales/stride

        ... in tesnor means all of the axis

        Arguements:
            preds (tensor): tensor of shape (N, 3, S, S, 5+C)
            target (tensor): tensor of shape (N, 3, S, S, 6)
            anchors (tensor): tensor of shape (3, 2)
        Prediction format:
            (x_raw, y_raw, w_raw, h_raw, conf, [classes...])
        Target format:
            (x_offset, y_offset, w_cell, h_cell, conf, class)

        """
        # pdb.set_trace()
        target = target.to(self.config.DEVICE)
        obj = target[...,4] == 1 # in dataset we assigned anchors which match the GT as 1 # targets[scale_idx][anchor_on_scale,i,j,0]=1
        # target[...,0] -> [3,S,S] -> S is gridspace not stride. obj-> 3xSxS
        noobj = target[...,4] == 0 # the target is initialized as tensor of zero

        # No object loss
        no_object_loss = self.bce((predicition[...,4:5][noobj]),(target[...,4:5][noobj]))

        anchors = anchors.reshape(1,3,1,1,2) # 3x2 where each anchor has h and width

        # prediciton conversion
        xyoffset = self.sigmoid(predicition[...,0:2])
        wh_cell = torch.exp(predicition[...,2:4])*anchors
        pred_bboxes = torch.cat([xyoffset,wh_cell],dim=-1)
        # ground truth is already converted while creating the dataset.
        xyoffset = target[...,0:2]
        wh_cell = target[...,2:4]
        target_bboxes = torch.cat([xyoffset,wh_cell],dim=-1)        
        iou = bbox_iou(pred_bboxes[obj],target_bboxes[obj],mode="giou")

        # object loss
        object_loss = self.bce((predicition[...,4:5][obj]),(target[...,4:5][obj]*iou.detach().clamp(0)))

        # box coordinate loss
        """
        To get the better gradient flow the authors do manipulate the target instead of the predicitons

        target[0:2] ->sigmoid()
        target[2:4] ->exp()

        pred[0:2] ->sigmoid
        target[2:4] ->inverse()
        """

        predicition[...,0:2] = self.sigmoid(predicition[...,0:2]) # x, y to be between [0,1]
        # take inverse of exp on the target to get w,h instead of performin expnential on the preidiction
        target[...,2:4] = torch.log((1e-6+target[...,2:4]/anchors))

        box_loss = self.bce(predicition[...,0:2][obj],target[...,0:2][obj])

        box_loss += self.mse(predicition[...,2:4][obj],target[...,2:4][obj])
        box_loss +=(1-iou).mean()
        # class loss
        class_loss = self.entropy((predicition[...,5:][obj]),(target[...,5][obj].long()))

        # Aggregate Loss
        loss = {
            'box_loss': self.lambda_box * box_loss,
            'obj_loss': self.lambda_obj * object_loss,
            'noobj_loss': self.lambda_noobj * no_object_loss,
            'class_loss': self.lambda_class * class_loss,
            'total_loss': (
                self.lambda_box * box_loss
                + self.lambda_obj * object_loss
                + self.lambda_noobj * no_object_loss
                + self.lambda_class * class_loss
            )
        }
        return loss