import torch
import torch.nn as nn
from utils import utils#intersection_over_union

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
        target = target.to(self.config.DEVICE)
        obj = target[...,4] == 1 # in dataset we assigned anchors which match the GT as 1 # targets[scale_idx][anchor_on_scale,i,j,0]=1
        # target[...,0] -> [3,S,S] -> S is gridspace not stride. obj-> 3xSxS
        noobj = target[...,4] == 0 # the target is initialized as tensor of zero

        # No object loss
        no_object_loss = self.bce((predicition[...,4:5][noobj]),(target[...,4:5][noobj]))

        # object loss
        """
        
        
        """
        anchors = anchors.reshape(1,3,1,1,2) # 3x2 where each anchor has h and width
        box_preds = torch.cat([self.sigmoid(predicition[...,0:2]),torch.exp(predicition[...,2:4])*anchors],dim=-1) # check yolov3 notion file to understand this

        # we need to calculate iou of only anchors where the object is present
        ious = utils.intersection_over_union(box_preds[obj],target[...,0:4][obj]).detach()
        object_loss = self.bce((predicition[...,4:5][obj]),(ious.clamp(0)*target[...,4:5][obj]))


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
        box_loss = self.mse(predicition[...,0:4][obj],target[...,0:4][obj])
        box_loss +=(1-ious).mean()
        # class loss
        class_loss = self.entropy((predicition[...,5:][obj]),(target[...,5][obj].long()))

        return (
            self.lambda_box*box_loss
            + self.lambda_obj*object_loss
            + self.lambda_noobj*no_object_loss
            + self.lambda_class*class_loss
        )