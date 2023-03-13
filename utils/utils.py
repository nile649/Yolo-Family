from config import config 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
import time
from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
from .plot import xywh2xyxy, plot_bbox
import pdb
import torch.nn.functional as F

def iou_wh(box1, box2):
    """Copmute IoU between two bbox sets with width & height only
    Arguments:
        box1 (tensor): tensor of shape (N, 2)
        box2 (tensor): tensor of shape (M, 2)
    Returns:
        tensor of shape (N, M) representing pair-by-pair iou values
        between two bbox sets.
    NOTES: box format is (w, h)
    """
    N = box1.size(0)
    M = box2.size(0)
    # Computer intersection area
    '''
    we need the shape of NxM, we use unsqueeze to add a dim at position 1 which will be broadcasted to NxM
    and in the box2, its of shapoe Mx2, so we select M samples, add a dim at position 0, 1XM and do the broadcasting to get NxM
    
    '''
    min_w = torch.min(
            box1[..., 0].unsqueeze(1).expand(N, M), # (N,) -> (N, M)
            box2[..., 0].unsqueeze(0).expand(N, M), # (M,) -> (N, M)
            )
    min_h = torch.min(
            box1[..., 1].unsqueeze(1).expand(N, M), # (N,) -> (N, M)
            box2[..., 1].unsqueeze(0).expand(N, M), # (M,) -> (N, M)
            )
    inter = min_w * min_h # (N, M)
    area1 = box1[..., 0]*box1[..., 1]       # (N,)
    area1 = area1.unsqueeze(1).expand(N,M)  # (N, M)
    area2 = box2[..., 0]*box2[..., 1]       # (M,)
    area2 = area2.unsqueeze(0).expand(N,M)  # (N, M)
    iou = inter / (area1+area2-inter)
    return iou
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): if boxes (x,y,w,h)

    Returns:
        tensor: Intersection over union for all examples
    """
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.maximum(box1_x1, box2_x1)
    y1 = torch.maximum(box1_y1, box2_y1)
    x2 = torch.minimum(box1_x2, box2_x2)
    y2 = torch.minimum(box1_y2, box2_y2)

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def NMS(bboxes,iou_threshold,class_n=2):
    # print(class_n)
    for n_c in range(0,class_n):
        # pdb.set_trace()
        if bboxes.size(0)==0:
            return []

        boxes = xywh2xyxy(bboxes[...,:4]) # in xywh
        score = bboxes[...,4:5]
        classes = bboxes[...,5]
        extras = bboxes[...,6:] if bboxes.size(1) > 6 else None 
        # Filter out target class
        mask = (classes==n_c)
        if torch.sum(mask)==0:
            continue
        
        boxes = boxes[mask]
        score = score[mask]
        extras=extras[mask] if extras is not None else None  
        # calculate IOU
        x1 = boxes[..., 0]
        y1 = boxes[..., 1]
        x2 = boxes[..., 2]
        y2 = boxes[..., 3]
        # Area of shape (N,)
        areas = (x2-x1)*(y2-y1)
        keep = []
        order = score.sort(0,descending=True)[1].squeeze(1)
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order[0].item())
                break
            else:
                i = order[0].item()
                keep.append(i)
            # Compute IoU with remaining boxes (N-1,)
            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            inter = (xx2-xx1).clamp(min=0)*(yy2-yy1).clamp(min=0)
            iou = (inter / (areas[i]+areas[order[1:]]-inter))
            idx = (iou <= iou_threshold).nonzero().squeeze(1)
            if idx.numel() == 0:
                break
            order = order[idx+1]
        boxes = boxes[keep]
        score = score[keep]
        extras = extras[keep] if extras is not None else None
        classes = torch.tensor([[n_c]]).repeat(boxes.size(0), 1) # (N,1) -> [[c],[c]]
        columns = [ boxes, score, classes ] + ([extras] if extras is not None else [])
        bboxes = torch.cat(columns, dim=1)

    return bboxes.tolist()        



def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=2
):
    """
    Video explanation of this function:
    https://youtu.be/FppOzcDvaDI

    This function calculates mean average precision (mAP)

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6
    # print(num_classes)
    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_samples(model,loader,sample,config,pred_plot):
    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)
    final_predict = []
    while(sample):
        sample-=1
        predict = []
        x, y = next(iter(loader))
        # Prediction on single image
        img = x
        targets = y
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # print(img.shape)
                preds = model(img.to(config.DEVICE))
                
        # Collect Bounding boxes
        true_bboxes = []
        pred_bboxes = []
        for scale_idx, (pred, target) in enumerate(zip(preds, targets)):                                              
            scale = pred.size(2)   
            anchors = scaled_anchors[scale_idx] # (3, 2)
            anchors = anchors.reshape(1, 3, 1, 1, 2) # (1, 3, 1, 1, 2)                                                                                                                   
            # Convert prediction to correct format                          
            pred[..., 0:2] = torch.sigmoid(pred[..., 0:2])      # (N, 3, S, S, 2)
            pred[..., 2:4] = torch.exp(pred[..., 2:4])*anchors  # (N, 3, S, S, 2)
            pred[..., 4:5] = torch.sigmoid(pred[..., 4:5])      # (N, 3, S, S, 1)
            pred_cls_probs = F.softmax(pred[..., 5:], dim=-1)   # (N, 3, S, S, C)
            probs, indices = torch.max(pred_cls_probs, dim=-1)  # (N, 3, S, S)
            indices = indices.unsqueeze(-1)                     # (N, 3, S, S, 1)
            pred[..., 4] *= probs
            pred = torch.cat([ pred[..., :5], indices ], dim=-1)# (N, 3, S, S, 6)
            # Convert coordinate system to normalized format (xywh)         
            pboxes = cells_to_bboxes(cells=pred, scale=scale)    # (N, 3, S, S, 6)
            tboxes = cells_to_bboxes(cells=target, scale=scale)  # (N, 3, S, S, 6)
            # Filter out bounding boxes with confidence threshold
            pred_mask = pboxes[..., 4] > config.CONF_THRESHOLD
            true_mask = tboxes[..., 4] == 1.
            pred_boxes = pboxes[pred_mask] # (N,n_anchors,S,S,6)
            true_boxes = tboxes[true_mask] 
            pred_bboxes.extend(pred_boxes.detach().cpu().numpy().tolist())
            true_bboxes.extend(true_boxes.detach().cpu().numpy().tolist())
        #     print(pred_mask.shape)
        # Collect prediction
        predict.append((img.detach().cpu(), pred_bboxes, true_bboxes))

        nms_predict = []
        for img, pred_bboxes, true_bboxes in predict:
            nms_pred_bboxes = []
            nms_true_bboxes = []
        #     for c in range(2):
            nms_pred_boxes = NMS(torch.tensor(pred_bboxes),config.NMS_IOU_THRESH,class_n=config.NUM_CLASSES)
            nms_true_boxes = NMS(torch.tensor(true_bboxes),config.NMS_IOU_THRESH,class_n=config.NUM_CLASSES)
            nms_pred_bboxes.extend(nms_pred_boxes)
            nms_true_bboxes.extend(nms_true_boxes)
            nms_predict.append((img, nms_pred_bboxes, nms_true_bboxes))
        final_predict.append(nms_predict)
    plot_bbox(final_predict,pred_plot,config)


def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
    class_n = 2
):
    # make sure model is in eval before get bboxes

    model.eval()
    all_pred_boxes = []
    all_true_boxes = []
    sample_idx=0
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)
        batch_size = x.shape[0]
        target_s1 = labels[0].to(device)
        target_s2 = labels[1].to(device)
        target_s3 = labels[2].to(device)
        targets = [target_s1,target_s2,target_s3]
        with torch.no_grad():
            predictions = model(x)

        true_bboxes = [[] for _ in range(batch_size)]
        pred_bboxes = [[] for _ in range(batch_size)]
        # targets [[N,3,13,13,6],[N,3,26,26,6],[N,3,52,52,6]]
        for scale_idx,(pred,target) in enumerate(zip(predictions,targets)):
            scale = pred.shape[2] # s = [13,26,52]
            anchor = torch.tensor([*anchors[scale_idx]]).to(device) * scale
            anchor = anchor.reshape(1, len(anchor), 1, 1, 2) # (1,3,1,1,2)

            # convert predicitions of a cell to box
            pred[...,0:2] = torch.sigmoid(pred[...,0:2]) # (N,3,S,S,2)
            # convert x & y in cell domain to box. 
            # Write a note from notion
            pred[...,2:4] = torch.exp(pred[...,2:4])*anchor # (N,3,S,S,2)
            pred[...,4:5] = torch.sigmoid(pred[...,4:5]) # (N,3,S,S,1)
            cls_prob =  F.softmax(pred[..., 5:], dim=-1)   # (N, 3, S, S, C)
            _, indices = torch.max(cls_prob, dim=-1)      # (N, 3, S, S)
            indices = indices.unsqueeze(-1)                     # (N, 3, S, S, 1)        
            pred = torch.cat([ pred[..., :5], indices ], dim=-1)# (N, 3, S, S, 6)

            # convert the coordinate system to a normalized format xywh

            pred_box = cells_to_bboxes(cells=pred,scale=scale)
            true_box = cells_to_bboxes(cells=target,scale=scale)

            # Filter out bbox from all cells.
            for idx, candidate_box in enumerate(pred_box):
                pred_bboxes[idx] += candidate_box[candidate_box[...,4]>threshold].tolist()
            for idx, candidate_box in enumerate(true_box):
                true_bboxes[idx] += candidate_box[candidate_box[...,4]>threshold].tolist()
        """
        Once we get the prediction of a batch of images.
        We clear the BBOX with low conf.
        Now we can perform NMS for batch of images in vectorize format.
        """
        for batch_idx in range(batch_size):
            pbboxes = torch.tensor(pred_bboxes[batch_idx])
            tbboxes = torch.tensor(true_bboxes[batch_idx])   
            for c in range(class_n):
                nms_pred_boxes = NMS(pbboxes,iou_threshold,class_n=class_n)
                nms_true_boxes = NMS(tbboxes,iou_threshold,class_n=class_n)
                # Note
                all_pred_boxes.extend([[sample_idx]+box
                                        for box in nms_pred_boxes])
                all_true_boxes.extend([[sample_idx]+box
                                        for box in nms_true_boxes])
            sample_idx += 1
    model.train()
    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(cells=None, scale=None):
    """
    Scale the cell grid coordinate to image coordinate with help of 
    scale. ex [13,26,52]
    Input :
        : cells [N,3,S,S,6]
        : S
    """
    N = cells.size(0)
    # Extract each dimension
    x_cells = cells[..., 0:1]   # (N, 3, scale, scale, 1)
    y_cells = cells[..., 1:2]   # (N, 3, scale, scale, 1)
    w_cells = cells[..., 2:3]   # (N, 3, scale, scale, 1)
    h_cells = cells[..., 3:4]   # (N, 3, scale, scale, 1)
    conf = cells[..., 4:5]      # (N, 3, scale, scale, 1)
    cls = cells[..., 5:6]       # (N, 3, scale, scale, 1)
    if cells.size(4) > 6:
        tails = cells[..., 6:]  # (N, 3, scale, scale, N)
    # Cell coordinates
    cell_indices = (            # (N, 3, scale, scale, 1)
        torch.arange(scale)
        .repeat(N, 3, scale, 1)
        .unsqueeze(-1)
        .to(cells.device)
        )
    # Convert coordinates
    x = (1/scale)*(x_cells+cell_indices)
    y = (1/scale)*(y_cells+cell_indices.permute(0, 1, 3, 2, 4))
    w = (1/scale)*(w_cells)
    h = (1/scale)*(h_cells)
    if cells.size(4) > 6:
        boxes = torch.cat([x, y, w, h, conf, cls, tails], dim=-1)
    else:
        boxes = torch.cat([x, y, w, h, conf, cls], dim=-1)
    return boxes

def check_class_accuracy(model, loader, threshold):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 4] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 4] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 4]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 4][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 4][noobj])
            tot_noobj += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
