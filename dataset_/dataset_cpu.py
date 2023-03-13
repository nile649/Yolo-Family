from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from utils import utils#iou_wh, non_max_suppression as nms
import random 
import os 
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True # PIL fuction to check if image is corrupted 


class YOLODataset(Dataset):
    '''
    bbox : [class, left_x, left_y, width, height]
    coco type dataset supported
    '''
    def __init__(self,path,anchors,img_size=416,S=[13,26,52],C=2,transform=None):
        super().__init__()
        self.path = path 
        self.img_size = img_size
        self.bbox_minsize = 0.01
        with open(path, "r") as file:
            self.img_files = [p.rstrip() for p in file.read().splitlines()]
            random.shuffle(self.img_files)
            self.img_files = self.img_files
        self.label_files = []
        for path in self.img_files:
            path = os.path.splitext(path)[0] + '.txt'
            path = path.replace('images','labels')
            self.label_files.append(path)  
        self.img_files = list(
            filter(lambda item: item is not None, self.img_files))
        self.label_files = list(
            filter(lambda item: item is not None, self.label_files))
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors//3
        self.C = C #number of class
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        label_path = self.label_files[index]
        try:
            bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        except:
            return self.__getitem__(index+1)
        # albumentation library require [xmin,ymin,xmax,ymax,Class], but coco  has [Class,x,y,w,h]
        #imgaug is also there
        # skipping all kinds of augmentation
        img_path = self.img_files[index]
        image = Image.open(img_path).convert("RGB")
        if image.height<250 or image.width<250:
            # skip the file
            return self.__getitem__(index+1)
        image = np.array(image)

        targets = [torch.zeros((self.num_anchors_per_scale,S,S,6)) for S in self.S] 
        if self.transform:
            try:
                aug = self.transform(image=image, bboxes=bboxes)
                image = aug["image"]
                bboxes = aug["bboxes"]
            except:
                return self.__getitem__(index+1)
        
        
        # [0,Num_Anchors,Grid_X,Grid_Y,prob,x,y,w,h,c]
        if len(bboxes)==0:
            return image, targets


        iou_anchors = utils.iou_wh(torch.tensor(bboxes)[...,2:4],self.anchors)
        anchors_indices = iou_anchors.argsort(descending=True,dim=-1)
        for idx, anchors_indice in enumerate(anchors_indices):
            
            x,y,w,h,class_lbl = bboxes[idx] 
            
            has_anchor = [False]*3
            # To check if 3 scales have at least one anchors matched to the boundingbox

            for anchor_idx in anchors_indice:
                scale_idx = anchor_idx // self.num_anchors_per_scale # 0,1,2
                """
                we need to match anchors to the target.
                We have 3 anchors per scale, total of 9 anchors on each scale applied to every gridspace.
                if anchor_idx = 8//3 -> idx 2
                so the anchor with idx 8 belongs to scale 2
                """
                anchor_on_scale = anchor_idx%self.num_anchors_per_scale # 0,1,2

                # We need map anchors to a specific cell on a gridspace to match the target
                S = self.S[scale_idx]
                i,j = int(S*y),int(S*x) # x=0.5, s = 13 -> cell (6.5)->6 in the gridspace 13
          
                anchor_taken = targets[scale_idx][anchor_on_scale,i,j,4]
                if not anchor_taken and not has_anchor[scale_idx]:
                    """
                    To make sure is anchor cell in a particular scale idx has not be taken by othe object.
                    Chances are super rare to happen. 
                    """
                    targets[scale_idx][anchor_on_scale,i,j,4]=1
                    # now we have set the grid space with the particular object as 1
                    # we need to set x,y relative to gridspace 
                    x_cell_offset,y_cell_offset = S*x-j, S*y-i #S*x-j-> 6.5-6 = 0.5 # in yolo output the sigmoid gives output from 0-1
                    width_cell, height_cell = (
                        w*S, # s=13, width=0.5 -> 13*0.5 = 6.5 
                        h*S
                    )
                    box_coordinates = torch.tensor(
                        [x_cell_offset,y_cell_offset,width_cell,height_cell]
                    )
                    targets[scale_idx][anchor_on_scale,i,j,:4] = box_coordinates
                    targets[scale_idx][anchor_on_scale][i,j,5] = int(class_lbl)

                elif not anchor_taken and iou_anchors[anchor_idx]>self.ignore_iou_thresh:
                    """
                    they are anchors which may share the same gridspace center as the one with higher iou, We need to ignore this anchors if its not already
                    assigned to another gridspace center.
                    
                    """
                    targets[scale_idx][anchor_on_scale,i,j,4]=-1 # ignore this prediction
        return image,targets

def coco2voc(coco):
    voc = []
    x, y, w, h, c = coco
    voc = [x, y, x+w, y+h,c]
    return voc