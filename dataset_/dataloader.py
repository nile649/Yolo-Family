import os
import random
import time

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import tfrecord

class DataLoader(Dataset):
    '''
    bbox : [class, left_x, left_y, width, height]
    
    '''
    def __init__(self,path,img_size=352):
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

        # Check cache
        f_1, f_2 = os.path.split(path)
        cache_path = str(f_1) + '/{}.cache'.format(f_2.split('.')[0])
        print(cache_path)

        cache_hash = get_hash(self.label_files + self.img_files)
        if os.path.isfile(cache_path):

            cache = torch.load(cache_path)  # load
            if cache['hash'] != cache_hash:  # dataset changed
                cache = self.cache_labels(cache_path, cache_hash)  # re-cache
        else:
            cache = self.cache_labels(cache_path, cache_hash)  # cache

        self.img_files = []
        self.labels = []
        for img_path, img_labels in cache.items():
            if img_path != "hash":
                self.img_files.append(img_path)
                self.labels.append(img_labels)

        self.current_batch = []
        del self.label_files

    def _getimg(self, index):
        img_path = self.img_files[index % len(self.img_files)]
        image = cv2.imread(img_path)
        return image

    def _getlbl(self, index):
        boxes = self.labels[index % len(self.img_files)]
        if len(boxes) > 0:
            boxes = torch.tensor(boxes).reshape(-1, 5)
            a = boxes[..., 3] > self.bbox_minsize
            b = boxes[..., 4] > self.bbox_minsize
            boxes = boxes[a & b]
        else:
            boxes = torch.zeros((0, 5), dtype=float)
        return boxes

    def __get_label__(self,idx):
        file = self.label_files[idx%len(self.img_files)]
        # Multiple bbox so we need to contain all the bbox in
        # tensor
        boxes = [] # list of tensor
        with open(file, 'r') as rf:
            for l in rf:
                box = l.rstrip().split(' ')
                box = [float(x) for x in box]
                box[0] = int(box[0]) # class id
                boxes.append(box)
        # tensor of tensor
        bbox_tenosr = torch.tensor(boxes)
        bbox_tenosr = bbox_tenosr.reshape(-1,5)
        a = bbox_tenosr[..., 3] > self.bbox_minsize
        b = bbox_tenosr[..., 4] > self.bbox_minsize
        bbox_tenosr = bbox_tenosr[a & b]
        bbox_target = torch.zeros((len(bbox_tenosr),6))
        bbox_target[:,1:] = bbox_tenosr
        return bbox_target

    def _get_data(self, index):
        image = self._getimg(index)
        boxes = self._getlbl(index)
        targets = torch.zeros((len(boxes), 6))
        if len(boxes):
            targets[:, 1:] = boxes

        return image,targets

    def __getitem__(self, index):
        """return an image and labels in (0,cls_id,x,y,w,h) format."""
        image, targets = self._get_data(index)

        image = image_cv2_to_tensor(image)

        return image, targets

    def collate_fn(self, batch):
        """function which adds an extra dimension to the data: the batch dimension."""
        imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        targets = torch.cat(targets, 0)
        return torch.stack(imgs), targets

    def __len__(self):
        return len(self.img_files)

    def get_batch(self):
        return self.current_batch

    def cache_labels(self, path='labels.cache', cache_hash=0):
        # Cache dataset labels, check images and read shapes
        img_cache = {}  # dict
        print("generating cache for", len(self.img_files), "samples")
        print("")
        for idx, (img_path, label) in enumerate(zip(self.img_files, self.label_files)):
            print("\r%d" % idx, end='')
            try:
                cache_list = []
                image = Image.open(img_path)
                image.verify()  # PIL verify
                shape = image.size  # image size
                assert (shape[0] > 100) & (shape[1] > 100), 'image size <100 pixels'
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        cache_list = np.array(
                            [x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
                if len(cache_list) == 0:
                    cache_list = np.zeros((0, 5), dtype=np.float32)
                img_cache[img_path] = cache_list
            except Exception as e:
                print('\nWARNING: %s: %s' % (img_path, e))
        print("")
        img_cache['hash'] = cache_hash
        print("saving cache ", path)
        torch.save(img_cache, path)  # save for next time
        return img_cache
    
    def tfrecord_cache(self):
        pass

def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))

def image_cv2_to_tensor(image):
    # convert cv2 BGR (H, W, 3) to pytorch RGB (3, H, W)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image)
    return image
