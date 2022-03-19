import os
import numpy as np
import torch
from PIL import Image
class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        annot_path = os.path.join(self.root)
        with open(annot_path ,'r',encoding="utf-8-sig") as lines:
            self.lineslist = lines.readlines()
        

    def __getitem__(self, idx):
        # load images and masks
        
        annot = self.lineslist[idx]
        ele = annot.split(' ')
        img_path = ele[0]
        
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        # convert the PIL Image into a numpy array
        # instances are encoded as different colors


        # get bounding box coordinates for each mask
        boxes = []
        labels= []
        for i in range(1,len(ele)-1):
            an = ele[i].split(',')
            xmin = int(an[0])
            xmax = int(an[2])
            ymin = int(an[1])
            ymax = int(an[3])
            label = int(an[4])
            labels.append(label)
            boxes.append([xmin, ymin, xmax,ymax])
        #print(img_path)
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(ele)-1,), dtype=torch.int64)
        labelss = torch.tensor(labels)
        # suppose all instances are not crowd
        image_id = torch.tensor([idx])
        target = {}
        target["boxes"] = boxes
        target["labels"] = labelss
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.lineslist)
    
    