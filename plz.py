import pdb
import time
import argparse
import os
# import datasets

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# from model import *
# # from loss import FocalLoss
# from utils import freeze_bn
# from utils2 import *
# from logger import Logger
# from encoder import DataEncoder

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from skimage.transform import resize

torch.set_default_tensor_type('torch.cuda.FloatTensor')
cuda = torch.cuda.is_available()
# FloatTensor = torch.cuda.FloatTensor
# LongTensor = torch.cuda.LongTensor
anchors = [[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]
anchors = anchors[::-1]
obj_thresh = 0.25
ignore_thresh = 0.5
nms_thresh = 0.4

num_classes = 80
num_workers = os.cpu_count()
batch_size = 1
lr = 0.001
momentum = 0.9
weight_decay = 1e-4
gpus = 0
is_best = 0
use_cuda = torch.cuda.is_available() 
step = 0
min_scale = 600
max_scale = 1000

def save_checkpoint(state,epoch):
    path = "./weights/yolov3_{}_.pth".format(epoch)
    torch.save(state,path)

class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index = index + 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # // : floor operation for division
        
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            x1 = x1 + pad[1][0]
            y1 = y1 + pad[0][0]
            x2 = x2 + pad[1][0]
            y2 = y2 + pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] = labels[:, 3] * w / padded_w
            labels[:, 4] = labels[:, 4] * h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)

def iou_calc(box11, box22):
    box1 = box11.to(torch.float)
    box2 = box22.to(torch.float)
    s_box1 = (box1[:,2] - box1[:,0] + 1) * (box1[:,3] - box1[:,1] + 1)
    s_box2 = (box2[:,2] - box2[:,0] + 1) * (box2[:,3] - box2[:,1] + 1)
    
    xmin = torch.max(box1[:,0],box2[:,0])
    ymin = torch.max(box1[:,1],box2[:,1])
    xmax = torch.min(box1[:,2],box2[:,2])
    ymax = torch.min(box1[:,3],box2[:,3])
    
    s_inter = torch.clamp((xmax - xmin),0) * torch.clamp((ymax - ymin),0)
    iou = torch.clamp(s_inter / (s_box1 + s_box2 - s_inter),0)
    return iou

def iou_wRatio(box11, box22):
    box1 = torch.zeros_like(box11)
    box2 = torch.zeros_like(box22)
    
    box1[:,0] = box11[:,0] - 0.5 * box11[:,2]
    box1[:,1] = box11[:,1] - 0.5 * box11[:,3]
    box1[:,2] = box11[:,0] + 0.5 * box11[:,2]
    box1[:,3] = box11[:,1] + 0.5 * box11[:,3]
    
    box2[:,0] = box22[:,0] - 0.5 * box22[:,2]
    box2[:,1] = box22[:,1] - 0.5 * box22[:,3]
    box2[:,2] = box22[:,0] + 0.5 * box22[:,2]
    box2[:,3] = box22[:,1] + 0.5 * box22[:,3]
    
    s_box1 = (box1[:,2] - box1[:,0] + 1) * (box1[:,3] - box1[:,1] + 1)
    s_box2 = (box2[:,2] - box2[:,0] + 1) * (box2[:,3] - box2[:,1] + 1)
    xmin = torch.max(box1[:,0],box2[:,0])
    ymin = torch.max(box1[:,1],box2[:,1])
    xmax = torch.min(box1[:,2],box2[:,2])
    ymax = torch.min(box1[:,3],box2[:,3])

    s_inter = torch.clamp((xmax - xmin),0) * torch.clamp((ymax - ymin),0)
    iou = torch.clamp(s_inter / (s_box1 + s_box2 - s_inter),0)
    return iou

def yolo_nms_box(boxes):
    init = 0
    for i in range(boxes.shape[0]):
        boxes_ = boxes[i].view(-1,7)
        nz = torch.nonzero(boxes_[:,4])
        nz_boxes = boxes_[nz.squeeze(),:]
        nz_idxs = torch.sort(nz_boxes[:,4],descending=True)[1]
        new_boxes = nz_boxes[nz_idxs]
        for cls in range(80):
            try:
                cls_idx = torch.nonzero(new_boxes[:,5] == cls)
                cls_box = new_boxes[cls_idx].squeeze(1)
                for j in range(cls_box.shape[0]):
                    try:
                        ious = iou_calc(cls_box[j].unsqueeze(0),cls_box[j+1:])
                    except IndexError:
                        break
                    mask = (ious < nms_thresh).float().unsqueeze(1)
                    cls_box[j+1:] = mask * cls_box[j+1:]
                    new_idx = torch.nonzero(cls_box[:,4])
                    cls_box = cls_box[new_idx].squeeze(1)
                if init==0:
                    init= init + 1
                    out = torch.cat([torch.zeros(cls_box.shape[0],1).fill_(i),cls_box],1)
                else:
                    out = torch.cat((out,torch.cat([torch.zeros(cls_box.shape[0],1).fill_(i),cls_box],1)))
            except RuntimeError:
                continue
    return out

def yolo_post_process(sbox,pc,pp,dim):    
    sbox_masked = sbox * (sbox[... , 4] > obj_thresh).unsqueeze(4).to(torch.float)
    
    xmin = (sbox[... , 0]  - 0.5 * sbox[... , 2]).clamp(0,dim-1).unsqueeze(4)
    xmax = (sbox[... , 0]  + 0.5 * sbox[... , 2]).clamp(0,dim-1).unsqueeze(4)
    ymin = (sbox[... , 1]  - 0.5 * sbox[... , 3]).clamp(0,dim-1).unsqueeze(4)
    ymax = (sbox[... , 1]  + 0.5 * sbox[... , 3]).clamp(0,dim-1).unsqueeze(4)
    objn = pc.unsqueeze(4)
    maxval = torch.max(pp,4)[0].unsqueeze(4)
    maxidx = torch.max(pp,4)[1].unsqueeze(4).to(torch.float)
    
    boxes = torch.cat([xmin,xmax,ymin,ymax,objn,maxidx,maxval],4)

    boxes0 = boxes[:,0,:,:,:].unsqueeze(1)
    boxes1 = boxes[:,1,:,:,:].unsqueeze(1)
    boxes2 = boxes[:,2,:,:,:].unsqueeze(1)
    
    nms_boxes0 = yolo_nms_box(boxes0)
    nms_boxes1 = yolo_nms_box(boxes1)
    nms_boxes2 = yolo_nms_box(boxes2)
    def minmax_to_xywh(boxes):
        xmin = boxes[... , 1]
        ymin = boxes[... , 2]
        xmax = boxes[... , 3]
        ymax = boxes[... , 4]
        
        boxes[:,1] = (xmin + xmax) / 2.
        boxes[:,2] = (ymin + ymax) / 2.
        boxes[:,3] = xmax - xmin
        boxes[:,4] = ymax - ymin

        return boxes
    
    ori_boxes0 = minmax_to_xywh(nms_boxes0)
    ori_boxes1 = minmax_to_xywh(nms_boxes1)
    ori_boxes2 = minmax_to_xywh(nms_boxes2)
    
    preds = torch.zeros(boxes.shape[0], 3, dim, dim, 7)
    p_cls = torch.zeros(boxes.shape[0], 3, dim, dim, num_classes)
    
    preds[list(map(int,ori_boxes0[:,0])), 0, list(map(int,ori_boxes0[:,1])), list(map(int,ori_boxes0[:,2])),:] = ori_boxes0[:,1:]
    preds[list(map(int,ori_boxes1[:,0])), 1, list(map(int,ori_boxes1[:,1])), list(map(int,ori_boxes1[:,2])),:] = ori_boxes1[:,1:]
    preds[list(map(int,ori_boxes2[:,0])), 2, list(map(int,ori_boxes2[:,1])), list(map(int,ori_boxes2[:,2])),:] = ori_boxes2[:,1:]
    
    return preds

def build_targets(pred, cls_yolo, targets, nY):
    #pred will be [5,3,13,13,7], cls_yolo will be [5,3,13,13,80], target will be [5, 50, 5], num_y will be 0 in this case
    #pred should be [5, 9, 416, 416]
    nGT = 0
    nOK = 0
    nT = targets.shape[1]
    nB = targets.shape[0]
    nA = 3
    dim = pred.shape[2]
    stride = 416 / dim
    mask        = torch.zeros(nB, nA, dim, dim)
    conf_mask   = torch.ones(nB, nA, dim, dim)
    tx          = torch.zeros(nB, nA, dim, dim)
    ty          = torch.zeros(nB, nA, dim, dim)
    tw          = torch.zeros(nB, nA, dim, dim)
    th          = torch.zeros(nB, nA, dim, dim)
    tconf       = torch.zeros(nB, nA, dim, dim)
    tcls        = torch.zeros(nB, nA, dim, dim, num_classes)
    for numB in range(nB):
        for numT in range(nT):
            if targets[numB, numT].sum() == 0:
                continue
            nGT =  nGT + 1
            tx_scaled = targets[numB, numT, 1] * dim
            ty_scaled = targets[numB, numT, 2] * dim
            tw_scaled = targets[numB, numT, 3] * dim
            th_scaled = targets[numB, numT, 4] * dim
            xCell = int(tx_scaled)
            yCell = int(ty_scaled)
            tBox = torch.tensor([0, 0, tw_scaled, th_scaled], requires_grad=False)
            aBox = torch.tensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))[nY * 3 : (nY+1) * 3]/stride
            at_ious = iou_calc(tBox.unsqueeze(0), aBox)
            conf_mask[numB, at_ious > ignore_thresh] = 0
            maxN = np.argmax(at_ious)
            tBox = torch.tensor([tx_scaled, ty_scaled, tw_scaled, th_scaled], requires_grad=False).unsqueeze(0)
            predBox = torch.tensor(pred[numB, maxN, xCell, yCell]).unsqueeze(0)
            mask[numB, maxN, xCell, yCell] = 1
            conf_mask[numB, maxN, xCell, yCell] = 1
            tx[numB, maxN, xCell, yCell] = tx_scaled - xCell
            ty[numB, maxN, xCell, yCell] = ty_scaled - yCell
            tw[numB, maxN, xCell, yCell] = torch.log(tw_scaled/(anchors[maxN][0]/stride) + 1e-16)
            th[numB, maxN, xCell, yCell] = torch.log(th_scaled/(anchors[maxN][1]/stride) + 1e-16)
            tcls[numB, maxN, xCell, yCell, int(targets[numB, numT, 0])] = 1

            iou = iou_wRatio(tBox, predBox)
            if predBox.sum() != 0:
                print(iou, tBox, predBox)
            tconf[numB, maxN, xCell, yCell] = 1
            
            if iou > 0.5:
                nOK = nOK + 1
                
    return nGT, nOK, mask, conf_mask, tx, ty, tw, th, tconf, tcls

def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0] 
    lines = [x for x in lines if x[0] != '#']  
    lines = [x.rstrip().lstrip() for x in lines]

    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

class ConvSet(nn.Module):
    def __init__(self,din,dout,kernel,stride,pad):
        super(ConvSet, self).__init__()
        self.conv = nn.Conv2d(din,dout,kernel,stride,pad)
        self.bn = nn.BatchNorm2d(dout)
        self.act = nn.LeakyReLU()
        self.size = dout
    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        out = self.act(bn)
        return out
    
class ShortCut(nn.Module):
    def __init__(self,size):
        super(ShortCut, self).__init__() 
        self.size = size
        
class UpSamp(nn.Module):
    def __init__(self,size):
        super(UpSamp,self).__init__()
        self.size = size
        self.up = nn.Upsample(scale_factor=2,mode='bilinear')
    def forward(self,x):
        out = self.up(x)
        return out
    
class Route(nn.Module):
    def __init__(self,route):
        super(Route, self).__init__() 
        self.route = []
        for i in range(len(route)):
            self.route.append(route[i])
            
class YOLO(nn.Module):
    def __init__(self, num):
        super(YOLO, self).__init__()
        self.num = num
    def forward(self, x, targets=None):
        dim = x.shape[2]
        stride = 416 / dim
        is_training = targets is not None
        x_view = x.view((x.shape[0], 3, x.shape[1] // 3, x.shape[2] * x.shape[3]))
        x_t = x_view.transpose(2,3).contiguous()
        x_f = x_t.view(x.shape[0],3,dim,dim,num_classes+5)
        #reshape x
#         x = torch.rand(5,3,13,13,85)

        #predicted(used for loss)
        px = torch.sigmoid(x_f[... , 0])
        py = torch.sigmoid(x_f[... , 1])
        pw = x_f[... , 2]
        ph = x_f[... , 3]
        pc = torch.sigmoid(x_f[... , 4])
        pp = torch.sigmoid(x_f[... , 5:])
        
        #scaled(nms, target build), unsqueeze가 해당 차원에 1 끼워준다
        anchors_temp = anchors[3*self.num : 3*(self.num+1)]
        sx = (px + torch.linspace(0,dim-1,dim).repeat(dim,1).t()).unsqueeze(4)
        sy = (py + torch.linspace(0,dim-1,dim).repeat(dim,1)).unsqueeze(4)
        sw0 = (torch.exp(pw[:,0,:,:]) * anchors_temp[0][0] / stride).unsqueeze(1)
        sh0 = (torch.exp(ph[:,0,:,:]) * anchors_temp[0][1] / stride).unsqueeze(1)
        sw1 = (torch.exp(pw[:,1,:,:]) * anchors_temp[1][0] / stride).unsqueeze(1)
        sh1 = (torch.exp(ph[:,1,:,:]) * anchors_temp[1][1] / stride).unsqueeze(1)       
        sw2 = (torch.exp(pw[:,2,:,:]) * anchors_temp[2][0] / stride).unsqueeze(1)
        sh2 = (torch.exp(ph[:,2,:,:]) * anchors_temp[2][1] / stride).unsqueeze(1)
        sw = torch.cat([sw0,sw1,sw2],1).unsqueeze(4)
        sh = torch.cat([sh0,sh1,sh2],1).unsqueeze(4)
        sc = pc.unsqueeze(4)
        sbox = torch.cat([sx,sy,sw,sh,sc],4)
  #         the original p_cls now becomes pp      
        yolo_pred = yolo_post_process(sbox,pc,pp,dim)
        
        nGT, nOK, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(yolo_pred,pp,targets,self.num)
        
        return px, py, pw, ph, pc, pp, tx, ty, tw, th, tconf, tcls, mask, conf_mask, nGT, nOK
        # loss_x = self.bceloss(mask * px,tx)
        # loss_y =self.bceloss(mask * py,ty)
        # loss_w = self.mseloss(mask * pw,tw)
        # loss_h = self.mseloss(mask * ph,th)
        # loss_conf = self.bceloss(conf_mask * pc,tconf)
        # loss_cls = self.bceloss(mask.unsqueeze(4) * pp,tcls)
        
        # loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        
        # return loss

    
class Darknet(nn.Module):
    def __init__(self,module):
        super(Darknet, self).__init__()
        self.module_list = module
    def forward(self, x, targets=None):
        is_training = targets is not None
        outputs = []
        yolo_output = []   
        for i in range(len(self.module_list)):
            if isinstance(self.module_list[i],ConvSet) == True:
                x = self.module_list[i](x)
                outputs.append(x)
            elif isinstance(self.module_list[i],UpSamp) == True:
                x = self.module_list[i](x)
                outputs.append(x)     
            elif isinstance(self.module_list[i],ShortCut) == True:
                x = torch.cat([outputs[i-1],outputs[i-3]],1)
                outputs.append(outputs[i-3])
            elif isinstance(self.module_list[i],Route) == True:
                if len(self.module_list[i].route) == 1:
                    x = outputs[self.module_list[i].route[0]]
                    outputs.append(x)
                elif len(self.module_list[i].route) == 2:
                    x = torch.cat([outputs[self.module_list[i].route[0]],outputs[self.module_list[i].route[1]]],1)
                    outputs.append(x)
            elif isinstance(self.module_list[i],YOLO) == True:
                outputs.append(x)
                px, py, pw, ph, pc, pp, tx, ty, tw, th, tconf, tcls, mask, conf_mask, nGT, nOK = self.module_list[i](x, targets)
                yolo_output.append([px, py, pw, ph, pc, pp, tx, ty, tw, th, tconf, tcls, mask, conf_mask, nGT, nOK])
        return yolo_output

def build_module(yolov3 = 'yolov3.cfg'):
    result = parse_cfg(yolov3)
    result = result[1:]   
    input_size = 3
    yolo_num = 0
    module_list = nn.ModuleList()

    for i in range(len(result)):
        if result[i]['type'] == 'convolutional':
            pad = int(result[i]['pad'])
            filter_size = int(result[i]['size'])
            out = int(result[i]['filters'])
            stride = int(result[i]['stride'])
            if filter_size ==1: pad = 0
            module_list.append(ConvSet(input_size,out,filter_size,stride,pad))
            input_size = out
        elif result[i]['type'] == 'shortcut':
            module_list.append(ShortCut(module_list[i-3].size))
            input_size = input_size * 2
        elif result[i]['type'] == 'upsample':
            module_list.append(UpSamp(module_list[i-1].size))
        elif result[i]['type'] == 'yolo':
            module_list.append(YOLO(yolo_num))
            yolo_num = yolo_num + 1
        elif result[i]['type'] == 'route':
            if "," in result[i]['layers']:
                val1, val2 = result[i]['layers'].split(",")
                input_size = module_list[int(val1)].size + module_list[int(val2)].size
                module_list.append(Route([int(val1),int(val2)]))
            else:
                input_size = module_list[int(result[i]['layers'])].size
                module_list.append(Route([int(result[i]['layers'])]))
    return module_list

model = Darknet(build_module())
model.cuda()
model.train()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=0)
criterionBCE = nn.BCELoss()
criterionMSE = nn.MSELoss()
trainlist = 'ver8.txt'
trainloader = torch.utils.data.DataLoader(ListDataset(trainlist),batch_size=1, shuffle=False, num_workers=0)
for epoch in range(50):
    for batch_idx, (_, imgs, targets) in enumerate(trainloader):
        imgs = imgs.cuda().float()
        targets = targets.cuda().float()
        optimizer.zero_grad()
        yolo_output = model(imgs, targets)

        def calc_loss(i):
            px, py, pw, ph, pc, pp, tx, ty, tw, th, tconf, tcls, mask, conf_mask, nGT, nOK = yolo_output[i]
            print(nGT, nOK)
            loc_loss = criterionBCE(mask*px, tx) + criterionBCE(mask*py, ty) + criterionMSE(mask*pw, tw) + criterionMSE(mask*ph, th)
            cls_loss = criterionBCE(conf_mask*pc, tconf) + criterionBCE(mask.unsqueeze(4)*pp, tcls)
            return loc_loss, cls_loss
        
        loss1 = calc_loss(0)
        loss2 = calc_loss(1)
        loss3 = calc_loss(2)
        loss_loc = loss1[0] + loss2[0] + loss3[0]
        loss_cls = loss1[1] + loss2[1] + loss3[1]
        loss = loss_loc + loss_cls   
        print(epoch,loss)
        loss.backward()
        optimizer.step()
        save_checkpoint({'epoch':epoch, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()},epoch)
end = time.time()