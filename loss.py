import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss

class SoftLoULoss(nn.Module):
    def __init__(self):
        super(SoftLoULoss, self).__init__()

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        smooth = 1

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)

        return loss

class SoftLoULoss1(nn.Module):
    def __init__(self, a= 0.):
        super(SoftLoULoss1, self).__init__()
        self.a = a
        if a < 0 or a > 1:
            raise ('loss error due to a:{}'.format(a))
        self.iou = None
        self.loss1 = 0.
        self.loss2 = 0.
    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        smooth = 0.00

        target = target.float()
        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)

        loss = 1 - torch.mean(loss)

        return loss

class HierBCELoss(nn.Module):
    def __init__(self):
        super(HierBCELoss, self).__init__()
        self.base_loss = nn.BCELoss(size_average=True)

    def forward(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.base_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.base_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.base_loss(preds, gt_masks)
            return loss
        

class ISNetLoss(nn.Module):
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = Get_gradient_nopadding()
        
    def forward(self, preds, gt_masks):
        edge_gt = self.grad(gt_masks.clone())
        
        ### img loss
        loss_img = self.softiou(preds[0], gt_masks)
        
        ### edge loss
        loss_edge = 10 * self.bce(preds[1], edge_gt)+ self.softiou(preds[1].sigmoid(), edge_gt)
        
        return loss_img + loss_edge



class Iou_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        iou = intersection_sum / (pred_sum + target_sum - intersection_sum + self.eps)
        if self.reduction == 'mean':
            return 1 - iou.mean()
        elif self.reduction == 'sum':
            return 1 - iou.mean()
        else: 
            raise NotImplementedError('reduction type {} not implemented'.format(self.reduction))


class HCFNetLoss(nn.Module):
    def __init__(self):
        super(HCFNetLoss, self).__init__()
        self.reduction = 'mean'
        self.eps = 1e-6
        self.cal_loss1 = nn.BCEWithLogitsLoss(reduction=self.reduction)
        self.cal_loss2 = Iou_loss(reduction=self.reduction)
    def forward(self, preds, gt_masks):

        if isinstance(preds, list):
            loss_total = 0
            # preds[0]= F.interpolate(preds[0], gt_masks.shape[2:], mode='bilinear', align_corners=False)
            for i in range(len(preds)):
                pred = preds[i]
                loss_fn1 = self.cal_loss1(pred, gt_masks)
                loss_fn2 = self.cal_loss2(pred, gt_masks)
                loss = (loss_fn2 + loss_fn1) * (0.5**i)
                loss_total = loss + loss_total
            return loss_total 

        elif isinstance(preds, tuple):
            a = []
            # preds[0]= F.interpolate(preds[0], gt_masks.shape[2:], mode='bilinear', align_corners=False)
            for i in range(len(preds)):
                pred = preds[i]
                loss_fn1 = self.cal_loss1(pred, gt_masks)
                loss_fn2 = self.cal_loss2(pred, gt_masks)
                loss = (loss_fn2 + loss_fn1) * (0.5**i)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] 
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        # print("You are using `CEL`!")
        self.eps = 1e-6

    def forward(self, pred, target):
        pred = pred.sigmoid()
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)
