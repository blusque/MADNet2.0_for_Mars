import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# import time


class GradientLoss(nn.Module):
    """ (1/rc) * (grad_x(h_gt, h_est) + grad_y(h_gt, h_est)) """

    def __init__(self, reduction='mean'):
        super(GradientLoss, self).__init__()
        sobel_x = torch.tensor([[[[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]]], dtype=torch.float32)
        self.sobel_x = nn.Parameter(data=sobel_x, requires_grad=False)
        self.sobel_y = nn.Parameter(data=sobel_y, requires_grad=False)
        self.reduction = reduction

    def forward(self, gt: torch.tensor, est: torch.tensor):
        grad_gt_x = torch.abs(F.conv2d(gt, self.sobel_x, stride=1, padding=1))
        grad_gt_y = torch.abs(F.conv2d(gt, self.sobel_y, stride=1, padding=1))
        grad_est_x = torch.abs(F.conv2d(est, self.sobel_x, stride=1, padding=1))
        grad_est_y = torch.abs(F.conv2d(est, self.sobel_y, stride=1, padding=1))
        delta_x = (grad_gt_x - grad_est_x) ** 2
        delta_y = (grad_gt_y - grad_est_y) ** 2
        if self.reduction == 'none':
            return delta_x + delta_y
        elif self.reduction == 'mean':
            return torch.mean(delta_x + delta_y)
        elif self.reduction == 'sum':
            return torch.sum(delta_x + delta_y)


class BerhuLoss(nn.Module):
    """ L1 when abs(h_gt - h_est) < tao, L2 when abs(h_gt - h_est) > tao """

    def __init__(self, reduction='mean'):
        super(BerhuLoss, self).__init__()
        self.tao = 255
        self.reduction = reduction

    def forward(self, gt, est):
        delta = torch.abs(gt - est)
        self.tao = 0.2 * torch.max(delta)
        # n = delta.shape[0]
        # c = delta.shape[1]
        # row = 0
        # col = 0
        result = Variable(torch.cuda.FloatTensor(delta.shape))
        # start = time.time()
        result = torch.where(delta <= self.tao, delta, 
                             0.5 * delta ** 2 / self.tao + 0.5 * self.tao)
        # end = time.time()
        # print("bh loss loop cost: {}s".format(end - start))
        if self.reduction == 'none':
            return result
        elif self.reduction == 'mean':
            return torch.mean(result)
        elif self.reduction == 'sum':
            return torch.sum(result)
        
        
class AdversarialLoss(nn.Module):
    """ """
    
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        
    def forward(self, v1, v2):
        return - torch.log(v1).mean() - torch.log(1 - v2).mean()
