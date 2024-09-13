import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):

        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)  # [b,n]
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)

        return loss_1/preds.shape[1] + loss_2/gts.shape[1]

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P


class GlobalAlignLoss(nn.Module):

    def __init__(self):
        super(GlobalAlignLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts, c):
        P = self.batch_pairwise_dist(gts, preds)

        mins, _ = torch.min(P, 1)
        mins = self.huber_loss(mins, c)
        # mins = self.mccr_loss(mins, c)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        mins = self.huber_loss(mins, c)
        # mins = self.mccr_loss(mins, c)
        loss_2 = torch.sum(mins)

        return loss_1+loss_2

    def huber_loss(self, x, c):
        x = torch.where(x < c, 0.5 * (x ** 2), c * x - 0.5 * (c ** 2))
        return x

    def mccr_loss(self, x, sigma):
        x = (sigma ** 2) * (1.0 - torch.exp(x / (sigma ** 2)))
        return x

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P
