import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import pairwise_distance_batch,get_graph_feature,get_knn_index
import numpy as np
from pointnet2 import pointnet2_utils
class DGCNN(nn.Module):
    def __init__(self, emb_dims=96):
        super(DGCNN, self).__init__()
        self.emb_dims = emb_dims
        self.conv1 = nn.Conv2d(3, 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(256, self.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(self.emb_dims)
        self.dp = nn.Dropout(p=0.3)

    def forward(self, points, k):
        """
            Simplified DGCNN.
            Args:
                points: Input point clouds. Size [B, 3, N]
                k: Number of nearest neighbors.
                relative_coordinates: Relative coordinates between nearest neighbors and the center point. Size [B, 3, N, K]
            Returns:
                x: Features. Size [B, self.emb_dims, N]
                idx: Nearest neighbor indices [B*N*k]
                idx2: Nearest neighbor indices [B, N, k]/[B, M, k]
                knn_points: [B, N, k, 3]/[B, M, k, 3]
        """
        idx, relative_coordinates, knn_points, idx2 = get_graph_feature(points, k)

        batch_size, num_dims, num_points, _ = relative_coordinates.size()

        x = F.relu(self.bn1(self.conv1(relative_coordinates)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x).view(batch_size, -1, num_points)

        return x, idx, knn_points, idx2



