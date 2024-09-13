import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.DGCNN import DGCNN
from utils import (pairwise_distance_batch,
                   transform_point_cloud,
                   compute_rigid_transformation,
                   get_keypoints,
                   get_knn_index,
                   get_graph_feature,
                   sinkhorn,
                   angle, square_distance,pairwise_distance)
from pointnet2 import pointnet2_utils
from losses.chamfer_loss import *
from model.InlierEncoder import InlierEncoder
import time
import open3d as o3d


class norm(nn.Module):
    def __init__(self, axis=2):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        mean = torch.mean(x, self.axis,keepdim=True)
        std = torch.std(x, self.axis,keepdim=True)
        x = (x-mean)/(std+1e-6)
        return x

class Gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input*8
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Modified_softmax(nn.Module):
    def __init__(self, axis=1):
        super(Modified_softmax, self).__init__()
        self.axis = axis
        self.norm = norm(axis = axis)
    def forward(self, x):
        x = self.norm(x)
        x = Gradient.apply(x)
        x = F.softmax(x, dim=self.axis)
        return x

class RRPNet(nn.Module):
    def __init__(self, knn_num_list1, knn_num_list2,nn_margin=0.7,num_keypoints=256,loss_margin=0.01,iedim=8,dgcnn_emb=256):
        super(RRPNet, self).__init__()
        self.embedding = DGCNN(emb_dims=dgcnn_emb)
        self.knn_num_list1=knn_num_list1
        self.knn_num_list2=knn_num_list2
        self.nn_margin=nn_margin
        self.num_keypoints=num_keypoints
        self.weight_eval=InlierEncoder(dim=iedim)
        self.input_pts=768
        self.loss_margin=loss_margin
        self.NoSinkhorn = nn.Sequential(
            nn.Conv1d(in_channels=self.input_pts, out_channels=768, kernel_size=1, stride=1,
                      bias=True),
            nn.ReLU(),
            norm(axis=1),
            nn.Conv1d(in_channels=768, out_channels=self.input_pts, kernel_size=1, stride=1,
                      bias=True),
            Modified_softmax(axis=2)
        )

    def forward(self, ref_points, src_points, max_iterations=2):
        """
        :param ref_points:    [B,3,M]
        :param src_points:    [B,3,N]
        :param max_iterations:
        :return:
        """
        batch_size, num_dims_ref, num_points_ref=ref_points.shape
        _, num_dims_src, num_points_src = src_points.shape
        rotation_ab_pred = torch.eye(3, device=src_points.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src_points.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)
        global_alignment_loss, consensus_loss, spatial_consistency_loss, cd_loss = 0.0, 0.0, 0.0, 0.0
        for i in range(max_iterations):
            #--------------------------Finely Initial Correspondences--------------------------
            src_embedding, src_idx, src_knn, _ = self.embedding(src_points, k=self.knn_num_list1[i])  #[B,C,N],[B*N*k],[B,N,k,3],[B,N,k]
            ref_embedding, _, ref_knn, _ = self.embedding(ref_points, k=self.knn_num_list1[i])
            src_idx1, _ = get_knn_index(src_points, self.knn_num_list2[i])
            _, ref_idx2 = get_knn_index(ref_points, self.knn_num_list2[i])
            distance_map = pairwise_distance_batch(src_embedding, ref_embedding)  # [B, N, M]
            scores = self.NoSinkhorn(distance_map.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            src_knn_scores = scores.view(batch_size * num_points_src, -1)[src_idx1, :]  # [B*N,M]->[B*N*k,M]
            src_knn_scores = src_knn_scores.view(batch_size, num_points_src, self.knn_num_list2[i], num_points_ref)  # [B, N, k, M]
            src_knn_scores = pointnet2_utils.gather_operation(
                src_knn_scores.view(batch_size * num_points_src, self.knn_num_list2[i], num_points_ref), # [B*N, k, M]
                ref_idx2.view(batch_size, 1, num_points_ref * self.knn_num_list2[i]).repeat(1, num_points_src, 1). # [B,M,k]->[B,1,M*k]->[B,N,M*k]->[B*N,M*k]
                view(batch_size * num_points_src, num_points_ref * self.knn_num_list2[i]).int()).view(batch_size, num_points_src, self.knn_num_list2[i], num_points_ref, self.knn_num_list2[i])[:, :, 1:, :, 1:]\
                                 .sum(-1).sum(2) / (self.knn_num_list2[i]-1)

            # [B,N,k,M*k]->[B,N,k,M,k]->[B,N,M]
            #--------------------------The closed point of each points in src and ref-----------------
            _, _, src_singleknn, _ = get_graph_feature(src_points, 1)
            _, _, ref_singleknn, _ = get_graph_feature(ref_points, 1)
            src_points_1nn = src_singleknn.squeeze(2).permute(0,2,1)
            ref_points_1nn = ref_singleknn.squeeze(2).permute(0,2,1)
            src_embedding_1nn, src_idx_1nn, src_knn_1nn, _ = self.embedding(src_points_1nn, k=self.knn_num_list1[i])  # [B,C,N],[B*N*k],[B,N,k,3],[B,N,k]
            ref_embedding_1nn, _, ref_knn_1nn, _ = self.embedding(ref_points_1nn, k=self.knn_num_list1[i])
            src_idx1_1nn, _ = get_knn_index(src_points_1nn, self.knn_num_list2[i])
            _, ref_idx2_1nn = get_knn_index(ref_points_1nn, self.knn_num_list2[i])
            distance_map_1nn = pairwise_distance_batch(src_embedding_1nn, ref_embedding_1nn)  # [B, N, M]

            scores_1nn=self.NoSinkhorn(distance_map_1nn.transpose(1,2).contiguous()).transpose(1,2).contiguous()

            src_knn_scores_1nn = scores_1nn.view(batch_size * num_points_src, -1)[src_idx1_1nn, :]  # [B*N,M]->[B*N*k,M]
            src_knn_scores_1nn = src_knn_scores_1nn.view(batch_size, num_points_src, self.knn_num_list2[i],num_points_ref)  # [B, N, k, M]
            src_knn_scores_1nn = pointnet2_utils.gather_operation(
                src_knn_scores_1nn.view(batch_size * num_points_src, self.knn_num_list2[i], num_points_ref),  # [B*N, k, M]
                ref_idx2_1nn.view(batch_size, 1, num_points_ref * self.knn_num_list2[i]).repeat(1, num_points_src,
                                                                                   1).  # [B,M,k]->[B,1,M*k]->[B,N,M*k]->[B*N,M*k]
                    view(batch_size * num_points_src, num_points_ref * self.knn_num_list2[i]).int()).view(batch_size,num_points_src,self.knn_num_list2[i],num_points_ref,self.knn_num_list2[i])[:, :, 1:, :, 1:] \
                                 .sum(-1).sum(2) / (self.knn_num_list2[i]-1)
            # ---------------------------Merge src knn scores and the closest point knn scores----------------------------------------
            src_knn_scores = (src_knn_scores + src_knn_scores_1nn)
            src_knn_scores = self.nn_margin - src_knn_scores
            refined_distance_map = torch.exp(src_knn_scores)* (distance_map + distance_map_1nn)
            refined_matching_map = torch.softmax(-refined_distance_map, dim=2) # [B, N, M]

            # reference point cloud copy
            src_corr = torch.matmul(ref_points, refined_matching_map.transpose(2, 1).contiguous())  # [B,3,N]

            #-------------------------------- Inlier Confidence Calibration --------------------------------

            # neighborhoods of reference point cloud copy
            src_knn_corr = src_corr.transpose(2, 1).contiguous().view(batch_size * num_points_src, -1)[src_idx, :]  #[B,N,3]->[B*N,3]->[B*N*k,3]
            src_knn_corr = src_knn_corr.view(batch_size, num_points_src, self.knn_num_list1[i], num_dims_src)  # [B, N, k, 3]

            # edge features of the reference point cloud copy neighborhoods and the source neighborhoods
            knn_distance = src_corr.transpose(2, 1).contiguous().unsqueeze(2) - src_knn_corr  # [B, N, k, 3]
            # angle features of the reference point cloud copy neughborhoods and the source neighborhoods
            knn_angle = angle(knn_distance, knn_distance)  # [B, N, k]
            src_knn_distance = src_points.transpose(2, 1).contiguous().unsqueeze(2) - src_knn  # [B, N, k, 3]
            # src_knn_dnorm=torch.norm(src_knn_distance,dim=-1)
            src_knn_angle=angle(src_knn_distance,src_knn_distance) # [B, N, k]

            knn_feat = torch.cat([knn_distance,knn_angle.unsqueeze(3)], dim=-1)  # [B, N, k, 4]
            src_knn_feat=torch.cat([src_knn_distance,src_knn_angle.unsqueeze(3)], dim=-1)  # [B, N, k, 4]
            weight = self.weight_eval(knn_feat, src_knn_feat)  # [B, 1, N]

            # compute rigid transformation
            R, t = compute_rigid_transformation(src_points, src_corr, weight)  # weighted SVD
            #--------------------------------- Preparation for the Loss Function ---------------------------
            # choose k keypoints with highest weights
            src_topk_idx, src_keypoints, ref_keypoints = get_keypoints(src_points, src_corr, weight, self.num_keypoints)
            #------------------------------------Compute loss function-----------------------------------------
            idx_ref_corr = torch.argmax(refined_matching_map, dim=-1).int()  # [B,N,M]->[B, N]
            identity = torch.eye(num_points_ref).cuda().unsqueeze(0).repeat(batch_size, 1, 1)  # [B, M, M]
            one_hot_number = pointnet2_utils.gather_operation(identity, idx_ref_corr)  # [B, M, N]
            src_keypoints_idx = src_topk_idx.repeat(1, num_points_ref, 1)  # [B, M, num_keypoints]
            keypoints_one_hot = torch.gather(one_hot_number, dim=2, index=src_keypoints_idx).transpose(2, 1).reshape(
                batch_size * self.num_keypoints, num_points_ref)
            # [B, M, num_keypoints] -> [B, num_keypoints, M] -> [B * num_keypoints, M]
            predicted_keypoints_scores = torch.gather(refined_matching_map.transpose(2, 1), dim=2,
                                                      index=src_keypoints_idx).transpose(2, 1).reshape(
                batch_size * self.num_keypoints, num_points_ref)
            loss_scl = (-torch.log(predicted_keypoints_scores + 1e-15) * keypoints_one_hot).sum(1).mean()

            # neighorhood information
            src_keypoints_idx2 = src_topk_idx.unsqueeze(-1).repeat(1, 3, 1, self.knn_num_list1[i])  # [B, 3, num_keypoints, k]
            ref_keypoints_knn = torch.gather(knn_distance.permute(0, 3, 1, 2), dim=2, index=src_keypoints_idx2)  # [B, 3, num_kepoints, k]

            src_transformed = transform_point_cloud(src_points, R, t.view(batch_size, 3))
            src_transformed_knn_corr = src_transformed.transpose(2, 1).contiguous().view(batch_size * num_points_src, -1)[src_idx, :]
            src_transformed_knn_corr = src_transformed_knn_corr.view(batch_size, num_points_src, self.knn_num_list1[i],num_dims_src)  # [B, N, k, 3]

            knn_distance2 = src_transformed.transpose(2, 1).contiguous().unsqueeze(2) - src_transformed_knn_corr  # [B, N, k, 3]
            src_keypoints_knn = torch.gather(knn_distance2.permute(0, 3, 1, 2), dim=2, index=src_keypoints_idx2)  # [B, 3, num_kepoints, k]
            rotation_ab_pred = torch.matmul(R, rotation_ab_pred)
            translation_ab_pred = torch.matmul(R, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + t.view(batch_size,3)
            src_points = transform_point_cloud(src_points, R, t.view(batch_size,3))

            global_alignment_loss_i=GlobalAlignLoss()(src_points.permute(0,2,1),ref_points.permute(0,2,1),self.loss_margin)

            transformed_srckps_forward=transform_point_cloud(src_keypoints,R,t.view(batch_size,3))
            keypoints_loss_i=nn.MSELoss(reduction='sum')(transformed_srckps_forward.permute(0,2,1),ref_keypoints.permute(0,2,1))

            knn_consensus_loss_i=nn.MSELoss(reduction='sum')(src_keypoints_knn,ref_keypoints_knn)\
                                 +nn.MSELoss(reduction='sum')(
                angle(src_keypoints_knn, src_keypoints_knn),angle(ref_keypoints_knn, ref_keypoints_knn)
            )

            neighborhood_consensus_loss_i=knn_consensus_loss_i /self.knn_num_list2[i]+ keypoints_loss_i

            global_alignment_loss += global_alignment_loss_i
            consensus_loss += neighborhood_consensus_loss_i
            spatial_consistency_loss += loss_scl


        result={
            "transformed_source":src_points,
            "pred_R":rotation_ab_pred,
            "pred_t":translation_ab_pred,
            "global_alignment_loss":global_alignment_loss,
            "consensus_loss":consensus_loss,
            "spatial_consistency_loss":spatial_consistency_loss
        }
        return result