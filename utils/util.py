import torch
import numpy as np
from scipy.spatial.transform import Rotation
import torch.nn as nn

def sinkhorn(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))

        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha

def square_distance(src, dst):
	"""Calculate Euclid distance between each two points.
		src^T * dst = xn * xm + yn * ym + zn * zm；
		sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
		sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
		dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
			 = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

	Args:
		src: source points, [B, N, C]
		dst: target points, [B, M, C]
	Returns:
		dist: per-point square distance, [B, N, M]
	"""
	B, N, _ = src.shape
	_, M, _ = dst.shape
	dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
	dist += torch.sum(src ** 2, dim=-1)[:, :, None]
	dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
	return dist

def pairwise_distance_batch(x,y):
    """
        pairwise_distance
        Args:
            x: Input features of source point clouds. Size [B, c, N]
            y: Input features of source point clouds. Size [B, c, M]
        Returns:
            pair_distances: Euclidean distance. Size [B, N, M]
    """
    xx = torch.sum(torch.mul(x,x), 1, keepdim = True)#[b,1,n]
    yy = torch.sum(torch.mul(y,y),1, keepdim = True) #[b,1,n]
    inner = -2*torch.matmul(x.transpose(2,1),y) #[b,n,n]
    pair_distance = xx.transpose(2,1) + inner + yy #[b,n,n]
    device = torch.device('cuda')
    zeros_matrix = torch.zeros_like(pair_distance,device=device)
    pair_distance_square = torch.where(pair_distance > 0.0,pair_distance,zeros_matrix)
    error_mask = torch.le(pair_distance_square,0.0)
    pair_distances = torch.sqrt(pair_distance_square + error_mask.float()*1e-16)
    pair_distances = torch.mul(pair_distances,(1.0-error_mask.float()))
    return pair_distances

def pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    # sq_distances = sq_distances.clamp(min=0.0)   #限制sq_distance中的值范围大于0
    return sq_distances

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_knn_index(x, k):
    """
        knn-graph.
        Args:
            x: Input point clouds. Size [B, 3, N]
            k: Number of nearest neighbors.
        Returns:
            idx: Nearest neighbor indices. Size [B * N * k]
            idx2: Nearest neighbor indices. Size [B, N, k]
    """
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    idx2 = idx
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size,device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)
    return idx, idx2

def get_graph_feature(x, k):
    """
        knn-graph.
        Args:
            x: Input point clouds. Size [B, 3, N]
            k: Number of nearest neighbors.
        Returns:
            idx: Nearest neighbor indices. Size [B * N * k]
            relative_coordinates: Relative coordinates between nearest neighbors and the center point. Size [B, 3, N, K]
            knn_points: Coordinates of nearest neighbors. Size[B, N, K, 3].
            idx2: Nearest neighbor indices. Size [B, N, k]
    """
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    idx2 = idx
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size,device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    knn_points = x.view(batch_size * num_points, -1)[idx, :]
    knn_points = knn_points.view(batch_size, num_points, k, num_dims)#[b, n, k, 3],knn
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)#[b, n, k, 3],central points

    relative_coordinates = (knn_points - x).permute(0, 3, 1, 2)

    return idx, relative_coordinates, knn_points, idx2

def compute_rigid_transformation(src, src_corr, weight):
    """
        Compute rigid transforms between two point sets
        Args:
            src: Source point clouds. Size (B, 3, N)
            src_corr: Pseudo target point clouds. Size (B, 3, N)
            weights: Inlier confidence. (B, 1, N)

        Returns:
            R: Rotation. Size (B, 3, 3)
            t: translation. Size (B, 3, 1)
    """
    src2 = (src * weight).sum(dim = 2, keepdim = True) / weight.sum(dim = 2, keepdim = True)
    src_corr2 = (src_corr * weight).sum(dim = 2, keepdim = True)/weight.sum(dim = 2,keepdim = True)
    src_centered = src - src2
    src_corr_centered = src_corr - src_corr2
    H = torch.matmul(src_centered * weight, src_corr_centered.transpose(2, 1).contiguous())

    R = []

    for i in range(src.size(0)):
        u, s, v = torch.svd(H[i])
        r = torch.matmul(v, u.transpose(1, 0)).contiguous()
        r_det = torch.det(r).item()
        diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                            [0, 1.0, 0],
                                            [0, 0, r_det]]).astype('float32')).to(v.device)
        r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous()
        R.append(r)

    R = torch.stack(R, dim = 0).cuda()

    t = torch.matmul(-R, src2.mean(dim = 2, keepdim=True)) + src_corr2.mean(dim = 2, keepdim = True)
    return R, t

def get_keypoints(src, src_corr, weight, num_keypoints):
    """
        Compute rigid transforms between two point sets
        Args:
            src: Source point clouds. Size (B, 3, N)
            src_corr: Pseudo target point clouds. Size (B, 3, N)
            weights: Inlier confidence. (B, 1, N)
            num_keypoints: Number of selected keypoints.

        Returns:
            src_topk_idx: Keypoint indices. Size (B, 1, num_keypoints)
            src_keypoints: Keypoints of source point clouds. Size (B, 3, num_keypoints)
            tgt_keypoints: Keypoints of target point clouds. Size (B, 3, num_keypoints)
    """
    src_topk_idx = torch.topk(weight, k = num_keypoints, dim = 2, sorted=False)[1]
    src_keypoints_idx = src_topk_idx.repeat(1, 3, 1)
    src_keypoints = torch.gather(src, dim = 2, index = src_keypoints_idx)
    tgt_keypoints = torch.gather(src_corr, dim = 2, index = src_keypoints_idx)
    return src_topk_idx, src_keypoints, tgt_keypoints

#-----------------------------------------------------------------------------------------------------

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')

def dcm2euler(mats, seq: str = 'zyx', degrees: bool = True):#: np.ndarray
    """Converts rotation matrix to euler angles

    Args:
        mats: (B, 3, 3) containing the B rotation matricecs
        seq: Sequence of euler rotations (default: 'zyx')
        degrees (bool): If true (default), will return in degrees instead of radians

    Returns:

    """

    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=degrees))
    return np.stack(eulers)

#----------------------------------------------------------------------------------

def angle(v1: torch.Tensor, v2: torch.Tensor):
    """Compute angle between 2 vectors

    For robustness, we use the same formulation as in PPFNet, i.e.
        angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
    This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0

    Args:
        v1: (B, *, 3)
        v2: (B, *, 3)

    Returns:

    """

    cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
                              v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
                              v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
    cross_prod_norm = torch.norm(cross_prod, dim=-1)+1e-5
    dot_prod = torch.sum(v1 * v2, dim=-1)

    return torch.atan2(cross_prod_norm, dot_prod)

def inv_R_t(R, t):
    inv_R = R.permute(0, 2, 1).contiguous()
    inv_t = - inv_R @ t[..., None]
    return inv_R, torch.squeeze(inv_t, -1)