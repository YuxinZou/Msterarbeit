import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def square_distance(a, b):
    """
    Input:
        a: source points, [B, N, C]
        b: target points, [B, M, C]
    Output:
        result: per-point square distance, [B, N, M]
    """
    B, N, _ = a.shape
    _, M, _ = b.shape
    res = -2 * torch.matmul(a, b.permute(0, 2, 1))
    res += torch.sum(a**2, -1).view(B, N, 1)
    res += torch.sum(b**2, -1).view(B, 1, M)
    return res

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, D1, D2, ..., Dn]
    Return:
        new_points:, indexed points data, [B, D1, D2, ..., Dn, C] add extra dims(value of channels)
    """
    device = points.device
    B = points.shape[0]
    #a list with 2 nummer B and S(only the first layer)
    view_shape = list(idx.shape)
    # change the rest channel all 1, [B,1,1,1,...]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    #change the first dim to 1, rest remain
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(raw_points, npoint):
    """
    Input:
        raw_points: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud data, [B, npoint, C]
    """
    device = raw_points.device
    B, N, C = raw_points.shape
    #each batch contain B sets, for each sets sample S points
    S = npoint
    centroids = torch.zeros(B, S, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    #simply choose B random point
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)#size is Batch_size
    batch_indices = torch.arange(B, dtype=torch.long).to(device)#0,1,2,3,4,5.........
    for i in range(S):
	#each col store the index of centroid of each batch
        centroids[:, i] = farthest
	#farthest is a list, use this list as index
        centroid = raw_points[batch_indices, farthest, :].view(B, 1, 2)
	#calculate distance between all point and centroid, 3th dim to reduce, dim[B,N]
        dist = torch.sum((raw_points - centroid)**2, -1)
        mask = dist < distance
	#each time update the distance, to make each centroid more dispersed
        distance[mask] = dist[mask]
	#recoard the index of max distance
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, raw_points, centroids):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        raw_points: all points, [B, N, C]        all N cloud points
        centroids: query points, [B, S, C]         all S centroids,
    Return:
        group_idx: grouped points index, [B, S, nsample]
	s circles, each circles includes nsample points, centroid includes
    """
    device = raw_points.device
    B, N, C = raw_points.shape
    _, S, _ = centroids.shape
    #each circle contains only K points
    K = nsample
    #size B*S*N
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    #sqrdists: B*S*N
    sqrdists = square_distance(centroids, raw_points)
    group_idx[sqrdists > radius**2] = N
    #sort according to index, note that not choose k nearest point, randomly k point in the circle
    #reduce the dims from N to K
    group_idx = group_idx.sort(dim=-1)[0][:,:,:K]
    group_first = group_idx[:,:,0].view(B, S, 1).repeat([1, 1, K])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D] D is channel
    Return:
        new_xyz: sampled points position data, [B, S, C]
        new_points: sampled points data, [B, S, K, C+D]
    """
    B, N, C = xyz.shape
    S = npoint

    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))#B,S,C centroid point
    idx = query_ball_point(radius, nsample, xyz, new_xyz) #B,S,K
    grouped_xyz = index_points(xyz, idx) #B, S, K, C
    grouped_xyz -= new_xyz.view(B, S, 1, C) #B, S, K, C
    if points is not None:
        grouped_points = index_points(points, idx) #B,N,D,
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1) #B,S,K, C+D
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """  this is the last layer, so no sample_and_group
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)#only one point, but with value 0
    grouped_xyz = xyz.view(B, 1, N, C) #see all the points as sampled point
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def make_nets(size_list):
    layers = []
    for size in size_list:
    	layers.append(nn.Conv2d(size[0], size[1], 1))
    	layers.append(nn.BatchNorm2d(size[1]))
    	layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)
"""
class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
	self.mlps = nn.ModuleList() 
	for elem in mlp_list:
	    self.mlps.append(make_nets(elem))
    
    def forward(self, xyz, points):
        
       # Input:
         #   xyz: input points position data, [B, C, N]
        #    points: input points data, [B, D, N]
       # Return:
        #    new_xyz: sampled points position data, [B, C, S]
        #    new_points_concat: sample points feature data, [B, D', S]
        
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            model_sub = self.mlps[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1) #[B, D, K, S]

	    grouped_points = model_sub(grouped_points)
            new_points = torch.max(grouped_points, 2)[0]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

"""

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 2
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
    
    def forward(self, xyz, points):
       
       # Input:
       #     xyz: input points position data, [B, C, N]
       ##     points: input points data, [B, D, N]
       # Return:
       #     new_xyz: sampled points position data, [B, C, S]
       #     new_points_concat: sample points feature data, [B, D', S]
       #
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1) #[B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0] #[B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
	self.mlp = make_nets(mlp)
	
        self.group_all = group_all
    
    def forward(self, xyz, points):
       
       # Input: 
       #     xyz: input points position data, [B, C, N]
       #     points: input points data, [B, D, N]
       # Return:
       #     new_xyz: sampled points position data, [B, C, S]
       #     new_points_concat: sample points feature data, [B, D', S]
       
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1)
	
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNet2ClsSsg(nn.Module):
    def __init__(self):
        super(PointNet2ClsSsg, self).__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.1, 32, [[2,64],[64, 64],[64, 128]], False)
        self.sa2 = PointNetSetAbstraction(128, 0.2, 64, [[128 + 2,128],[128, 128],[128, 256]] , False)
        self.sa3 = PointNetSetAbstraction(None, None, None, [[256 + 2,256],[256, 512],[512, 1024]], True)
    
    def forward(self, xyz):
        B = xyz.shape[0]
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
	return x
"""
class PointNet2ClsMsg(nn.Module):
    def __init__(self):
        super(PointNet2ClsMsg, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(1024, [0.1,0.2,0.4], [16,32,128], [[[2, 32],[32, 32],[32, 64]],[[2, 64],[64, 64],[64, 128]],[[2, 64],[64, 96],[96, 128]]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.2,0.4,0.8], [32,64,128], [[[322,64],[64,64],[64,128]],[[322,128],[128,128],[128,256]],[[322,128],[128,128],[128,256]]])
        self.sa3 = PointNetSetAbstraction(None, None, None, [[640 + 2,256], [256, 512],[512, 1024]], True)


    def forward(self, xyz):
        B = xyz.shape[0]
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
	return x
"""
class PointNet2ClsMsg(nn.Module):
    def __init__(self):
        super(PointNet2ClsMsg, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1,0.2,0.4], [16,32,128], 0, [[32,32,64], [64,64,128], [64,96,128]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.2,0.4,0.8], [32,64,128], 320, [[64,64,128], [128,128,256], [128,128,256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, [[640 + 2,256], [256, 512],[512, 1024]], True)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
	return x
if __name__ == "__main__":
    print('...........test function square_distance...........')
    torch.manual_seed(999)
    src= torch.randn(8,10,2)
    dst= torch.randn(8,5,2)
    #print(a,a.shape)
    #print(b,b.shape)
    print(square_distance(src,dst),square_distance(src,dst).shape)

    print('...........test function farthest_point_sample...........')
    
    xyz = torch.randn(8,10,2)
    print('xyz,',xyz)
    idx = farthest_point_sample(xyz,4)
    print('result,',idx,idx.shape)

    print('...........test function index_points...........')
    res = index_points(xyz,idx)
    print(res,res.shape)

    print('...........test function query_ball_point...........')
    ball_points = query_ball_point(0.5, 3, xyz, res)
    print(ball_points,ball_points.shape)

    print('...........test function sample_and_group...........')
    new_xyz,new_points = sample_and_group(4, 0.5, 3, xyz, None)
    print('new_xyz,',new_xyz,new_xyz.shape)
    print('new_points,',new_points,new_points.shape)


    print('...........test function PointNetSetAbstraction...........')
    #model = PointNetSetAbstraction(512, 0.2, 32, [[2,64],[64, 64],[64, 128]], False)
    points = torch.randn(8,2,2048)
    #new_xyz,new_points = model(points,None)
    #print('new_points,',new_points,new_points.shape)


    print('...........test function PointNet2ClsSsg...........')
    model = PointNet2ClsSsg()
    output = model(points)
    print(output,output.shape)
 
