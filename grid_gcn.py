import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import dgl
import dgl.function as fn
import math

'''
Part of the code are adapted from
https://github.com/yanx27/Pointnet_Pointnet2_pytorch
'''

'''
configs
save_model_prefix: modelnet40
voxel_size_lst: [[0.05, 0.05, 0.05], [0.25, 0.25, 0.25], [2.0, 2.0, 2.0]]
grid_size_lst: [[40, 40, 40], [8, 8, 8], [1, 1, 1]]
lidar_coord: [1.0, 1.0, 1.0]
max_p_grid_lst: [64, 64, 128]
max_o_grid_lst: [1024, 128, 1]
kernel_size_lst: [7, 3, 1]
'''

def square_distance(src, dst):
    '''
    Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    '''
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    '''
    Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    '''
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class VoxelModule(nn.Module):
    def __init__(self, voxel_size):
        super(VoxelModule, self).__init__()
        #self.point_cloud = point_cloud
        self.voxel_size = voxel_size
        self.voxels = None
        self.index_voxels = []
        for i in range(voxel_size):
            self.index_voxels.append([])
            for j in range(voxel_size):
                self.index_voxels[i].append([])
                for k in range(voxel_size):
                    self.index_voxels[i][j].append([])
        # self.voxels = torch.div(self.point_cloud, self.voxel_size, out=None)
        #print("/////")

    
    def forward(self, point_cloud):
        #self.voxels = torch.div(point_cloud, self.voxel_size, out=None)
        size = point_cloud.size()
        #print(size)
        # torch.Size([32, 1024, 3])
        xyz_max = torch.max(point_cloud, 1)
        xyz_min = torch.min(point_cloud, 1)
        self.voxels = torch.rand(size[0],size[1],size[2])
        #print(xyz_max.values[1][2].item())
        indexes = [0,0,0]
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                    indexes[k] = int((point_cloud[i][j][k].item()-xyz_min.values[i][k].item())/((xyz_max.values[i][k].item()-xyz_min.values[i][k].item())/40.0))
                    self.voxels[i][j][k] = indexes[k]
                    #print(self.voxels[i][j][k].item()) 
                self.index_voxels[indexes[0]][indexes[1]][indexes[2]].append([i,j]) 
                print(self.index_voxels[indexes[0]][indexes[1]][indexes[2]])                 



        return self.voxels
        
        # Here !!!!
        # conduct voxels!!!
        # map point cloud to voxels 

    # this function gets point cloud initial xyz
    def get_point(self, point_cloud, voxel_size):
        #torch.div(input,, out=None)
        print("111")

    # 
    def get_context_points(self):
        print("111")


     

class FarthestPointSampler(nn.Module):
    '''
    Sample the farthest point iteratively
    '''

    # Why not use the function in ModelNetDataLoader? 


    def __init__(self, npoints):
        super(FarthestPointSampler, self).__init__()
        self.npoints = npoints

    def forward(self, pos):
        '''
        Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
        '''
        device = pos.device
        B, N, C = pos.shape
        centroids = torch.zeros(B, self.npoints, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(self.npoints):
            centroids[:, i] = farthest
            centroid = pos[batch_indices, farthest, :].view(B, 1, C)
            dist = torch.sum((pos - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        #print(self.npoints)
        #print(centroids.size())

        return centroids



# CAS Module
class CAS(nn.Module):
    def __init__(self, npoints):
        super(CAS, self).__init__()
        self.npoints = npoints

# RVS Module
class RVS(nn.Module):
    def __init__(self, npoints):
        super(RVS, self).__init__()
        self.npoints = npoints

    def forward(self, pos):
        return 0     


class FixedRadiusNearNeighbors(nn.Module):
    '''
    Find the neighbors with-in a fixed radius
    '''
    def __init__(self, radius, n_neighbor):
        super(FixedRadiusNearNeighbors, self).__init__()
        self.radius = radius
        self.n_neighbor = n_neighbor

    def forward(self, pos, centroids):
        '''
        Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
        '''
        device = pos.device
        B, N, _ = pos.shape
        center_pos = index_points(pos, centroids)
        _, S, _ = center_pos.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = square_distance(center_pos, pos)
        group_idx[sqrdists > self.radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :self.n_neighbor]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, self.n_neighbor])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        return group_idx


# This function find fixed number neighbours from context points
class FixedNumberNeighbors(nn.Module):
    def __init__(self, n_neighbor):
        super(FixedNumberNeighbors, self).__init__()
        self.n_neighbor = n_neighbor

    


class FixedRadiusNNGraph(nn.Module):
    '''
    Build NN graph
    '''
    
    # Remain the same? 

    def __init__(self, radius, n_neighbor):
        super(FixedRadiusNNGraph, self).__init__()
        self.radius = radius
        self.n_neighbor = n_neighbor
        self.frnn = FixedRadiusNearNeighbors(radius, n_neighbor)

    def forward(self, pos, centroids, feat=None):
        dev = pos.device
        group_idx = self.frnn(pos, centroids)
        B, N, _ = pos.shape
        glist = []
        for i in range(B):
            center = torch.zeros((N)).to(dev)
            center[centroids[i]] = 1
            src = group_idx[i].contiguous().view(-1)
            dst = centroids[i].view(-1, 1).repeat(1, self.n_neighbor).view(-1)

            unified = torch.cat([src, dst])
            uniq, idx, inv_idx = np.unique(unified.cpu().numpy(), return_index=True, return_inverse=True)
            src_idx = inv_idx[:src.shape[0]]
            dst_idx = inv_idx[src.shape[0]:]

            g = dgl.DGLGraph((src_idx, dst_idx), readonly=True)
            g.ndata['pos'] = pos[i][uniq]
            g.ndata['center'] = center[uniq]
            if feat is not None:
                g.ndata['feat'] = feat[i][uniq]
            glist.append(g)
        bg = dgl.batch(glist)
        return bg


class RelativePositionMessage(nn.Module):
    '''
    Compute the input feature from neighbors
    '''

    # Remain the same

    def __init__(self, n_neighbor):
        super(RelativePositionMessage, self).__init__()
        self.n_neighbor = n_neighbor

    def forward(self, edges):
        pos = edges.src['pos'] - edges.dst['pos']
        if 'feat' in edges.src:
            res = torch.cat([pos, edges.src['feat']], 1)
        else:
            res = pos
        return {'agg_feat': res}


class Grid_GCN_Conv(nn.Module):
    '''
    Feature aggregation
    '''
    def __init__(self, sizes, batch_size):
        super(Grid_GCN_Conv, self).__init__()
        self.batch_size = batch_size
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(1, len(sizes)):
            self.conv.append(nn.Conv2d(sizes[i-1], sizes[i], 1))
            self.bn.append(nn.BatchNorm2d(sizes[i]))

    def forward(self, nodes):
        shape = nodes.mailbox['agg_feat'].shape
        h = nodes.mailbox['agg_feat'].view(self.batch_size, -1, shape[1], shape[2]).permute(0, 3, 1, 2)
        for conv, bn in zip(self.conv, self.bn):
            h = conv(h)
            h = bn(h)
            h = F.relu(h)
        h = torch.max(h, 3)[0]
        feat_dim = h.shape[1]
        h = h.permute(0, 2, 1).reshape(-1, feat_dim)
        return {'new_feat': h}
    
    def group_all(self, pos, feat):
        '''
        Feature aggretation and pooling for the non-sampling layer
        '''
        '''
        Here we concatenate x_c, x_i, w_i to h_geo
        concatenate fcxt, fi to h_sematic
        '''

        '''
        Q&A: How could we concatenate the weight matrix?
        '''

        # h_geo = 
        # h_semantic = 

        if feat is not None:
            h = torch.cat([pos, feat], 2)


        else:
            h = pos
        shape = h.shape
        h = h.permute(0, 2, 1).view(shape[0], shape[2], shape[1], 1)
        # h_geo = 
        # h_sematic = 
        for conv, bn in zip(self.conv, self.bn):
            h = conv(h)
            h = bn(h)
            h = F.relu(h)
        h = torch.max(h[:, :, :, 0], 2)[0]
        
        # Should we do the same thing to h_geo and h_semantic


        return h

class SAModule(nn.Module):
    """
    The Set Abstraction Layer
    """
    def __init__(self, npoints, batch_size, radius, mlp_sizes, n_neighbor=64,
                 group_all=False):
        super(SAModule, self).__init__()
        self.group_all = group_all
        if not group_all:
            self.fps = FarthestPointSampler(npoints)
            self.rvs = RVS(npoints)
            self.frnn_graph = FixedRadiusNNGraph(radius, n_neighbor)
        self.message = RelativePositionMessage(n_neighbor)
        self.conv = Grid_GCN_Conv(mlp_sizes, batch_size)
        self.batch_size = batch_size

    def forward(self, pos, feat):
        if self.group_all:
            return self.conv.group_all(pos, feat)

        #centroids = self.fps(pos)
        centroids = self.rvs(pos)
        g = self.frnn_graph(pos, centroids, feat)
        g.update_all(self.message, self.conv)
        mask = g.ndata['center'] == 1
        pos_dim = g.ndata['pos'].shape[-1]
        feat_dim = g.ndata['new_feat'].shape[-1]
        pos_res = g.ndata['pos'][mask].view(self.batch_size, -1, pos_dim)
        feat_res = g.ndata['new_feat'][mask].view(self.batch_size, -1, feat_dim)
        return pos_res, feat_res



class Grid_GCN(nn.Module):
    def __init__(self, output_classes, batch_size, input_dims=3, dropout_prob=0.4):
        super(Grid_GCN, self).__init__()
        self.input_dims = input_dims

        self.sa_module1 = SAModule(512, batch_size, 0.2, [input_dims, 64, 64, 128])
        self.sa_module2 = SAModule(128, batch_size, 0.4, [128 + 3, 128, 128, 256])
        self.sa_module3 = SAModule(None, batch_size, None, [256 + 3, 256, 512, 1024],
                                   group_all=True)

        self.mlp1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout_prob)

        self.mlp2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout_prob)

        self.mlp_out = nn.Linear(256, output_classes)

        self.selfvoxels = VoxelModule(40)

    def forward(self, x):
        #print("----------")
        voxels = self.selfvoxels(x[:, :, :3])
        #print(voxels)
        #print("----------")
        if x.shape[-1] > 3:
            pos = x[:, :, :3]
            feat = x[:, :, 3:]
        else:
            pos = x
            feat = None
        pos, feat = self.sa_module1(pos, feat)
        pos, feat = self.sa_module2(pos, feat)
        #print(self.sa_module3(pos, feat))
        h = self.sa_module3(pos, feat)

        h = self.mlp1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.drop1(h)
        h = self.mlp2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.drop2(h)

        out = self.mlp_out(h)
        return out

