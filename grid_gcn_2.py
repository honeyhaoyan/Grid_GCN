import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import dgl
import dgl.function as fn
import math
import random

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
        self.voxel_size = voxel_size
        self.voxels = None

    # append val to self.index_voxels
    def set_voxel_value(self, index_voxels, val, x, y, z):
        index1 = self.voxel_size * self.voxel_size
        index2 = self.voxel_size
        index_voxels[x * index1 + y * index2 + z].append(val)

    def forward(self, point_cloud):
        #self.voxels = torch.div(point_cloud, self.voxel_size, out=None)
        size = point_cloud.size()
        # torch.Size([32, 1024, 3])
        index_voxels = []
        for i in range(size[0]): # batch
            xyz_max = torch.max(point_cloud[i,:,:], 0)
            xyz_min = torch.min(point_cloud[i,:,:], 0)
            index_voxels_tmp = []
            for _ in range(self.voxel_size):
                for _ in range(self.voxel_size):
                    for _ in range(self.voxel_size):
                        index_voxels_tmp.append([])             
            for j in range(size[1]):
                index = (point_cloud[i,j,:]-xyz_min.values)/(xyz_max.values-xyz_min.values)*(self.voxel_size-1)
                # print(index)
                # print(int(index[0]))
                self.set_voxel_value(index_voxels_tmp, j, int(index[0]), int(index[1]), int(index[2]))
            index_voxels.append(index_voxels_tmp)
        return index_voxels

   

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
        return centroids


# CAS Module
class CAS(nn.Module):
    def __init__(self, npoints):
        super(CAS, self).__init__()
        self.npoints = npoints

    def forward(self, pos, index_voxels):
        pass

# RVS Module
class RVS(nn.Module):
    def __init__(self, npoints):
        super(RVS, self).__init__()
        self.npoints = npoints

    def forward(self, pos, index_voxels):
        B = len(index_voxels) # batch_size
        device = pos.device
        vs = int(np.cbrt(len(index_voxels[0]))) # 64 -> 4, voxel_size
        centroids = torch.zeros(B, self.npoints, dtype=torch.long).to(device)
        centroids_index = []

        for batch in range(B):
            occupied = []
            for i in range(vs):
                for j in range(vs):
                    for k in range(vs):
                        if len(self.get_voxel_value(index_voxels, vs, batch, i, j, k)) != 0:
                            occupied.append([i, j, k])
            if self.npoints <= len(occupied):
                selected = random.sample(occupied, self.npoints)
            
            indexs = []
            if self.npoints <= len(occupied):
                for i in range(self.npoints):
                    val = self.get_voxel_value(index_voxels, vs, batch, selected[i][0], selected[i][1], selected[i][2])
                    index = int(random.sample(val, 1)[0])
                    centroids[batch,i] = index #index
                    indexs.append([batch, selected[i][0], selected[i][1], selected[i][2], val.index(index)])
            else:
                added = []
                for i in range(len(occupied)):
                    val = self.get_voxel_value(index_voxels, vs, batch, occupied[i][0], occupied[i][1], occupied[i][2])
                    index = int(random.sample(val, 1)[0])
                    centroids[batch,i] = index
                    added.append(index)
                    indexs.append([batch, occupied[i][0], occupied[i][1], occupied[i][2], val.index(index)])
                add_num = 0
                while add_num < (self.npoints-len(occupied)):
                    index = int(random.sample(range(pos.shape[1]), 1)[0])
                    if index not in added:
                        centroids[batch, len(occupied)+add_num] = index
                        added.append(index)
                        add_num += 1

            centroids_index.append(indexs)
        return centroids, centroids_index # centroid_index is not used

    # get value from self.index_voxels
    def get_voxel_value(self, index_voxels, voxel_size, batch, x, y, z):
        index1 = voxel_size * voxel_size
        index2 = voxel_size
        return index_voxels[batch][x * index1 + y * index2 + z]


class FixedRadiusNearNeighbors(nn.Module):
    '''
    Find the neighbors with-in a fixed radius
    '''
    def __init__(self, radius, n_neighbor):
        super(FixedRadiusNearNeighbors, self).__init__()
        self.radius = radius
        self.n_neighbor = n_neighbor

    def forward(self, pos, centroids, centroids_index, index_voxels):
        '''
        Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
        TODO: Need to update the select neighbor operation
        '''
        device = pos.device
        B, N, _ = pos.shape
        center_pos = index_points(pos, centroids)
        # print(center_pos.shape)
        _, S, _ = center_pos.shape
        # print(B, N, S)
        # print(torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).shape)
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        # print(group_idx.shape)
        sqrdists = square_distance(center_pos, pos)
        # print(sqrdists.shape)
        group_idx[sqrdists > self.radius ** 2] = N
        # print(group_idx.shape)
        group_idx = group_idx.sort(dim=-1)[0][:, :, :self.n_neighbor]
        # print(group_idx.shape)
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, self.n_neighbor])
        # print(group_first.shape)
        mask = group_idx == N
        # print(mask.shape)
        group_idx[mask] = group_first[mask]
        # print('group_idx', group_idx.shape)
        print(group_idx.shape)
        print(group_idx[0,:,:])
        print(group_idx[0,0,:])
        return group_idx

class GridGCNNearNeighbors(nn.Module):
    '''
    Find the neighbors with-in a fixed radius
    '''
    def __init__(self, radius, n_neighbor):
        super(GridGCNNearNeighbors, self).__init__()
        self.radius = radius
        self.n_neighbor = n_neighbor

    def forward(self, pos, centroids, centroids_index, index_voxels):
        '''
        Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
        TODO: Need to update the select neighbor operation
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

    def forward(self, pos, centroids, centroids_index, index_voxels, feat=None):
        dev = pos.device
        group_idx = self.frnn(pos, centroids, centroids_index, index_voxels)
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
            #print("=========== in if ===========")
            res = torch.cat([pos, edges.src['feat']], 1)
        else:
            #print("in else")
            res = pos
        # print(pos.shape, edges.src['pos'].shape, edges.src['pos'].shape, edges.src['feat'].shape)
        # print(edges.src.keys())
        # print(edges.src['center'].shape)
        geo_feat = torch.cat([edges.src['pos'], edges.dst['pos']], 1)
        print('send feat shape ', res.shape, geo_feat.shape)
        return {'agg_feat': res, 'geo_feat': geo_feat}


class Grid_GCN_Conv(nn.Module):
    '''
    Feature aggregation
    '''
    def __init__(self, sizes, batch_size):
        super(Grid_GCN_Conv, self).__init__()
        self.batch_size = batch_size
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.sizes = sizes
        print("sizes: "+str(sizes))
        for i in range(1, len(sizes)):
            self.conv.append(nn.Conv2d(sizes[i-1], sizes[i], 1))
            self.bn.append(nn.BatchNorm2d(sizes[i]))
        # geo
        self.conv_geo = nn.ModuleList()
        self.bn_geo = nn.ModuleList()
        for i in range(1, len(sizes)):
            if i == 1:
                self.conv_geo.append(nn.Conv2d(6, sizes[i], 1))
                self.bn_geo.append(nn.BatchNorm2d(sizes[i])) 
            else:               
                self.conv_geo.append(nn.Conv2d(sizes[i-1], sizes[i], 1))
                self.bn_geo.append(nn.BatchNorm2d(sizes[i]))

    def forward(self, nodes):
        shape = nodes.mailbox['agg_feat'].shape
        h = nodes.mailbox['agg_feat'].view(self.batch_size, -1, shape[1], shape[2]).permute(0, 3, 1, 2)
        print('here shape: ', h.shape)
        for conv, bn in zip(self.conv, self.bn):
            h = conv(h)
            h = bn(h)
            h = F.relu(h)
            # print("H shape: ")=
        h = torch.max(h, 3)[0]
        feat_dim = h.shape[1]
        h = h.permute(0, 2, 1).reshape(-1, feat_dim)
        print('h shape: ', h.shape)

        # geo
        shape = nodes.mailbox['geo_feat'].shape
        print('22 ', shape)
        h_geo = nodes.mailbox['geo_feat'].view(self.batch_size, -1, shape[1], shape[2]).permute(0, 3, 1, 2)
        print('here shape: ', h_geo.shape)
        for conv, bn in zip(self.conv_geo, self.bn_geo):
            h_geo = conv(h_geo)
            h_geo = bn(h_geo)
            h_geo = F.relu(h_geo)
        h_geo = torch.max(h_geo, 3)[0]
        feat_dim = h_geo.shape[1]
        h_geo = h_geo.permute(0, 2, 1).reshape(-1, feat_dim)
        
        h_all = torch.cat([h, h_geo], 0)
        print('h_geo.shape: ', h_geo.shape)
        return {'new_feat': h_all}
    
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
    def __init__(self, npoints, batch_size, radius, voxel_size, mlp_sizes, n_neighbor=64,
                 group_all=False):
        super(SAModule, self).__init__()
        self.group_all = group_all
        self.voxel_size = voxel_size
        if not group_all:
            self.fps = FarthestPointSampler(npoints)
            self.rvs = RVS(npoints)
            self.frnn_graph = FixedRadiusNNGraph(radius, n_neighbor)
        self.message = RelativePositionMessage(n_neighbor)
        self.conv = Grid_GCN_Conv(mlp_sizes, batch_size)
        self.batch_size = batch_size
        self.selfvoxels = VoxelModule(voxel_size)

    def forward(self, pos, feat, index_voxels):
        if self.group_all:
            return self.conv.group_all(pos, feat)

        centroids, centroids_index = self.rvs(pos, index_voxels)
        # centroids = self.fps(pos)
        # centroids_index = None
        g = self.frnn_graph(pos, centroids, index_voxels, centroids_index, feat)
        g.update_all(self.message, self.conv)
        mask = g.ndata['center'] == 1
        pos_dim = g.ndata['pos'].shape[-1]
        feat_dim = g.ndata['new_feat'].shape[-1]
        pos_res = g.ndata['pos'][mask].view(self.batch_size, -1, pos_dim)
        feat_res = g.ndata['new_feat'][mask].view(self.batch_size, -1, feat_dim)
        index_voxels_res = self.selfvoxels(pos_res)
        return pos_res, feat_res, index_voxels_res



class Grid_GCN(nn.Module):
    def __init__(self, output_classes, batch_size, input_dims=3, dropout_prob=0.4):
        super(Grid_GCN, self).__init__()
        self.input_dims = input_dims

        self.voxel_size = 40

        self.sa_module1 = SAModule(512, batch_size, 0.2, self.voxel_size, [input_dims, 64, 64, 128])
        self.sa_module2 = SAModule(128, batch_size, 0.4, self.voxel_size, [128 + 3, 128, 128, 256])
        self.sa_module3 = SAModule(None, batch_size, None, self.voxel_size, [256 + 3, 256, 512, 1024],
                                   group_all=True)

        self.mlp1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout_prob)

        self.mlp2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout_prob)

        self.mlp_out = nn.Linear(256, output_classes)

        self.selfvoxels = VoxelModule(self.voxel_size)

    def forward(self, x):
        #print("----------")
        #print(voxels)
        #print("----------")
        if x.shape[-1] > 3:
            pos = x[:, :, :3]
            feat = x[:, :, 3:]
        else:
            pos = x
            feat = None
        index_voxels = self.selfvoxels(pos)
        pos, feat, index_voxels = self.sa_module1(pos, feat, index_voxels)
        pos, feat, index_voxels = self.sa_module2(pos, feat, index_voxels)
        #print(self.sa_module3(pos, feat))
        h = self.sa_module3(pos, feat, index_voxels)

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

