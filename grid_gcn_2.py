import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import dgl
import dgl.function as fn
import math
import random
from ModelNetDataLoader import normalization
from pyinstrument import Profiler

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
        self.neighbour_voxel_list = torch.empty(voxel_size,voxel_size,voxel_size,27,3)
        for i in range(voxel_size):
            for j in range(voxel_size):
                for k in range(voxel_size):
                    center_voxel_list = torch.from_numpy(np.array([[i,j,k]]).repeat(27,axis=0))
                    #print(center_voxel_list)
                    neighbour_movement_list = torch.tensor([[-1,-1,-1],[-1,-1,0],[-1,-1,1],[-1,0,-1],[-1,0,0],[-1,0,1],[-1,1,-1],[-1,1,0],[-1,1,1],[0,-1,-1],[0,-1,0],[0,-1,1],[0,0,-1],[0,0,0],[0,0,1],[0,1,-1],[0,1,0],[0,1,1],[1,-1,-1],[1,-1,0],[1,-1,1],[1,0,-1],[1,0,0],[1,0,1],[1,1,-1],[1,1,0],[1,1,1]])
                    neighbour_list = center_voxel_list + neighbour_movement_list
                    self.neighbour_voxel_list[i][j][k] = neighbour_list
        



    # append val to self.index_voxels
    '''
    def set_voxel_value(self, index_voxels, val, x, y, z, mask):
        #key_value = x*10000 + y*100 + z
        key_value = tuple([x,y,z])
        previous_value = index_voxels.get(key_value)
        if previous_value is None:
            value_list = []
            value_list.append(val)
            new_value = value_list
            mask[x][y][z] = 1
        else:
            previous_value.append(val)
            new_value = previous_value
        index_voxels.update({key_value:new_value})

    def set_context_points(self, key, index_voxel, mask):
        x = int(key[0])
        y = int(key[1])
        z = int(key[2])
        center_voxel_list = np.array([[key[0],key[1],key[2]]]).repeat(27,axis=0)
        neighbour_movement_list = np.array([[-1,-1,-1],[-1,-1,0],[-1,-1,1],[-1,0,-1],[-1,0,0],[-1,0,1],[-1,1,-1],[-1,1,0],[-1,1,1],[0,-1,-1],[0,-1,0],[0,-1,1],[0,0,-1],[0,0,0],[0,0,1],[0,1,-1],[0,1,0],[0,1,1],[1,-1,-1],[1,-1,0],[1,-1,1],[1,0,-1],[1,0,0],[1,0,1],[1,1,-1],[1,1,0],[1,1,1]])
        neighbour_list = center_voxel_list + neighbour_movement_list
        #print("neighbour voxel list: "+str(neighbour_list))
        context_points = []
        for voxel in neighbour_list:
            if (voxel[0]<0 or voxel[0]>39 or voxel[1]<0 or voxel[1]>39 or voxel[2]<0 or voxel[2]>39):
                continue
            if (mask[voxel[0]][voxel[1]][voxel[2]]==0):
                continue
            points = index_voxel.get(tuple(voxel))
            for point in points:
                context_points.append(point)
        #print(key)
        #print(context_points)
        return context_points

        




    def forward(self, point_cloud):
        size = point_cloud.size()
        index_voxels = []
        context_points = []
        mask = []
        for i in range(size[0]): # batch
            index_voxels_tmp = dict()
            mask_tmp = np.zeros([self.voxel_size, self.voxel_size, self.voxel_size])
            context_points_tmp = dict()

            for j in range(size[1]):
                index  = point_cloud[i,j,:]*(self.voxel_size-1)
                #print(self.voxel_size)
                #print(index)
                self.set_voxel_value(index_voxels_tmp, j, int(index[0]), int(index[1]), int(index[2]),mask_tmp)
            print("--------- "+str(i))
            index_voxels.append(index_voxels_tmp)
            mask.append(mask_tmp)

            for key in index_voxels_tmp.keys():
                voxel_context_points = self.set_context_points(key, index_voxels_tmp,mask_tmp)
                context_points_tmp.update({key:voxel_context_points})
            
            context_points.append(context_points_tmp)
                




            


        return index_voxels, context_points
        '''
# self.set_voxel_value(index_voxels_tmp, current_list, pre_index, mask_tmp)

    def set_voxel_value(self, index_voxels, current_list, index, mask):
        #key_value = x*10000 + y*100 + z
        #key_value = tuple([x,y,z])
        #previous_value = index_voxels.get(key_value)
        #if previous_value is None:
        #    value_list = []
        #    value_list.append(val)
        #    new_value = value_list
        #    mask[x][y][z] = 1
        #else:
        #    previous_value.append(val)
        #    new_value = previous_value
        #index_voxels.update({key_value:new_value})
        #print(mask)
        #print(index[0])
        #print(index[1])
        #print(index[2])
        mask[int(index[0])][int(index[1])][int(index[2])] = 1
        #if (len(current_list)>10):
        #    print("----------------------------------------")
        #    print(tuple(index))
        #    print(current_list)
        index_voxels.update({tuple(index):current_list})

    '''
    def set_context_points(self, key, index_voxel, mask):
        x = int(key[0])
        y = int(key[1])
        z = int(key[2])
        center_voxel_list = np.array([[key[0],key[1],key[2]]]).repeat(27,axis=0)
        neighbour_movement_list = np.array([[-1,-1,-1],[-1,-1,0],[-1,-1,1],[-1,0,-1],[-1,0,0],[-1,0,1],[-1,1,-1],[-1,1,0],[-1,1,1],[0,-1,-1],[0,-1,0],[0,-1,1],[0,0,-1],[0,0,0],[0,0,1],[0,1,-1],[0,1,0],[0,1,1],[1,-1,-1],[1,-1,0],[1,-1,1],[1,0,-1],[1,0,0],[1,0,1],[1,1,-1],[1,1,0],[1,1,1]])
        neighbour_list = center_voxel_list + neighbour_movement_list
        #print("neighbour voxel list: "+str(neighbour_list))
        context_points = []
        for voxel in neighbour_list:
            if (voxel[0]<0 or voxel[0]>39 or voxel[1]<0 or voxel[1]>39 or voxel[2]<0 or voxel[2]>39):
                continue
            if (mask[voxel[0]][voxel[1]][voxel[2]]==0):
                continue
            points = index_voxel.get(tuple(voxel))
            for point in points:
                context_points.append(point)
        #print(key)
        #print(context_points)
        return context_points
    '''

    def forward(self, point_cloud):
        size = point_cloud.size()
        index_voxels = []
        #context_points = []
        mask = []
        for i in range(size[0]): # batch
            #sorted_points, sorted_indexes = torch.sort(point_cloud[i],-2)
            #print(sorted_points)
            #print(sorted_indexes)
            index_voxels_tmp = dict()
            mask_tmp = torch.zeros([self.voxel_size, self.voxel_size, self.voxel_size])
            #context_points_tmp = dict()

            #pre_index = (point_cloud[i,0,:]*(self.voxel_size-1)).int()

            point_to_voxels = (point_cloud[i]*(self.voxel_size-1)).int()

            new_point_to_voxels = point_to_voxels[:,0]*10000+point_to_voxels[:,1]*100+point_to_voxels[:,2]

            sorted_point_to_voxels, sorted_point_indexes = torch.sort(new_point_to_voxels)
            #print(sorted_point_to_voxels)
            current_list = []
            pre_index = (point_cloud[i,sorted_point_indexes[0],:]*(self.voxel_size-1)).int()

            for point in sorted_point_indexes:
                index  = (point_cloud[i,point,:]*(self.voxel_size-1)).int()

                if (torch.all(torch.eq(index, pre_index))):
                    #print("in if")
                    current_list.append(point)
                    #print("============================================================")
                    #print(current_list)
                else:
                    #print("in else")
                    self.set_voxel_value(index_voxels_tmp, current_list, pre_index, mask_tmp)
                    current_list = [point]
                    pre_index = index




                #print(self.voxel_size)
                #print(index)
                #self.set_voxel_value(index_voxels_tmp, j, int(index[0]), int(index[1]), int(index[2]),mask_tmp)
            print("--------- "+str(i))
            index_voxels.append(index_voxels_tmp)
            mask.append(mask_tmp)

            #for key in index_voxels_tmp.keys():
            #    voxel_context_points = self.set_context_points(key, index_voxels_tmp,mask_tmp)
            #    context_points_tmp.update({key:voxel_context_points})
            
            #context_points.append(context_points_tmp)
        self.neighbour_voxel_list = self.neighbour_voxel_list.repeat([size[0],1,1,1,1,1])
        return index_voxels, self.neighbour_voxel_list, mask


   

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
    '''
    def forward(self, pos, index_voxels):
        B = len(index_voxels) # batch_size
        device = pos.device
        vs = int(np.cbrt(len(index_voxels[0]))) # 64 -> 4, voxel_size
        centroids = torch.zeros(B, self.npoints, dtype=torch.long).to(device)
        centroids_index = []

        for batch in range(B):
            occupied = []
            # dictionary
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
    '''

    
    def forward(self, pos, index_voxels):
        B = len(index_voxels) # batch_size
        #print(B)
        device = pos.device
        vs = int(np.cbrt(len(index_voxels[0]))) # 64 -> 4, voxel_size
        centroids = torch.zeros(B, self.npoints, dtype=torch.long).to(device)
        centroids_index = []
        #print(index_voxels[0])
        #print('-------------------------------------------------------------')

        for batch in range(B):
            #print(batch)
            voxels_per_batch = index_voxels[batch]

            indexes = []

            dict_keys = voxels_per_batch.keys()
            len_key = len(dict_keys)
            if self.npoints <= len_key:
                #print(list(voxels_per_batch.items()))
                #print("npoints: "+str(self.npoints)+" "+"length: "+str(len_key))
                selected_keys = random.sample(dict_keys,self.npoints)
                #print(selected_keys)
                i = 0
                for per_key in selected_keys:
                    #int_index = int(per_key)
                    #indexes.append([batch, int_index//10000, int_index//100, int_index%100])
                    indexes.append([batch, per_key[0],per_key[1],per_key[2]])
                    val = voxels_per_batch.get(per_key)
                    index = int(random.sample(val, 1)[0])      
                    centroids[batch, i] = index
                    i = i + 1   
            else:
                #self.npoints > len(voxels_per_batch):
                #print(list(voxels_per_batch.items()))
                selected_keys = dict_keys
                i = 0
                added = []
                for per_key in selected_keys:
                    #int_index = int(per_key)
                    #indexes.append([batch, int_index//10000, int_index//100, int_index%100])
                    indexes.append([batch, per_key[0],per_key[1],per_key[2]])
                    val = voxels_per_batch.get(per_key)
                    #print(val)
                    index = int(random.sample(val, 1)[0])      
                    centroids[batch, i] = index
                    added.append(index)
                    i = i + 1     


                #added = indexes
                #for i in range(len(voxels_per_batch)):
                #    val = self.get_voxel_value(index_voxels, vs, batch, voxels_per_batch[i][0], voxels_per_batch[i][1], voxels_per_batch[i][2])
                #    index = int(random.sample(val, 1)[0])
                #    centroids[batch,i] = index
                #    added.append(index)
                #    indexes.append([batch, voxels_per_batch[i][0], voxels_per_batch[i][1], voxels_per_batch[i][2], val.index(index)])
                add_num = 0
                while add_num < (self.npoints-len_key):
                    index = int(random.sample(range(pos.shape[1]), 1)[0])
                    #print(index)
                    if index not in added:
                        centroids[batch, len_key+add_num] = index
                        indexes.append(index)
                        add_num += 1
                        added.append(index)

            centroids_index.append(indexes)
            i = 0
            #for xyz in indexes:
            #    centroids[batch, i] = xyz
            #    i = i + 1
        #print("--------------------------------------------------------------")
        #print(centroids.size())
        #print(centroids)
        return centroids, centroids_index # centroid_index is not used

    # get value from self.index_voxels
    def get_voxel_value(self, index_voxels, voxel_size, batch, key):
        #index1 = voxel_size * voxel_size
        #index2 = voxel_size
        #key_value = x*10000 + y*100 + z
        return index_voxels[batch].get(key)


class FixedRadiusNearNeighbors(nn.Module):
    '''
    Find the neighbors with-in a fixed radius
    '''
    def __init__(self, radius, n_neighbor):
        super(FixedRadiusNearNeighbors, self).__init__()
        self.radius = radius
        self.n_neighbor = n_neighbor

    '''
    def forward(self, pos, centroids, centroids_index, index_voxels, voxel_size):
        
        #Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
        #TODO: Need to update the select neighbor operation
        
        device = pos.device
        B, N, _ = pos.shape
        center_pos = index_points(pos, centroids)
        # print(center_pos.shape)
        _, S, _ = center_pos.shape
        print(B, N, S)
        #print(torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).shape)
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        print("group_idx shape: " + str(group_idx.shape))
        print(group_idx[0])
        sqrdists = square_distance(center_pos, pos)
        print("sqrdists shape: " + str(sqrdists.shape))
        print(sqrdists[0])
        group_idx[sqrdists > self.radius ** 2] = N
        print("group_idx shape: " + str(group_idx.shape))
        print(group_idx[0])
        group_idx = group_idx.sort(dim=-1)[0][:, :, :self.n_neighbor]
        print("group_idx shape: " + str(group_idx.shape))
        print(group_idx[0])
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, self.n_neighbor])
        print("group first shape: " + str(group_first.shape))
        print(group_first[0])
        mask = group_idx == N
        print("mask shape: " + str(mask.shape))
        print(mask[0])
        group_idx[mask] = group_first[mask]
        print('group_idx ', group_idx.shape)
        print(group_idx[0])
        # print(group_idx.shape)
        # print(group_idx[0,:,:])
        # print(group_idx[0,0,:])
        print(group_idx.shape)
        return group_idx
    
    '''

    '''

    def find_voxel_neighbour(self, voxel):
        voxel_list = []
        for i in range (3):
            for j in range (3):
                for k in range (3):
                    if (i==2 and j == 1 and k == 1):
                        continue
                
                    else:
                        new_voxel = [voxel[0]+i-1, voxel[1]+j-1, voxel[2]+k-1]
                        voxel_list.append(new_voxel)
        #print(voxel_list)
        return voxel_list

    def get_context_points(self, batch, voxel_neighbour_list, index_voxels):
        profiler = Profiler()
        profiler.start()
        voxels = index_voxels[batch]
        neighbour_voxel_list = []
        for item in voxel_neighbour_list:
            key = int(item[0]*10000+item[1]*100+item[2])
            #print(key)
            neighbour_voxel = voxels.get(key)
            #print(neighbour_voxel)
            if neighbour_voxel is None:
                continue
            for single_point in neighbour_voxel:
                neighbour_voxel_list.append(single_point)
        profiler.stop()

        print(profiler.output_text(unicode=True, color=True,show_all = True))
        return neighbour_voxel_list
                    

    def forward(self, pos, centroids, centroids_index, index_voxels, voxel_size):
        profiler = Profiler()
        profiler.start()
        
        #Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
        #TODO: Need to update the select neighbor operation
        
        device = pos.device
        B, N, _ = pos.shape
        center_pos = index_points(pos, centroids)
        # print(center_pos.shape)
        _, S, _ = center_pos.shape
        #print(B, N, S)
        #print(torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).shape)
        #group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        group_idx = torch.ones(B, S, self.n_neighbor)
        
        #print("group_idx shape: " + str(group_idx.shape))
        #print(group_idx[0])
        i = 0

        neighbour_movement_list = np.array([[-1,-1,-1],[-1,-1,0],[-1,-1,1],[-1,0,-1],[-1,0,0],[-1,0,1],[-1,1,-1],[-1,1,0],[-1,1,1],[0,-1,-1],[0,-1,0],[0,-1,1],[0,0,-1],[0,0,0],[0,0,1],[0,1,-1],[0,1,0],[0,1,1],[1,-1,-1],[1,-1,0],[1,-1,1],[1,0,-1],[1,0,0],[1,0,1],[1,1,-1],[1,1,0],[1,1,1]])
        print(neighbour_movement_list)
        

        for batch in centroids:
            print(i)
            j = 0
            for center in batch:
                point = pos[i][j]
                voxel = point*(voxel_size-1)
                voxel[0] = int(voxel[0])
                voxel[1] = int(voxel[1])
                voxel[2] = int(voxel[2])
                voxel_neighbour_list = self.find_voxel_neighbour(voxel)
                neighbors = self.get_context_points(i, voxel_neighbour_list, index_voxels)
                if (len(neighbors)>self.n_neighbor):
                    selected_neighbours = random.sample(neighbors,self.n_neighbor)
                    neighbors = selected_neighbours
                k = 0
                for item in neighbors:
                    #print(item)
                    group_idx[i][j][k] = item
                    k = k+1
                while (k<self.n_neighbor):
                    group_idx[i][j][k] = center
                    k = k+1
                j = j+1
            print(group_idx.shape)
            i = i+1
        group_idx = group_idx.float().to(device)
        print(group_idx.shape)

        profiler.stop()

        print(profiler.output_text(unicode=True, color=True,show_all = True))
        return group_idx
        '''

    
    #def find_voxel_neighbour(self, voxel):
    #    voxel_list = []
    #    for i in range (3):
    #        for j in range (3):
    #            for k in range (3):
    #                if (i==2 and j == 1 and k == 1):
    #                    continue
                
    #                else:
    #                    new_voxel = [voxel[0]+i-1, voxel[1]+j-1, voxel[2]+k-1]
    #                    voxel_list.append(new_voxel)
        #print(voxel_list)
    #    return voxel_list
    '''
    def get_context_points(self, batch, voxel_neighbour_list, index_voxels):
        profiler = Profiler()
        profiler.start()
        voxels = index_voxels[batch]
        neighbour_voxel_list = []
        for item in voxel_neighbour_list:
            key = int(item[0]*10000+item[1]*100+item[2])
            #print(key)
            neighbour_voxel = voxels.get(key)
            #print(neighbour_voxel)
            if neighbour_voxel is None:
                continue
            for single_point in neighbour_voxel:
                neighbour_voxel_list.append(single_point)
        profiler.stop()

        print(profiler.output_text(unicode=True, color=True,show_all = True))
        return neighbour_voxel_list
    '''
                    
    '''
    def forward(self, pos, centroids, centroids_index, index_voxels, voxel_size, context_points):
        profiler = Profiler()
        profiler.start()
        device = pos.device
        B, N, _ = pos.shape
        center_pos = index_points(pos, centroids)
        _, S, _ = center_pos.shape
        group_idx = torch.ones(B, S, self.n_neighbor).to(device)
        i = 0

        #neighbour_movement_list = np.array([[-1,-1,-1],[-1,-1,0],[-1,-1,1],[-1,0,-1],[-1,0,0],[-1,0,1],[-1,1,-1],[-1,1,0],[-1,1,1],[0,-1,-1],[0,-1,0],[0,-1,1],[0,0,-1],[0,0,0],[0,0,1],[0,1,-1],[0,1,0],[0,1,1],[1,-1,-1],[1,-1,0],[1,-1,1],[1,0,-1],[1,0,0],[1,0,1],[1,1,-1],[1,1,0],[1,1,1]])
        #print(neighbour_movement_list)
        

        for batch in centroids:
            print(i)
            voxel_set = set()
            voxels = index_voxels[i]
            j = 0
            
            #center_voxel_id = get_voxel_id(center)
            #sorted_v_id, sorted_c_id = sort(cat(center_voxel_id, center))
            #for center in sorted_c_id:
            #    if current_v_id != last_v_id:
            #        preprocess
            #    sampling
            

            for center in batch:
                point = pos[i][j]
                point_voxel = (point*(voxel_size-1)).int()
                point_voxel = tuple(point_voxel)
                neighbors = context_points[i].get(point_voxel)
                #print("?????????????")
                #neighbors = []
                if ((neighbors) and (len(neighbors)>self.n_neighbor)):
                    selected_neighbours = random.sample(neighbors,self.n_neighbor)
                    neighbors = selected_neighbours
                k = 0
                if neighbors:
                    for item in neighbors:
                        #print(item)
                        group_idx[i][j][k] = item
                        k = k+1
                while (k<self.n_neighbor):
                    group_idx[i][j][k] = center
                    k = k+1
                j = j+1
            print(group_idx.shape)
            i = i+1
            
        group_idx = group_idx.float().to(device)
        print(group_idx.shape)

        profiler.stop()

        print(profiler.output_text(unicode=True, color=True,show_all = True))
        return group_idx
        '''
    def forward(self, pos, centroids, centroids_index, index_voxels, voxel_size, neighbour_voxel_list, mask):
        profiler = Profiler()
        profiler.start()
        device = pos.device
        B, N, _ = pos.shape
        center_pos = index_points(pos, centroids)
        _, S, _ = center_pos.shape
        group_idx = torch.ones(B, S, self.n_neighbor).to(device)
        i = 0

        #neighbour_movement_list = np.array([[-1,-1,-1],[-1,-1,0],[-1,-1,1],[-1,0,-1],[-1,0,0],[-1,0,1],[-1,1,-1],[-1,1,0],[-1,1,1],[0,-1,-1],[0,-1,0],[0,-1,1],[0,0,-1],[0,0,0],[0,0,1],[0,1,-1],[0,1,0],[0,1,1],[1,-1,-1],[1,-1,0],[1,-1,1],[1,0,-1],[1,0,0],[1,0,1],[1,1,-1],[1,1,0],[1,1,1]])
        #print(neighbour_movement_list)
        

        for batch in center_pos:
            print(i)
            voxel_set = set()
            voxels = index_voxels[i]
            j = 0
            
            #center_voxel_id = get_voxel_id(center)
            #sorted_v_id, sorted_c_id = sort(cat(center_voxel_id, center))
            #for center in sorted_c_id:
            #    if current_v_id != last_v_id:
            #        preprocess
            #    sampling
            
            center_voxel_id = (batch*(voxel_size-1)).int()
            #print(center_voxel_id)
            #print(center_voxel_id.size())
            #print(center_voxel_id)

            new_center_voxel_id = center_voxel_id[:,0]*10000+center_voxel_id[:,1]*100+center_voxel_id[:,2]

            sorted_centers, center_indexes = torch.sort(new_center_voxel_id)
            #for item in sorted_centers:
            #    print(item)
            #print(sorted_centers)
            #print(center_indexes)
            
            current_voxel = None
            current_context_points = []
            j = 0
            for index in center_indexes:
                #print(index[0])
                #x = center_voxel_id[index][0].item()
                #y = center_voxel_id[index][1].item()
                #z = center_voxel_id[index][2].item()
                #self_voxel = new_center_voxel_id[x][y][z]
                self_voxel = center_voxel_id[index]
                #print(self_voxel)
                if((not current_voxel==None) and torch.all(torch.eq(self_voxel, current_voxel))):
                    self_context_points = current_context_points
                else:
                    #self_neighbour_voxels = neighbour_voxel_list[i].get(tuple(self_voxel))
                    x_1 = self_voxel[0].item()
                    y_1 = self_voxel[1].item()
                    z_1 = self_voxel[2].item()
                    self_neighbour_voxels = neighbour_voxel_list[i][x_1][y_1][z_1]
                    for voxel in self_neighbour_voxels:
                        #voxel = voxel.int()
                        x = int(voxel[0].item())
                        y = int(voxel[1].item())
                        z = int(voxel[2].item())
                        if (x<0 or x>39 or y<0 or y>39 or z<0 or z>39):
                            continue
                        if (mask[i][x][y][z].item()==0):
                            continue
                        #print(mask[i][x][y][z])
                        points = voxels.get(tuple(voxel))
                        current_context_points = []
                        #print(str(x_1)+' '+str(y_2)+' '+str(z_1))
                        if (points is None):
                            continue
                        for point in points:
                            #context_points.append(point)
                            current_context_points.append(point)
                    self_context_points = current_context_points
                k = 0
                if self_context_points:
                    for item in self_context_points:
                        #print(item)
                        group_idx[i][index][k] = item
                        k = k+1
                while (k<self.n_neighbor):
                    #print(index)
                    #print(centroids[i][index])
                    group_idx[i][index][k] = centroids[i][index]
                    k = k+1
                j = j+1
                
                        

            i = i+1




            '''
            for center in batch:
                point = pos[i][j]
                point_voxel = (point*(voxel_size-1)).int()
                point_voxel = tuple(point_voxel)
                neighbors = context_points[i].get(point_voxel)
                #print("?????????????")
                #neighbors = []
                if ((neighbors) and (len(neighbors)>self.n_neighbor)):
                    selected_neighbours = random.sample(neighbors,self.n_neighbor)
                    neighbors = selected_neighbours
                k = 0
                if neighbors:
                    for item in neighbors:
                        #print(item)
                        group_idx[i][j][k] = item
                        k = k+1
                while (k<self.n_neighbor):
                    group_idx[i][j][k] = center
                    k = k+1
                j = j+1
            print(group_idx.shape)
            i = i+1
        '''
            
        group_idx = group_idx.float().to(device)
        print(group_idx.shape)

        profiler.stop()

        print(profiler.output_text(unicode=True, color=True,show_all = True))
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

    def forward(self, pos, centroids, index_voxels, centroids_index, voxel_size, context_points, mask, feat=None):
        dev = pos.device
        
        group_idx = self.frnn(pos, centroids, centroids_index, index_voxels, voxel_size, context_points, mask)
        B, N, _ = pos.shape
        glist = []
        for i in range(B):
            center = torch.zeros((N)).to(dev)
            center[centroids[i]] = 1
            src = group_idx[i].contiguous().view(-1)
            src = src.to(dev)
            dst = centroids[i].view(-1, 1).repeat(1, self.n_neighbor).view(-1).float()
            dst = dst.to(dev)

            unified = torch.cat([src, dst])
            uniq, idx, inv_idx = np.unique(unified.cpu().numpy(), return_index=True, return_inverse=True)
            src_idx = inv_idx[:src.shape[0]]
            dst_idx = inv_idx[src.shape[0]:]

            g = dgl.DGLGraph((src_idx, dst_idx), readonly=True)
            #import pdb; pdb.set_trace()
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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pos = pos.to(device)
        #import pdb; pdb.set_trace()
        if 'feat' in edges.src:
            #print("=========== in if ===========")
            #edges.src['feat'].to(device)
            res = torch.cat([pos, edges.src['feat']], 1)
        else:
            #print("in else")
            res = pos
        #print(res.shape, pos.shape, edges.src['pos'].shape, edges.src['pos'].shape, edges.src['feat'].shape)
        # print(edges.src.keys())
        # print(edges.src['center'].shape)
        geo_feat = torch.cat([edges.src['pos'], edges.dst['pos']], 1)
        #a = edges.dst['pos'].view(3, -1, 512, 64)
        #print(a[0,:,0,:].view(64,3))
        #print(a[0,:,0,:].shape)

        return {'agg_feat': res, 'geo_feat': geo_feat}
        #return {'agg_feat': res}

    #def get_edge_num(self, )


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
        #print('shape', shape)
        #print('here shape:', shape, self.batch_size)
        #print('sizes: ', self.sizes, self.batch_size)
        h = nodes.mailbox['agg_feat'].view(self.batch_size, -1, shape[1], shape[2]).permute(0, 3, 1, 2)
        #print('here shape', h.shape)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h = h.to(device)
        for conv, bn in zip(self.conv, self.bn):
            h = conv(h)
            h = bn(h)
            h = F.relu(h)
            # print("H shape: ")=
        h = torch.max(h, 3)[0]
        feat_dim = h.shape[1]
        h = h.permute(0, 2, 1).reshape(-1, feat_dim)

        
        # geo
        shape = nodes.mailbox['geo_feat'].shape
        h_geo = nodes.mailbox['geo_feat'].view(self.batch_size, -1, shape[1], shape[2]).permute(0, 3, 1, 2)
        h_geo = h_geo.to(device)
        for conv, bn in zip(self.conv_geo, self.bn_geo):
            h_geo = conv(h_geo)
            h_geo = bn(h_geo)
            h_geo = F.relu(h_geo)
        h_geo = torch.max(h_geo, 3)[0]
        feat_dim = h_geo.shape[1]
        h_geo = h_geo.permute(0, 2, 1).reshape(-1, feat_dim)
        
        h_all = torch.cat([h, h_geo], 0)
        return {'new_feat': h_all}
        
        #return h
    
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

    def forward(self, pos, feat, index_voxels,context_points, voxel_mask):
        if self.group_all:
            return self.conv.group_all(pos, feat)

        centroids, centroids_index = self.rvs(pos, index_voxels)
        # centroids = self.fps(pos)
        # centroids_index = None
        #self_index_voxel = index_voxels
        g = self.frnn_graph(pos, centroids, index_voxels, centroids_index, self.voxel_size,context_points, voxel_mask, feat)
        g.update_all(self.message, self.conv)
        mask = g.ndata['center'] == 1
        pos_dim = g.ndata['pos'].shape[-1]
        feat_dim = g.ndata['new_feat'].shape[-1]
        pos_res = g.ndata['pos'][mask].view(self.batch_size, -1, pos_dim)
        feat_res = g.ndata['new_feat'][mask].view(self.batch_size, -1, feat_dim)
        #index_voxels_res = self.selfvoxels(pos_res)
        #return pos_res, feat_res, index_voxels_res
        return pos_res, feat_res



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
        profiler = Profiler()
        profiler.start()
        #import pdb; pdb.set_trace()
        x = normalization(x)
        if x.shape[-1] > 3:
            pos = x[:, :, :3]
            feat = x[:, :, 3:]
        else:
            pos = x
            feat = None
        index_voxels, context_points, mask = self.selfvoxels(pos)
        print("============ "+str(len(index_voxels)))
        #pos, feat, index_voxels = self.sa_module1(pos, feat, index_voxels, context_points)
        pos, feat = self.sa_module1(pos, feat, index_voxels, context_points, mask)
        index_voxels, context_points, mask = self.selfvoxels(pos)
        print("============ "+str(len(index_voxels)))
        #pos, feat, index_voxels = self.sa_module2(pos, feat, index_voxels, context_points)
        pos, feat  = self.sa_module2(pos, feat, index_voxels, context_points, mask)
        #print(self.sa_module3(pos, feat))
        index_voxels, context_points, mask = self.selfvoxels(pos)
        h = self.sa_module3(pos, feat, index_voxels, context_points, mask)

        h = self.mlp1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.drop1(h)
        h = self.mlp2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.drop2(h)

        out = self.mlp_out(h)
        profiler.stop()

        print(profiler.output_text(unicode=True, color=True))
        return out

