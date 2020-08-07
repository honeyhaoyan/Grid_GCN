import torch

x = torch.Tensor([[12,13,14],[34,35,12],[11,56,90],[1,9,90]])

y = x[:,0]*10000+x[:,1]*100+x[:,2]

print(x)

print("-----------------------")

new_y, indices = torch.sort(y)

print(y)

print(new_y)

print(indices)



#sorted, indices = torch.sort(x[:,0],-1)
#print("-----------------------")
#print(sorted)
#print(indices)
#sorted, indices = torch.sort(x, -2)
#print(sorted)
#print(indices)