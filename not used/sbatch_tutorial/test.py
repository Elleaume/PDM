import torch
print('Creating tensor...')
t = torch.ones(10)
print(t)

d = torch.rand(10).cuda()
print(d.is_leaf)