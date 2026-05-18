import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# cat: 沿现有维度拼接
c = torch.cat([a, b])          # tensor([1, 2, 3, 4, 5, 6])
c = torch.cat([a, b], dim=-1)
print(c) 
