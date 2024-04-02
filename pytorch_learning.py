'''import  torch as tr
import numpy as np

tensor=tr.rand(3,4)
# We move our tensor to the GPU if available
if tr.cuda.is_available():
  tensor = tensor.to('cuda')
  print("cuda is available")
print("shape of tensor:"+str(tensor.shape))
print("datatype:"+str(tensor.dtype))
print("Device tensor was stored on "+str(tensor.device))
print(tensor)
print(tensor[:,0])
print((tensor[1]))'''

import torch
from pathlib import Path

print("tensor:")
tensor_rand = torch.rand(3, 4)
print(tensor_rand)
if torch.cuda.is_available():
    tensor_rand = tensor_rand.to('cuda')
    print(str(tensor_rand.device))
print("slice of tensors:")
print("first row:" + str(tensor_rand[0]))  # print(tensor_rand[0])
print("first column:" + str(tensor_rand[:, 0]))
print("second column:" + str(tensor_rand[:, 1]))
print("joining tensors:")
# 创建两个示例Tensor
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor1)
tensor2 = torch.tensor([[7, 8, 9], [10, 11, 12]])
print(tensor2)
# 将两个Tensor在维度0上连接
result = torch.cat((tensor1, tensor2), dim=0)

print(result)
print("computation of tensors:")
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
tensor = torch.rand(3, 4)
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(y1, y2, y3)
# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1, z2, z3)
tensor_ones = torch.ones(3, 4)
agg = tensor_ones.sum()
agg = agg.item()
print(type(agg))
print("-------------------")

# 以下为新增改动
try:
    import matplotlib.pyplot as plt
    print("matplotlib is available")
except ModuleNotFoundError:
    print("matplotlib is not available")
