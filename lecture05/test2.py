import torch
import numpy as np
t1 = torch.rand (2, 3 ,4)
print(t1)

n = np.ones((2, 4, 5), dtype= np.float32)

t2  = torch.from_numpy(n)

res = t1.matmul (t1, t2)

print (res.dtype, res.shape)