
import torch

tens = torch.rand(3,5,4)
print('tens :',tens)

# t = tens[:,:,-1].reshape(3,2,1)
# print(t)
b = torch.tensor([[4,2,3,1,0],
                  [4,3,2,1,0],
                  [3,2,1,0,4]])
b = b.reshape(3,1,5)

print(b)
a = torch.index_select(tens, 0, b)
# for i in range(3): 
#     print(b[i])
#     print(tens[i][:][:])
#     a = torch.index_select(tens[i][:][:], 0, b[i])
print (a)

