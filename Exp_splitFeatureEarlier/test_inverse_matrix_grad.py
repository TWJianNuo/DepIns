import torch
import torch.optim as optim

testM = torch.rand_like(torch.zeros([10,10])).cuda()
testM = torch.nn.Parameter(testM)
optimizer = optim.Adam([testM], lr=1e-3)
for i in range(1000):
    testM_inv = torch.inverse(testM)
    loss = torch.sum(torch.abs(testM_inv))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
