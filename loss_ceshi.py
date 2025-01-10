import torch

def loss_GP(fake,real):
    _,_,D,H,W = fake.shape
    L_D = (1 / D )* torch.sum(torch.trace(torch.mm(real[:, :, 1:, :, :], fake[:, :, 1:, :, :].t())) / H + torch.trace(torch.mm(real[:, :, 1:, :, :].t(), fake[:, :, 1:, :, :])) / W)
    L_H = (1 / H) * torch.sum(torch.trace(torch.mm(real[:, :, :, 1:, :], fake[:, :, :, 1:, :].t())) / D + torch.trace(torch.mm(real[:, :, :, 1:, :].t(), fake[:, :, :, 1:, :])) / W)
    L_W = (1 / W) * torch.sum(torch.trace(torch.mm(real[:, :, :, :, 1:], fake[:, :, :, :, 1:].t())) / H + torch.trace(torch.mm(real[:, :, :, :, 1:].t(), fake[:, :, :, :, 1:])) / D)
    L = (L_H + L_W + L_D) / 3
    return L

real = torch.randn(1,1,128,128,128)
fake = torch.randn(1,1,128,128,128)

real_slice = real[:, :, 1:, :, :]
# fake_t = real_slice.transpose(fake)

test1 = torch.randn(1,1,1,2,3)
test2 = torch.randn(1,1,1,2,3)
test11 = test1.transpose(2,3)
out = torch.tensordot(test1,test11,dims=2)
print('11')