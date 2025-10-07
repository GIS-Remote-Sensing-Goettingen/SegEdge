import torch
print(torch.__file__)
print(torch.version.cuda)
print(torch.cuda.is_available())

a = torch.randn((1000, 1000), device='cuda')
b = torch.randn((1000, 1000), device='cuda')
c = a @ b
print(c)
print(c.device)
