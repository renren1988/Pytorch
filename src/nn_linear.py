import torchvision
import torch
from torch import nn

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, drop_last=True)

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, x):
        x = self.linear1(x)
        return x
    
myModule = MyModule()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.reshape(imgs, (1, 1, 1, -1))
    print(output.shape)
    output = myModule(output)
    print(output.shape)