import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)
        
    def forward(self, x):
        x = self.maxpool1(x)
        return x
    
tudui = Tudui()

step = 0
writer = SummaryWriter("nn_maxpoollogs")

for data in dataloader:
    imgs, targets = data
    writer.add_images("pre", imgs, step)
    output = tudui(imgs)
    writer.add_images("nn_maxpool", output, step)
    step += 1
    
writer.close()