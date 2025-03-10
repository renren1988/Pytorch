import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
tudui = Tudui()
#print(tudui)

step = 0
writer = SummaryWriter("nn_conv2dlogs")

for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    writer.add_images("nn_conv2d", output, step)
    step += 1
    
writer.close()