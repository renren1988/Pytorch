import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(x)
        return x
    
mymodule = MyModule()

writer = SummaryWriter("./nn_relulogs")

step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = mymodule(imgs)
    writer.add_images("output", output, global_step=step)
    step += 1

writer.close()