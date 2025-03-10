import torch
from torch import nn
import torchvision

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x
    
myModule = MyModule()

loss = nn.CrossEntropyLoss()
for data in dataloader:
    imgs, targets = data
    outputs = myModule(imgs)
    result_loss = loss(outputs, targets)
    #print(resultloss)
    result_loss.backward()
    print('ok')
