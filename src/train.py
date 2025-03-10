import torchvision
import torch
from torch import nn
from model import *

train_data = torchvision.datasets.CIFAR10("./dataset", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64)
test_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64)

# train_data_size = len(train_data)
# test_data_size = len(test_data)

# print(f"训练集长度:{train_data_size}")
# print(f"测试集长度:{test_data_size}")

myModule = MyModule()

loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-2
optimizer = torch.optim.SGD(myModule.parameters(), lr = learning_rate)

train_step = 0
test_step = 0

epoch = 10

for i in range(epoch):
    print(f"-----第{i + 1}次训练开始------")
    
    for data in train_dataloader:
        imgs, targets = data
        outputs = myModule(imgs)
        loss = loss_fn(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_step += 1
        if train_step % 100 == 0:
            print(f"训练次数:{train_step},loss:{loss}")
        
    test_step += 1
    test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = myModule(imgs)
            test_loss += loss_fn(outputs, targets)
        
        print(f"测试集loss:{test_loss}")
    torch.save(myModule, f"myModule_{i}.pth")