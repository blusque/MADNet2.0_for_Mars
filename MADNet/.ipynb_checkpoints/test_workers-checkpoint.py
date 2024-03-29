from time import time
import multiprocessing as mp
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import DEMDataset
import os
 
 
transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])
 
# train_set = DEMDataset("../../../data/training_dataset.hdf5")

train_set = torchvision.datasets.MNIST(
    root='dataset/',
    train=True,  #如果为True，从 training.pt 创建数据，否则从 test.pt 创建数据。
    download=True, #如果为true，则从 Internet 下载数据集并将其放在根目录中。 如果已下载数据集，则不会再次下载。
    transform=transform
)
 
print(f"num of CPU: {mp.cpu_count()}")
for num_workers in range(2, mp.cpu_count() + 1, 2):  
    train_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=64, pin_memory=True,
                                      shuffle=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
