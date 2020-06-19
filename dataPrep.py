import os
import random

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

def getDataloader(path):
    os.chdir(path=path)
    files = os.listdir()
    WBC = []
    notWBC = []
    Data = []
    for file in files:
        if file.startswith('WBC'):
            WBC.append(file)
        if file.startswith('not'):
            notWBC.append(file)

    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor()
    ])
    for i in range(100):
        # Equal number of samples for both classes 0 and 1
        if i < 50:
            image1 = random.choice(WBC)
            while True:
                image2 = random.choice(WBC)
                if image1 != image2:
                    break
            image1 = Image.open(image1)
            image2 = Image.open(image2)

            image1 = transform(image1)
            image2 = transform(image2)
            image1 = image1[:3, :, :]
            image2 = image2[:3, :, :]
            label = torch.tensor(data=0, dtype=torch.long)

            data = {'image1': image1, 'image2': image2, 'label': label}
            Data.append(data)

        else:
            image1 = random.choice(WBC)
            image2 = random.choice(notWBC)

            image1 = Image.open(image1)
            image2 = Image.open(image2)

            image1 = transform(image1)
            image2 = transform(image2)
            image1 = image1[:3, :, :]
            image2 = image2[:3, :, :]
            label = torch.tensor(data=1, dtype=torch.long)

            data = {'image1': image1, 'image2': image2, 'label': label}
            Data.append(data)

    return DataLoader(Data, batch_size=1)
