import cv2
import numpy
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from Siamese_Model import SiameseNetwork
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy

#A rough script for data exploration and processing
image = Image.open('/home/atharva/Downloads/images/images/0011.jpg')
image = ImageOps.autocontrast(image)
image.show()
trans = transforms.Compose([
    transforms.ToTensor()
])


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mean = numpy.mean(image)
print(mean)
print(numpy.amax(image))
_, thresh = cv2.threshold(image, thresh=mean, maxval=255, type=cv2.THRESH_BINARY)
strel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3, 3))
print(image.shape)
eroded = (cv2.erode(thresh, kernel=strel, iterations=1))
eroded_inverse = cv2.bitwise_not(eroded)
cv2.imshow('thresh', thresh)
cv2.imshow('closed', eroded)
cv2.imshow('closed_inverse', eroded_inverse)
cv2.imshow('original', image)
cv2.waitKey()

a = torch.randn((1, 512, 512))
a = torch.nn.functional.relu(a)
pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
a = pool(a)
print(a.shape)
a = a.permute(2, 1, 0).squeeze().numpy()
plt.imshow(a)
plt.show()
