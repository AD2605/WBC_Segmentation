import torch

from Siamese_Model import SiameseNetwork
from dataPrep import getDataloader


path = '/home/atharva/Pictures/WBC'
model = SiameseNetwork().cuda()
Data = getDataloader(path=path)
model.train_model(model=model, train_Data=Data, epochs=30)
model.load_state_dict(torch.load('/home/atharva/wbc.pth'))
model.indentify(model=model, images='/home/atharva/Downloads/images/images/')
