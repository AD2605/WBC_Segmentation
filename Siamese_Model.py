import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os
import cv2

class Contrastive_Loss(nn.Module):
    def __init__(self, margin=1.0):
        super(Contrastive_Loss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        distance = nn.functional.pairwise_distance(out1, out2)
        return torch.mean((1 - label) * torch.pow(distance, 2) +
                          (label * (torch.clamp(self.margin - distance, min=0))))


class perceptualFeatures(nn.Module):
    def __init__(self):
        super(perceptualFeatures, self).__init__()
        print('Using VGG16 for extracting texture and content')
        layers = []
        layers.append(torchvision.models.vgg16(pretrained=True).cuda().features[:4].eval())
        layers.append(torchvision.models.vgg16(pretrained=True).cuda().features[4:9].eval())
        layers.append(torchvision.models.vgg16(pretrained=True).cuda().features[9:16].eval())
        layers.append(torchvision.models.vgg16(pretrained=True).cuda().features[16:23].eval())

        for layer in layers:
            for parameters in layer:
                parameters.required_grad = False

        self.layers = nn.ModuleList(layers)

    def gramMatrix(self, image):
        (b, c, h, w) = image.size()
        f = image.view(b, c, w * h)
        G = f.bmm(f.transpose(1, 2)) / c * w * h
        return G

    def forward(self, image):
        features = []
        #image = image.cuda()
        for layer in self.layers:
            image = layer(image)
            features.append(image)
        return features


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.content = nn.Linear(in_features=1875, out_features=2048)
        self.texture = nn.Linear(in_features=16384, out_features=4096)
        self.texture_2 = nn.Linear(in_features=4096, out_features=2048)
        self.linear_1 = nn.Linear(in_features=4096, out_features=4096)
        self.linear_2 = nn.Linear(in_features=4096, out_features=2048)
        self.final = nn.Linear(in_features=2048, out_features=512)
        self.conv_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.features = perceptualFeatures()
        self.min_loss = 100
        self.loss_plot = []

    def forward(self, gramMatrix, content):
        gramMatrix = nn.functional.relu(gramMatrix)
        gramMatrix = self.pool(gramMatrix)
        gramMatrix = nn.functional.relu(gramMatrix)
        gramMatrix = self.pool(gramMatrix)
        gramMatrix = gramMatrix.view(gramMatrix.size(0), -1)
        gramMatrix = self.texture(gramMatrix)
        gramMatrix = self.texture_2(gramMatrix)

        content = self.conv_1(content)
        content = self.conv_2(content)
        content = content.view(content.size(0), -1)
        content = self.content(content)
        feature = torch.cat((gramMatrix, content), dim=1)
        feature = self.linear_1(feature)
        feature = self.linear_2(feature)
        feature = self.final(feature)
        return feature

    def train_model(self, train_Data, epochs, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        criterion = Contrastive_Loss()
        model.train()
        model.cuda()

        for epoch in range(0, epochs+1):
            train_loss = 0
            for _, Data in enumerate(train_Data):
                optimizer.zero_grad()

                image1 = Data['image1'].cuda()
                image2 = Data['image2'].cuda()
                label = Data['label'].cuda()

                image1_features = self.features(image1)
                image2_features = self.features(image2)

                image1_content = image1_features[1]
                image2_content = image2_features[1]

                image1_gram = self.features.gramMatrix(image1_features[3])
                image2_gram = self.features.gramMatrix(image2_features[3])

                image1_out = model(image1_gram, image1_content)
                image2_out = model(image2_gram, image2_content)
                loss = criterion(image1_out, image2_out, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.loss_plot.append(train_loss)
            if train_loss < self.min_loss:
                print('----------')
                print('saving')
                print(train_loss)
                print(epoch)
                torch.save(model.state_dict(), '/home/atharva/wbc.pth')
                self.min_loss = train_loss

        plt.plot(self.loss_plot)
        plt.show()

    def indentify(self, images, model):
        # images is a path to blood sample images...
        # model is the pre-trained image to be inferred on
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        os.chdir(images)
        images = os.listdir()
        transform = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor()
        ])
        transform_image = transforms.Compose([
            transforms.ToTensor()
        ])

        WBC_sample = Image.open('/home/atharva/Pictures/WBC/WBC_1.png')
        WBC_sample = transform(WBC_sample).unsqueeze(dim=0).cuda()
        WBC_sample = WBC_sample[:, :3, :, :]
        WBC_features = self.features(WBC_sample)
        WBC_content = WBC_features[1]
        WBC_gram = self.features.gramMatrix(WBC_features[3])
        WBC_vector = model(WBC_gram, WBC_content)
        for image in images:
            name = image
            image_cv = cv2.imread(image)
            print('--------------------------------------------')
            print('file name - ', image)
            position = []
            count = 0
            draw_counter = 0
            image = Image.open(image)
            image = transform_image(image).unsqueeze(dim=0).cuda()
            image = image[:, :3, :, :]
            height = image.shape[2]
            width = image.shape[3]

            for i in range(25, height - 25, 25):
                for j in range(25, width - 25, 25):
                    subimage = image[:, :, i - 25: i + 25, j - 25: j + 25]
                    subimage_features = self.features(subimage)
                    subimage_gram = self.features.gramMatrix(subimage_features[3])
                    subimage_content = subimage_features[1]
                    out = model(subimage_gram, subimage_content)
                    dissimilarity = nn.functional.l1_loss(out, WBC_vector)
                    if dissimilarity < 0.008:
                        count += 1
                        # Draw rectangles around WBCs
                        image_cv = cv2.rectangle(image_cv, (j - 25, i - 25), (j + 25, i + 25),
                                                 (50, 50, 255), 3)

                        cv2.imshow('image', image_cv)
                        draw_counter += 1
                        cv2.putText(image_cv, str(dissimilarity), (i + 25, j), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                                    (255, 255, 255), 1)
                        position.append([i, j, dissimilarity.item()])
            print(count)
            print(draw_counter)
            cv2.imwrite('/home/atharva/inferred/' + name + '.png', image_cv)
