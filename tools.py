import torch
import torch.nn as nn
import cv2
import numpy as np


class lock_classification_model(nn.Module):
    def __init__(self):
        super(lock_classification_model, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, 5, 1, 2),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 128, 5, 1, 2),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU())
        self.out = nn.Sequential(nn.Dropout(0.2),
                                 nn.Linear(32*32*32, 64),
                                 nn.Linear(64, 3))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bottleneck(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

class lock_detector(object):
    def __init__(self, model_path, img_width=128, img_height=128):
        model = lock_classification_model()
        model.load_state_dict(torch.load(model_path))
        if torch.cuda.is_available():
            model = model.cuda()
        model = model.eval()
        self.model = model
        self.width = img_width
        self.height = img_height

    def detect(self, patch):
        input = self.preprocess(patch)
        output = self.model(input)
        pred = torch.max(output, 1)[1].cpu().numpy()
        return pred

    def preprocess(self, patch):
        patch = cv2.resize(patch, (self.width, self.height))
        patch = np.expand_dims(np.array(patch), axis=0)/255.0
        tensor = torch.Tensor(patch).permute(0, 3, 1, 2)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor


if __name__ == '__main__':
    detector = lock_detector('lock_classifier_2019_3_18.pth')
    img = cv2.imread('/home/xddz/Datasets/container_dataset/lock_detection/lock/lock_00047.jpg')
    result = detector.detect(img)
    import matplotlib.pyplot as plt
    print('LCH: the detection result is ', result)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()