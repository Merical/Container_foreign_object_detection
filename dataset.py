import torch
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
import numpy as np
import cv2
import os
from sklearn.utils import shuffle

dataset_dir = '/home/xddz/Datasets/container_dataset/lock_detection'
test_size = 300

images = []
labels = []
for step, dir in enumerate(['lock', 'clear', 'background']):
    patch_list = os.listdir(os.path.join(dataset_dir, dir))
    for im in patch_list:
        img = cv2.resize(cv2.imread(os.path.join(dataset_dir, dir, im)), (128, 128))
        images.append(img)
        labels.append(step)

images, labels = np.array(images), np.array(labels)
images, labels = shuffle(images, labels, random_state=2)
print(images.shape, labels.shape)
images = images/255.0

# for i in range(1,26):
#     plt.subplot(5, 5, i)
#     plt.imshow(images[i*5])
#     print(labels[i*5])
# plt.show()

x_train = torch.Tensor(images[:-test_size]).permute(0, 3, 1, 2)
y_train = torch.LongTensor(labels[:-test_size])
x_test = torch.Tensor(images[-test_size:]).permute(0, 3, 1, 2)
y_test = torch.LongTensor(labels[-test_size:])
train_dataset = Data.TensorDataset(x_train, y_train)
torch.save(train_dataset, "./train_dataset.pt")
test_dataset = Data.TensorDataset(x_test, y_test)
torch.save(test_dataset, "./test_dataset.pt")