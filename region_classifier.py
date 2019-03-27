import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from tools import lock_classification_model


EPOCH = 600
BATCH_SIZE = 128
LR = 0.001

train_dataset = torch.load('./train_dataset.pt')
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = torch.load('./test_dataset.pt')
x_test, y_test = test_dataset.tensors
x_test = x_test.cuda()
y_test = y_test.cuda()

model = lock_classification_model()
model = nn.DataParallel(model)
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99), eps=1e-06, weight_decay=0.0005)
loss_func = nn.CrossEntropyLoss()
best_accuracy = 0
best_loss = 1

for epoch in range(EPOCH):
    if epoch % 20 == 0:
        LR = LR * 0.9
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99), eps=1e-06, weight_decay=0.0005)

    for step, (x, y) in enumerate(train_loader):
        b_x = x.cuda()
        b_y = y.cuda()

        output = model(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            test_output = model(x_test)
            pred_y = torch.max(test_output, 1)[1].cuda().data
            accuracy = torch.sum(pred_y == y_test).type(torch.FloatTensor) / y_test.size(0)
            if best_accuracy <= accuracy and best_loss >= loss:
                torch.save(model.module.state_dict(), 'best_weights.pth')
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.4f' % accuracy)

del model
model = lock_classification_model()
model.load_state_dict(torch.load('best_weights.pth'))
model = model.cuda()
model.eval()

test_output = model(x_test[:100])
pred_y = torch.max(test_output, 1)[1].cuda().data

print(pred_y, 'prediction number')
print(y_test[:100], 'real number')
print('real accuracy: ', float(torch.sum(pred_y == y_test[:100]).type(torch.FloatTensor) / y_test[:100].size(0)))

