import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math
import sys
from sklearn.manifold import TSNE


BATCH_SIZE = 50
EPOCHS = 1
LR = 0.001
STEPS=200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False,
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/', transform=torchvision.transforms.ToTensor(), train=False)

test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE * 10, shuffle=True)

class CNN(nn.Module):
    '''
    第一层:28x28x1
    第二层:14x14x16
    第三层:7x7x32
    '''
    def __init__(self):
        super(CNN, self).__init__()
        # 第一层卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 输出层
        self.out = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)

        return output, x

cnn = CNN().to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()
        

def plot_with_labels(lowDWeights, labels):
    fig = plt.figure()
    fig.set_size_inches(15, 15)
    ax = fig.gca(projection='3d')

    X, Y, Z = lowDWeights[:, 0], lowDWeights[:, 1], lowDWeights[:, 2]
    
    for x, y, z, s in zip(X, Y, Z, labels):
        c = cm.rainbow(int(255 * s / 9))
        ax.text(x, y, z, s, backgroundcolor=c)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    plt.axis('off')
    plt.show()

# 权重初始化
cnn.apply(weight_reset)

# 建立迭代器iter
dataiter = iter(test_loader)

train_acc_list = []
test_acc_list = []
train_loss_list = []
test_loss_list = []

for epoch in range(EPOCHS):
    for step, data in enumerate(train_loader):
        # 数据提取并转为GPU形式
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 结果预测并获取预测值
        outputs = cnn(inputs)[0]
        train_pred_y = torch.max(outputs, 1)[1].data.squeeze()
        # 获取准确率
        train_acc = (train_pred_y == labels).sum().item() / float(labels.size(0))
        # 获取损失值
        loss = loss_func(outputs, labels)
        train_loss_list.append(float(loss))
        train_acc_list.append(train_acc)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 异常处理标志
        try:
            images, targets = next(dataiter)
        except StopIteration:
            dataiter = iter(test_loader)
            images, targets = next(dataiter)
        
        # 数据提取并转为GPU形式
        images, targets = next(dataiter)
        images, targets = images.to(device), targets.to(device)
        # 结果预测并获取预测值
        test_outputs, last_layer = cnn(images)
        test_pred_y = torch.max(test_outputs, 1)[1].data.squeeze()
        # 获取准确率
        test_acc = (test_pred_y == targets).sum().item() / float(targets.size(0))
        test_acc_list.append(test_acc)
        # 获取损失值
        test_loss = loss_func(test_outputs, targets)
        test_loss_list.append(float(test_loss))
        
        sys.stdout.write(('\rOnly EPOCH 1, step={:3d}/{:3d}, train_loss={:.4f}, '
               'train_acc={:.4f}, valid_loss={:.4f}, val_acc={:.4f}').format(step + 1,
                                                                    STEPS,
                                                                    loss,
                                                                    train_acc,
                                                                    test_loss,
                                                                    test_acc))
        if step == STEPS-1:
            # 绘制出图像的数量最大值取决于数据的batchsize大小
            tsne = TSNE(perplexity=30, n_components=3, init='pca', n_iter=5000)
            low_dim_embs = tsne.fit_transform(last_layer.cpu().detach().numpy())
            labels = targets.cpu().numpy()
            plot_with_labels(low_dim_embs, labels)
            break