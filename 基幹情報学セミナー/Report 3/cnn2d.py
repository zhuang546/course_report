import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

class ScalableCNN(nn.Module):
    def __init__(self, num_layers=2, channel_factor=2, use_pooling=False, use_gap=False):
        super(ScalableCNN, self).__init__()
        self.use_pooling = use_pooling
        self.use_gap = use_gap

        layers = []

        # 第一个卷积层（stem层），输出通道数为16
        in_channels = 3  # CIFAR10数据集的图像为RGB三通道
        out_channels = 16
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, padding_mode="replicate"))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels

        # 记录池化层的数量以计算特征图尺寸
        num_pools = 1

        # 构建指定数量的卷积块
        for i in range(num_layers - 1):
            out_channels = int(in_channels * channel_factor)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="replicate"))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if self.use_pooling and i % 2 == 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                num_pools += 1
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # 计算全连接层的输入尺寸
        # CIFAR10图像尺寸为32x32
        feature_map_size = 32
        for _ in range(num_pools):
            feature_map_size = feature_map_size // 2  # 每个池化层尺寸减半

        if self.use_gap:
            fc_input_dim = out_channels  # GAP后的特征尺寸为通道数
        else:
            fc_input_dim = out_channels * feature_map_size * feature_map_size  # 展开2D特征

        # 两个全连接层和softmax层
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10),
            nn.Softmax(dim=1)  # CIFAR10有10个分类
        )

    def forward(self, x):
        x = self.features(x)
        if self.use_gap:
            x = F.adaptive_avg_pool2d(x, (1, 1))  # 全局平均池化
            x = x.view(x.size(0), -1)
        else:
            x = x.view(x.size(0), -1)  # 展开2D特征
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    num_workers = 0
    num_epoch = 10

    random.seed(42)
    torch.manual_seed(42)

    transform = transforms.Compose( [
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=32, shuffle=False, num_workers=num_workers)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=num_workers)

    exps = {
        'net_L2': {'num_layers': 2, 'channel_factor': 2, 'use_pooling': False, 'use_gap': False},
        'net_L4': {'num_layers': 4, 'channel_factor': 2, 'use_pooling': False, 'use_gap': False},
        'net_L6': {'num_layers': 6, 'channel_factor': 2, 'use_pooling': False, 'use_gap': False},
        'net_L6_pool': {'num_layers': 6, 'channel_factor': 2, 'use_pooling': True, 'use_gap': False},
        'net_L6_gap': {'num_layers': 6, 'channel_factor': 2, 'use_pooling': False, 'use_gap': True},
        'net_L6_pool_gap': {'num_layers': 6, 'channel_factor': 2, 'use_pooling': True, 'use_gap': True},
    }
    accuracy_dict = {}
    for net_name, net_opt in exps.items():
        net = ScalableCNN(**net_opt)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        accuracy_dict[net_name] = []
        
        start_time = time.time()

        for epoch in range(num_epoch):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                net.train()
                if torch.cuda.is_available():
                    net = net.cuda()
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                optimizer.zero_grad()

                outputs = net(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 200 == 199:
                    net.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for (inputs, labels) in testloader:
                            if torch.cuda.is_available():
                                net = net.cuda()
                                inputs = inputs.cuda()
                                labels = labels.cuda()

                            outputs = net(inputs) 
                            
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                    accuracy = 100 * float(correct/total)

                    elapsed_time = time.time() - start_time

                    print('[%d, %5d] Loss: %.3f '% (epoch + 1, i + 1, running_loss / 200), 
                          'Accuracy: {:.2f} %'.format(accuracy), 
                          'Time: {:.2f} seconds'.format(elapsed_time))
                    accuracy_dict[net_name].append(accuracy)
                    running_loss = 0.0

        print(f'Total training time: {elapsed_time:.2f}')

        PATH = f'./{net_name}.pth'
        torch.save(net.state_dict(), PATH)
    
    # 绘制折线图
    for name, acc_list in accuracy_dict.items():
        iterations = range(1, len(acc_list) + 1)
        plt.plot(iterations, acc_list, label=name)

    plt.xlabel('Iterations')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy over Iterations for Different Models')
    plt.legend()
    try:
        plt.savefig('./result.jpg')
        plt.show()
    except:
        plt.show()