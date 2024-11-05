import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
print("当前工作目录:", os.getcwd())

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_Layers = 2 # Number of layers in the FNN
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# FNN with [num_Layers] layers
class FNN(nn.Module):
    def __init__(self, use_Xavier = True, use_Relu = True):
        super(FNN, self).__init__()
        current_size = 1024
        layers = []

        # Construct the model
        for i in range(num_Layers):
            if i == 0:
                layers.append(nn.Linear(784, current_size >> 1)) # 784 512 256 128 64 32 10
            elif i == num_Layers - 1:
                layers.append(nn.Linear(current_size, 10))
            else:
                layers.append(nn.Linear(current_size, current_size >> 1))

            if use_Xavier:
                nn.init.xavier_normal_(layers[-1].weight)
            else:
                nn.init.kaiming_normal_(layers[-1].weight)

            if i != num_Layers - 1:
                if use_Relu:
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.Tanh())
            current_size = current_size >> 1

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

accuracy_dict = {}

experiments = {
        'model_Xavier_Tanh': {'use_Xavier': True, 'use_Relu': False},
        'model_Xavier_Relu': {'use_Xavier': True, 'use_Relu': True},
        'model_He_Tanh': {'use_Xavier': False, 'use_Relu': False},
        'model_He_Relu': {'use_Xavier': False, 'use_Relu': True},
    }

for model_name, model_args in experiments.items():
    model = FNN(**model_args).to(device)
    accuracy_dict[model_name] = []

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                accuracy_dict[model_name].append(loss.item())

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('{}: Accuracy of the network on the 10000 test images: {} %'.format(model_name, 100 * correct / total))
    
# Plot the loss over iterations
for name, acc_list in accuracy_dict.items():
    iterations = range(1, len(acc_list) + 1)
    plt.plot(iterations, acc_list, label = name)

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss over Iterations for Different Models with {} Layers'.format(num_Layers))
plt.legend()
try:
    plt.savefig('./result_{}.jpg'.format(num_Layers))
except Exception as e:
    print(f"error: {e}")
plt.show()