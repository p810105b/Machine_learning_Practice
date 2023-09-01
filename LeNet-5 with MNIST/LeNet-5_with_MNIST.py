import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

# build network 
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # layer 1, 2 (Convolution and max pool)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # layer 3, 4 (Convolution and max pool)
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # layer 5 (full connection)
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        # layer 6 (full connection)
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        # layer 7 (soft max)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)
        return x


def Training(model, device, trainloader, optimizer):
    # training mode
    model.train()
    
    total    = 0
    correct  = 0.0
    accuracy = 0.0
    Batch    = 0
    i        = -1
    for inputs, labels in trainloader:
        
        inputs = inputs.to(device) # (x_train) imgs for handwritten digits 
        labels = labels.to(device) # (y_hat)   the actual digits coressponding to imgs 
        
        outputs = model(inputs)    # (y_train) training result  
        
        # cross_entropy for binary classification
        loss = F.cross_entropy(outputs, labels)
        
        # pick the label of maximun value foe each row (the higest possibility number)
        predict = outputs.argmax(dim = 1)
        
        total    += labels.size(0)
        correct  += (predict == labels).sum().item()
        accuracy = correct / total
        
        # get grad for loss function 
        loss.backward()
        
        # update parameters 
        optimizer.step()
        
        # clear the grad after updating the parameters
        optimizer.zero_grad()
        
        Loss.append(loss.item())
        
        i += 1 
        if i % 100 == 0:
            Batch += 1
            print(" Batch：{} / 60000 | Loss: {:.4f}, accuracy: {:.4f} %" .format(6400 * (Batch-1), loss.item(), 100 * accuracy))
            
    return loss.item(), accuracy


def Testing(model, device, testloader):
    # evaluation mode
    model.eval()
    
    correct   = 0.0
    test_loss = 0.0
    total     = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            
            inputs = inputs.to(device) # (x_train) imgs for handwritten digits 
            labels = labels.to(device) # (y_hat)   the actual digits coressponding to imgs 
            
            output = model(inputs)
            
            test_loss += F.cross_entropy(output, labels).item()
            
            predict = output.argmax(dim=1)
            
            total += labels.size(0)
            correct += (predict == labels).sum().item()
            accuracy = correct / total
            
        print(" Testing avarage loss: {:.4f}, Testing accuracy: {:.4f} %" .format(test_loss / total, 100 * (accuracy)))
        
    
train_form = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
test_form  = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

train_set = datasets.MNIST(root="./data", train = True,  download = True, transform = train_form)
test_set  = datasets.MNIST(root="./data", train = False, download = True, transform = test_form)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size = 32, shuffle = False)  
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = LeNet().to(device)
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

Loss = []

print('|------------------------Training-------------------------|')
Training(model, device, train_loader, optimizer)
print('|------------------------Training-------------------------| \n')

print('|-------------------------Testing-------------------------|')
Testing(model, device, test_loader)
print('|-------------------------Testing-------------------------|')

# plot mean loss | window size = 50
plt.title('Mean Loss | Window size = 50')
window_size = 50
Loss_series = pd.Series(Loss)
Loss_rolling = Loss_series.rolling(window = window_size)
rolling_mean = Loss_rolling.mean()
plt.plot(rolling_mean, 'r')

