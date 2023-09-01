import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import copy
import pandas as pd

def get_data_loaders(batch_size , train = False):
    if train:
        transform = transforms.Compose([
            transforms.Resize(196),
            transforms.CenterCrop(180),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p = 0.3),
            transforms.RandomVerticalFlip(p = 0.01),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(), transforms.GaussianBlur(3)]), p = 0.1),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        train_set    = datasets.ImageFolder("dataset/train/", transform = transform)
        train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
        
        return train_loader, len(train_set)
    
    else:
        transform = transforms.Compose([
            transforms.Resize(196),
            transforms.CenterCrop(180),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        vali_set = datasets.ImageFolder("dataset/valid/", transform = transform)
        test_set = datasets.ImageFolder("dataset/test/",  transform = transform)
        
        vali_loader = DataLoader(vali_set, batch_size = batch_size, shuffle = True)
        test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True)
        
        return vali_loader, test_loader, len(vali_set), len(test_set)
    
def train_model(model, criterion, optimizer, scheduler):
    model.train()
                
    wrongs   = 0.0
    corrects = 0
            
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # initailization
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):        
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
                        
            loss.backward()
            optimizer.step()
           
        wrongs   += loss.item() * inputs.size(0)
        corrects += torch.sum(pred == labels.data)
            
    epoch_loss = wrongs / len_train_data
    epoch_accu = corrects.double() / len_train_data
    
    scheduler.step()        
    Training_Loss.append(float(epoch_loss))
    Training_Accuracy.append(float(epoch_accu))
    
    print("Testing Loss: {:.4f} | Accuracy: {:.4f}".format(epoch_loss, epoch_accu))
    
def test_model(model):
    model.eval()
    
    wrongs   = 0.0
    corrects = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            wrongs   += loss.item() * inputs.size(0)
            corrects += torch.sum(pred == labels.data)
                
        epoch_loss = wrongs / len_test_data
        epoch_accu = corrects.double() / len_test_data
        
        Testing_Loss.append(float(epoch_loss))
        Testing_Accuracy.append(float(epoch_accu))
        
        print("Testing Loss: {:.4f} | Accuracy: {:.4f}".format(epoch_loss, epoch_accu))
    return epoch_accu             

# cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_dict = pd.read_csv("dataset/class_dict.csv")
classes    = list(class_dict['class'])
num_class  = len(classes)

# parameters
batch_size = 64
epochs     = 10    
Learning_rate = 0.005

# data loader
train_loader, len_train_data = get_data_loaders(batch_size = batch_size, train = True)
valid_loader, test_loader, len_valid_data, len_test_data = get_data_loaders(batch_size = batch_size, train = False)

Training_Loss = []
Testing_Loss  = []
Training_Accuracy = []
Testing_Accuracy  = []


#------------------- resnet18 -----------------#
# use resnet18 structure
model = models.resnet18(pretrained = True)

for param in model.parameters():
    param.requires_grad = False

# append output features = num_class(400)  (transfer learning)
num_ftrs   = model.fc.in_features # inputs of the lastest fc layer 
model.fc   = nn.Linear(num_ftrs , num_class) 
#------------------- resnet18 -----------------#


'''
#------------------   vgg16  ------------------#
# use VGG16 structure
model = models.vgg16(pretrained = True)

for param in model.features.parameters():
    param.requires_grad = False
    
# append output features = num_class(400)  (transfer learning)
num_ftrs = model.classifier[6].in_features
feature_model = list(model.classifier.children())
feature_model.pop()            
feature_model.append(nn.Linear(num_ftrs, num_class))
model.classifier = nn.Sequential(*feature_model)
#------------------   vgg16  ------------------#
'''

# condition settings
model      = model.to(device)
criterion  = nn.CrossEntropyLoss()
optimizer  = optim.SGD(model.parameters(), lr = Learning_rate, momentum = 0.9)
exp_lr_sch = lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1) # learning rate regulated 

# main
epoch_accu = 0.0
best_accu  = 0.0
for epoch in range(epochs):
    best_weight = copy.deepcopy(model.state_dict())
    print("Epoch : {} / {}".format(epoch + 1, epochs))
    print("-" * 40)
    # training
    train_model(model, criterion, optimizer, exp_lr_sch)
    # testing
    epoch_accu = test_model (model)
    print('\t')
    
    if(epoch_accu > best_accu):
        best_accu   = epoch_accu
        best_weight = copy.deepcopy(model.state_dict())

model.load_state_dict(best_weight)

print("Best Accuracy : {}".format(best_accu))

'''
torch.save(model,'trained_model.pt')
'''

# plot
epoch = [x for x in range(1, 11)]
plt.figure(1)
plt.plot(epoch, Training_Loss, color = 'blue')
plt.plot(epoch, Testing_Loss,  linestyle='-', marker = '*', color = 'red')
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Training_Loss', 'Testing_Loss'], title_fontsize = 15)

plt.figure(2)
plt.plot(epoch, Training_Accuracy, color = 'blue')
plt.plot(epoch, Testing_Accuracy,  linestyle='-', marker = '*', color = 'red')
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['Training_Accuracy', 'Testing_Accuracy'], title_fontsize = 15)