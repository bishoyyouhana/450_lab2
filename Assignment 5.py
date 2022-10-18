# %%
from __future__ import division
import argparse
import os, sys
import time
import datetime
import numpy as np

# Import pytorch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm

# %%
# You cannot change this line.
from dataloader import CIFAR10
import dataloader
import imp
imp.reload(dataloader)

# %%
############################################  FIRST ATTEMPT   ###########################################################
# ##### until 35 epochs it gives us 85 percent with lr =0.01, sgd, no transforms,
# ##### dropouts of 0.5 prove to be too much 

# class model(nn.Module):
#     def __init__(self):
#         super(model, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
#         #relu
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         #relu
#         #pool 2,2
#         self.bn1 = nn.BatchNorm2d(64)
        
        
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         #relu
#         self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
#         #relu
#         #pool 2,2
#         self.bn2 = nn.BatchNorm2d(128)
#         self.drop2 = nn.Dropout2d(p=0.05)
        
#         self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
#         #relu
#         self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
#         #relu
#         #pool 2,2
#         self.bn3 = nn.BatchNorm2d(256)
        
#         self.flat = nn.Flatten()
        
#         self.fc1 =nn.Linear(256*4*4, 1024)
#         #relu
#         self.drop1 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(1024, 512)
#         #relu
#         self.drop3 = nn.Dropout(p=0.1)
#         self.fc3 = nn.Linear(512, 10)
#         #relu
        
        
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = F.relu(self.conv2(out))
#         out = self.bn1(F.max_pool2d(out, 2))
        
#         out = F.relu(self.conv3(out))
#         out = F.relu(self.conv4(out))
#         out = self.bn2(F.max_pool2d(out, 2))
#         out = self.drop2(out)
        
#         out = F.relu(self.conv5(out))
#         out = F.relu(self.conv6(out))
#         out = self.bn3(F.max_pool2d(out, 2))
        
#         out = self.flat(out)

# #         out = out.view(out.size(0), -1)
        
#         out = F.relu(self.fc1(out))
#         out = self.drop1(out)
#         out = F.relu(self.fc2(out))
# #         out = self.drop3(out)
#         out = F.relu(self.fc3(out))
#         return out

# %%
############################################  SECOND ATTEMPT   ###########################################################

# ##### until 35 epochs it gives us 85 percent with lr =0.01, sgd, no transforms,
# ##### dropouts of 0.5 prove to be too much 

# ##### dropouts 0.5 and 0.1, lr = 0.1, 100 epochs. sgd, 87 percent
# class model(nn.Module):
#     def __init__(self):
#         super(model, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
#         #relu
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         #relu
#         #pool 2,2
#         self.bn1 = nn.BatchNorm2d(64)
        
        
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         #relu
#         self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
#         #relu
#         #pool 2,2
#         self.bn2 = nn.BatchNorm2d(128)
#         self.drop2 = nn.Dropout2d(p=0.05)
        
#         self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
#         #relu
#         self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
#         #relu
#         #pool 2,2
#         self.bn3 = nn.BatchNorm2d(256)
        
#         self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
#         #relu
#         self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
#         #relu
#         #pool 2,2
#         self.bn4 = nn.BatchNorm2d(512)
        
#         self.flat = nn.Flatten()
        
#         self.fc1 =nn.Linear(512*2*2, 1024)
#         #relu
#         self.drop1 = nn.Dropout(p=0.5)
#         self.fc2 = nn.Linear(1024, 512)
#         #relu
#         self.drop3 = nn.Dropout(p=0.1)
#         self.fc3 = nn.Linear(512, 10)
#         #relu
        
        
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = F.relu(self.conv2(out))
#         out = self.bn1(F.max_pool2d(out, 2))
        
#         out = F.relu(self.conv3(out))
#         out = F.relu(self.conv4(out))
#         out = self.bn2(F.max_pool2d(out, 2))
#         out = self.drop2(out)
        
#         out = F.relu(self.conv5(out))
#         out = F.relu(self.conv6(out))
#         out = self.bn3(F.max_pool2d(out, 2))
        
#         out = F.relu(self.conv7(out))
#         out = F.relu(self.conv8(out))
#         out = self.bn4(F.max_pool2d(out, 2))
        
#         out = self.flat(out)

# #         out = out.view(out.size(0), -1)
        
#         out = F.relu(self.fc1(out))
#         out = self.drop1(out)
#         out = F.relu(self.fc2(out))
# #         out = self.drop3(out)
#         out = F.relu(self.fc3(out))
#         return out

# %%
############################################  FINAL ATTEMPT   ###########################################################

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.neuralnet = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
#             nn.Dropout2d(p=0.05),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
    
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Dropout2d(p=0.05),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Flatten(), 
            nn.Linear(512*2*2, 1024),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, x):
        return self.neuralnet(x)

# %%
#hyperparameters
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 30
INITIAL_LR = 0.05
MOMENTUM = 0.9
REG = 1e-5
EPOCHS = 130
DATAROOT = "./data"
CHECKPOINT_PATH = "./saved_model"

# %%
"""
Reference value for mean/std:

mean(RGB-format): (0.4914, 0.4822, 0.4465)
std(RGB-format): (0.2023, 0.1994, 0.2010)
"""

transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
#      transforms.CenterCrop(10),
        transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    ])

transform_val = transforms.Compose(
    [
transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

# %%
# Call the dataset Loader
trainset = CIFAR10(root=DATAROOT, train=True, test=False,download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=1)
valset = CIFAR10(root=DATAROOT, train=False,test=False, download=True, transform=transform_val)
valloader = torch.utils.data.DataLoader(valset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=1)

# %%
# data loader for testing process
testset = CIFAR10(root="", train=False, test=True, transform=transform_val)
testloader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=1)

# %%
# Specify the device for computation
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
net = model()
net = net.to(device)
print(device)
if device =='cuda:1':
    print("Train on GPU...")
else:
    print("Train on CPU...")

# %%

# FLAG for loading the pretrained model
TRAIN_FROM_SCRATCH = False
# Code for loading checkpoint and recover epoch id.
CKPT_PATH = "./saved_model/model.h5"
def get_checkpoint(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path)
    except Exception as e:
        print(e)
        return None
    return ckpt

ckpt = get_checkpoint(CKPT_PATH)
if ckpt is None or TRAIN_FROM_SCRATCH:
    if not TRAIN_FROM_SCRATCH:
        print("Checkpoint not found.")
    print("Training from scratch ...")
    start_epoch = 0
    current_learning_rate = INITIAL_LR
else:
    print("Successfully loaded checkpoint: %s" %CKPT_PATH)
    net.load_state_dict(ckpt['net'])
    start_epoch = ckpt['epoch'] + 1
    current_learning_rate = ckpt['lr']
    print("Starting from epoch %d " %start_epoch)

print("Starting from learning rate %f:" %current_learning_rate)


# %%
criterion =  nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=REG)

# adam proved to be weaker
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(net.parameters(),lr=INITIAL_LR,weight_decay=REG)


# %%
# Start the training/validation process
# The process should take about 5 minutes on a GTX 1070-Ti
# if the code is written efficiently.
global_step = 0
best_val_acc = 0

for i in range(start_epoch, EPOCHS):
    print(datetime.datetime.now())
    # Switch to train mode
    net.train()
    print("Epoch %d:" %i)

    total_examples = 0
    correct_examples = 0

    train_loss = 0
    train_acc = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        
#         print(len(inputs))
#         print(len(len(inputs)))
        # Copy inputs to device
        inputs = inputs.requires_grad_().to(device)
        targets = targets.to(device)
        # Zero the gradient
        optimizer.zero_grad()
        # Generate output
#         print(inputs)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # Now backward loss
        loss.backward()
        # Apply gradient
        optimizer.step()
        # Calculate predicted labels
        _, predicted = torch.max(outputs, 1)
        # Calculate accuracy
        total_examples += targets.size(0)
        correct_examples += (predicted == targets).sum().item()

        train_loss += loss

        global_step += 1
        if global_step % 100 == 0:
            avg_loss = train_loss / (batch_idx + 1)
        pass
    avg_acc = correct_examples / total_examples
    print(total_examples)
    print("Training loss: %.4f, Training accuracy: %.4f" %(avg_loss, avg_acc))
    print(datetime.datetime.now())
    # Validate on the validation dataset
    print("Validation...")
    total_examples = 0
    correct_examples = 0
    
    net.eval()
    
    val_loss = 0
    val_acc = 0
    # Disable gradient during validation
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
#             print("----------------------------------------------------------------------------------")
#             print(inputs)
            # Copy inputs to device
            inputs = inputs.requires_grad_().to(device)
            targets = targets.to(device)
            # Zero the gradient
            optimizer.zero_grad()
            # Generate output from the DNN.
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # Calculate predicted labels
            _, predicted = outputs.max(1)
            # Calculate accuracy
            total_examples +=targets.size(0)
            correct_examples += (predicted == targets).sum().item()
            val_loss += loss

    avg_loss = val_loss / len(valloader)
    avg_acc = correct_examples / total_examples
    print("Validation loss: %.4f, Validation accuracy: %.4f" % (avg_loss, avg_acc))
    

    DECAY_EPOCHS = 5
    DECAY = 1.00
    
    if i % DECAY_EPOCHS == 0 and i != 0:
#         current_learning_rate = (pow(0.95,(i/DECAY_EPOCHS)))*INITIAL_LR
        current_learning_rate = (1/(1+(i/DECAY_EPOCHS)))*INITIAL_LR
#         current_learning_rate = (0.95^(i/DECAY_EPOCHS)*INITIAL_LR
        for param_group in optimizer.param_groups:
            # Assign the learning rate parameter
            torch.optim.SGD(net.parameters(), lr=current_learning_rate, momentum=MOMENTUM, weight_decay=REG)
            
        print("Current learning rate has decayed to %f" %current_learning_rate)
    
    # Save for checkpoint
    if avg_acc > best_val_acc:
        best_val_acc = avg_acc
        if not os.path.exists(CHECKPOINT_PATH):
            os.makedirs(CHECKPOINT_PATH)
        print("Saving ...")
        state = {'net': net.state_dict(),
                 'epoch': i,
                 'lr': current_learning_rate}
        torch.save(state, os.path.join(CHECKPOINT_PATH, 'model.h5'))

print("Optimization finished.")

# %%
#to create Kaggle compatible soltuions
import csv

# open the file in the write mode
f = open('soltuion.csv', 'w')
writer = csv.writer(f)

with torch.no_grad():
#     print(enumerate(testloader))

    for batch_idx, (inputs, targets) in enumerate(testloader):
#         print(inputs)
        # Copy inputs to device
        inputs = inputs.requires_grad_().to(device)
#         targets = targets.to(device)
        # Zero the gradient
        optimizer.zero_grad()
        # Generate output from the DNN.
        outputs = net(inputs)
#         loss = criterion(outputs, targets)
        # Calculate predicted labels
        _, predicted = outputs.max(1)
        # Calculate accuracy
        #total_examples +=targets.size(0)
        #correct_examples += (predicted == targets).sum().item()
#         val_loss += loss
#         predictions += predicted
        writer.writerow([batch_idx, predicted.item()])
        print(predicted.item())

f.close()
# avg_loss = val_loss / len(valloader)
# avg_acc = correct_examples / total_examples
# print("Validation loss: %.4f, Validation accuracy: %.4f" % (avg_loss, avg_acc))

# %%



