from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import models as models

import os
import sys
import time
import argparse
import datetime

from torch.autograd import Variable

import dataloader

parser = argparse.ArgumentParser(description='PyTorch Clothing-1M Training')
parser.add_argument('--lr', default=0.0008, type=float, help='learning_rate')
parser.add_argument('--start_epoch', default=2, type=int)
parser.add_argument('--num_epochs', default=3, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--optim_type', default='SGD')
parser.add_argument('--seed', default=7)
parser.add_argument('--gpuid', default=1, type=int)
parser.add_argument('--id', default='cross_entropy')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
  
# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    learning_rate = args.lr
    if epoch > args.start_epoch:
        learning_rate=learning_rate/10        
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print('\n=> %s Training Epoch #%d, LR=%.4f' %(args.id,epoch, learning_rate))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, args.num_epochs, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, loss.data[0], 100.*correct/total))
        sys.stdout.flush()
        if batch_idx%1000==0:
            val(epoch)
            net.train()
            
def val(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data[0], acc))
    record.write('Validation Acc: %f\n'%acc)
    record.flush()
    if acc > best_acc:
        best_acc = acc
        print('| Saving Best Model ...')
        save_point = './checkpoint/%s.pth.tar'%(args.id)
        save_checkpoint({
            'state_dict': net.state_dict(),
        }, save_point) 

def test():
    global test_acc
    test_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = test_net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    acc = 100.*correct/total   
    test_acc = acc
    record.write('Test Acc: %f\n'%acc)
    
os.mkdir('checkpoint')     
record=open('./checkpoint/'+args.id+'_test.txt','w')
record.write('learning rate: %f\n'%args.lr)
record.flush()
     
loader = dataloader.clothing_dataloader(batch_size=args.batch_size,num_workers=5,shuffle=True)
train_loader,val_loader,test_loader = loader.run()

best_acc = 0
test_acc = 0
# Model
print('\nModel setup')
print('| Building net')
net = models.resnet50(pretrained=True)
net.fc = nn.Linear(2048,14)
test_net = models.resnet50(pretrained=True)
test_net.fc = nn.Linear(2048,14)
if use_cuda:
    net.cuda()
    test_net.cuda()
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

print('\nTraining model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(args.optim_type))

for epoch in range(1, 1+args.num_epochs):
    train(epoch)
    val(epoch)

print('\nTesting model')
checkpoint = torch.load('./checkpoint/%s.pth.tar'%args.id)
test_net.load_state_dict(checkpoint['state_dict'])
test()

print('* Test results : Acc@1 = %.2f%%' %(test_acc))
record.write('Test Acc: %.2f\n' %test_acc)
record.flush()
record.close()
