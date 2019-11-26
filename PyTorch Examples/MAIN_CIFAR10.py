'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from Opt_PyGOLS import PyGOLS

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import scipy.io as scio
import numpy as np
#from models import *
from __init__ import *
from utils import progress_bar


nEpoch = 100
ManLRs = (1e-3,1e-2,1e-1) # manually set learning rates for comparison
loops = 1 + len(ManLRs)
optMethod = 'SGD'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainBatchS = 128
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainBatchS, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


criterion = nn.CrossEntropyLoss()
iterations = round(50000/trainBatchS)

for l in range(loops):
    print('loop: '+str(l))

    if l == 0:
        IGOLSFlag = True
        MLR = None
    elif l > 0:
        IGOLSFlag = False
        MLR = ManLRs[l-1]


    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


    def train(epoch,a_old,stateO):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        As = np.zeros((iterations))
        Fvs = np.zeros((iterations))

        for its in range(iterations):
            def Closure():
                optimizer.zero_grad()
                inputs, targets = next(iter(trainloader))
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                return loss



            if IGOLSFlag == True:
                optimizer = PyGOLS(net.parameters(), init_guess=a_old, state=stateO, alg=optMethod)
                loss, stateO = optimizer.step(Closure)
                a = stateO.get('a')
                fevs = stateO.get('func_evals')
            else:
                optimizer = optim.SGD(net.parameters(), lr=MLR)
                loss = optimizer.step(Closure)
                a = MLR
                fevs = (epoch)*iterations+its+1

            As[its] = a
            Fvs[its] = fevs
            a_old = a*1.0


            inputs, targets = next(iter(trainloader))
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)

            train_loss += loss
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(its, iterations, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(its+1), 100.*correct/total, correct, total))

        trErr = train_loss/(its+1)
        trAcc = 100.*correct/total

        return trErr,trAcc,As,Fvs,a_old,stateO


    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            best_acc = acc

        terr = (test_loss/(batch_idx+1))
        tacc = 100.*correct/total
        return terr,tacc


    # Training
    trErr = np.zeros((nEpoch,1))
    teErr = np.zeros((nEpoch,1))
    trAcc = np.zeros((nEpoch,1))
    teAcc = np.zeros((nEpoch,1))
    dataSt = np.zeros((nEpoch*iterations,1))
    dataFv = np.zeros((nEpoch*iterations,1))
    a_old = 1e-8
    stateO = {}

    for epoch in range(start_epoch, start_epoch+nEpoch):
        trErr[epoch,0],trAcc[epoch,0],Sts,Fvs,a_old,stateO = train(epoch,a_old,stateO)
        dataSt[epoch*iterations:(epoch+1)*iterations,0] = Sts[:]
        dataFv[epoch*iterations:(epoch+1)*iterations,0] = Fvs[:]
        teErr[epoch,0],teAcc[epoch,0] = test(epoch)

    name = 'CIFAR10_loop'+str(l)+'_Eps'+str(nEpoch)+'_N'+str(trainBatchS)+'.mat'
    scio.savemat(name,dict([('dataTr', trErr), ('dataTe',teErr), ('dataSt',dataSt), ('dataFv',dataFv), ('dataAccTr',trAcc), ('dataAccTe',teAcc)]))

