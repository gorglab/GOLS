'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from Opt_PyGOLS import PyGOLS
import torchvision.datasets as dsets
from torch.autograd import Variable

import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from cifar_Batch1 import CIFAR10_B1
#from sklearn.preprocessing import StandardScaler



# TUNABLE PARAMETERS

nSamp = 50
nSampT = 1000 # test dataset sample size


ManLRs = (1e-1,1e-0,10) # manually set learning rates
#ManLRs = (1e-2,1e-1,1) # manually set learning rates
#ManLRs = (1e-3,1e-2,1e-1) # manually set learning rates



dataset = 'mnist'
#dataset = 'cifar'

NetType = 'NI'
#NetType = 'NII'

samps = 1 # samples of test data for each training update
loops = 1 + len(ManLRs)
overP = 0.9
optMethod = 'SGD'

saveSeeds = 1
loadSeeds = 0

capFrac = 10
print_int = 10 # must be a multiple of capFrac


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if dataset == 'mnist':
#    MaxFvals = 40000
    MaxFvals = 1000
else:
#    MaxFvals = 10000
    MaxFvals = 1000

print('Samp size: '+str(nSamp))


# DATA
transform_T = transforms.Compose([
        transforms.ToTensor(),
    ])
def get_mean_and_std(train_dataset,test_dataset):
        '''Compute the mean and std value of dataset.'''
        BatchSize = 1000
        Tr_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BatchSize,shuffle=False)
        Te_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BatchSize,shuffle=False)
        print('==> Computing mean and std..')
        LTr = len(train_dataset)
        LTe = len(test_dataset)
        inputs, targets = next(iter(Tr_loader))
        inp_dims = np.shape(inputs)
        mean = torch.zeros((1,inp_dims[1]))
        std = torch.zeros((1,inp_dims[1]))
        for d in range(int(LTr/BatchSize)):
            inputs, targets = next(iter(Tr_loader))
            for i in range(inp_dims[1]):
                mean[0,i] += inputs[:,i,:,:].view((BatchSize,inp_dims[2]*inp_dims[3])).mean()
                std[0,i] += inputs[:,i,:,:].view((BatchSize,inp_dims[2]*inp_dims[3])).std()
        for d in range(int(LTe/BatchSize)):
            inputs, targets = next(iter(Te_loader))
            for i in range(inp_dims[1]):
                mean[0,i] += inputs[:,i,:,:].view((BatchSize,inp_dims[2]*inp_dims[3])).mean()
                std[0,i] += inputs[:,i,:,:].view((BatchSize,inp_dims[2]*inp_dims[3])).std()

        mean.div_(LTr/BatchSize + LTe/BatchSize)
        std.div_(LTr/BatchSize + LTe/BatchSize)

        print('Mean: ',mean)
        print('Std: ',std)

        return mean, std


print('==> Preparing data..')
if dataset == 'cifar':
    trainset = CIFAR10_B1(root='./data', train=True, transform=transform_T)
    testset = CIFAR10_B1(root='./data', train=False, transform=transform_T)

    mean, std = get_mean_and_std(trainset,testset)

    transform_Both = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean[0,0],mean[0,1],mean[0,2]), (std[0,0],std[0,1],std[0,2])),

    ])

    trainset = CIFAR10_B1(root='./data', train=True, transform=transform_Both)
    testset = CIFAR10_B1(root='./data', train=False, transform=transform_Both)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=nSamp, shuffle=True)
    trainC_loader = torch.utils.data.DataLoader(trainset, batch_size=nSampT, shuffle=True)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=nSampT, shuffle=True)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

elif dataset == 'mnist':
    train_dataset = dsets.MNIST(root ='./data',train = True,transform = transform_T,download = True)
    test_dataset = dsets.MNIST(root ='./data',train = False,transform = transform_T)

    mean, std = get_mean_and_std(train_dataset,test_dataset)

    transform_Both = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean[0,0],), (std[0,0],)),

    ])

    train_dataset = dsets.MNIST(root ='./data',train = True,transform = transform_Both,download = True)
    test_dataset = dsets.MNIST(root ='./data',train = False,transform = transform_Both)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=nSamp, shuffle=True)
    trainC_loader = torch.utils.data.DataLoader(train_dataset, batch_size=nSampT, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=nSampT, shuffle = True)



# MODEL CONSTRUCTION

if dataset == 'mnist':
    channels = 1
    inputdim = 28*28*channels
elif dataset == 'cifar':
    channels = 3
    inputdim = 32*32*channels

if NetType == 'NI':
    Structure = [inputdim,800,10]
    Acts = 'sigmoid'
    criterion = nn.BCELoss(reduction='mean')
elif NetType == 'NII':
    Structure = [inputdim,1000,500,250,10]
    Acts = 'tanh'
    criterion = nn.MSELoss(reduction='mean')

NNdeets = {}
NNdeets['Str'] = Structure
NNdeets['Acts'] = Acts

# DATA MEMORY
trunc = int(MaxFvals/capFrac)
TrErrs = np.zeros((trunc,loops))
TeErrs = np.zeros((trunc,loops))
TrAccs = np.zeros((trunc,loops))
TeAccs = np.zeros((trunc,loops))
StepTrunc = np.zeros((trunc,loops))
TrLoss = np.zeros((MaxFvals,loops))
StepS = np.zeros((MaxFvals,loops))
Fvals = np.zeros((MaxFvals,loops))



for l in range(loops):
    print('loop: '+str(l))

    if l == 0:
        IGOLSFlag = True
        MLR = None
    elif l > 0:
        IGOLSFlag = False
        MLR = ManLRs[l-1]


    print('==> Building model..')
    def CreateModel(NNdeets):
        class CNet(nn.Module):
            def __init__(self,Str,Acts):
                super(CNet, self).__init__()

                self.lStr = len(Str)
                self.lins = nn.ModuleList([nn.Linear(int(Str[i]), int(Str[i+1])) for i in range(len(Str)-1)])

                if Acts == 'tanh':
                    self.act = nn.Tanh()
                else:
                    self.act = nn.Sigmoid()


            def forward(self, x):
                for i in range(self.lStr-1):
                    mapnow = self.lins[i]
                    x = self.act(mapnow(x))
                return x

        Str = NNdeets['Str']
        Acts = NNdeets['Acts']

        model = CNet(Str,Acts)
        return model

    net = CreateModel(NNdeets)

    ## SAVE OR LOADING NET
    #PATH = './seeds/Net'+NetType+'_dataset_'+dataset+'_seed'+str(l)+'.pt'
    #if saveSeeds == 1:
        #if NetType == 'NI':
            #def weights_init(m):
                #if isinstance(m, nn.Linear):
                    #torch.nn.init.normal_(m.weight.data)
                    #torch.nn.init.normal_(m.bias.data)
            #net.apply(weights_init)
        #torch.save(net.state_dict(), PATH)
    #if loadSeeds == 1:
        #net.load_state_dict(torch.load(PATH))

    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # TRAINING FUNCTION
    def trainNet(stateO,a_old,upd,IGOLSFlag,ManLR):
        net.train()

        def Closure():
            optimizer.zero_grad()
            inputs, targets = next(iter(train_loader))
            inputs = Variable(inputs.view(-1, inputdim))
            targets = torch.zeros(nSamp,10).scatter_(1,targets.view(-1,1),1)

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
            a = ManLR
            fevs = upd+1

        return loss,a,fevs,stateO



    # TEST FUNCTION
    def Evaluate(loader,TrTe):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            inputs, targets1 = next(iter(loader))
            inputs = Variable(inputs.view(-1, inputdim))
            total = targets1.size(0)
            targets = torch.zeros(total,10).scatter_(1,targets1.view(-1,1),1)
            inputs, targets, targets1 = inputs.to(device), targets.to(device), targets1.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets1).sum().item()

        if (upd % print_int) == 0:
            print(TrTe+' at: '+str(upd)+' Loss: '+str(test_loss)+' Acc: '+str(100.*correct/total))


        terr = (test_loss)
        tacc = 100.*correct/total
        return terr,tacc



    # COMMENCE TRAINING
    a_old = 1e-8
    stateO = {}
    fevsNow = 0
    upd = 0
    while fevsNow < MaxFvals:
        if (upd % capFrac) == 0:
            index = int(upd/capFrac)
            StepTrunc[index,l] = fevsNow
            TrTe = 'Training'
            TrErrs[index,l],TrAccs[index,l] = Evaluate(trainC_loader,TrTe)
            TrTe = 'Test'
            TeErrs[index,l],TeAccs[index,l] = Evaluate(test_loader,TrTe)

        loss,a_old,fevsNow,stateO = trainNet(stateO,a_old,upd,IGOLSFlag,MLR)
        TrLoss[upd,l] = loss
        StepS[upd,l] = a_old
        Fvals[upd,l] = fevsNow
        if upd == MaxFvals-1 and fevsNow < upd:
            fevsNow = MaxFvals + 1
        upd += 1




# PLOTTING

font = {'size'   : 12}
plt.rc('font', **font)
lwth = 1

colours = ('b','k','m','c','r','m','y','g')
labels = {}

for pli in range(loops):
    if pli == 0:
        labels[pli] = 'GOLS'
    else:
        labels[pli] = 'LR: '+str(ManLRs[pli-1])


plt.figure(1)
for pli in range(loops):
    Fvals_tmp = Fvals[:,pli]
    Fvals_p = Fvals_tmp[Fvals_tmp!=0]
    TrLoss_tmp = TrLoss[:,pli]
    TrLoss_p = TrLoss_tmp[TrLoss_tmp!=0]

    plt.plot(Fvals_p,np.log10(TrLoss_p),colours[pli],label=labels[pli], linewidth=lwth)
plt.xlabel('Function Evaluations')
plt.ylabel('Training Loss')
plt.legend()


plt.figure(2)
for pli in range(loops):
    Fvals_tmp = Fvals[0::capFrac,pli]
    Fvals_p = Fvals_tmp[Fvals_tmp!=0]
    TrErrs_tmp = TrErrs[:,pli]
    TrErrs_p = TrErrs_tmp[TrErrs_tmp!=0]

    plt.plot(Fvals_p,np.log10(TrErrs_p),colours[pli],label=labels[pli], linewidth=lwth)
plt.xlabel('Function Evaluations')
plt.ylabel('Large Batch Training Loss')
plt.legend()


plt.figure(3)
for pli in range(loops):
    Fvals_tmp = Fvals[0::capFrac,pli]
    Fvals_p = Fvals_tmp[Fvals_tmp!=0]
    TeErrs_tmp = TeErrs[:,pli]
    TeErrs_p = TeErrs_tmp[TeErrs_tmp!=0]

    plt.plot(Fvals_p,np.log10(TeErrs_p),colours[pli],label=labels[pli], linewidth=lwth)
plt.xlabel('Function Evaluations')
plt.ylabel('Large Batch Test Loss')
plt.legend()


plt.figure(4)
for pli in range(loops):
    Fvals_tmp = Fvals[0::capFrac,pli]
    Fvals_p = Fvals_tmp[Fvals_tmp!=0]
    TrAccs_tmp = TrAccs[:,pli]
    TrAccs_p = TrAccs_tmp[TrAccs_tmp!=0]

    plt.plot(Fvals_p,(TrAccs_p),colours[pli],label=labels[pli], linewidth=lwth)
plt.xlabel('Function Evaluations')
plt.ylabel('Large Batch Training Accuracy')
plt.legend()


plt.figure(5)
for pli in range(loops):
    Fvals_tmp = Fvals[0::capFrac,pli]
    Fvals_p = Fvals_tmp[Fvals_tmp!=0]
    TeAccs_tmp = TeAccs[:,pli]
    TeAccs_p = TeAccs_tmp[TeAccs_tmp!=0]

    plt.plot(Fvals_p,(TeAccs_p),colours[pli],label=labels[pli], linewidth=lwth)
plt.xlabel('Function Evaluations')
plt.ylabel('Large Batch Test Accuracy')
plt.legend()


plt.figure(6)
for pli in range(loops):
    Fvals_tmp = Fvals[:,pli]
    Fvals_p = Fvals_tmp[Fvals_tmp!=0]
    StepS_tmp = StepS[:,pli]
    StepS_p = StepS_tmp[StepS_tmp!=0]

    plt.plot(Fvals_p,np.log10(StepS_p),colours[pli],label=labels[pli], linewidth=lwth)
plt.xlabel('Function Evaluations')
plt.ylabel('Log Step Size')
plt.legend()


plt.show()


