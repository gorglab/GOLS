#%matplotlib inline

#https://www.kaggle.com/graymant/breast-cancer-diagnosis-with-pytorch/notebook

import numpy as np
import pandas as pd
from sympy import *
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Opt_PyGOLS import PyGOLS


nSamp = 100 # mini-batch size
ManLRs = (1e-0,10,100) # manually set learning rates for comparison
optMethod = 'SGD' # chosen algorithm
MaxFvals = 10000 # maximum number of function evaluations
capFrac = 10 # interval of evaluating large batch stats
print_int = 10 # print text at given intervals, must be multiple of capFrac

# whether to use pre-saved or newly generated seeds
saveSeeds = 1
loadSeeds = 0



loops = 1 + len(ManLRs)
init_printing(use_unicode=True)

data = pd.read_csv('./data/data.csv')
data.head()

x = data.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)
diag = { "M": 1, "B": 0}
y = data["diagnosis"].replace(diag)
scaler = StandardScaler()
xTrTrans = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(xTrTrans, y, test_size=0.297, random_state=85)

nTr = np.shape(x_train)[0]
nTe = np.shape(x_test)[0]

train = data_utils.TensorDataset(torch.Tensor(x_train),torch.from_numpy(y_train.as_matrix()).float())
dataloader = data_utils.DataLoader(train, batch_size=nSamp, shuffle=True)


test_set = torch.from_numpy(x_test).float()
train_set = torch.from_numpy(x_train).float()
train_valid = torch.from_numpy(y_train.as_matrix()).float()
test_valid = torch.from_numpy(y_test.as_matrix()).float()

truncSize = int(MaxFvals/capFrac)
TrErrs = np.zeros((truncSize,loops))
TeErrs = np.zeros((truncSize,loops))
TrAccs = np.zeros((truncSize,loops))
TeAccs = np.zeros((truncSize,loops))
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

    def create_model(layer_dims):
        model = torch.nn.Sequential()
        for idx, dim in enumerate(layer_dims):
            if (idx < len(layer_dims) - 1):
                module = torch.nn.Linear(dim, layer_dims[idx + 1])
                model.add_module("linear" + str(idx), module)
            else:
                model.add_module("sig" + str(idx), torch.nn.Sigmoid())

        return model

    ## Create model and hyper parameters
    dim_in = x_train.shape[1]
    dim_out = 1

    layer_dims = [dim_in, dim_out]

    model = create_model(layer_dims)
    #print(model)


    # SAVE OR LOADING NET
    PATH = './seeds/cancer_seed'+str(l)+'.pt'
    if saveSeeds == 1:
        def weights_init(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight.data)
                torch.nn.init.normal_(m.bias.data)
        model.apply(weights_init)
        torch.save(model.state_dict(), PATH)
    if loadSeeds == 1:
        model.load_state_dict(torch.load(PATH))

    loss_fn = torch.nn.BCELoss(reduction='mean')

    ## Now run model
    stateO = {}
    a_old = 1e-8
    fevsOld = 0
    history = { "loss": [], "accuracy": [], "loss_val": [], "accuracy_val": [] }
    loss = None

    def Evaluate(train_set,test_set,model):
        y_tr_pred = model(Variable(train_set))
        LossValTr = loss_fn(y_tr_pred[:,0], Variable(train_valid.float()))
        predictionTr = [1 if x > 0.5 else 0 for x in y_tr_pred.data.numpy()]
        correctTr = (predictionTr == train_valid.numpy()).sum()

        y_te_pred = model(Variable(test_set))
        LossValTe = loss_fn(y_te_pred[:,0], Variable(test_valid.float()))
        predictionTe = [1 if x > 0.5 else 0 for x in y_te_pred.data.numpy()]
        correctTe = (predictionTe == test_valid.numpy()).sum()

        TrLossV = LossValTr.item()/nTr
        TeLossV = LossValTe.item()/nTe
        TrAcc = 100 * correctTr / nTr
        TeAcc = 100 * correctTe / nTe


        if (i % print_int) == 0:
            print("Loss, accuracy, test loss, test acc; at update ", i,": ",TrLossV,
                  TrAcc, TeLossV, TeAcc )

        return TrLossV, TeLossV, TrAcc, TeAcc

    #############################################################
    # TRAINING
    #############################################################

    i = 0
    fevs = 0
    while fevs < (MaxFvals):

        if (i % capFrac) == 0:
            index = int(i/capFrac)
            TrErrs[index,l],TeErrs[index,l],TrAccs[index,l],TeAccs[index,l] = Evaluate(train_set,test_set,model)

        def Closure():
            optimizer.zero_grad()
            minibatch, target = next(iter(dataloader))
            y_pred = model(Variable(minibatch))
            loss = loss_fn(y_pred[:,0], Variable(target.float()))
            loss.backward()
            return loss

        if IGOLSFlag:
            optimizer = PyGOLS(model.parameters(), init_guess=a_old, state=stateO, alg=optMethod)
            loss, stateO = optimizer.step(Closure)
            a = stateO.get('a')
            fevs = stateO.get('func_evals')

        else:
            if optMethod == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=MLR)
            loss = optimizer.step(Closure)
            a = MLR
            fevs = i + 1

        StepS[i,l] = a
        Fvals[i,l] = fevs
        TrLoss[i,l] = loss

        a_old = a*1.0

        fevsOld = fevs

        if i == MaxFvals-1 and fevs < i:
            fevs = MaxFvals + 1

        i += 1


    aveFevs = fevs/(i)
    print('ave. f/evs: '+str(aveFevs))




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
    TrLoss_tmp = TrLoss[:,pli]
    TrLoss_p = TrLoss_tmp[TrLoss_tmp!=0]
    Fvals_tmp = Fvals[:,pli]
    Fvals_p = Fvals_tmp[TrLoss_tmp!=0]

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





