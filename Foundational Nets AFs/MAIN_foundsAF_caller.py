# main NN caller for different algorithms

# Python libs
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

# PyTorch libs
from torch import nn
import torch

# custom libs
from utils_functions import ProbTrainer
from utils_functions import GetData
from utils_functions import StrDef


###############################################################################
# CODE PARAMETERS
###############################################################################

params = {}

minIts = 3e3      # Minimum number of iterations (for fair comparison between datasets)
epochsManual = 10 # Manually set the number of epochs (for this mode of comparison)
itsOn = 1         # switch between 1: its or 0: epochs


# DATASET
# ranked from smallest (1, Iris) to largest (22, MNIST)
dataset = 1

# SAMPLE SIZE
nSampM = 32 # manually set sample size

manLRs = (1e-1,1e-0,1e1)

# OPTIMIZATION ALGORITHM:
# options: 'SGD', 'LBFGS', 'Adagrad', 'Adam'
opt = 'SGD'

# GOLS ALGORITHM:
# Boolean to activate GOLS (True) or use a manually determined learning rate (False)
params['GOLSFlag'] = True
# options: 'Bisection', 'Inexact', 'Back', 'Max'
params['GOLS_var'] = 'Inexact'

# MANUAL LEARNING RATE PARAMETER
params['MLR'] = 0.005


# ACTIVATION FUNCS
acts = ('sigmoid', 'tanh', 'softsign', 'relu', 'leakyrelu', 'elu')


# TORCH PARAMS
criterion = nn.MSELoss()
params['SubSamp'] = 1 # 0: Epoch training with random shuffle, 1: Random subsampling


# OTHER
params['max_step'] = 1e7
params['prInt'] = 100 # print interval
randgen = 1           # whether to generate a new random starting point or used the saved point
rand_no_bound = 0.2   # range of the initial guess, centred around 0
K_fold = 3            # Determines the ratio; training data : test/validation data
nfac = 1.5            # factor for the ratio between observations and weights
nHL = 1               # Number of hidden layers
nHNmin = 3            # Minimum number of hidden nodes
HLRedFac = 1.5        # taper factor between successive layers
MMode = 0             # 0: old school-no Taper, 1: M=D-1, 2: n/nfac (with taper factor), 3: min of 1 and 2, 4: M=D/2, 5: M=D*2

print('number of threads: '+str(torch.get_num_threads()))

HLC = (nHL+1)*2

params['optMethod'] = opt

nActs = len(acts)
dataTr = np.zeros((int(minIts),nActs))
dataVa = np.zeros((int(minIts),nActs))
dataTe = np.zeros((int(minIts),nActs))
dataSt = np.zeros((int(minIts),nActs))
dataFe = np.zeros((int(minIts),nActs))



###############################################################################
# DATA PROCESSING
###############################################################################
params = GetData(dataset,K_fold,params)
datasetNo = params['datasetNo']

nTrain = params['N_train']
if nSampM > nTrain:
    nSamp = int(nTrain*1.0)
else:
    nSamp = nSampM
print('Samp size: '+str(nSamp))
params['nSamp'] = nSamp

nTest = params['N_test']
if nSampM > nTest:
    nSampT = int(nTest*1.0-1)
else:
    nSampT = nSampM
print('Test Samp size: '+str(nSampT))
params['nSampT'] = nSampT

###############################################################################
# DETERMINE NN STRUCTURE
###############################################################################
params = StrDef(K_fold,nHL,nHNmin,HLRedFac,nfac,MMode,params)
print('NN Structure')
print(params['Str'])
Str = params['Str']

actCnt = 0
for act in acts:
    params['Acts'] = act

    x_len_sum = 0
    for lstr in range(len(Str)-1):
        x_len_sum = x_len_sum + ((Str[lstr]+1)*Str[lstr+1])


    ###############################################################################
    # INFO
    ###############################################################################
    #	print((params['X_train_all']).size()[0])
    prob_size = round((params['X_train_all']).size()[0])
    print('Number of weights: '+str(x_len_sum))
    print('Number of training observations: '+str(prob_size))


    ###############################################################################
    # TRAINING AND TESTING
    ###############################################################################
    if itsOn == 1:
        params['UpDits'] = int(minIts)
    else:
        params['UpDits'] = int((nTrain//nSamp)*epochsManual)


    err_train,err_tests,err_valids,steps,evals = ProbTrainer(criterion,params)

    dataTr[0:int(minIts),actCnt] = err_train.detach().numpy()[0:int(minIts),0]
    dataVa[0:int(minIts),actCnt] = err_tests[0:int(minIts),0]
    dataTe[0:int(minIts),actCnt] = err_valids[0:int(minIts),0]
    dataSt[0:int(minIts),actCnt] = steps[0:int(minIts),0]
    dataFe[0:int(minIts),actCnt] = evals[0:int(minIts),0]


    actCnt += 1



# PLOTTING


font = {'size'   : 12}
plt.rc('font', **font)
lwth = 1

colours = ('k','c','b','r','g','m')
labels = acts

plot_its_fe = 0

xs_its = np.zeros(np.shape(dataFe))
for a in range(nActs):
    xs_its[:,a] = np.linspace(1,minIts,minIts)

if plot_its_fe == 0:
    xs = xs_its
else:
    xs = dataFe

plt.figure(1)
for pli in range(nActs):
    plt.plot(xs[:,pli],np.log10(dataTr[:,pli]),colours[pli],label=labels[pli], linewidth=lwth)
if plot_its_fe == 0:
    plt.xlabel('Iterations')
else:
    plt.xlabel('Function Evaluations')

plt.ylabel('Training Loss')
plt.legend()


plt.figure(2)
for pli in range(nActs):
    plt.plot(xs[:,pli],np.log10(dataVa[:,pli]),colours[pli],label=labels[pli], linewidth=lwth)
if plot_its_fe == 0:
    plt.xlabel('Iterations')
else:
    plt.xlabel('Function Evaluations')
plt.ylabel('Validation Loss')
plt.legend()


plt.figure(3)
for pli in range(nActs):
    plt.plot(xs[:,pli],np.log10(dataTe[:,pli]),colours[pli],label=labels[pli], linewidth=lwth)
if plot_its_fe == 0:
    plt.xlabel('Iterations')
else:
    plt.xlabel('Function Evaluations')
plt.ylabel('Test Loss')
plt.legend()



plt.figure(5)
for pli in range(nActs):
    plt.plot(xs_its[:,pli],np.log10(dataSt[:,pli]),colours[pli],label=labels[pli], linewidth=lwth)
plt.xlabel('Iterations')
plt.ylabel('Log Step Sizes')
plt.legend()


plt.figure(6)
for pli in range(nActs):
    plt.plot(xs_its[:,pli],(dataFe[:,pli]),colours[pli],label=labels[pli], linewidth=lwth)
plt.xlabel('Iterations')
plt.ylabel('Cumulative Function Evaluations')
plt.legend()


plt.show()
