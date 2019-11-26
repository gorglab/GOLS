import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np



ManLRs = (1e-5,1e-4,1e-3) # manually set learning rates for comparison
nEpoch = 1
trainBatchS = 128
loops = 1 + len(ManLRs)
alg = 'SGD'
capFrac = 1

NTot = 50000
runS = np.ceil(NTot/trainBatchS).astype(int)
gSize = int((runS)*nEpoch)

TrErrs = np.zeros((gSize,loops))
TeErrs = np.zeros((gSize,loops))
TrAccs = np.zeros((gSize,loops))
TeAccs = np.zeros((gSize,loops))
StepTrunc = np.zeros((gSize,loops))
StepS = np.zeros((gSize,loops))
Fvals = np.zeros((gSize,loops))


for l in range(loops):
    name = 'CIFAR10_loop'+str(l)+'_Eps'+str(nEpoch)+'_N'+str(trainBatchS)+'.mat'
    data = scio.loadmat(name)
    TrErrs[:,l] = data['dataTr'][:,0]
    TeErrs[:,l] = data['dataTe'][:,0]
    StepS[:,l] = data['dataSt'][:,0]
    Fvals[:,l] = data['dataFv'][:,0]
    TrAccs[:,l] = data['dataAccTr'][:,0]
    TeAccs[:,l] = data['dataAccTe'][:,0]



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


# XXX reduce redundency in plotting (especially the function evals)




plt.figure(2)
for pli in range(loops):
    #Fvals_tmp = Fvals[0::capFrac,pli]
    #Fvals_p = Fvals_tmp[Fvals_tmp!=0]
    #TrErrs_tmp = TrErrs[:,pli]
    #TrErrs_p = TrErrs_tmp[TrErrs_tmp!=0]

    plt.plot(Fvals[:,pli],np.log10(TrErrs[:,pli]),colours[pli],label=labels[pli], linewidth=lwth)
plt.xlabel('Function Evaluations')
plt.ylabel('Large Batch Training Loss')
plt.legend()


plt.figure(3)
for pli in range(loops):
    Fvals_tmp = Fvals[0::capFrac,pli]
    Fvals_p = Fvals_tmp[Fvals_tmp!=0]
    TeErrs_tmp = TeErrs[:,pli]
    TeErrs_p = TeErrs_tmp[TeErrs_tmp!=0]

    plt.plot(Fvals[:,pli],np.log10(TeErrs[:,pli]),colours[pli],label=labels[pli], linewidth=lwth)
plt.xlabel('Function Evaluations')
plt.ylabel('Large Batch Test Loss')
plt.legend()


plt.figure(4)
for pli in range(loops):
    Fvals_tmp = Fvals[0::capFrac,pli]
    Fvals_p = Fvals_tmp[Fvals_tmp!=0]
    TrAccs_tmp = TrAccs[:,pli]
    TrAccs_p = TrAccs_tmp[TrAccs_tmp!=0]

    plt.plot(Fvals[:,pli],(TrAccs[:,pli]),colours[pli],label=labels[pli], linewidth=lwth)
plt.xlabel('Function Evaluations')
plt.ylabel('Large Batch Training Accuracy')
plt.legend()


plt.figure(5)
for pli in range(loops):
    Fvals_tmp = Fvals[0::capFrac,pli]
    Fvals_p = Fvals_tmp[Fvals_tmp!=0]
    TeAccs_tmp = TeAccs[:,pli]
    TeAccs_p = TeAccs_tmp[TeAccs_tmp!=0]

    plt.plot(Fvals[:,pli],(TeAccs[:,pli]),colours[pli],label=labels[pli], linewidth=lwth)
plt.xlabel('Function Evaluations')
plt.ylabel('Large Batch Test Accuracy')
plt.legend()


plt.figure(6)
for pli in range(loops):
    Fvals_tmp = Fvals[:,pli]
    Fvals_p = Fvals_tmp[Fvals_tmp!=0]
    StepS_tmp = StepS[:,pli]
    StepS_p = StepS_tmp[StepS_tmp!=0]

    plt.plot(Fvals[:,pli],np.log10(StepS[:,pli]),colours[pli],label=labels[pli], linewidth=lwth)
plt.xlabel('Function Evaluations')
plt.ylabel('Log Step Size')
plt.legend()


plt.show()


