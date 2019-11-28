#import torch.utils.data
from torch import nn, optim
#from torch.nn import functional as F
import torch
import torch.utils.data as tuda
from Opt_PyGOLS import PyGOLS
import scipy.io as scio
import numpy as np
#import PyTorch_Inits as inits


def GetData(indNo,K_fold,params):

    maparr = np.array([12,5,7,14,19,9,2,1,3,6,17,4,11,16,13,10,8,18,15,20,21,22])
    datasetNo = maparr[indNo-1].astype(int)

    path = './datasets/'
    if datasetNo == 1:
        name = 'cancer1'
    elif datasetNo == 2:
        name = 'card1'
    elif datasetNo == 3:
        name = 'diabetes1'
    elif datasetNo == 4:
        name = 'gene1'
    elif datasetNo == 5:
        name = 'glass1'
    elif datasetNo == 6:
        name = 'heartc1'
    elif datasetNo == 7:
        name = 'horse1'
    elif datasetNo == 8:
        name = 'mushroom1'
    elif datasetNo == 9:
        name = 'soybean1'
    elif datasetNo == 10:
        name = 'thyroid1'
    elif datasetNo == 11:
        name = 'Abalone'
    elif datasetNo == 12:
        name = 'iris1'
    elif datasetNo == 13:
        name = 'companies_1year'
    elif datasetNo == 14:
        name = 'forests'
    elif datasetNo == 15:
        name = 'FT_clave'
    elif datasetNo == 16:
        name = 'Wilt'
    elif datasetNo == 17:
        name = 'biodeg'
    elif datasetNo == 18:
        name = 'HAR'
    elif datasetNo == 19:
        name = 'pop_failures'
    elif datasetNo == 20:
        name = 'default'
    elif datasetNo == 21:
        name = 'Sensorless_drive_diagnosis'
    elif datasetNo == 22:
        name = 'MNIST'
    print('Dataset: '+name)
    dataset = scio.loadmat(path+name)

    X1 = np.vstack((dataset['training']['inputs'][0][0],dataset['test']['inputs'][0][0]))
    X = np.vstack((X1,dataset['validation']['inputs'][0][0]))

    Y1 = np.vstack((dataset['training']['outputs'][0][0],dataset['test']['outputs'][0][0]))
    Y = np.vstack((Y1,dataset['validation']['outputs'][0][0]))

    #if datasetNo < 11:
        #inds = range(N_total)
    #else:
        #inds = np.random.permutation(N_total)

    #dev = params['device']
    nTotal = np.shape(X)[0]
    inds = np.array(range(nTotal)).astype(int)
    X = X[inds]
    Y = Y[inds]
    N_test = int(np.floor(nTotal/(K_fold+1)))
    #N_valid = np.floor(N_total/(K_fold+1))
    N_train = int(nTotal- 2*N_test)
    X_train_all = torch.Tensor(X[inds[0:N_train],:])
    Y_train_all = torch.Tensor(np.float64(Y[inds[0:N_train],:]))
    X_test = torch.Tensor(X[inds[N_train+1:N_train+N_test],:])
    Y_test = torch.Tensor(np.float64(Y[inds[N_train+1:N_train+N_test],:]))
    X_valid = torch.Tensor(X[inds[N_train+N_test+1:nTotal],:])
    Y_valid = torch.Tensor(np.float64(Y[inds[N_train+N_test+1:nTotal],:]))

    params['X_train_all'] = X_train_all
    params['Y_train_all'] = Y_train_all
    params['X_test'] = X_test
    params['Y_test'] = Y_test
    params['X_valid'] = X_valid
    params['Y_valid'] = Y_valid
    params['nTotal'] = nTotal
    params['N_train'] = N_train
    params['N_test'] = N_test
    params['D'] = dataset['input_count'][0,0]
    params['K'] = dataset['output_count'][0,0]
    params['datasetNo'] = datasetNo

    return params



def StrDef(K_fold,nHL,nHNmin,HLRedFac,nfac,MMode,params):

    nTotal = params['nTotal']
    D = params['D']
    K = params['K']

    n = nTotal - 2*np.floor(nTotal/(K_fold+1))

    VarT = n/nfac

    # sum size if each layer decreases by 1
    len_sum = 0
    for ms in range(nHL+2):
        if ms == 0:
            Mo = D*1.0
            Mn = D*1.0-1
        elif ms == nHL+1:
            Mo = Mn*1.0
            Mn = K*1.0
        else:
            Mo = Mn*1.0
            Mn = Mn*1.0-1
        len_sum = len_sum + ((Mo+1)*Mn)

    if MMode == 4:
        M45 = np.ceil(D/2.0)
    if MMode == 5:
        M45 = np.ceil(D*2.0)

    if MMode == 3:
        if VarT > len_sum:
            MModen = 1
        else:
            MModen = 2
    else:
        MModen = MMode

    Str = np.zeros((nHL+2))
    Str[0] = D
    Str[nHL+1] = K

    if MModen == 2:
        Mt = VarT - K
        fac = 1/HLRedFac
        fa = 0
        fb = D + 1
        for h in range(nHL):
            fa = fa*1.0 + fac**(2*h-1)
            fb = fb*1.0 + fac**h
        fb = fb*1.0 + fac**3*K
        sq = np.sqrt(fb**2 + 4*fa*Mt)
        Mp1 = (-fb - sq)/(2.0*fa)
        Mp2 = (-fb + sq)/(2.0*fa)
        MSt = round(max(Mp1,Mp2))

    M1 = np.ceil((n/nfac - K)/(D+K+1))
    M2 = D - 1
    M = min(M1,M2)

    for hl in range(nHL):
        if MModen == 0:
            Str[hl+1] = M
        if MModen == 1:
            Str[hl+1] = D-(hl+1)
        if MModen == 2:
            if hl == 0:
                MAdd = MSt
            else:
                MAdd = round(MAdd/HLRedFac)
            if MAdd < nHNmin:
                MAdd = nHNmin
            Str[hl+1] = MAdd
        if MModen == 4 or MModen == 5:
            Str[hl+1] = M45

    params['Str'] = torch.Tensor(Str)
    return params



def CreateModel(NNdeets):
    class CNet(nn.Module):
        def __init__(self,Str,Acts):
            super(CNet, self).__init__()

            self.lStr = len(Str)
            self.lins = nn.ModuleList()

            for i in range(self.lStr-1):
                self.lins.append(nn.Linear(int(Str[i]), int(Str[i+1])))

            if Acts == 'relu':
                self.act = nn.ReLU()
                print('Act: relu')
            elif Acts == 'elu':
                self.act = nn.ELU()
                print('Act: elu')
            elif Acts == 'leakyrelu':
                self.act = nn.LeakyReLU()
                print('Act: Lrelu')
            elif Acts == 'tanh':
                self.act = nn.Tanh()
                print('Act: tanh')
            elif Acts == 'softsign':
                print('Act: softsign')
                self.act = nn.Softsign()
            else:
                self.act = nn.Sigmoid()
                print('Act: default-sigmoid')



        def forward(self, x):
            for i in range(self.lStr-1):
                mapnow = self.lins[i]
                x = self.act(mapnow(x))
            return x

    Str = NNdeets['Str']
    Acts = NNdeets['Acts']

    model = CNet(Str,Acts)
    return model


def ProbTrainer(criterion,params):

    X_train_all = params['X_train_all']
    Y_train_all = params['Y_train_all']
    X_test = params['X_test']
    Y_test = params['Y_test']
    X_valid = params['X_valid']
    Y_valid = params['Y_valid']
    nSamp = params['nSamp']
    nSampT = params['nSampT']
    prInt = params['prInt']
    optMethod = params['optMethod']
    UpDits = params['UpDits']
    SubSamp = params['SubSamp']
    MLR = params['MLR']
    GOLSFlag = params['GOLSFlag']
    GOLS_var = params['GOLS_var']

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: '+str(dev))

    acts = params['Acts']

    NNdeets = {}
    NNdeets['Str'] = params['Str']
    NNdeets['Acts'] = acts
    model = CreateModel(NNdeets).to(dev)


    #def weights_init(m):
        ##print(m)
        #if isinstance(m, nn.Linear):

            #if acts == 'relu':

                ##inits.kaiming_uniform_(m.weight.data, nonlinearity=acts, bound_mode=ReLUBounds)
                ##fan_in, _ = inits._calculate_fan_in_and_fan_out(m.weight.data)
                ##bound = 1 / np.sqrt(fan_in)
                ##print(BiasB,bound)
                #BiasB = 'p'
                #bound = 0.0001
                #if BiasB == 'p':
                    #aVal = 0
                    #bVal = bound
                #elif BiasB == 'n':
                    #aVal = -bound
                    #bVal = 0
                #else:
                    #aVal = -bound
                    #bVal = bound
                ##nn.init.uniform_(m.bias.data,a=aVal,b=bVal)
                #nn.init.uniform_(m.weight.data,a=aVal,b=bVal)
                #print('inited')

    #model.apply(weights_init)


    dataTr = tuda.TensorDataset(X_train_all,Y_train_all)
    dataVa = tuda.TensorDataset(X_valid,Y_valid)
    dataTe = tuda.TensorDataset(X_test,Y_test)


    threads = torch.get_num_threads()
    if SubSamp == 1:
        trainLoader = torch.utils.data.DataLoader(dataTr,sampler=tuda.sampler.RandomSampler(dataTr),batch_size=nSamp,drop_last=True,pin_memory=True)
        validLoader = torch.utils.data.DataLoader(dataVa,sampler=tuda.sampler.RandomSampler(dataVa),batch_size=nSampT,drop_last=True,pin_memory=True)
        testLoader = torch.utils.data.DataLoader(dataTe,sampler=tuda.sampler.RandomSampler(dataTe),batch_size=nSampT,drop_last=True,pin_memory=True)

    else:
        trainLoader = torch.utils.data.DataLoader(dataTr,sampler=None,batch_size=nSamp,shuffle=True,num_workers=threads,drop_last=True,pin_memory=True)
        validLoader = torch.utils.data.DataLoader(dataVa,sampler=None,batch_size=nSampT,shuffle=True,num_workers=threads,drop_last=True,pin_memory=True)
        testLoader = torch.utils.data.DataLoader(dataTe,sampler=None,batch_size=nSampT,shuffle=True,num_workers=threads,drop_last=True,pin_memory=True)

    #print('init weights begin')
    #for param in model.parameters():
        #print(np.min(np.array(param.data)))
        #print(np.max(np.array(param.data)))
    #print('init weights end')
    #input('pause')


    def NetEval(Loader):
        model.eval()
        vt_loss = 0
        with torch.no_grad():
            for i, (Xvt, Yvt) in enumerate(Loader):
                Xvt = Xvt.to(dev)
                Yvt = Yvt.to(dev)
                Yhvt = model(Xvt)
                vt_loss += criterion(Yhvt, Yvt).item()

        vt_loss /= len(Loader.dataset)
        return vt_loss



    def train(UpDits,a_old,stateO):
        model.train()
        train_loss = 0
        errsTr = torch.zeros((int(UpDits),1))
        errsVa = torch.zeros((int(UpDits),1))
        errsTe = torch.zeros((int(UpDits),1))
        steps = torch.zeros((int(UpDits),1))
        evals = torch.zeros((int(UpDits),1))



        for i in range(UpDits):
            def Closure():
                optimizer.zero_grad()
                Xtr, Ytr = next(iter(trainLoader))
                Xtr = Xtr.to(dev)
                Ytr = Ytr.to(dev)
                Yhat = model(Xtr)
                loss = criterion(Yhat, Ytr)
                loss.backward()
                return loss



            mParams = list(model.parameters())
            if GOLSFlag:
                optimizer = PyGOLS(model.parameters(), init_guess=a_old, state=stateO, LineSearch=GOLS_var, BisectionBracketing='Conservative', alg=optMethod)
                loss, stateO = optimizer.step(Closure)
                a = stateO.get('a')
                fevs = stateO.get('func_evals')

            else:
                if optMethod == 'SGD':
                    optimizer = optim.SGD(model.parameters(), lr=MLR)
                elif optMethod == 'LBFGS':
                    optimizer = optim.LBFGS(mParams,lr=MLR)
                elif optMethod == 'Adagrad':
                    optimizer = optim.Adagrad(mParams,lr=MLR)
                elif optMethod == 'Adam':
                    optimizer = optim.Adam(mParams,lr=MLR)
                a = MLR
                fevs = i + 1
                loss = optimizer.step(Closure)




            steps[i,0] = a
            evals[i,0] = fevs
            errsTr[i,0] = loss / (nSamp)
            errsVa[i,0] = NetEval(validLoader)
            errsTe[i,0] = NetEval(testLoader)
            a_old = a*1.0
            train_loss += loss

            if (i+1) % prInt == 0:
                print('Train it: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i+1, (i+1) * (nSamp), nSamp*UpDits,
                    100. * (i+1) / UpDits,
                    train_loss / (nSamp*i)))


        return errsTr, errsVa, errsTe, steps, evals




    a_old = 1e-8
    stateO = {}

    glErrTr,glErrVa,glErrTe,glSteps,glEvals = train(UpDits,a_old,stateO)

    #print('init weights begin')
    #for param in model.parameters():
        #print(np.min(np.array(param.data)))
        #print(np.max(np.array(param.data)))
    #print('init weights end')
    #input('pause')

    return glErrTr,glErrVa,glErrTe,glSteps,glEvals
