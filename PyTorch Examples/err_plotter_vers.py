import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np


plot_vers = 3 # 1: consts, 2: ball, 3: stages
probno = 4
Pman = 0.9
B = 0
fv_plot = 0

alg = 'SGD'
#alg = 'LBFGS'
#alg = 'Adagrad'
#alg = 'Adam'

#vers = ('_Bac_','_Bac2_','_x0_','_orig_','_origQ_','_agrr_','')
#vlab = ('BackT-SC','BackT-Ov','Exp init','I-GOLS','I-GOLSQ','GOLS-Iagrr','GOLS-I')

#vers = ('_Bac_','_Bac2_','_x0_','_orig_','_o_','_c_','')
#vlab = ('BackT-SC','BackT-Ov','init','I-GOLS','o','c','GOLS-I')

#vers = ('_Bac_','_x0_','_orig_','_agrr_','')
#vlab = ('BackT-SC','Exp init','I-GOLS','GOLS-Iagrr','GOLS-I')

#vers = ('_32_',)
#vlab = ('64',)


#vers = ('_o_','_Bac_','_Bac2_','_c_','','_origQ_','_g_','_53vO_','_53vO5_')
#vlab = ('Overshoot','Backtrack','BacktrackO','Conservative','GOLS-I','OrigQ','Growing','2stage-2','2stage-5')

if plot_vers == 1:
    vers = ('_o_','_H_','_M_','_L_','')
    vlab = ('GOLS-MAX','Const. Step $10^{-3}$','Const. Step $10^{-4}$','Const. Step $10^{-5}$','GOLS-I')
elif plot_vers == 2:
    vers = ('_o_','_golsip0_','_Bac_','')
    vlab = ('GOLS-MAX','GOLS-I $c_2=0$','GOLS-Back','GOLS-I')
elif plot_vers == 3:
    vers = ('_o_','','_53vO2_','_53vO5_')
    vlab = ('GOLS-MAX','GOLS-I','M2I-20','M2I-50')


if plot_vers == 1:
    colours = ('b','k','m','c','r','m','y','g','c')
elif plot_vers == 2:
    colours = ('b','k','g','r','c','m','y','g','c')
elif plot_vers == 3:
    colours = ('b','r','g','c','c','m','y','g','c')


lv = len(vers)

batchS = 128
NTot = 60000
Eps = 10
runS = np.ceil(NTot/batchS).astype(int)
gSize = int((runS)*Eps)

errs = np.zeros((gSize,lv))
steps = np.zeros((gSize,lv))
fvs = np.zeros((gSize,lv))
vcount = 0

for ver in vers:
    if ver == '_orig_' or ver == '_origQ_' or ver == '_golsip0_':
        overP = 0.0
    else:
        overP = Pman
    if B == 0:
        name = 'err_dataFv'+ver+alg+str(probno)+'_p'+str(overP)+'.mat'
    else:
        name = 'err_dataFv'+ver+alg+str(probno)+'B_p'+str(overP)+'.mat'
    print(name)
    data = scio.loadmat(name)
    errs[:,vcount] = data['err_train'][:,0]
    steps[:,vcount] = data['steps'][:,0]
    if fv_plot == 1:
        fvs[:,vcount] = data['fvals'][:,0]
    else:
        fvs[:,vcount] = np.linspace(1,gSize,gSize)

    vcount += 1




#font = {'size'   : 23}
font = {'size'   : 12}

plt.rc('font', **font)
#lwth = 3.8
lwth = 1


plt.figure(1)
vcount = 0
for ver in vers:
    plt.plot(fvs[:,vcount],(errs[:,vcount]),colours[vcount],label=vlab[vcount], linewidth=lwth)
    vcount += 1
plt.ylim((100,300))
#plt.xlim((0,10))
#plt.xticks(range(1,20))
plt.yticks([100,150,200,250,300])
if fv_plot == 1:
    plt.xlabel('Function Evaluations')
else:
    plt.xlabel('Iterations')

plt.ylabel('Training Loss')
plt.legend()

print(fvs)
print(errs)
print(steps)



plt.figure(2)
vcount = 0
for ver in vers:
    print(ver)
    print(vers[vcount])
    plt.semilogy(fvs[:,vcount],steps[:,vcount],colours[vcount],label=vlab[vcount], linewidth=lwth)
    vcount += 1

plt.legend()

if fv_plot == 1:
    plt.xlabel('Function Evaluations')
else:
    plt.xlabel('Iterations')
plt.ylabel('Step size')
#plt.xlim((0,10))

plt.show()
