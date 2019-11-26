# Original from:
# https://github.com/pytorch/examples/blob/master/vae/main.py


from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt


from Opt_PyGOLS import PyGOLS

ManLRs = (1e-4,1e-3,1e-2) # manually set learning rates for comparison
batchS = 128 # mini-batch size
Eps = 2 # number of Epochs
# algorithm = 'SGD' # chosen algorithm
algorithm = 'Adagrad' # chosen algorithm


NTot = 60000
loops = 1 + len(ManLRs)
runS = np.ceil(NTot/batchS).astype(int)

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=batchS, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=Eps, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

print('number of threads: '+str(torch.get_num_threads()))


gSize = (runS)*Eps
incS = int(runS)

TrLoss = np.zeros((int(gSize),loops))
StepS = np.zeros((int(gSize),loops))
Fvals = np.zeros((int(gSize),loops))


for l in range(loops):
    print('loop: '+str(l))

    if l == 0:
        IGOLSFlag = True
        MLR = None
    elif l > 0:
        IGOLSFlag = False
        MLR = ManLRs[l-1]



    print('==> Building model..')
    class VAE(nn.Module):
        def __init__(self):
            super(VAE, self).__init__()

            self.fc1 = nn.Linear(784, 400)
            self.fc21 = nn.Linear(400, 20)
            self.fc22 = nn.Linear(400, 20)
            self.fc3 = nn.Linear(20, 400)
            self.fc4 = nn.Linear(400, 784)

        def encode(self, x):
            h1 = torch.sigmoid(self.fc1(x))
            return self.fc21(h1), self.fc22(h1)

        def reparameterize(self, mu, logvar):
            if self.training:
                std = torch.exp(0.5*logvar)
                eps = torch.randn_like(std)
                return eps.mul(std).add_(mu)
            else:
                return mu

        def decode(self, z):
            h3 = torch.sigmoid(self.fc3(z))
            return torch.sigmoid(self.fc4(h3))

        def forward(self, x):
            mu, logvar = self.encode(x.view(-1, 784))
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar


    model = VAE().to(device)


    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


    def train(epoch,a_old,stateO):
        model.train()
        train_loss = 0
        lerrs = np.zeros((int(runS),1))
        steps = np.zeros((int(runS),1))
        fvals = np.zeros((int(runS),1))
        cnt = 0


        for i in range(runS):
            def Closure():
                optimizer.zero_grad()
                data, _ = next(iter(train_loader))
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                loss = loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                return loss


            if IGOLSFlag == True:
                optimizer = PyGOLS(model.parameters(), init_guess=a_old, state=stateO, alg=algorithm)
                loss, stateO = optimizer.step(Closure)
                a = stateO.get('a')
                fevs = stateO.get('func_evals')
            else:
                if algorithm == 'SGD':
                    optimizer = optim.SGD(model.parameters(), lr=MLR)
                elif algorithm == 'Adagrad':
                    optimizer = optim.Adagrad(model.parameters(), lr=MLR)

                loss = optimizer.step(Closure)
                a = MLR
                fevs = (epoch-1)*runS+cnt+1

            steps[cnt,0] = a
            fvals[cnt,0] = fevs
            a_old = a*1.0

            optimizer.zero_grad()
            data, _ = next(iter(train_loader))
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()

            lerrs[cnt,0] = loss.item()/len(data)
            cnt += 1


            if i % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), len(train_loader.dataset),
                    100. * i / len(train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))

        return lerrs, steps, fvals


    def test(epoch):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                test_loss += loss_function(recon_batch, data, mu, logvar).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                          recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                             'results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))




    a_old = 1e-8
    stateO = {}

    for epoch in range(1, args.epochs + 1):

        lerr, steps, fvals = train(epoch,a_old,stateO)
        ep = epoch-1
        TrLoss[ep*incS:(ep+1)*incS,l] = lerr[:,0]
        StepS[ep*incS:(ep+1)*incS,l] = steps[:,0]
        Fvals[ep*incS:(ep+1)*incS,l] = fvals[:,0]
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')





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
    Fvals_tmp = Fvals[:,pli]
    Fvals_p = Fvals_tmp[Fvals_tmp!=0]
    StepS_tmp = StepS[:,pli]
    StepS_p = StepS_tmp[StepS_tmp!=0]

    plt.plot(Fvals_p,np.log10(StepS_p),colours[pli],label=labels[pli], linewidth=lwth)
plt.xlabel('Function Evaluations')
plt.ylabel('Log Step Size')
plt.legend()


plt.show()




