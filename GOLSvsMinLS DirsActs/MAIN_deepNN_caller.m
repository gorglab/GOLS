% main NN caller for different algorithms

close all, clear all, clc;

% DATASETS, choose one
datasetName = 'cancer1'
% datasetName = 'card1'
% datasetName = 'diabetes1'
% datasetName = 'gene1'
% datasetName = 'glass1'
% datasetName = 'heartc1'
% datasetName = 'horse1'
% datasetName = 'mushroom1'
% datasetName = 'Abalone'
% datasetName = 'iris1'
% datasetName = 'forests'
% datasetName = 'FT_clave'
% datasetName = 'Wilt'
% datasetName = 'biodeg'
% datasetName = 'pop_failures'
% datasetName = 'companies_1year'
% datasetName = 'default'
% datasetName = 'Sensorless_drive_diagnosis'
% datasetName = 'HAR'
% datasetName = 'MNIST'

% DATA RELATED PARAMETERS
N_samp = 32; % sample size
sel_version = 2; % 1: full-batch sampling, 2: dynamic mini-batch sub-sampling
K_fold = 3; % Determines the ratio; training data: test data: validation data --- (K_fold-1):1:1


% OPTIMIZATION ALGORITHM: 
opt_method = 'SGD'; 
% opt_method = 'SGDM'; 
% opt_method = 'NAG'; 
% opt_method = 'Adagrad'; 
% opt_method = 'Adadelta'; 
% opt_method = 'Adam'; 

% OPTIMIZATION/TRAINING SETTINGS
options = optimset;
options.maxiter = 1000; % maximum number of iterations
options.exact = 0; % exact or inexact line search methods
options.grads = 1; % use of the gradient only line search methods, -1: small fixed step, -2: medium fixed step, -3: large fixed step
% exact=0, grads=0: Armijo's condition inexact line search
% exact=1, grads=0: Golden Section line search
% exact=0, grads=1: GOLS-I - Gradient-Only Line Search that is Inexact
% exact=1, grads=1: GOLS-B - Gradient-Only Line Search with Bisection
options.max_step = 1e7; % maximum permissible step size


% ARCHITECTURE RELATED PARAMETERS
nHL = 1; % number of hidden layers
randgen = 1; % whether to generate a new random starting point or used the saved point
rand_no_bound = 0.02; % range of the initial guess, centred around 0
nfac = 1.5; % factor for the ratio between observations and weights

% ACTIVATION FUNCTION
actF = 'sigmoid'; 
% actF = 'tanh';
% actF = 'softsign';
% actF = 'relu';
% actF = 'lrelu'; % leaky ReLU
% actF = 'elu';


print_inc = 100; % intervals at which results are printed to screen



func = @f_deepNN; 

% LOADING AND SEPARATING DATA IN TRAINING, VALIDATION AND TEST DATASETS
datLoadName = ['data/',datasetName];
dataset = load(datLoadName); % dataset used

X = [dataset.training.inputs;dataset.test.inputs;dataset.validation.inputs];
Y = [dataset.training.outputs;dataset.test.outputs;dataset.validation.outputs];
N_total = size(X,1);
inds = 1:N_total; 
X = X(inds,:);
Y = Y(inds,:);
N_test = floor(N_total/(K_fold+1));
N_valid = floor(N_total/(K_fold+1));
N_train = N_total- 2*N_test;
X_train_all = X(inds(1:N_train),:);
Y_train_all = Y(inds(1:N_train),:);
X_test = X(inds(N_train+1:N_train+N_test),:);
Y_test = Y(inds(N_train+1:N_train+N_test),:);
X_valid = X(inds(N_train+N_test+1:end),:);
Y_valid = Y(inds(N_train+N_test+1:end),:);

n = N_total - 2*floor(N_total/(K_fold+1));   

% DETERMINE NETWORK ARCHITECTURE
D = dataset.input_count;
K = dataset.output_count;

M1 = ceil((n/nfac - K)/(D+K+1));
M2 = D - 1;
M = min(M1,M2);

Str = zeros(1,nHL+2);
Str(1) = D;
Str(nHL+2) = K;
for m = 1:nHL
    Str(1+m) = M;
end

%Str = [D,M,M,M,K] % manually define Str if desired

% GENERATE INITIAL GUESS FOR NETWORK WEIGHTS
x_len_sum = 0;
for lstr = 1:(length(Str)-1)
    x_len_sum = x_len_sum + ((Str(lstr)+1)*Str(lstr+1));
end

name = ['seeds/seed_',datasetName,'_HL',num2str(nHL),'.mat'];
if randgen == 1
    X0 = -0.5*rand_no_bound + rand_no_bound*rand(x_len_sum,1);
    save(name,'X0')
else
    load(name)
end


% SAVE AND INITIALIZE REQUIRED PARAMETERS
inds = randperm(N_total); 
X = X(inds,:);
Y = Y(inds,:);
params.X_train_all = X_train_all;
params.Y_train_all = Y_train_all;
params.X_test = X_test;
params.Y_test = Y_test;
params.X_valid = X_valid;
params.Y_valid = Y_valid;
params.Str = Str;
params.K_fold = K_fold;
params.N_samp = N_samp;
params.sel_version = sel_version;
params.print_inc = print_inc;
params.actF = actF;


prob_size = size(X_train_all);
disp(['Number of weights: ',num2str(x_len_sum)])
disp(['Number of training observations: ',num2str(prob_size)])

% TRAINING USING DESIRED ALGORITHM
tic
if strcmp(opt_method,'SGD') 
    [x0,fhs,err_tests,err_valids,run_fcount,run_alpha] = steep_desc(func,X0,params,options);
elseif strcmp(opt_method,'SGDM')
    [x0,fhs,err_tests,err_valids,run_fcount,run_alpha] = momentum(func,X0,params,options);
elseif strcmp(opt_method,'NAG')
    [x0,fhs,err_tests,err_valids,run_fcount,run_alpha] = NAG(func,X0,params,options);
elseif strcmp(opt_method,'Adagrad')
    [x0,fhs,err_tests,err_valids,run_fcount,run_alpha] = Adagrad(func,X0,params,options);
elseif strcmp(opt_method,'Adadelta')
    [x0,fhs,err_tests,err_valids,run_fcount,run_alpha] = Adadelta(func,X0,params,options);
elseif strcmp(opt_method,'Adam')
    [x0,fhs,err_tests,err_valids,run_fcount,run_alpha] = Adam(func,X0,params,options);
end
toc



% PLOTTING SECTION

xs = linspace(1,options.maxiter,options.maxiter);
fontsize = 18;

figure(3)
plot(xs,run_fcount,'LineWidth',2)
xlabel('Iterations')
ylabel('Function Evaluations')
set(gca,'FontSize',fontsize)

figure(2)
semilogy(xs,run_alpha,'LineWidth',2)
xlabel('Iterations')
ylabel('Step size')
set(gca,'FontSize',fontsize)

figure(1)    
plot(xs,fhs,xs,err_tests,xs,err_valids,'LineWidth',2)
xlabel('Iterations')
ylabel('Loss')
set(gca,'FontSize',fontsize)
legend('Training error','Test error','Validation error') 



