% main NN caller for comparing PrLS and GOLS-I 

close all, clear all, clc;


% choose dataset: 1: Cancer, 2: MNIST, 3: CIFAR10
dataset_no = 1; 

% choose architecture
NET_TYPE = 'NetPI'

% set sample size
N_samp = 50; 

% whether to generate or load an initial guess
randgen = 1;

% set the number of function evaluations
N_FE = 3000;

% interval in which outputs are printed (iterations)
print_int = 10;

% interval in which data is captured (iterations)
capFrac = 10;






more off
%[a, MSGID] = lastwarn()
warning('off')

func = @f_deepNN_var_B;

if strcmp(NET_TYPE,'NetI')
    % NETI
    HLStr = [800];
    actF = 1; % Sigmoid activation function
    cost_func = 3 % BCE
    rand_no_bound = 1e0; % range of the initial guess, centred around 0

elseif strcmp(NET_TYPE,'NetII')
    % NETII
    HLStr = [1000,500,250];
    actF = 2; % Tanh activation function
    cost_func = 1; % MSE
    rand_no_bound = 1e-2; % range of the initial guess, centred around 0

elseif strcmp(NET_TYPE,'LogR')
    HLStr = [];
    actF = 1; % Sigmoid activation function
    cost_func = 3; % BCE
    rand_no_bound = 1e0; % range of the initial guess, centred around 0

elseif strcmp(NET_TYPE,'NetPI')
    HLStr = [32];
    actF = 1;
    cost_func = 3 ;% BCE
    rand_no_bound = 1e0; % range of the initial guess, centred around 0

elseif strcmp(NET_TYPE,'NetPII')
    HLStr = [32];
    actF = 1;
    cost_func = 1; % MSE
    rand_no_bound = 1e0; % range of the initial guess, centred around 0
else
    HLStr = [32];
    actF = 1;
    cost_func = 1; % MSE
    rand_no_bound = 1e0; % range of the initial guess, centred around 0
end    
    

options = optimset;
options.maxiter = N_FE;
options.maxfval = N_FE;
options.maxupdate = N_FE*2;
options.tol = 0;
options.hi = 1; 
options.steplimit = 1e-0;
options.conservativetol = 1e-1;
options.max_step = 1e7;
options.exact = 0;
options.grads = 1;


N_sampT = 1000;
N_test = 10000;


sel_version = 4; % 4
nfac = 1.5; % factor for the ratio between observations and weights
K_fold = 3; % folds for crossvalidation


% calculate variance: eq 17 and 18.
% in batch variance nablaS, do by taking out the last matrix before average
% gradient collapse




if dataset_no == 1
    dataset = load('Cancer2'); 
elseif dataset_no == 2
    dataset = load('MNIST10'); 
elseif dataset_no == 3
    dataset = load('Cifar'); 
end


X = [dataset.training.inputs;dataset.test.inputs;dataset.validation.inputs];
Y = [dataset.training.outputs;dataset.test.outputs;dataset.validation.outputs];
N_total = size(X,1);
inds = 1:N_total; 
X = X(inds,:);
Y = Y(inds,:);

N_test_det = 0;
if dataset_no == 2 || dataset_no == 3
    N_test_det = 1; % 1: N_test manually determined, 2: calcualated by K_fold
end


if N_test_det == 0
    N_test = N_total/(K_fold);
    N_sampT = N_test;
end

if dataset_no == 1 
    N_test = 169;
    N_sampT = N_test;
end

%     N_test = 10000;
N_train = N_total - N_test;
X_train_all = X(inds(1:N_train),:);
Y_train_all = Y(inds(1:N_train),:);
X_test = X(inds(N_train+1:N_train+N_test),:);
Y_test = Y(inds(N_train+1:N_train+N_test),:);


% structure size variables for the neural network
D = dataset.input_count;
K = dataset.output_count;

Str = [D,HLStr,K]

x_len_sum = 0;
for lstr = 1:(length(Str)-1)
    x_len_sum = x_len_sum + ((Str(lstr)+1)*Str(lstr+1));
end



params = {};
params.X_train_all = X_train_all;
params.Y_train_all = Y_train_all;
params.X_test = X_test;
params.Y_test = Y_test;
params.X_valid = X_test;
params.Y_valid = Y_test;

params.Str = Str;
params.K_fold = K_fold;
params.sel_version = sel_version;
params.hold = 0;
params.use = 0;
params.skipVar = 0;
params.actF = actF;
params.N_sampT = N_sampT;
params.capFrac = capFrac;
params.cost_func = cost_func;
params.print_int = print_int;


N_samp_corr = min(N_samp,(N_train))
params.N_samp = N_samp_corr;

tic
name = ['randno_d',num2str(dataset_no),'_B',num2str(N_samp),'.mat'];
if randgen == 1
   X0 = rand_no_bound*randn(x_len_sum,1);
    save(name,'X0')        
else
    load(name)
end


for PrLSvGOLSI = 1:2
    if PrLSvGOLSI == 1
        [Xs,loss_train_p,loss_test_p,loss_valid_p,Acc_tests_p,Acc_train_p,run_count_p,run_alpha_p] = steep_desc_prob_M(func,X0,params,options);
    else
        [Xs,loss_train_g,loss_test_g,loss_valid_g,Acc_tests_g,Acc_train_g,run_count_g,run_alpha_g] = steep_desc_M_upd(func,X0,params,options);
    end
end



% PLOTTING SECTION

Err_train_p = 1 - Acc_train_p;
Err_train_g = 1 - Acc_train_g;
Err_test_p = 1 - Acc_tests_p;
Err_test_g = 1 - Acc_tests_g;

Err_train_p(Err_train_p==1)= 0;
Err_train_g(Err_train_g==1) = 0;
Err_test_p(Err_test_p==1) = 0;
Err_test_g(Err_test_g==1) = 0;

L_tr_p = loss_train_p(loss_train_p~=0);
L_te_p = loss_test_p(loss_test_p~=0);
E_tr_p = Err_train_p(loss_train_p~=0);
E_te_p = Err_test_p(loss_test_p~=0);
R_co_p = run_count_p(run_count_p~=0);
R_al_p = run_alpha_p(run_alpha_p~=0);

L_tr_g = loss_train_g(loss_train_g~=0);
L_te_g = loss_test_g(loss_test_g~=0);
E_tr_g = Err_train_g(loss_train_g~=0);
E_te_g = Err_test_g(loss_test_g~=0);
R_co_g = run_count_g(run_count_g~=0);
R_al_g = run_alpha_g(run_alpha_g~=0);





fontsize = 18;
alph = 0.09;
max_y = 3;
min_y = 0.0;


colours = {'b','r','g','m','y'};
LS = {'PrLS','GOLS-I'};


nugget = 1e-4;
    

figure(1)
hold on;
plot(R_co_p,log10(L_tr_p),colours{1},'LineWidth',2)
plot(R_co_g,log10(L_tr_g),colours{2},'LineWidth',2)
hold off;
legend(LS,'Location', 'Best')
xlabel('Function Evaluations')
ylabel('Training Loss')
set(gca,'FontSize',fontsize)

figure(2)
hold on;
plot(R_co_p,log10(L_te_p),colours{1},'LineWidth',2)
plot(R_co_g,log10(L_te_g),colours{2},'LineWidth',2)
hold off;
legend(LS,'Location', 'Best')
xlabel('Function Evaluations')
ylabel('Test Loss')
set(gca,'FontSize',fontsize)

figure(3)
hold on;
plot(R_co_p,log10(E_tr_p+nugget),colours{1},'LineWidth',2)
plot(R_co_g,log10(E_tr_g+nugget),colours{2},'LineWidth',2)
hold off;
legend(LS,'Location', 'Best')
xlabel('Function Evaluations')
ylabel('Training Accuracy')
set(gca,'FontSize',fontsize)

figure(4)
hold on;
plot(R_co_p,log10(E_te_p+nugget),colours{1},'LineWidth',2)
plot(R_co_g,log10(E_te_g+nugget),colours{2},'LineWidth',2)
hold off;
legend(LS,'Location', 'Best')
xlabel('Function Evaluations')
ylabel('Test Accuracy')
set(gca,'FontSize',fontsize)

figure(5)
hold on;
plot(R_co_p,log10(R_al_p),colours{1},'LineWidth',2)
plot(R_co_g,log10(R_al_g),colours{2},'LineWidth',2)
hold off;
legend(LS,'Location', 'Best')
xlabel('Function Evaluations')
ylabel('Log Step Sizes')
set(gca,'FontSize',fontsize)

