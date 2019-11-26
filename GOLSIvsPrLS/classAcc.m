function [Acc] = classAcc(x,params)
% function evaluation for natural methods, feedforward step, gradient and cost
% function evaluation.

Str = params.Str;

actF = params.actF;

%K_fold = params.K_fold;
N_samp = params.N_samp;
sel_version = params.sel_version;

if sel_version == 1
    trteva = params.trteva;
    if trteva == 1
        X_train = params.X_train_all;
        Y_train = params.Y_train_all;
    elseif trteva == 2
        X_train = params.X_test;
        Y_train = params.Y_test;
    elseif trteva == 3
        X_train = params.X_valid;
        Y_train = params.Y_valid;
    end
    
elseif sel_version == 4
    trteva = params.trteva;
    if trteva == 1
        X_train_all = params.X_train_all;
        Y_train_all = params.Y_train_all;
        N_train = size(X_train_all,1);
        inds2 = randperm(N_train);
        X_train = X_train_all(inds2(1:N_samp),:);
        Y_train = Y_train_all(inds2(1:N_samp),:);
    elseif trteva == 2
        X_train = params.X_test;
        Y_train = params.Y_test;
        N_sampT = params.N_sampT;        
        N_train = size(X_train,1);
        inds2 = randperm(N_train);
        X_train = X_train(inds2(1:N_sampT),:);
        Y_train = Y_train(inds2(1:N_sampT),:);
    elseif trteva == 3
        X_train_all = params.X_train_all;
        Y_train_all = params.Y_train_all;
        N_sampT = params.N_sampT;        
        N_train = size(X_train_all,1);
        inds2 = randperm(N_train);
        X_train = X_train_all(inds2(1:N_sampT),:);
        Y_train = Y_train_all(inds2(1:N_sampT),:);
    end

end

[B,dims] = size(Y_train);

%costfunc_fac = (2*100)/(B*(size(Y_train,2)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_layers = length(Str);
sumStr = 0;
Ws = cell(1, N_layers - 1);
for i = 1:(N_layers - 1)
    Ws{i} = reshape(x(sumStr+1 : sumStr+((Str(i)+1)*Str(i+1))),[Str(i)+1,Str(i+1)]);
    sumStr = sumStr + ((Str(i)+1)*Str(i+1));
end



%%%% old version
zs = cell(1, N_layers - 1);
as = cell(1, N_layers - 1);

[yhat0,~,~,~] = feedforward_deep(X_train,Ws,zs,as,N_layers,actF);

[~,yhat] = max(yhat0,[],2);
[~,Y_t] = max(Y_train,[],2);
equ = yhat==Y_t;
Acc = sum(equ)/(B);

cl_net = zeros(dims,1);
cl_targ = zeros(dims,1);
for i = 1:dims
    cl_net(i) = sum(yhat==i);
    cl_targ(i) = sum(Y_t==i);
end
%cl_ratio = cl_net./cl_targ

%disp('yes')


%yhat = round(yhat0);
%Acc = 1 - sum(sum(abs(yhat-Y_train)))/(B*dims);
 
% yhat0(1,:)
% yhat(1,:)
% Y_train(1,:)
% 1 - sum(sum(abs(yhat(1,:)-Y_train(1,:))))/(dims) 
% pause


% variance version1        
% zs = cell(1, N_layers - 1);
% as = cell(1, N_layers - 1);
% 
% x_s = size(x,1);
% 
% dJdW_all = zeros(x_s,B);
% f_all = zeros(B,1);
% 
% for v = 1:B
%     X_in = X_train(v,:);
%     Y_in = Y_train(v,:);
%     
%     [yhat2,Ws,zs2,as2] = feedforward_deep(X_in,Ws,zs,as,N_layers);
%     f_all(v) = costfunc_pr(Y_in,yhat2);
%     [dJdW] = backprop_deep(X_in,Y_in,yhat2,Ws,Str,zs2,as2);
%     dJdW2 = dJdW'*costfunc_fac*B;
%     
%     dJdW_all(:,v) = dJdW2;
% end
% 
% S = mean(f_all.^2); 
% f = mean(f_all);
% fsig = 1/(B - 1)*(S-f^2);
% 
% %dJdW_sc = dJdW_all*(costfunc_fac*N_samp);
% dJdW_sc = dJdW_all;
% dJdW_sum = mean(dJdW_sc,2);
% dJdWS = 1/(N_samp-1)*(mean(dJdW_sc.^2,2)-dJdW_sum.^2);



%[f,dJdW_sum,fsig,dJdWS] = feedforward_backprop(x,X_train,Y_train,Str,actF);

% B
% size(dJdW0)
% size(dJdW1)

% dJdW0
% norm(as{1}-as{1})
% norm(zs{1}-zs{1})

% gDiff = dJdW0-dJdW_sum;
% gDiv = dJdW0./dJdW_sum
% gSDiv = dJdWS0./dJdWS
% fSDiv = fsig0./fsig
% 
% norm(gDiff)
% f
% f0/f
% 
% pause



% if norm(dJdWS) < 1e-7
%     dJdWS = dJdWS + 1e-7*rand(size(x));
%     disp('low g var')
% end
% if fsig < 1e-7
%     fsig = fsig + 1e-7*rand();
%     disp('low f var')
% end

% dJdWS = 1e-6*rand(size(x));
% fsig = 1e-6*rand();

%dJdWS = dJdWS';

% trteva
% vf
% norm(vdf)

%dJdW = dJdW';
