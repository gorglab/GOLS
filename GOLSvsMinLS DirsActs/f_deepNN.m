function [err_train,err_test,err_valid,dJdW] = f_deepNN(x,params)
% function evaluation for natural methods, feedforward step, gradient and cost
% function evaluation.

Str = params.Str;
%K_fold = params.K_fold;
N_samp = params.N_samp;
sel_version = params.sel_version;

if sel_version == 1
    X_train = params.X_train_all;
    Y_train = params.Y_train_all;
    X_test = params.X_test;
    Y_test = params.Y_test;
    X_valid = params.X_valid;
    Y_valid = params.Y_valid;
elseif sel_version == 2
    X_train_all = params.X_train_all;
    Y_train_all = params.Y_train_all;
    X_test = params.X_test;
    Y_test = params.Y_test;
    X_valid = params.X_valid;
    Y_valid = params.Y_valid;
    N_train = size(X_train_all,1);
    inds2 = randperm(N_train);
    X_train = X_train_all(inds2(1:N_samp),:);
    Y_train = Y_train_all(inds2(1:N_samp),:);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_layers = length(Str);
sum = 0;
Ws = cell(1, N_layers - 1);
for i = 1:(N_layers - 1)
    Ws{i} = reshape(x(sum+1 : sum+((Str(i)+1)*Str(i+1))),[Str(i)+1,Str(i+1)]);
    sum = sum + ((Str(i)+1)*Str(i+1));
end

zs = cell(1, N_layers - 1);
as = cell(1, N_layers - 1);
actF = params.actF;

[yhat_train,~,~,~] = feedforward_deep(X_train,Ws,zs,as,N_layers,actF);
err_train = costfunc_pr(Y_train,yhat_train);
%disp('train')
[yhat_test,~,~,~] = feedforward_deep(X_test,Ws,zs,as,N_layers,actF);
err_test = costfunc_pr(Y_test,yhat_test);
%disp('test')
[yhat_valid,~,~,~] = feedforward_deep(X_valid,Ws,zs,as,N_layers,actF);
err_valid = costfunc_pr(Y_valid,yhat_valid);
%disp('valid')

[yhat,Ws,zs,as] = feedforward_deep(X_train,Ws,zs,as,N_layers,actF);
[dJdW] = backprop_deep(X_train,Y_train,yhat,Ws,Str,zs,as,actF);
%disp('tr grad')


costfunc_fac = (2*100)/(size(Y_train,1)*(size(Y_train,2)));
dJdW = dJdW*costfunc_fac;

%dJdW = dJdW';
