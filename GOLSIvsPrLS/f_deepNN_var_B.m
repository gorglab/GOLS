function [f,dJdW_sum,fsig,dJdWS] = f_deepNN_var_B(x,params)
% function evaluation for natural methods, feedforward step, gradient and cost
% function evaluation.

Str = params.Str;
N_samp = params.N_samp;
sel_version = params.sel_version;
actF = params.actF;
cost_func = params.cost_func;

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
        X_train = params.X_valid;
        Y_train = params.Y_valid;
        N_sampT = params.N_sampT;
        N_train = size(X_train,1);
        inds2 = randperm(N_train);
        X_train = X_train(inds2(1:N_sampT),:);
        Y_train = Y_train(inds2(1:N_sampT),:);
    end

end

%[f0,dJdW_sum0,fsig0,dJdWS0] = feedforward_backprop(x,X_train,Y_train,Str,actF);
%[f,dJdW_sum,fsig,dJdWS] = forback_cofu(x,X_train,Y_train,Str,actF,cost_func);


[f,dJdW_sum,fsig,dJdWS] = forback_cofu_mem(x,X_train,Y_train,Str,actF,cost_func);


% f-f2
% fsig - fsig2
% norm(dJdW_sum2 - dJdW_sum)
% norm(dJdWS2 - dJdWS)
% 
% 
% pause

% df = f0-f
% dJ = norm(dJdW_sum0-dJdW_sum0)
% dsf = fsig0-fsig
% dSd = norm(dJdWS0-dJdWS0)


% mag = 1e1;
% fsig = rand()*mag;
% dJdWS = mag*rand(size(dJdWS));

% 
% tlen = 10;
% grad = zeros(tlen,1);
% %[f,dJdW_sum,fsig,dJdWS] = feedforward_backprop(x,X_train,Y_train,Str,actF);
% 
% h = 1j*1e-8;
% 
% for i = 1:tlen
%     x_mod = x;
%     x_mod(i) = x_mod(i) + h;
%     [fi,djdw2,~,~] = forback_cofu(x_mod,X_train,Y_train,Str,actF,cost_func);
%     grad(i) = imag(fi)/imag(h);
% end
% 
% grad
% 
% dJdW_sum(1:tlen)
% 
% norm(grad-dJdW_sum(1:tlen))
% 
% grad2 = zeros(tlen,1);
% hr = 1e-6;
% for i = 1:tlen
%     x_mod = x;
%     x_mod(i) = x_mod(i) + hr;
%     %x_mod(i)
%     [fi,djdw2,~,~] = forback_cofu(x_mod,X_train,Y_train,Str,actF,cost_func);
%     grad2(i) = (fi-f)/(hr);
% end
% (grad2)
% norm(grad-grad2)
% norm(grad2-dJdW_sum(1:tlen))



%length(x)


%norm(real(djdw2(1:tlen))-grad)

% norm(real(djdw2(1:tlen))-dJdW_sum(1:tlen))
% pause