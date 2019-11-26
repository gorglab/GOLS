function [dJdW] = backprop_deep(X,Y,yhat,Ws,Str,zs,as,actF)
% back propagation step to find derivatives


if strcmp(actF,'tanh')
    df_act = @dtanh_act;
elseif strcmp(actF,'softsign')
    df_act = @dsoftsign;
elseif strcmp(actF,'relu')
    df_act = @dReLU;
elseif strcmp(actF,'lrelu')
    df_act = @dleakyReLU;
elseif strcmp(actF,'elu')
    df_act = @dELU;
else 
    df_act = @dsigmoid;
end


N_layers = length(Str);

diffs = (yhat-Y);    
fac = diffs;
dJdWs = cell(1, N_layers - 1);

for i = 1:(N_layers-1)
    j = (N_layers) - i;
    if i == 1
        del = fac.*df_act(zs{j});
    else
        del = del*Ws{j+1}(1:end-1,1:end)'.*df_act(zs{j});
    end
    if i ~= (N_layers-1)
        dJdWs{i} = as{j-1}'*del;
    else
        dJdWs{i} = [X,ones(size(X,1),1)]'*del;
    end
end
    
dJdW2 = cell(1,N_layers-1);

for k = 1:(N_layers - 1)
    l = N_layers - k;  
    dJdW_v = reshape(dJdWs{l},[(Str(k)+1)*Str(k+1),1]);
    dJdW2{k} = dJdW_v';
end
dJdW = cell2mat(dJdW2);
%dJdW = dJdWf;

