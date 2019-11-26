function [f,dJdW_sum,fsig,dJdWS] = feedforward_backprop(x,X,Y,Str,actF)

if actF == 2
    f_act = @tanh_act;
elseif actF == 3
    f_act = @softsign;
elseif actF == 4
    f_act = @ReLU;
elseif actF == 5
    f_act = @leakyReLU;
elseif actF == 6
    f_act = @ELU;
else 
    f_act = @sigmoid;
end

if actF == 2
    df_act = @dtanh_act;
elseif actF == 3
    df_act = @dsoftsign;
elseif actF == 4
    df_act = @dReLU;
elseif actF == 5
    df_act = @dleakyReLU;
elseif actF == 6
    df_act = @dELU;
else 
    df_act = @dsigmoid;
end



B = size(X,1);
f_all = zeros(B,1);

N_layers = length(Str);
sumStr = 0;
Ws = cell(1, N_layers - 1);
for i = 1:(N_layers - 1)
    Ws{i} = reshape(x(sumStr+1 : sumStr+((Str(i)+1)*Str(i+1))),[Str(i)+1,Str(i+1)]);
    sumStr = sumStr + ((Str(i)+1)*Str(i+1));
end

dJdW_all = zeros(sumStr,B);

costfunc_fac = (2*100)/(B*(size(Y,2)));

zs = cell(1, N_layers - 1);
as = cell(1, N_layers - 1);

for b = 1:B
    
    for j = 1:(N_layers - 1)
        if j == 1
            X_bias = [X(b,:),1];
            zs{j} = X_bias*Ws{j};
        else            
            zs{j} = as{j-1}*Ws{j};
        end

        if j ~= (N_layers - 1)
            as_temp = f_act(zs{j});
            as{j} = [as_temp, ones(size(as_temp,1),1)];
        else
            yhat = f_act(zs{j});
        end
    end

    Y_in = Y(b,:);
    f_all(b) = costfunc_pr(Y_in,yhat);
    fac = (yhat-Y_in);

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
            dJdWs{i} = [X(b,:),1]'*del;
        end
    end

    dJdW2 = cell(1,N_layers-1);
    for k = 1:(N_layers - 1)
        l = N_layers - k;  
        dJdW_v = reshape(dJdWs{l},[(Str(k)+1)*Str(k+1),1]);
        dJdW2{k} = dJdW_v';
    end
    dJdW_all(:,b) = cell2mat(dJdW2);
    
end

S = mean(f_all.^2); 
f = mean(f_all);
fsig = 1/(B - 1)*(S-f^2);

dJdW_sc = dJdW_all*(costfunc_fac*B);
dJdW_sum = mean(dJdW_sc,2);
dJdWS = 1/(B-1)*(mean(dJdW_sc.^2,2)-dJdW_sum.^2);

%dJdW = dJdWf;

