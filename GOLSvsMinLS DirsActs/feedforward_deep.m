function [yhat,Ws,zs,as] = feedforward_deep(X,Ws,zs,as,N_layers,actF)


if strcmp(actF,'tanh')
    f_act = @tanh_act;
elseif strcmp(actF,'softsign')
    f_act = @softsign;
elseif strcmp(actF,'relu')
    f_act = @ReLU;
elseif strcmp(actF,'lrelu')
    f_act = @leakyReLU;
elseif strcmp(actF,'elu')
    f_act = @ELU;
else 
    f_act = @sigmoid;
end

% Str has length the number of layers, with each entry being the number of
% units in that layer
% Vectorized feedforward structure


for j = 1:(N_layers - 1)
    if j == 1
        X_bias = [X,ones(size(X,1),1)];
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

