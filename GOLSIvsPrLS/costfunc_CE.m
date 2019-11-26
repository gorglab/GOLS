function [J] = costfunc_CE(Y,yhat)
% cross entropy cost function as prescribed in https://gombru.github.io/2018/05/23/cross_entropy_loss/

[dim] = size(Y,2);
esi = exp(Y*yhat');
Sesj = exp(yhat)*ones(dim,1);

J = -log(esi./Sesj);

% for b = 1:B
%     softmax(b,1) = sum(esi)/sum(yhat(b,:))
% end




