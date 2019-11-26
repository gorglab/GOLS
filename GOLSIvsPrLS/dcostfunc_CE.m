function [J] = dcostfunc_CE(Y,yhat)
% cross entropy cost function as prescribed in https://gombru.github.io/2018/05/23/cross_entropy_loss/
%J = 100/(size(Y,1)*size(Y,2))*sum(sum((Y-yhat).^2));

[dim] = size(Y,2);
%esi = Y*yhat';
Sesi = exp(yhat)*ones(dim,1);

J = (exp(yhat)./Sesi - Y);





