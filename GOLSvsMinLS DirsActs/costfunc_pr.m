function [J] = costfunc_pr(Y,yhat)
% cost function as prescribed in Proben1 documentation
J = 100/(size(Y,1)*size(Y,2))*sum(sum((Y-yhat).^2));
