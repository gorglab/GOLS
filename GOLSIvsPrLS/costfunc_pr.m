function [J] = costfunc_pr(Y,yhat)
% cost function as prescribed in Proben1 documentation
[B,dim] = size(Y);
J = 100/(B*dim)*sum(sum((Y-yhat).^2));

% Js = 100/(size(Y,2))*sum((Y-yhat).^2,2);
% 
% S = mean(Js.^2);
% J = mean(Js);
% 
% JS = 1/(size(S,1) - 1)*(S-J^2);

