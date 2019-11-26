function [out] = dsoftsign(z)
% logistic sigmoid function
out = 1./(1+abs(z)).^2; 