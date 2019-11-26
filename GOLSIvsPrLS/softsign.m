function [out] = softsign(z)
% logistic sigmoid function
out = z./(1+abs(z)); 