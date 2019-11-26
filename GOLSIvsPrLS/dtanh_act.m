function [out] = dtanh_act(z)
% logistic sigmoid function
out = 1 - (tanh_act(z)).^2; 