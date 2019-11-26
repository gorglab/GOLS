function [out] = tanh_act(z)
% logistic sigmoid function
out = (exp(z) - exp(-z))./(exp(z) + exp(-z)); 