function [out] = sigmoid(z)
% logistic sigmoid function
out = 1./(1+exp(-z));
out(isnan(out)) = 0;
