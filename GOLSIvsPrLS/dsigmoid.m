function [out] = dsigmoid(z)
% derivative of the sigmoid function
out = (exp(-z))./(1+exp(-z)).^2;
out(isnan(out)) = 0;
