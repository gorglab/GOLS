function [out] = dgaussian(z)
% logistic sigmoid function
out = -2*z.*exp(-z.^2);