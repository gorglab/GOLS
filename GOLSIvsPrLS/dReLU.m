function [out] = dReLU(z)
% ReLU activation function


out = zeros(size(z));
out(z >= 0) = 1;
% z
% out
