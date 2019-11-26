function [out] = ReLU(z)
% ReLU activation function

out = zeros(size(z));
out(z >= 0) = z(z >= 0);
% z
% out
% pause