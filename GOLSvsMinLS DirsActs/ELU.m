function [out] = ELU(z)
% ReLU activation function

out = zeros(size(z));
out(z >= 0) = z(z >= 0);
out(z < 0) = exp(z(z<0))-1;

% z
% out
% pause