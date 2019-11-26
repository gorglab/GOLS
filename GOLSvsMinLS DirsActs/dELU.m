function [out] = dELU(z)
% ReLU activation function

out = zeros(size(z));
out(z >= 0) = 1;
out(z < 0) = exp(z(z<0));

% z
% out
% pause