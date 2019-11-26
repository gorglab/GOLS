function [out] = dnleakyReLU(z)
% ReLU activation function
% if z < 0
%     out = -1e-8;
% elseif z >= 0 
%     out = 1;
% end

out = -1e-4*ones(size(z));
out(z >= 0) = 1;