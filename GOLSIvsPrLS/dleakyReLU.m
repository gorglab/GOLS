function [out] = dleakyReLU(z)
% ReLU activation function

% if z < 0
%     out = 0.01;
% elseif z >= 0 
%     out = 1;
% end


out = 0.01*ones(size(z));
out(z >= 0) = 1;
% z
% out
