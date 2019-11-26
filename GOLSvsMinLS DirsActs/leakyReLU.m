function [out] = leakyReLU(z)
% ReLU activation function

% if z < 0
%     out = 0.01*z;
% elseif z >= 0 
%     out = z;
% end

out = zeros(size(z));
out(z < 0) = 0.01*z(z < 0);
out(z >= 0) = z(z >= 0);
% z
% out
% pause