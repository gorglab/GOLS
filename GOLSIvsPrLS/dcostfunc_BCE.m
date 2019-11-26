function [J] = dcostfunc_BCE(Y,yhat)
% Binary cross entropy cost function as prescribed in
% https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right

[n_obs,dim] = size(Y);
nugget = ones(n_obs,dim)*1e-15;
term1 = yhat-Y;
term2 = yhat-(yhat.*yhat) + nugget;

J = 1/n_obs.*(term1./term2);

%J = term3';




