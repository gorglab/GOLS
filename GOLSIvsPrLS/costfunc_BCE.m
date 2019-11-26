function [J] = costfunc_BCE(Y,yhat)
% cross entropy cost function as prescribed in https://gombru.github.io/2018/05/23/cross_entropy_loss/

% [n_obs,dim] = size(Y);
% nugget = ones(n_obs,dim)*1e-12;
% term1 = (Y'*log10(yhat+ nugget));
% term2 = (ones(n_obs,dim)-Y)'*log10(nugget + ones(n_obs,dim)-yhat);
% 
% term3 = -1/n_obs * (term1 + term2);
% 
% J = mean(diag(term3));


%term3
%pause
% for b = 1:B
%     softmax(b,1) = sum(esi)/sum(yhat(b,:))
% end

[n_obs,dim] = size(Y);
nugget = ones(n_obs,1)*1e-12;
term3s = zeros(dim,1);
for d = 1:dim
    term1 = (Y(:,d)'*log10(yhat(:,d)+ nugget));
    term2 = (ones(n_obs,1)-Y(:,d))'*log10(nugget + ones(n_obs,1)-yhat(:,d));

    term3s(d,1) = -1/n_obs * (term1 + term2);
end
J = mean(term3s);



