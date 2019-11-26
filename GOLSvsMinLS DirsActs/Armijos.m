function [a,k] = Armijos(f,q,eta,dir,max_step,init_guess)
% Armijo's rule for inexact line search

tol = 1/max_step;
if norm(dir) > tol
    max_step = 1/norm(dir);
else
    max_step = 1/tol;
end
min_step = 1e-8;
a = init_guess;

if f(a) < q(a)
    flag = 2; % increase
else
    flag = 1; % decrease
end

k = 0;

while flag
   k = k + 1;
   if flag == 2
       a = a*eta;
       if f(a) > q(a)
           flag = 0;
           a = a/eta;
       end
   else
       a = a/eta;
       if f(a) < q(a)
           flag = 0;
       end
   end
   if a < min_step
       flag = 0;
       a = min_step;
       
   elseif a > max_step
       flag = 0;
       a = max_step;
       %disp('maxed')
   end

end

if a>max_step
    a
    max_step
    pause
end
