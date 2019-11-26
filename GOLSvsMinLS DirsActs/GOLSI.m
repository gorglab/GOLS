% Copyright (c) Kafka and Wilke 2019
% dominic.kafka@gmail.com, wilkedn@gmail.com
% All Rights Reserved 
% Any commercial use is restricted to licensing from the authors.

% Redistribution and use in source and binary forms, with or without
% modification, are permitted for ACADEMIC USE ONLY provided that the following 
% conditions are met:
%     * Redistributions of source code and/or modifications of source code 
%       for academic purposes must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in the
%       documentation and/or other materials provided with the distribution.
%     * Neither the name of the <organization> nor the
%       names of its contributors may be used to endorse or promote products
%       derived from this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
% DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



% INPUTS
% -- F: Function that returns a function value and a gradient (can also be
% reduced to only a gradient)
% -- dir: The search direction along which to conduct the line search
% -- grad0: Given gradient evaluation (last gradient evaluation of previous
% line search) to save one function call per iteration
% -- p: Modified strong wolfe condition parameter (Eq. 8), commonly used as 0.9 
% -- eta: Factor by which step sizes are increased or decreased
% -- max_step: Absolute maximum step size
% -- init_guess: Initial guess for step size (generally taken as the step
% size from the previous iteration

% OUTPUTS
% -- a: Final step size
% -- i: Number of function evaluations (increments) performed


function [a,i] = GOLSI(F,dir,grad0,p,eta,max_step,init_guess)

tol = 1/max_step;
if norm(dir) > tol
    max_step = 1/norm(dir);
else
    max_step = 1/tol;
end
min_step = 1e-8;

i = 0;

dirD0 = grad0*dir';

% set gradient tolerance, dependent on allowed overshoot
grad_tol = abs(p*dirD0);

% check initial bounds
a = init_guess;
if a > max_step
    a = max_step;
end
if a < min_step
    a = min_step;
end
% evaluation at initial guess
[~,~,~,dira] = F(a);
dirD = dira*dir';
i = i + 1;


% set increase or decrease mode
if dirD < 0 && a < max_step
    flag = 2;
elseif dirD >= 0 && a > min_step
    flag = 1;
else
    flag = 0;
end

% immediate accept condition
if dirD > 0 && dirD < grad_tol
    flag = 0;
end

%#################################################################
% IGOLS LOOP
%#################################################################
while flag > 0
    if flag == 2
        % INCREASE STEP SIZE
        a = a*eta;
        [~,~,~,dira] = F(a);
        dirD = dira*dir';
        i = i + 1;

        if dirD >= 0
            flag = 0;
        elseif a > max_step/eta
            flag = 0;
        end

    elseif flag == 1
        % DECREASE STEP SIZE
        a = a/eta;
        [~,~,~,dira] = F(a);
        dirD = dira*dir';
        i = i + 1;
        if dirD < 0
            flag = 0;
        elseif a < min_step*eta
            flag = 0;
        end
    end
end


