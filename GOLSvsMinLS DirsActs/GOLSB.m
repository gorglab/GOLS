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
% -- max_step: Absolute maximum step size

% OUTPUTS
% -- a: Final step size
% -- k: Number of function evaluations (increments) performed


function [a,k] = GOLSB(F,dir,max_step)

tol = 1/max_step;
if norm(dir) > tol
    max_step = 1/norm(dir);
else
    max_step = 1/tol;
end
min_step = 1e-8;
max_iter = 1000;

delta = 5;
l = min_step;
m = delta;
r = (sqrt(5)+1)/2;
u = m + r*delta;
flag = 1;

k = 0;

[~,~,~,dirm] = F(m);
[~,~,~,diru] = F(u);
gradm = dirm*dir';
gradu = diru*dir';
k = k + 2;

if u > max_step
    u = max_step;
    I = u-l;
    m = l + 0.5*I;
    [~,~,~,dirm] = F(m);
    [~,~,~,diru] = F(u);
    gradm = dirm*dir';
    gradu = diru*dir';
    k = k + 2;
end


while sign(gradu) == -1 && flag  && k < max_iter
    m = u;
    u = m + r^(k)*delta;
    %u = m + r*delta;
    [~,~,~,diru] = F(u);
    gradu = diru*dir';
    k = k + 1;
    if u > max_step
        flag = 0;
        a = max_step;
    end
end

  
if flag == 1
    % Phase 2 - Reduce the interval
    I = u-l;
    while I > 1e-12 && u <= max_step && u > min_step && k < max_iter
        if sign(gradm) == -1 && sign(gradu) == 1
            l = m;
            I = u-l;
        elseif sign(gradm) == 1
            u = m;
            gradu = gradm;
            I = u-l;
        end
        m = l + 0.5*I;
        [~,~,~,dirm] = F(m);
        gradm = dirm*dir';
        k = k + 1;

    end
    a = (u+l)/2;
end
 





