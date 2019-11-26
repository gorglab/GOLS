function [xstar,k] = GoldenSection(F,dir,~,max_step)

tol = 1/max_step;
if norm(dir) > tol
    max_step = 1/norm(dir);
else
    max_step = 1/tol;
end
min_step = 1e-8;
max_iter = 1000;
tol = 1e-12;

% Phase 1 - Bracket an interval
delta = 0.5;
l = 0;
m = delta;
r = (sqrt(5)+1)/2;
u = m + r*delta;
flag = 1;
k = 2;

if u > max_step
    u = max_step;
    I = u-l;
    m = l + (1-1/r)*I;
end

while F(m) > F(u) && flag
    l = m;
    m = u;
    u = m + r^(k)*delta;
    %u = m + r*delta;
    k = k + 1;
    if u > max_step
        flag = 0;
        xstar = max_step;
    end
end

if flag

    % Phase 2 - Reduce the interval
    I = u-l;
    a = l + (1-1/r)*I;
    b = l + 1/r*I;

    while I > tol && isnan(F(a)) ~= 1 && isnan(F(b)) ~= 1  && u < max_step && u > min_step && k < max_iter
        if F(a) < F(b)
            u = b;
            I = u-l;
            b = a;
            a = l + (1-1/r)*I;
        elseif F(a) > F(b)
            l = a;
            I = u-l;
            a = b;
            b = l + 1/r*I;
        else
            l = a;
            u = b;
            I = u-l;
            b = l + 1/r*I;
            a = l + (1-1/r)*I;
        end
        k = k + 1;
    end
    
    xstar = (b+a)/2;
    if xstar > max_step
        xstar = max_step;
    end

end

if xstar>max_step
    xstar
    max_step
    pause
end




