function [Xs,fh,err_tests,err_valids,run_count,run_alpha] = Adadelta(FUN,x,params,options)


% step 1
maxiter = options.maxiter;
exact = options.exact;
grads = options.grads;
max_step = options.max_step;

FunctionCount = 0;
GradientCount = 0;
print_int = params.print_inc;
disp('Adadelta Algorithm')

x0 = zeros(size(x'));
xm1 = x';

count = 0;

fh = zeros(maxiter,1);
err_tests = zeros(maxiter,1);
err_valids = zeros(maxiter,1);
run_count = zeros(maxiter,1);
run_alpha = zeros(maxiter,1);
% outside loop

Egt = zeros(size(x'));
Ett = zeros(size(x'));
Eps = 1e-8*ones(size(x'));
delta_x_old = ones(size(x'));

gamma = 0.9;
alpha_old = 1e-8;
err_test_min = 1e2;

while count<maxiter

    count = count+1;
    x0 = xm1;
    [err_train,err_test,err_valid,grad] = FUN(x0,params);
    
    
    Egt = gamma.*Egt + (1-gamma).*(grad.^2);
    Ett = gamma.*Ett + (1-gamma).*(delta_x_old.^2);
    factor = ((Ett + Eps).^(0.5)) .* (((Egt + Eps).^(0.5)).^(-1));
    direction = - factor.*grad;
    
    FunctionCount = FunctionCount + 1;
    GradientCount = GradientCount + 1;

    fh(count) = err_train;
    err_tests(count) = err_test;
    err_valids(count) = err_valid;
    
    if err_test < err_test_min
        err_test_min = err_test;
        X_min = x0;
    end
    

    F = @(x) FUN(x0 + x*direction,params);
    if grads == 0
        % function value line search 
        if exact == 1
            [alpha,its] = GoldenSection(F,direction,0.1,max_step);
        else
            p = 0.2;
            q = @(x) F(0) + x*p*(grad*direction');
            [alpha,its] = Armijos(F,q,2,direction,max_step,alpha_old);
        end
    elseif grads == 1
        % gradient only line search
        if exact == 1
            [alpha,its] = GOLSB(F,direction,0.1,max_step);
        else
            p = 0.9;
            [alpha,its] = GOLSI(F,direction,grad,p,2,max_step,alpha_old);
        end
    elseif grads == -1
        alpha = 0.1 ;
        its = 0;
    elseif grads == -2
        alpha = 1;
        its = 0;
    elseif grads == -3
        alpha = 10;
        its = 0;
    end    

    FunctionCount = FunctionCount + its;
    run_count(count,1) = FunctionCount;
    
    alpha_old = alpha;
    run_alpha(count,1) = alpha;
    
    delta_x = alpha*direction;
    xm1 = x0 + delta_x;
    delta_x_old = delta_x;
    
    if mod(count,print_int) == 0
        disp(['Alg. its. ',num2str(count),' Func. Evals. ',num2str(FunctionCount),' Grad. Norm ',num2str(norm(direction)), '   Func. Val. ',num2str(fh(count)),' Tol. ',num2str(norm(xm1-x0)),' Step ',num2str(alpha)])
    end
    
    
end

Xs = [X_min;x0];



