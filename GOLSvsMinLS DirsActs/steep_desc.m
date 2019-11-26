function [Xs,fh,err_tests,err_valids,run_count,run_alpha] = steep_desc(FUN,x,params,options)


% step 1
maxiter = options.maxiter;
exact = options.exact;
grads = options.grads;
max_step = options.max_step;

FunctionCount = 0;
GradientCount = 0;
print_int = params.print_inc;
disp('Steepest Descent Algorithm')


fh = zeros(maxiter,1);
err_tests = zeros(maxiter,1);
err_valids = zeros(maxiter,1);
run_count = zeros(maxiter,1);
run_alpha = zeros(maxiter,1);
gh = zeros(maxiter,length(x));
x0 = zeros(size(x'));
xm1 = x';

count = 0;
alpha_old = 1e-8;
t_sum = 0;
err_test_min = 1e2;

% outside loop
while count<maxiter
    
    count = count+1;

    x0 = xm1;
    
    [err_train,err_test,err_valid,grad] = FUN(x0,params);
    direction = -1*grad;
    FunctionCount = FunctionCount + 1;
    GradientCount = GradientCount + 1;
    
    
    fh(count) = err_train;
    err_tests(count) = err_test;
    err_valids(count) = err_valid;
    gh(count,:) = direction;    

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
        if exact == 1
            [alpha,its] = GOLSB(F,direction,max_step);
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
    
    alpha_old = alpha;
    run_alpha(count,1) = alpha;
        
    FunctionCount = FunctionCount + its;
    run_count(count,1) = FunctionCount;
    
    % steepest descent update
    xm1 = x0 + alpha*direction;
   
    
    
    if mod(count,print_int) == 0
        disp(['Alg. its. ',num2str(count),' Func. Evals. ',num2str(FunctionCount),' Grad. mean abs. ',num2str(mean(abs(direction))), '   Func. Val. ',num2str(fh(count)),' Tol. ',num2str(norm(xm1-x0)),' Step ',num2str(alpha)])
    end
     
    

end
Xs = [X_min;x0];





