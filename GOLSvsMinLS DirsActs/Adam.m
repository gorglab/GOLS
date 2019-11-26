function [Xs,fh,err_tests,err_valids,run_count,run_alpha] = Adam(FUN,x,params,options)


% step 1
maxiter = options.maxiter;
exact = options.exact;
grads = options.grads;
max_step = options.max_step;

FunctionCount = 0;
GradientCount = 0;
print_int = params.print_inc;
disp('Adam Algorithm')

x0 = zeros(size(x'));
xm1 = x';

count = 0;

fh = zeros(maxiter,1);
err_tests = zeros(maxiter,1);
err_valids = zeros(maxiter,1);
run_count = zeros(maxiter,1);
run_alpha = zeros(maxiter,1);

Eps = 1e-8*ones(size(x'));

v_t = zeros(size(x'));
m_t = zeros(size(x'));

beta_1 = 0.9;
beta_2 = 0.999;

t_sum = 0;
alpha_old = 1e-8;
err_test_min = 1e2;

while count<maxiter
    
    tic;
    count = count+1;
    x0 = xm1;
    [err_train,err_test,err_valid,grad] = FUN(x0,params);
    
    
    m_t = beta_1.*m_t + (1-beta_1).*grad;
    v_t = beta_2.*v_t + (1- beta_2).*grad.^2;
    mh_t = m_t./(1-beta_1);
    vh_t = v_t./(1-beta_2);
    factor = 1./(vh_t.^(0.5) + Eps);
    direction = - factor.*mh_t;
    
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
        alpha = 0.001 ;
        its = 0;
    elseif grads == -2
        alpha = 0.01;
        its = 0;
    elseif grads == -3
        alpha = 0.1;
        its = 0;
    elseif grads == 11
        p = 1.0;
        [alpha,its] = GradOnlyInex(F,direction,p,2,max_step,alpha_old);

    end    

    FunctionCount = FunctionCount + its;
    run_count(count,1) = FunctionCount;
    
    alpha_old = alpha;
    run_alpha(count,1) = alpha;
    
    delta_x = alpha*direction;
    xm1 = x0 + delta_x;
    %delta_x_old = delta_x;
    
    time_taken = toc;
    t_sum = t_sum + time_taken;
    tpit = t_sum/count;
    time_remaining = tpit*(maxiter-count)/60;
    
    if mod(count,print_int) == 0
        disp(['Alg. its. ',num2str(count),' Func. Evals. ',num2str(FunctionCount),' Grad. mean abs. ',num2str(mean(abs(direction))), '   Func. Val. ',num2str(fh(count)),' Tol. ',num2str(norm(xm1-x0)),' Step ',num2str(alpha),' Time rem. ',num2str(time_remaining)])
    end
    
    
end
disp(['Time taken: ',num2str(t_sum/60)])

Xs = [X_min;x0];


