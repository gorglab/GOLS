function [Xs,fh,err_tests,err_valids,Acc_tests,Acc_train,run_count,run_alpha] = steep_desc_M_upd(FUN,x,params,options)


% step 1
maxiter = options.maxiter;
maxfval = options.maxfval;
tol = options.tol;
max_step = options.max_step;

FunctionCount = 0;
print_int = params.print_int;
disp('GOLS-I')

capFrac = params.capFrac;
fh = zeros(maxiter/capFrac,1);
err_tests = zeros(maxiter/capFrac,1);
err_valids = zeros(maxiter/capFrac,1);
Acc_tests = zeros(maxiter/capFrac,1);
Acc_train = zeros(maxiter/capFrac,1);
run_count = zeros(maxiter/capFrac,1);
run_alpha = zeros(maxiter/capFrac,1);
%gh = zeros(maxiter,length(x));
x0 = zeros(size(x));
xm1 = x;

count = 0;
ccnt = 1;
alpha_old = 1e-8;
%alpha_old = 1e-4;
t_sum = 0;
params.skipVar = 1;


params.trteva = 1;
[err_train,grad,~,~] = FUN(x0,params);    
%direction = -1*grad;
FunctionCount = FunctionCount + 1;

% outside loop
while (count<maxiter) && (FunctionCount < maxfval)
    tic;
    
    x0 = xm1;
    
    direction = -1*grad;
    
    F = @(x) FUN(x0 + x*direction,params);

    p = 0.9;
    [alpha,its,grad,err_train] = GOLSI_M(F,direction,grad,p,2,max_step,alpha_old);
    
    
    alpha_old = alpha;

    FunctionCount = FunctionCount + its;
    
    if mod(count,capFrac) == 0 
        params.trteva = 2;
        [err_test,~,~,~] = FUN(x0,params);    
        te_Acc = classAcc(x0,params);
        Acc_tests(ccnt,1) = te_Acc;
        
        params.trteva = 3;
        tr_Acc = classAcc(x0,params);
        Acc_train(ccnt,1) = tr_Acc;
        
        params.trteva = 1;

        fh(ccnt) = err_train;
        err_tests(ccnt) = err_test;
        err_valids(ccnt) = 0;
        
        run_alpha(ccnt,1) = alpha;
        run_count(ccnt,1) = FunctionCount;
        
        ccnt = ccnt + 1;
    end
    
    % steepest descent update
    xm1 = x0 + alpha*direction;
   
    count = count+1;
    
    time_taken = toc;
    t_sum = t_sum + time_taken;
    tpit = t_sum/FunctionCount;
    time_remaining = tpit*(maxfval-FunctionCount)/60;
    
    if mod(count,print_int) == 0
        disp(['Alg. its. ',num2str(count),' Func. Evals. ',num2str(FunctionCount),' Grad. mean abs. ',num2str(mean(abs(direction))), '   Func. Val. ',num2str(err_train),' Tol. ',num2str(norm(xm1-x0)),' Step ',num2str(alpha),' Time rem. ',num2str(time_remaining)])
    end

end
Xs = [x0];





