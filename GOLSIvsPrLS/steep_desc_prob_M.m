function [Xs,fh,err_tests,err_valids,Acc_tests,Acc_train,run_count,run_alpha] = steep_desc_prob_M(FUN,x,params,options)


% step 1
maxiter = options.maxiter;
maxfval = options.maxfval;
tol = options.tol;

print_int = params.print_int;
disp('Prob. LS')


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
xm = x;
 

count = 0;
ccnt = 1;
alpha_old = 1e-4;
t_sum = 0;
err_test_min = 1e2;
outs.counter = 0;
FunctionCount = outs.counter;

params.trteva = 1;
%[f0,test,valid,df0] = f_deepNN(x0,params)

[f,df,var_f,var_df] = FUN(xm,params);
% [f,df,var_f,var_df] = FUN(x0,params);
direction = -1*df;
outs.counter = outs.counter + 1;

% outside loop
while (count<maxiter) && (FunctionCount < maxfval)
    tic;
    
    x0 = xm;
    
    [outs, alpha, f, df, xm, var_f, var_df] = probLineSearch...
    (FUN, xm, f, df, direction, alpha_old, 0, outs, params, var_f, var_df);

%     x0 = xm;

    direction = -1*df;
    
    alpha_old = alpha;
    alpha = alpha/1.3;
    
    FunctionCount = outs.counter;
    
    if mod(count,capFrac) == 0 
        params.trteva = 2;
        [err_test,~,~,~] = FUN(x0,params);    
        te_Acc = classAcc(x0,params);
        Acc_tests(ccnt,1) = te_Acc;
        
        params.trteva = 3;
        tr_Acc = classAcc(x0,params);
        Acc_train(ccnt,1) = tr_Acc;
        
        params.trteva = 1;

        fh(ccnt) = f;
        err_tests(ccnt) = err_test;
        err_valids(ccnt) = 0;
        
        run_alpha(ccnt,1) = alpha;
        run_count(ccnt,1) = FunctionCount;
        
        ccnt = ccnt + 1;
    end
    
    count = count+1;
    
    % steepest descent update
    %xm1 = x0; % update done internally in probLS
    %xm1 = x0 + alpha*direction;
   
    time_taken = toc;
    t_sum = t_sum + time_taken;
    tpit = t_sum/FunctionCount;
    time_remaining = tpit*(maxfval-FunctionCount)/60;
    
    if mod(count,print_int) == 0
        disp(['Alg. its. ',num2str(count),' Func. Evals. ',num2str(FunctionCount),' Grad. mean abs. ',num2str(mean(abs(direction))), '   Func. Val. ',num2str(f),' Tol. ',num2str(norm(x0-xm)),' Step ',num2str(alpha),' Time rem. ',num2str(time_remaining)])
    end
     
    %max(xm1 - x0)
    

end
Xs = [xm];





