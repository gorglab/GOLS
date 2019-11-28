# Copyright (c) Kafka and Wilke 2019
# dominic.kafka@gmail.com, wilkedn@gmail.com
# All Rights Reserved
# Any commercial use is restricted to licensing from the authors.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted for ACADEMIC USE ONLY provided that the following
# conditions are met:
#     * Redistributions of source code and/or modifications of source code
#       for academic purposes must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



import torch
from functools import reduce
from torch.optim import Optimizer
import math
import numpy as np


class PyGOLS(Optimizer):

    """Implements Steepest Descent with Inexct Gradient-Only Line Search.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    Arguments:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
    """



    def __init__(self, params, overP=0.9, eta=2.0,min_step=1e-8, max_step=1E7, init_guess=1e-8, max_iter=1, max_eval=2000,
                 tolerance_grad=1e-32, alg='SGD', LineSearch='Inexact', BisectionBracketing='Aggressive', betas=(0.0, 0.999), epsC=1e-8,
                 amsgrad=False, state=False, tolerance_change=1e-14,history_size=21):


        defaults = dict(overP=overP, eta=eta,min_step=min_step, max_step=max_step, init_guess=init_guess,
                        max_iter=max_iter, max_eval=max_eval, tolerance_grad=tolerance_grad, alg=alg,
                        LineSearch=LineSearch, BisectionBracketing=BisectionBracketing, betas=betas, epsC=epsC,
                        amsgrad=amsgrad,history_size=history_size, tolerance_change=tolerance_change, state=state)

        super(PyGOLS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("GOLS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None


    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    def _addcdiv_(self, step_size, numerator, denominator):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.addcdiv_(step_size, numerator[offset:offset + numel].view_as(p.data),
                denominator[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()





    def step(self, closure):

        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """


        ##################################################################
        # DEFINITIONS AND RECALLING OF PARAMETERS section
        ##################################################################

        # FUNCTION FOR THE LOSS EVALUATION
        def lossEval(a, aold, direction, closure):
            self._add_grad(a-aold, direction)
            loss = closure()
            flat_grad = self._gather_flat_grad()
            abs_grad_sum = flat_grad.abs().sum()
            dirD = torch.dot(flat_grad,direction)
            return loss, abs_grad_sum, dirD, flat_grad

        # RETRIEVE PARAMETERS
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        tolerance_grad = group['tolerance_grad']
        overP = group['overP']
        eta = group['eta']
        max_step = group['max_step']
        min_step = group['min_step']
        init_guess = group['init_guess']
        alg = group['alg']
        LineSearch = group['LineSearch']
        BisectionBracketing = group['BisectionBracketing']
        state = group['state']

        # Adam parameters
        amsgrad = group['init_guess']

        # LBFGS parameters
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_change = group['tolerance_change']
        history_size = group['history_size']


        # EVALUATE INITIAL LOSS AND DIRECTIONAL DERIVATIVE



        # NOTE: GOLS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict

        # INITIALIZE THE STATE DICTIONARY IN FIRST UPDATE STEP
        if len(state) == 0:
            print('STATE _INIT')
            orig_loss = closure()
            #print(orig_loss.item())
            loss = orig_loss
            flat_grad = self._gather_flat_grad()
            abs_grad_sum = flat_grad.abs().sum()
            grad = flat_grad*1.0
            current_evals = 1

            state.setdefault('func_evals', 0)
            state.setdefault('n_iter', 0)
            state['step'] = 0
            state['loss'] = loss
            state['gradient'] = flat_grad
            state['abs_grad_sum'] = abs_grad_sum
            ## Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(flat_grad)
            ## Exponential moving average of gradient values
            state['sum'] = torch.zeros_like(flat_grad)
            ## Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(flat_grad)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(flat_grad)
        else:
            loss = state['loss']
            #print('read loss ',loss)
            flat_grad = grad = state['gradient']
            abs_grad_sum = state['abs_grad_sum']
            current_evals = 0


        if alg != 'LBFGS':
            max_iter = 1
        else:
            # tensors cached in state (for tracing)
            direction = state.get('d')
            a = state.get('a')

            old_dirs = state.get('old_dirs')
            old_stps = state.get('old_stps')
            H_diag = state.get('H_diag')
            prev_flat_grad = state.get('prev_flat_grad')
            prev_loss = state.get('prev_loss')



        ##################################################################
        # ALGORITHM section
        # compute descent direction
        ##################################################################

        n_iter = 0
        # SUB ITERATIONS LOOP
        # keepS track of no. of iterations (multiples for BFGS, but singles for all others)
        while n_iter < max_iter:

            n_iter += 1
            state['n_iter'] += 1


            if alg == 'LBFGS':
                if state['n_iter'] == 1:
                    direction = flat_grad.neg()
                    old_dirs = []
                    old_stps = []
                    H_diag = 1

                else:
                    pre_mod = torch.norm(direction).item()

                    # do lbfgs update (update memory)
                    y = flat_grad.sub(prev_flat_grad)
                    # tee
                    # s = direction.mul(t)
                    s = direction.mul(a)
                    ys = y.dot(s)  # y*s
                    if ys > 1e-10:
                        # updating memory
                        if len(old_dirs) == history_size:
                            # shift history by one (limited-memory)
                            old_dirs.pop(0)
                            old_stps.pop(0)

                        # store new direction/step
                        old_dirs.append(s)
                        old_stps.append(y)

                        # update scale of initial Hessian approximation
                        H_diag = ys / y.dot(y)  # (y*y)

                    # compute the approximate (L-BFGS) inverse Hessian
                    # multiplied by the gradient
                    num_old = len(old_dirs)

                    if 'ro' not in state:
                        state['ro'] = [None] * history_size
                        state['al'] = [None] * history_size
                    ro = state['ro']
                    al = state['al']

                    for i in range(num_old):
                        ro[i] = 1. / old_stps[i].dot(old_dirs[i])

                    # iteration in L-BFGS loop collapsed to use just one buffer
                    q = flat_grad.neg()
                    for i in range(num_old - 1, -1, -1):
                        al[i] = old_dirs[i].dot(q) * ro[i]
                        q.add_(-al[i], old_stps[i])

                    # multiply by initial Hessian
                    # r/d is the final direction
                    direction = r = torch.mul(q, H_diag)
                    for i in range(num_old):
                        be_i = old_stps[i].dot(r) * ro[i]
                        r.add_(al[i] - be_i, old_dirs[i])

                    post_mod = torch.norm(direction).item()
                    if (np.log(post_mod) - np.log(pre_mod)) > 2:
                        init_guess = 0.01/torch.norm(direction).item()


                if prev_flat_grad is None:
                    prev_flat_grad = flat_grad.clone()
                else:
                    prev_flat_grad.copy_(flat_grad)
                prev_loss = loss*1.0


            elif alg == 'Adagrad':
                state['step'] += 1
                state['sum'].addcmul_(1, grad, grad)
                std = state['sum'].sqrt().add_(1e-10)
                dnum = grad*1.0
                dden = std*1.0
                Zds = torch.zeros_like(flat_grad)
                direction = torch.addcdiv(Zds, -1, dnum, dden)


            elif alg == 'Adam':
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['epsC'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['epsC'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                fac = 1 * math.sqrt(bias_correction2) / bias_correction1

                dnum = exp_avg*1.0
                dden = denom*1.0
                Zds = torch.zeros_like(flat_grad)
                direction = torch.addcdiv(Zds, -fac, dnum, dden)*1.0
            else: # alg == 'SGD'
                direction = flat_grad*(-1.0)



            ##################################################################
            # LINE SEARCH section
            ##################################################################

            # directional derivative before IGOLS
            dirD0 = torch.dot(flat_grad,direction)
            state['preDD'] = dirD0.item()

            # break if initial gradient is below tolerance
            if abs_grad_sum <= tolerance_grad:
                #print('GRADIENT TOO SMALL: '+str(abs_grad_sum))
                loss, abs_grad_sum, dirD0, flat_grad = lossEval(0,0,direction,closure)
                current_evals += 1
                state['gradient'] = flat_grad
                state['d'] = direction
                state['func_evals'] += current_evals
                state['a'] = min_step
                state['postDD'] = dirD0.item()
                state['loss'] = loss
                state['abs_grad_sum'] = abs_grad_sum
                abs_grad_sum = state['abs_grad_sum']
                #print('Re-evaluated: '+str(abs_grad_sum.item()))

                return loss, state



            ##################################################################
            # BISECTION LINE SEARCH
            ##################################################################
            #print(LineSearch)
            # check if max_step needs to be capped for Adam and Adagrad???
            if LineSearch=='Bisection':

                # set maximum step tolerance
                tol = 1/max_step;
                if torch.norm(direction) > tol:
                    max_step = 1/torch.norm(direction).item()
                else:
                    max_step = 1/tol

                #print('max_step ',max_step)

                if BisectionBracketing == 'Aggressive':
                    flag = 1
                    u = max_step
                    loss_u, abs_grad_sum, dirD_u, flat_grad_u = lossEval(u,0,direction,closure)
                    current_evals += 1
                    aold = u*1.0
                    #print(dirD_u.item())

                    dirD_x = dirD_u*1.0
                    x = u*1.0
                    while dirD_x > 0:
                        # INCREASE STEP SIZE
                        x = x/eta
                        loss_x, abs_grad_sum, dirD_x, flat_grad_x = lossEval(x,aold,direction,closure)
                        current_evals += 1
                        aold = x*1.0

                        if dirD_x >= 0:
                            dirD_u = dirD_x*1.0
                            loss_u = loss_x*1.0
                            flat_grad_u = flat_grad_x
                            u = x*1.0
                        if dirD_x < 0:
                            dirD_0 = dirD_x*1.0
                            loss_0 = loss_x*1.0
                            flat_grad_0 = flat_grad_x
                            l = x*1.0

                       # print(x,dirD_x.item())



                else:

                    flag = 1
                    l = min_step*1.0
                    loss_0, abs_grad_sum, dirD_0, flat_grad_0 = lossEval(min_step,0,direction,closure)
                    current_evals += 1
                    aold = min_step*1.0
                    #print(loss_0.item())

                    if n_iter != 1:
                        init_guess = a*1.0

                    if BisectionBracketing == 'Memory':

                        u = init_guess*1.0
                        loss_u, abs_grad_sum, dirD_u, flat_grad_u = lossEval(u,aold,direction,closure)
                        current_evals += 1
                        aold = u*1.0

                        while dirD_u<0 and u<(max_step/eta):
                            # INCREASE STEP SIZE
                            u = u*eta
                            loss_u, abs_grad_sum, dirD_u, flat_grad_u = lossEval(u,aold,direction,closure)
                            current_evals += 1
                            aold = u*1.0

                    elif BisectionBracketing == 'Conservative':
                        # CONSERVATIVE BRACKETING STRATEGY
                        #print('conservative')
                        delta = 5*min_step
                        m = delta
                        r = (np.sqrt(5)+1)/2
                        u = m + r*delta

                        if u > max_step:
                            u = max_step*1.0
                            I = u-l
                            m = l + 0.5*I

                        loss_u, abs_grad_sum, dirD_u, flat_grad_u = lossEval(u,aold,direction,closure)
                        current_evals += 1
                        aold = u*1.0

                        k_count = 0

                        while dirD_u < 0 and flag and k_count < max_eval:
                            m = u*1.0
                            u = m + r**(k_count)*delta

                            loss_u, abs_grad_sum, dirD_u, flat_grad_u = lossEval(u,aold,direction,closure)
                            current_evals += 1
                            k_count += 1
                            aold = u*1.0
                            #print(loss_u.item(),dirD_u.item())



                # UPPER BOUND CHECK
                if u > max_step:
                    flag = 0
                    a = max_step/eta

                if dirD_u < 0:
                    flag = 0
                    a = u*1.0
                    #print('truncated')

                if flag:
                    # SET INITIAL MIDPOINT
                    I = u-l
                    m = l + 0.5*I

                    loss_m, abs_grad_sum, dirD_m, flat_grad_m = lossEval(m,aold,direction,closure)
                    current_evals += 1
                    aold = m*1.0

                # REDUCE INTERVAL
                if flag == 1:
                    I = u-l
                    while I > 1e-12 and current_evals < max_eval and abs(dirD_m) > tolerance_grad:
                        if dirD_m < 0 and dirD_u > 0:
                            l = m*1.0
                            I = u-l
                        elif dirD_m > 0:
                            u = m*1.0
                            dirD_u = dirD_m*1.0
                            I = u-l

                        m = l + 0.5*I

                        loss_m, abs_grad_sum, dirD_m, flat_grad_m = lossEval(m,aold,direction,closure)
                        current_evals += 1
                        aold = m*1.0
                        #print(loss_m.item(),dirD_m.item())
                        #print(aold)

                    a = (u+l)/2

                #print('local_grad ',np.linalg.norm(flat_grad_m))
                #print('direction ',np.linalg.norm(direction))

                # FINAL POINT
                loss, abs_grad_sum, dirD, flat_grad = lossEval(a,aold,direction,closure)
                current_evals += 1
                aold = a*1.0

                #print(loss.item(),a)


            ##################################################################
            # INEXACT LINE SEARCH
            ##################################################################
            if LineSearch=='Inexact':

                # set maximum step tolerance
                tol = 1/max_step;
                if torch.norm(direction) > tol:
                    max_step = 1/torch.norm(direction).item();
                else:
                    max_step = 1/tol;

                # set gradient tolerance, dependent on allowed overshoot
                grad_tol = abs((overP)*dirD0)

                # check initial bounds
                a = init_guess*1.0
                if a > max_step:
                    a = max_step*1.0

                if a < min_step:
                    a = min_step*1.0

                #loss_0, abs_grad_sum, dirD_0, flat_grad_0 = lossEval(min_step,0,direction,closure)
                #current_evals += 1
                #aold = min_step*1.0
                #print(loss_0.item())

                # evaluation at initial guess
                loss, abs_grad_sum, dirD, flat_grad = lossEval(a,0,direction,closure)
                current_evals += 1
                aold = a*1.0

                # set increase or decrease mode
                if dirD.item() < 0 and a < max_step:
                    flag = 2
                elif dirD.item() >= 0 and a > min_step:
                    flag = 1
                else:
                    flag = 0

                # immediate accept condition
                if dirD.item() > 0 and dirD.item() < grad_tol:
                    flag = 0

                # GOLS-I LOOP
                while flag and abs_grad_sum>=tolerance_grad and current_evals< max_eval:
                    if flag == 2:
                        # INCREASE STEP SIZE
                        a = a*eta
                        loss, abs_grad_sum, dirD, flat_grad = lossEval(a,aold,direction,closure)
                        current_evals += 1
                        aold = a*1.0

                        if dirD.item() >= 0:
                            flag = 0

                        elif a > max_step/eta:
                            flag = 0

                    elif flag == 1:
                        # DECREASE STEP SIZE
                        a = a/eta
                        loss, abs_grad_sum, dirD, flat_grad = lossEval(a,aold,direction,closure)
                        current_evals += 1
                        aold = a*1.0
                        if dirD.item() < 0:
                            flag = 0
                        elif a < min_step*eta:
                            flag = 0


            ##################################################################
            # BACKTRACKING LINE SEARCH
            ##################################################################
            if LineSearch=='Back':

                tol = 1/max_step;
                if torch.norm(direction) > tol:
                    a = 1/torch.norm(direction).item()
                else:
                    a = 1/tol

                grad_tol = abs((overP)*dirD0)

                # evaluation at initial guess
                loss, abs_grad_sum, dirD, flat_grad = lossEval(a,0,direction,closure)
                current_evals += 1
                aold = a*1.0

                # accept if negative, else decrease
                if dirD.item() < 0:
                    flag = 0
                elif dirD.item() >= 0 and a > min_step:
                    flag = 1


                # GOLS-Back LOOP
                while flag and abs_grad_sum>=tolerance_grad and current_evals<max_eval:

                        # DECREASE STEP SIZE
                        a = a/eta
                        loss, abs_grad_sum, dirD, flat_grad = lossEval(a,aold,direction,closure)
                        current_evals += 1
                        aold = a*1.0
                        if dirD.item() <= grad_tol:
                            flag = 0
                        elif a < min_step*eta:
                            flag = 0


            ##################################################################
            # MAXIMIZING STEP LINE SEARCH
            ##################################################################
            if LineSearch=='Max':

                # set maximum step tolerance
                tol = 1/max_step;
                if torch.norm(direction) > tol:
                    max_step = 1/torch.norm(direction).item();
                else:
                    max_step = 1/tol;

                # set gradient tolerance, dependent on allowed overshoot
                grad_tol = abs((overP)*dirD0)

                # check initial bounds
                a = init_guess*1.0
                if a > max_step:
                    a = max_step*1.0

                if a < min_step:
                    a = min_step*1.0

                # evaluation at initial guess
                loss, abs_grad_sum, dirD, flat_grad = lossEval(a,0,direction,closure)
                current_evals += 1
                aold = a*1.0


                # set increase or decrease mode
                if dirD.item() < grad_tol and a < max_step:
                    flag = 2
                elif dirD.item() >= grad_tol and a > min_step:
                    flag = 1
                else:
                    flag = 0

                # immediate accept condition
                #if dirD.item() > grad_tol*0.0 and dirD.item() < grad_tol:
                    #flag = 0


                ##################################################################
                # IGOLS LOOP
                ##################################################################
                while flag and abs_grad_sum>=tolerance_grad:
                    if flag == 2:
                        # INCREASE STEP SIZE
                        a = a*eta
                        loss, abs_grad_sum, dirD, flat_grad = lossEval(a,aold,direction,closure)
                        current_evals += 1
                        aold = a*1.0

                        if dirD.item() >= grad_tol:
                            flag = 0
                            #a = a/eta;
                            #loss, abs_grad_sum, dirD, flat_grad = lossEval(a,aold,direction,closure)
                            #aold = a*1.0
                        elif a > max_step/eta:
                            flag = 0

                    elif flag == 1:
                        # DECREASE STEP SIZE
                        a = a/eta
                        loss, abs_grad_sum, dirD, flat_grad = lossEval(a,aold,direction,closure)
                        current_evals += 1
                        aold = a*1.0

                        if dirD.item() < grad_tol:
                            flag = 0
                        elif a < min_step*eta:
                            flag = 0



            # END OF LINE SEARCH
            state['postDD'] = dirD.item()

            # LBFGS check conditions
            if alg == 'LBFGS':

                if n_iter == max_iter:
                    break
                if current_evals >= max_eval:
                    print('break2, max. evals.')
                    break
                if abs_grad_sum <= tolerance_grad:
                    print('break3, grad tol.')
                    break
                if direction.mul(a).abs_().sum() <= tolerance_change:
                    print('break4, update tol.')
                    break
                if abs(loss - prev_loss) < tolerance_change:
                    print('break5, loss diff. tol.')
                    break


        ##################################################################
        # SAVING OF PARAMETERS
        ##################################################################


        # tensors cached in state (for tracing)
        state['loss'] = loss
        state['gradient'] = flat_grad
        state['abs_grad_sum'] = abs_grad_sum

        state['func_evals'] += current_evals
        state['d'] = direction
        state['a'] = a


        if alg == 'LBFGS':
            state['old_dirs'] = old_dirs
            state['old_stps'] = old_stps
            state['H_diag'] = H_diag
            state['prev_flat_grad'] = prev_flat_grad
            state['prev_loss'] = prev_loss



        return loss, state






