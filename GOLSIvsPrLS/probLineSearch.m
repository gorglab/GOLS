function [outs, alpha0_out, y_tt, dy_tt, x_tt, var_f_tt, var_df_tt] = ...
        probLineSearch(func, x0, f0, df0, search_direction, alpha0, verbosity, ...
        outs, paras, var_f0, var_df0)
% probLineSearch.m -- A probabilistic line search algorithm for nonlinear
% optimization problems with noisy gradients. 
%
% == INPUTS ===============================================================
% [f,f', var_f, var_df] = func(x) -- function handle 
% input: 
%     x -- column vectors (positions) (Dx1) 
% output: 
%     f -- scalar function values
%     df -- vector gradients (Dx1)
%     var_f -- estimated noise for function values (scalar)
%     var_df -- estimated noise for gradients (Dx1)
% x0 -- current position of optimizer (Dx1)
% f0 -- function value at x0 (scalar, previous output y_tt)
% df0 -- gradient at x0 ((Dx1), previous output dy_tt)
% search_direction -- (- df(x0) does not need to be normalized)
% alpha0: initial step size (scalar, previous output alpha0_out)
% var_f0 -- variance of function values at x0. (scalar, previous output var_f_tt)
% var_df0 -- variance of gradients at x0. ((Dx1), previous output var_df_tt)
% verbosity -- level of stdout output. 
%         0 -- no output
%         1 -- steps, function values, state printed to stdout
%         2 -- plots with only function values
%         3 -- plots including Gaussian process and Wolfe condition beliefs.
% paras -- possible parameters that func needs.
% outs -- struct with collected statistics 
%
% == OUTPUTS ==============================================================
% outs -- struct including counters and statistics
% alpha0_out -- accepted stepsize * 1.3 (initial step size for next step)
% x_tt -- accepted position
% y_tt -- functin value at x_tt
% dy_tt -- gradient at x_tt
% var_f_tt -- variance of function values at x_tt
% var_df_tt -- variance of gradients values at x_tt
%
% Copyright (c) 2015 (post NIPS 2015 release 4.0), Maren Mahsereci, Philipp Hennig 
% mmahsereci@tue.mpg.de, phennig@tue.mpg.de
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%     * Redistributions of source code must retain the above copyright
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


% -- vain plotting stuff --------------------------------------------------
if verbosity >=2
    mpg = [0,0.4717,0.4604]; % color [0,125,122]
    dre = [0.4906,0,0]; % color [130,0,0]
    ora = [255,153,51] ./ 255;
    blu = [0,0,0.509];
    gra = 0.5 * ones(3,1);

    lightmpg = [1,1,1] - 0.5 * ([1,1,1] - mpg);
    lightdre = [1,1,1] - 0.5 * ([1,1,1] - dre);
    lightblu = [1,1,1] - 0.5 * ([1,1,1] - blu);
    lightora = [1,1,1] - 0.5 * ([1,1,1] - ora);

    mpg2white = bsxfun(@minus,[1,1,1],bsxfun(@times,(linspace(0,0.6,2024)').^0.5,[1,1,1]-mpg));
    dre2white = bsxfun(@minus,[1,1,1],bsxfun(@times,(linspace(0,0.6,2024)').^0.5,[1,1,1]-dre));
    blu2white = bsxfun(@minus,[1,1,1],bsxfun(@times,(linspace(0,0.6,2024)').^0.5,[1,1,1]-blu));
    ora2white = bsxfun(@minus,[1,1,1],bsxfun(@times,(linspace(0,0.6,2024)').^0.5,[1,1,1]-ora));

    cya2black = bsxfun(@times,(linspace(0,0.6,2024)').^0.5,lightmpg);
    red2black = bsxfun(@times,(linspace(0,0.6,2024)').^0.5,lightdre);
    blu2black = bsxfun(@times,(linspace(0,0.6,2024)').^0.5,lightblu);
    ora2black = bsxfun(@times,(linspace(0,0.6,2024)').^0.5,lightora);

    GaussDensity = @(y,m,v) ...
        (bsxfun(@rdivide,exp(-0.5*bsxfun(@rdivide,bsxfun(@minus,y,m').^2,v'))./sqrt(2*pi),sqrt(v')));
end

% -- setup fixed parameters -----------------------------------------------
if ~isfield(outs, 'counter')
    outs.counter = 1;
end
if isempty(verbosity)
    verbosity = 0;
end
if ~isfield(outs, 'alpha_stats')
    outs.alpha_stats = alpha0; % running average over accepted step sizes
end
if ~isfield(outs, 'counter')
    outs.counter = 1; % counts total number of function evaluations
end
limit = 6; % maximum #function evaluations in one line search (+1)

% constants for Wolfe conditions (must be chosen 0 < c1 < c2 < 1)
c1 = 0.05;   % <---- DECIDED FIXED 0.05
c2 = 0.5;    % <---- DECIDED FIXED 0.8
% c2 = 0 extends until ascend location reached: lots of extrapolation
% c2 = 1 accepts any point of increased gradient: almost no extrapolation

WolfeThreshold = 0.3; % <---- DECIDED FIXED (0.3)
% the new free parameter of this method relative to sgd: 
% search is terminated when probability(Wolfe conditions hold) > WolfeThreshold
% not sensitive between 0.1 and 0.8 (0 = accept everyting, 1= accept nothing)

offset  = 10; % off-set, for numerical stability. 

EXT = 1; % extrapolation factor
tt  = 1; % initial step size in scaled space

% -- set up GP ------------------------------------------------------------
% create variables with shared scope. Ugly, but necessary because
% matlab does not discover shared variables if they are first created in a
% nested function.
d2m = []; d3m = []; V = []; Vd = []; dVd = [];
m0  = []; dm0 = []; V0= [];Vd0 = []; dVd0= [];
V0f = []; Vd0f= []; V0df=[]; Vd0df = [];

% kernel:
k  = @(a,b) (3 \ min(a+offset,b+offset).^3 + 0.5 * abs(a-b) .* min(a+offset,b+offset).^2);
kd = @(a,b) ((a<b) .* ((a+offset).^2/2) + (a>=b) .* ((a+offset)*(b+offset) - 0.5 .* (b+offset).^2));
dk = @(a,b) ((a>b) .* ((b+offset).^2/2) + (a<=b) .* ((a+offset)*(b+offset) - 0.5 .* (a+offset).^2));
dkd= @(a,b) (min(a+offset,b+offset));

% further derivatives
ddk = @(a,b) (a<=b) .* (b-a);
ddkd= @(a,b) (a<=b);
dddk= @(a,b) -(a<=b);

% -- helper functions -----------------------------------------------------

GaussCDF = @(z) 0.5 * (1 + erf(z/sqrt(2)));
GaussPDF = @(z) exp( - 0.5 * z.^2 ) ./ sqrt(2*pi);
EI       = @(m,s,eta) (eta - m) .* GaussCDF((eta-m)./s) + s .* GaussPDF((eta-m)./s);

% -- scale ----------------------------------------------------------------
beta = abs(search_direction'*df0); % scale f and df according to 1/(beta*alpha0)

% disp('ab b')
% beta
% alpha0
% disp('ab e')

% -- scaled noise ---------------------------------------------------------
sigmaf  = sqrt(var_f0)/(alpha0*beta); 
sigmadf = sqrt((search_direction.^2)'*var_df0)/beta; 

% -- initiate data storage ------------------------------------------------
T            = 0;
Y            = 0; 
dY           = df0;
dY_projected = (df0'*search_direction)/beta;  
Sigmaf       = var_f0;
Sigmadf      = var_df0;
N            = 1;

% -- update GP with new datapoint -----------------------------------------
updateGP();

% -- search (until budget used or converged) ------------------------------
while N < limit+1

    % -- evaluate function (change minibatch!) ----------------------------
    evaluate_function()    
    
    % -- update GP with new datapoint -------------------------------------
    updateGP(); % store, update belief
    
    % -- check last evaluated point for acceptance ------------------------
    if probWolfe(tt) > WolfeThreshold % are we done?
        if verbosity > 0; makePlot(); disp('found acceptable point.'); end
        make_outs(y, dy, var_f, var_df);
        return; % done.   
    end
    
    % -- find candidates (set up for EI) ----------------------------------
    % decide where to probe next: evaluate expected improvement and prob
    % Wolfe conditions at a number of candidate points, take the most promising one.
    
    % lowest mean evaluation, and its gradient (needed for EI):
    M  = zeros(N,1); 
    dM = zeros(N,1);
    for l = 1:N
        M(l)  = m(T(l)); 
        dM(l) = d1m(T(l)); 
    end
    [minm,minj] = min(M);    % minm: lowest GP mean, minj: index in candidate list
    tmin        = T(minj);   % tt of point with lowest GP mean of function values
    dmin        = dM(minj);  % GP mean of gradient at lowest point
    
    % -- check this point as well for acceptance --------------------------
    if abs(dmin) < 1e-5 && Vd(tmin) < 1e-4 % nearly deterministic
        tt = tmin; dy = dY(:, minj); y = Y(minj); var_f = Sigmaf(minj); var_df = Sigmadf(:, minj);        
        disp('found a point with almost zero gradient. Stopping, although Wolfe conditions not guaranteed.')
        make_outs(y, dy, var_f, var_df);
        return;
    end
    
    % -- find candidates --------------------------------------------------
    % CANDIDATES 1: minimal means between all evaluations:
    % iterate through all `cells' (O[N]), check minimum mean locations.
    Tcand  = []; % positions of candidate points
    Mcand  = []; % means of candidate points
    Scand  = []; % standard deviation of candidate points
    Tsort  = sort(T); 
    Wolfes = []; % list of acceptable points.
    for cel = 1:N-1 % loop over cells
        Trep = Tsort(cel) + 1e-6 * (Tsort(cel+1) - Tsort(cel));
        cc   = cubicMinimum(Trep);
        % add point to candidate list if minimum lies in between T(cel) and T(cel+1)
        if cc > Tsort(cel) && cc < Tsort(cel+1); 
            Tcand = [Tcand, cc];
            Mcand = [Mcand, m(cc)];
            Scand = [Scand, sqrt(V(cc))];
            
        else % no minimum, just take half-way
            if cel==1 && d1m(0) > 0 % only in first cell
                if verbosity > 0; disp 'function seems very steep, reevaluating close to start.'; end;
                Tcand = 0.01 * (Tsort(cel) + Tsort(cel+1)); 
                Mcand = [Mcand, m(0.01 * (Tsort(cel) + Tsort(cel+1)))];
                Scand = [Scand, sqrt(V(0.01 * (Tsort(cel) + Tsort(cel+1))))];                            
                
                tt = Tcand;
                % -- return instead ---------------------------------------
                evaluate_function();

                % -- prepare output -----------------------------------------------
                dy = dY(:, T == tt); y = Y(T == tt); var_f = Sigmaf(T == tt); var_df = Sigmadf(:, T==tt);    
                make_outs(y, dy, var_f, var_df);
                return; % done 
            end
        end
        
        % check whether there is an acceptable point among already
        % evaluated points (since this changes each time the GP gets updated)
        if cel > 1 && (probWolfe(Tsort(cel)) > WolfeThreshold) 
            Wolfes = [Wolfes,Tsort(cel)]; % list of acceptable points.
        end
    end
    
    % -- check if at least on point is acceptable -------------------------
    if ~isempty(Wolfes)
        if verbosity > 0 
            makePlot(); 
            disp('found acceptable point.'); 
        end
        
        % -- chose best point among Wolfes, return. -----------------------
        MWolfes = 0 * Wolfes;
        for o = 1:length(Wolfes)
            MWolfes(o) = m(Wolfes(o)); % compute GP means of acceptable points
        end
        tt = Wolfes(MWolfes == min(MWolfes));
                
        % find corresponding gradient and variances
        dy = dY(:, T == tt); y = Y(T == tt); var_f = Sigmaf(T == tt); var_df = Sigmadf(:, T==tt);    
        make_outs(y, dy, var_f, var_df);
        return; 
    end
        
    % Candidate 2: one extrapolation step
    Tcand = [Tcand, max(T) + EXT];
    Mcand = [Mcand, m(max(T) + EXT)];
    Scand = [Scand, sqrt(V(max(T)+EXT))]; 
    
    % -- discriminate candidates through EI and prob Wolfe ----------------
    EIcand = EI(Mcand, Scand, minm); % minm: lowest GP mean of collected evaluations (not candidates)
    PPcand = zeros(size(EIcand));
    for ti = 1:length(EIcand)
        PPcand(ti) = probWolfe(Tcand(ti)); 
    end
    
    [~,idx_best] = max(EIcand .* PPcand); % find best candidate
    
    if Tcand(idx_best) == tt + EXT; % extrapolating. Extend extrapolation step
       EXT = 2 * EXT;
    end

    tt = Tcand(idx_best);
    
    makePlot(); 
    
end

% -- algorithm reached limit without finding acceptable point. ------------
% Evaluate a final time, return "best" point (one with lowest function value)
evaluate_function()

% -- update GP with new datapoint -----------------------------------------
updateGP(); 

% -- check last evaluated point for acceptance ----------------------------
if probWolfe(tt) > WolfeThreshold % are we done?
    if verbosity > 0; disp('found acceptable point right at end of budget. Phew!'); end
    make_outs(y, dy, var_f, var_df);
    return; % done.
end

% -- return point with lowest mean ----------------------------------------
M  = inf(N,1); 
for ii = 2:N
    M(ii) = m(T(ii)); % compute all GP means of all evaluated locations
end
[~, minj] = min(M);  % index of minimal GP mean
if verbosity > 0 
    warning('reached evaluation limit. Returning ''best'' known point.');
end

% find corresponding tt, gradient and noise
tt = T(minj); dy = dY(:, minj); y = Y(minj);  var_f = Sigmaf(minj); var_df = Sigmadf(:, minj);
make_outs(y, dy, var_f, var_df);

makePlot();

% *************************************************************************
function evaluate_function()

    outs.counter = outs.counter + 1;
    [y, dy, var_f, var_df] = func(x0 + tt*alpha0*search_direction, paras); % y: function value at tt
    
    if isinf(y) || isnan(y)
        % this does not happen often, but still needs a fix
        % e.g. if function return inf or nan (probably too large step), 
        % do function value clipping relative to the intial value, 
        % e.g. y = 1e3*f0. 
        y = 1e3*f0; 
        %error('function values is inf or nan.')
    end
    
    % -- scale f and df ---------------------------------------------------
    y            = (y - f0)/(alpha0*beta);        % substract offset    
    dy_projected = (dy'*search_direction)/beta;   % projected gradient 
    
    % -- store ------------------------------------------------------------
    T            = [T; tt]; 
    Y            = [Y; y]; 
    dY           = [dY, dy];
    dY_projected = [dY_projected; dy_projected]; 
    Sigmaf       = [Sigmaf; var_f];
    Sigmadf      = [Sigmadf, var_df];
    N            = N + 1;
    
end

% -- helper functions -----------------------------------------------------
function updateGP() % using multiscope variables to construct GP

    % build Gram matrix
    kTT   = zeros(N); 
    kdTT  = zeros(N); 
    dkdTT = zeros(N);
    for i = 1:N
        for j = 1:N
            kTT(i,j)   = k(T(i),  T(j));
            kdTT(i,j)  = kd(T(i), T(j));
            dkdTT(i,j) = dkd(T(i),T(j));
        end
    end
    
    % build noise matrix
    Sig = sigmaf^2 * ones(2*N, 1); Sig(N+1:end) = sigmadf^2;
    
    % build Gram matrix
%     sigmaf^2
    G = diag(Sig) + [kTT, kdTT; kdTT', dkdTT];
%     diagM = diag(Sig)
%    addM = [kTT, kdTT; kdTT', dkdTT]
%     rank(G)   
%     var_f0
%     norm(var_df0)
%     rank([Y; dY_projected])
%     pause
    A = G \ [Y; dY_projected];

    % posterior mean function and all its derivatives
    m   = @(t) [k(t, T')   ,  kd(t,  T')] * A;
    d1m = @(t) [dk(t, T')  , dkd(t,  T')] * A;
    d2m = @(t) [ddk(t, T') ,ddkd(t,  T')] * A;
    d3m = @(t) [dddk(t, T'),zeros(1, N)]  * A;
    
    
    
    % posterior marginal covariance between function and first derivative
    V   = @(t) k(t,t)   - ([k(t, T') ,  kd(t, T')] * (G \ [k(t, T') , kd(t, T')]'));
    Vd  = @(t) kd(t,t)  - ([k(t, T') ,  kd(t, T')] * (G \ [dk(t, T'),dkd(t, T')]'));
    dVd = @(t) dkd(t,t) - ([dk(t, T'), dkd(t, T')] * (G \ [dk(t, T'),dkd(t, T')]'));
       
    % belief at starting point, used for Wolfe conditions
    m0   = m(0);
    dm0  = d1m(0);    
    V0   = V(0);
    Vd0  = Vd(0);
    dVd0 = dVd(0);
    
    % covariance terms with function (derivative) values at origin
    V0f   = @(t) k(0,t)  - ([k(0, T') ,  kd(0, T')] * (G \ [k(t, T') , kd(t, T')]'));
    Vd0f  = @(t) dk(0,t) - ([dk(0, T'), dkd(0, T')] * (G \ [k(t, T') , kd(t, T')]'));
    V0df  = @(t) kd(0,t) - ([k(0, T'),   kd(0, T')] * (G \ [dk(t, T'),dkd(t, T')]'));
    Vd0df = @(t) dkd(0,t)- ([dk(0, T'), dkd(0, T')] * (G \ [dk(t, T'),dkd(t, T')]'));
end

function [p,p12] = probWolfe(t) % probability for Wolfe conditions to be fulfilled

    % marginal for Armijo condition
    ma  = m0 - m(t) + c1 * t * dm0;
    Vaa = V0 + (c1 * t).^2 * dVd0 + V(t) + 2 * (c1 * t * (Vd0 - Vd0f(t)) - V0f(t));
%     Vaa

%     Mata = [k(t, T') , kd(t, T')];
%      Matb = [dk(t, T'),dkd(t, T')];
%     Matc = [k(t, T') , kd(t, T')];
%     Matd = [dk(t, T'),dkd(t, T')];
%     
%     rank(Mata)
%     rank(Matb)
%     rank(Matc)
%     rank(Matd)
%     size(Mata)
%     size(Matb)
%     size(Matc)
%     size(Matd)


    % marginal for curvature condition
    mb  = d1m(t) - c2 * dm0;
    Vbb = c2^2 * dVd0 - 2 * c2 * Vd0df(t) + dVd(t);
    
    % covariance between conditions
    Vab = -c2 * (Vd0 + c1 * t * dVd0) + V0df(t) + c2 * Vd0f(t) + c1 * t * Vd0df(t) - Vd(t);
                                     
    if (Vaa < 1e-9) && (Vbb < 1e-9) % deterministic evaluations
        p = (ma >= 0) .* (mb >= 0);
        return
    end
    
    % joint probability 
    rho = Vab / sqrt(Vaa * Vbb);
    if Vaa <= 0 || Vbb <= 0
        p   = 0; 
        p12 = [0,0,0]; 
        return;
    end
    upper = (2 * c2 * (abs(dm0)+2*sqrt(dVd0))-mb)./sqrt(Vbb);
    p = bvn(-ma / sqrt(Vaa), inf, -mb / sqrt(Vbb), upper, rho);
    
    if nargout > 1
        % individual marginal probabilities for each condition 
        % (for debugging)
        p12 = [1 - GaussCDF(-ma/sqrt(Vaa)), ....
            GaussCDF(upper)-GaussCDF(-mb/sqrt(Vbb)),...
            Vab / sqrt(Vaa * Vbb)];
    end
end

function tm = cubicMinimum(ts) 
    % mean belief at ts is a cubic function. It is defined up to a constant by
    d1mt = d1m(ts);
    d2mt = d2m(ts);
    d3mt = d3m(ts);
    
    a = 0.5 * d3mt;
    b = d2mt - ts * d3mt;
    c = d1mt - d2mt * ts + 0.5 * d3mt * ts^2;
    
    if abs(d3mt) < 1e-9 % essentially a quadratic. Single extremum
        tm = - (d1mt - ts * d2mt) / d2mt;
        return; 
    end
    
    % compute the two possible roots:
    detmnt = b^2 - 4*a*c;
    if detmnt < 0 % no roots
        tm = inf; 
        return;
    end
    LR = (-b - sign(a) * sqrt(detmnt)) ./ (2*a);  % left root
    RR = (-b + sign(a) * sqrt(detmnt)) ./ (2*a);  % right root
    
    % and the two values of the cubic at those points (up to constant)
    Ldt = LR - ts; % delta t for left root
    Rdt = RR - ts; % delta t for right root
    LCV = d1mt * Ldt + 0.5 * d2mt * Ldt.^2 + 6 \ d3mt * Ldt.^3; % left cubic value
    RCV = d1mt * Rdt + 0.5 * d2mt * Rdt.^2 + 6 \ d3mt * Rdt.^3; % right cubic value
    
    if LCV < RCV
        tm = LR; 
    else
        tm = RR; 
    end;
    
end

function makePlot()
    
if verbosity == 2
    clf; hold on;  
    ymin = min(Y) - 0.1*(max(Y)-min(Y));
    ymax = max(Y) + 0.1*(max(Y)-min(Y));
    
    xmin = -0.1;
    xmax = max([T;tt])+0.5;
    
    % plot evaluation points
    plot(T,Y,'o','Color',blu)
    for i = 1:N
       plot(T(i) + 0.1* [-1,1],Y(i) + 0.1*dY_projected(i) * [-1,1],'-','Color',blu);
    end
    plot(tt,m(tt),'o','Color',dre,'MarkerFaceColor',dre);
    
    xlim([xmin,xmax]);
    ylim([ymin,ymax]);
    drawnow;
end
    
if verbosity > 2
    clf; 
    subplot(321); hold on; 
    title('belief over function'); 
    
    ymin = min(Y) - 1.5*(max(Y)-min(Y));
    ymax = max(Y) + 1.5*(max(Y)-min(Y));
    
    xmin = -0.1;
    xmax = max([T;tt])+0.5;
    
    % also plot GP
    Np = 120;
    tp = linspace(xmin,xmax,Np)';
    ts = unique([tp; T(:)]);
    Ns = length(ts);
    ms = zeros(Ns,1);
    Vs = zeros(Ns,1);
    PP = zeros(Ns,1);
    Pab = zeros(Ns,3);
    mp = zeros(Np,1);
    Vp = zeros(Np,1);
    for i = 1:Np
        mp(i) = m(tp(i));
        Vp(i) = V(tp(i));
    end
    for i = 1:Ns
        ms(i) = m(ts(i));
        Vs(i) = V(ts(i));
        [PP(i),Pab(i,:)] = probWolfe(ts(i));
    end
    yp= linspace(ymin,ymax,250)';
    P = GaussDensity(yp,mp,Vp+1e-4);

    imagesc(tp,yp,P); colormap(ora2white);
    plot(ts,ms,'-','Color',ora);
    plot(ts,ms+2*sqrt(max(Vs,0)),'-','Color',lightora);
    plot(ts,ms-2*sqrt(max(Vs,0)),'-','Color',lightora);
    
    % plot evaluation points
    plot(T,Y,'o','Color',blu)
    for i = 1:N
       plot(T(i) + 0.1* [-1,1],Y(i) + 0.1*dY_projected(i) * [-1,1],'-','Color',blu);
    end
    plot(tt,m(tt),'o','Color',dre,'MarkerFaceColor',dre);
    plot([tt,tt],[ymin,ymax],'-','Color',dre);

    
    
    try
    % plot candidate points
    plot(Tcand,Mcand,'o','Color',gra);
    catch
    end
    
    xlim([xmin,xmax]);
    ylim([ymin,ymax]);
    
    subplot(322); hold on; 
    title('belief over Wolfe conditions'); 
    plot(ts,PP,'-','Color',dre);
    plot(ts,Pab(:,1),'--','Color',dre);
    plot(ts,Pab(:,2),'-.','Color',dre);
    plot(ts,0.5 + 0.5*Pab(:,3),':','Color',dre);
    
    plot(ts,0*ts + WolfeThreshold,'-','Color',gra)
    for i = 1:N
        plot([T(i),T(i)],[0,1],'-','Color',blu);
    end
    plot([tt,tt],[0,1],'-','Color',dre);
    ylim([0,1]);
    xlim([xmin,xmax])
    
    subplot(323); hold on;
    title('Expected Improvement')
    
    eta = min(Y);
    Ss  = sqrt(Vs + sigmaf^2);
    plot(ts,EI(ms,Ss,eta),'-','Color',mpg);
    plot(ts,EI(ms,Ss,eta) .* PP,'--','Color',blu); 
    yli = ylim();
    try
    for o = 1:length(Tcand)
        plot([Tcand(o),Tcand(o)],yli,'-','Color',gra);
    end
    catch
    end
    plot([tt,tt],yli,'-','Color',dre);
    xlim([xmin,xmax]);

    % for debugging:
    subplot(324); hold on; title('beliefs over derivatives')
    dms = zeros(Ns,1); 
    Vms = zeros(Ns,1);
    rhoD = zeros(Ns,1);
    
    V0s   = zeros(Ns,1);
    Vd0s  = zeros(Ns,1);
    V0ds  = zeros(Ns,1);
    Vd0ds = zeros(Ns,1);
    
    ma  = zeros(Ns,1);
    mb  = zeros(Ns,1);
    Vaa = zeros(Ns,1);
    Vbb = zeros(Ns,1);
    Vab = zeros(Ns,1);
    for i = 1:Ns
        dms(i)   = d1m(ts(i));
        Vms(i)   = dVd(ts(i));
        rhoD(i)  = Vd(ts(i)) ./ sqrt(V(ts(i)) * dVd(ts(i)));
        V0s(i)   = V0f(ts(i)) ./ sqrt(V0 * V(ts(i)));
        Vd0s(i)  = Vd0f(ts(i))./ sqrt(dVd0 * V(ts(i)));
        V0ds(i)  = V0df(ts(i))./ sqrt(V0 * dVd(ts(i)));
        Vd0ds(i) = Vd0df(ts(i))./sqrt(dVd0 * dVd(ts(i)));
        
        ma(i)    = m0 - m(ts(i)) + c1 * ts(i) * dm0;
        mb(i)    = d1m(ts(i)) - c2 * dm0;
        Vaa(i)   = V0 + (c1 * ts(i)).^2 * dVd0 + V(ts(i)) + 2 * (c1 * ts(i) * (Vd0 - Vd0f(ts(i))) - V0f(ts(i)));
        Vbb(i)   = c2^2 * dVd0 - 2 * c2 * Vd0df(ts(i)) + dVd(ts(i));
        Vab(i)   = -c2 * (Vd0 + c1 * ts(i) * dVd0) + (1+c2) * Vd0f(ts(i)) + c1 * ts(i) * Vd0df(ts(i)) - Vd(ts(i));
    end
    plot(ts,dms,'-','Color',ora);
    plot(ts,dms+2*sqrt(Vms),'-','Color',lightora);
    plot(ts,dms-2*sqrt(Vms),'-','Color',lightora);
    plot(T,dY_projected,'o','Color',blu);
    
    xlim([xmin,xmax]); 
    
    subplot(325); hold on; title('covariances')
    plot(ts,rhoD,'-','Color',ora);
    plot(ts,V0s,'-','Color',mpg);
    plot(ts,Vd0s,'-.','Color',mpg);
    plot(ts,V0ds,'--','Color',mpg);
    plot(ts,Vd0ds,':','Color',mpg);
    plot(ts,-1+0*ts,'-k');
    plot(ts,1+0*ts,'-k');
    
    xlabel 't';
    legend('\rho_{f\partial}(t)','\rho_{f_0f}(t)','\rho_{\partial_0f}(t)',...
        '\rho_{f_0\partial}(t)','\rho_{\partial_0\partial}(t)')
    
    xlim([xmin,xmax]); ylim([-1.1;1.1]);
    
    subplot(326); hold on; title('Wolfe terms')
    plot(ts,ma,'-','Color',mpg);
    plot(ts,mb,'-','Color',dre);
    plot(ts,ma+2*sqrt(Vaa),'-','Color',lightmpg);
    plot(ts,ma-2*sqrt(Vaa),'-','Color',lightmpg);
    plot(ts,mb+2*sqrt(Vbb),'-','Color',lightdre);
    plot(ts,mb-2*sqrt(Vbb),'-','Color',lightdre);
    plot(ts,sqrt(abs(Vab)),':','Color',blu);
    
    xlim([xmin,xmax]); 
    drawnow; 
    
end
end

function make_outs(y, dy, var_f, var_df)
    
    x_tt      = x0 + tt*alpha0*search_direction; % accepted position
    y_tt      = y*(alpha0*beta) + f0;            % function value at accepted position
    dy_tt     = dy;                              % gradient at accepted position
    var_f_tt  = var_f;                           % variance of function value at accepted position
    var_df_tt = var_df;                          % variance of gradients at accepted position
    
    % set new set size
    % next initial step size is 1.3 times larger than last accepted step size
    alpha0_out = tt*alpha0 * 1.3;

    % running average for reset in case the step size becomes very small
    % this is a saveguard
    gamma = 0.95;
    outs.alpha_stats = gamma*outs.alpha_stats + (1-gamma)*tt*alpha0;

    % reset NEXT initial step size to average step size if accepted step
    % size is 100 times smaller or larger than average step size
    if (alpha0_out > 1e2*outs.alpha_stats)||(alpha0_out < 1e-2*outs.alpha_stats)
        if verbosity > 0; disp 'making a very small step, resetting alpha0'; end;
        alpha0_out = outs.alpha_stats; % reset step size
    end
end
end


