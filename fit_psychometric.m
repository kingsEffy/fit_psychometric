function fit = fit_psychometric(coh, choiceRight, varargin)
% Fit Marques et al. (2018) psychometric in MATLAB (trial-wise MLE).
% coh:           Nx1 signed coherences (e.g., [-100 ... 100])
% choiceRight:   Nx1 binary (1 = Right/Temporal, 0 = Left/Nasal)
% Optional name/value: 'MaxLapse', default 0.20
% BASED ON _Marques2018;

opts.MaxLapse = 0.20;
if ~isempty(varargin), opts = parseOpts(opts, varargin{:}); end

coh = coh(:);             % column
y   = logical(choiceRight(:));
assert(numel(coh)==numel(y), 'coh and choiceRight must match length');

% drop NaNs or "no-choice" rows if any are encoded that way
valid = ~isnan(coh) & ~isnan(y);
coh = coh(valid); y = y(valid);

% Re-parameterise to avoid bounds:
% sigma = softplus(s) + eps; lambda = maxLapse * sigmoid(a)
softplus   = @(x) log1p(exp(-abs(x))) + max(x,0);        % numerically stable
sigmoid    = @(x) 1./(1+exp(-x));
Phi        = @(z) 0.5*(1+erf(z./sqrt(2)));

negLL = @(u) nll_from_u(u, coh, y, Phi, softplus, sigmoid, opts.MaxLapse);

% Initial guess: [mu, s, a_left, a_right]
u0 = [0, log(20), logit(0.02/opts.MaxLapse), logit(0.02/opts.MaxLapse)];
% Nelder–Mead (fminsearch) = no toolboxes
opt = optimset('Display','off','TolX',1e-5,'TolFun',1e-5,'MaxFunEvals',5e4,'MaxIter',5e4);
[u_hat, fval] = fminsearch(negLL, u0, opt);

% Map back to real parameters
[mu, sigma, lamL, lamR] = unpack(u_hat, softplus, sigmoid, opts.MaxLapse);

% Outputs
fit.mu        = mu;
fit.sigma     = sigma;
fit.lambda_left  = lamL;
fit.lambda_right = lamR;
fit.negloglik = fval;
fit.model     = @(c) lamR + (1-lamR-lamL).*Phi((c-mu)./sigma);
% Lapse-corrected 75% threshold (point where Phi = 0.75)
fit.c75       = mu + 0.67448975 * sigma;
% Slope at PSE (derivative of J at c = mu)
phi0          = 1/sqrt(2*pi);
fit.slope_at_mu = (1 - lamR - lamL) * (phi0 / sigma);

% Simple bootstrap SEs (optional): set nBoot>0 to enable
fit.bootstrap = @(nBoot) bootstrapSE(coh, y, nBoot, u_hat, opts.MaxLapse);

% Helper functions
    function v = logit(p)
        p = min(max(p,1e-6),1-1e-6);
        v = log(p./(1-p));
    end
    function nll = nll_from_u(u, c, yy, Phi, softplus, sigmoid, maxLapse)
        [mu_, sigma_, lamL_, lamR_] = unpack(u, softplus, sigmoid, maxLapse);
        p = lamR_ + (1-lamR_-lamL_) .* Phi((c - mu_)./sigma_);
        p = min(max(p, 1e-6), 1-1e-6);
        nll = -sum( yy.*log(p) + (1-yy).*log(1-p) );
    end
    function [mu_, sigma_, lamL_, lamR_] = unpack(u, softplus, sigmoid, maxLapse)
        mu_    = u(1);
        sigma_ = softplus(u(2)) + eps;
        lamL_  = maxLapse * sigmoid(u(3));
        lamR_  = maxLapse * sigmoid(u(4));
        % keep total lapse <= 0.4 implicitly via maxLapse (each <=0.2 by default)
    end
    function se = bootstrapSE(c, yy, nBoot, ustar, maxL)
        if nargin<3 || nBoot<=0, se=[]; return; end
        pars = zeros(nBoot,4);
        for b=1:nBoot
            idx = randi(numel(yy), numel(yy),1);
            cb  = c(idx); yb = yy(idx);
            nb  = @(u) nll_from_u(u, cb, yb, Phi, softplus, sigmoid, maxL);
            ub  = fminsearch(nb, ustar, opt);
            [mu_b, sig_b, lamL_b, lamR_b] = unpack(ub, softplus, sigmoid, maxL);
            pars(b,:) = [mu_b, sig_b, lamL_b, lamR_b];
        end
        se = struct('mu',std(pars(:,1)), 'sigma',std(pars(:,2)), ...
                    'lambda_left',std(pars(:,3)), 'lambda_right',std(pars(:,4)));
    end
end

function S = parseOpts(S, varargin)
for k=1:2:numel(varargin)
    S.(varargin{k}) = varargin{k+1};
end
end
