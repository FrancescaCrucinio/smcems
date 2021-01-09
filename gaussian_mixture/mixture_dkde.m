% Comparison of MISE and runtime for deterministic and stochastic
% discretisation of EMS

% set seed
rng('default');

% f, h, g are Normals
h = @(y) 2*normpdf(y, 0.3, sqrt(0.043^2 + 0.045^2))./3 + ...
    normpdf(y, 0.5, sqrt(0.015^2 + 0.045^2))./3;
g = @(x,y) normpdf(y, x, 0.045);
f = @(x) normpdf(x, 0.3, 0.015)/3 + normpdf(x, 0.5, 0.043)*2/3;

% number of bins/particles
N = 1000;
% bin centers
fBin = 1/(2*N):1/N:1;
% sample from h
hSample = Ysample_gaussian_mixture(10^5);

% DKDE
y = randsample(hSample, N, true);
sigU = 0.045;
varU = sigU^2;
%PI bandwidth of Delaigle and Gijbels
hPI = PI_deconvUknownth4(y, 'norm', varU, sigU);

%DKDE estimator with rescaling
dx = fBin(2) - fBin(1);
fdecPI = fdecUknown(fBin, y, hPI, 'norm', sigU, dx);

%CV bandwidth of Stefanski and Carroll
hCV = CVdeconv(y, 'norm', sigU);
fdecCV = fdecUknown(fBin, y, hCV, 'norm', sigU, dx);

fplot(f, [0, 1])
hold on
plot(fBin, fdecPI)
plot(fBin, fdecCV)