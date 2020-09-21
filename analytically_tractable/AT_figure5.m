%%% Comparison of reconstruction for EM, EMS and SMC

% set seed
rng('default');

% variances for f, g, h
sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaG + sigmaF;
% f, g, h are Normals
h = @(y) normpdf(y, 0.5, sqrt(sigmaH));
g = @(x,y) normpdf(y, x, sqrt(sigmaG));
f = @(x) normpdf(x, 0.5, sqrt(sigmaF));

%%% set parameters
% EM/EMS
% number of iterations
Niter = 100;
% number of bins
Nbins = 100;
% bin centers
pp = 1/(2*Nbins):1/Nbins:1;
% scale for smoothing
epsilon = 1e-02;
% smoothing kernel
K = @(x1, x2) normpdf(x2, x1, epsilon);
% smoothing matrix
Kmatrix = smoothingMatrix(K, pp);
% SMC
% number of particles
Nparticles = 10000;

% get exact potenrial
% initial variance of approximation of f
sigmaF1 = rand(1);
% exact variance for f_n(x)
[~, exactVarianceF] = AT_exact_variance_recursion(sigmaH, sigmaG, ...
    sigmaF1, epsilon^2, Niter);
% exact variance for h_n(y)
exactVarianceH = exactVarianceF + sigmaG;

% discretisation of h
hDisc = h(pp);
% discretisation of g
gDisc = zeros(Nbins);
for i=1:Nbins
   for j=1:Nbins
      gDisc(i, j) = g(pp(j), pp(i));       
   end    
end

% starting point
f0EM = rand(Nbins, 1);
f0SMC = 0.5 + sqrt(sigmaF1) * randn(Nparticles, 1);

% EM
EMres = em(gDisc, hDisc, Niter, f0EM);
% EMS
EMSres = ems(gDisc, hDisc, Niter, Kmatrix, f0EM);
% SMC
[x, W] = smc_AT_exact_potential(Nparticles, Niter, epsilon,...
    exactVarianceH, f0SMC, Nparticles);
% KDE
% bandwidth
bw = sqrt(epsilon^2 + optimal_bandwidthESS(x(Niter, :), W(Niter, :))^2);
[KDEy, KDEx] = ksdensity(x(Niter, :), 'weight', W(Niter, :), ...
    'Bandwidth', bw, 'Function', 'pdf');

% plot
close all;
figure(1);
fplot(f, '-k', [0, 1], 'LineWidth', 5)
hold on
plot(pp, EMres(Niter, :), '--', 'color', [0.7, 0.7, 0.7], 'LineWidth', 2)
plot(pp, EMSres(Niter, :), '-.r', 'LineWidth', 2)
plot(KDEx, KDEy, ':b', 'LineWidth', 2)
legend('$f$', 'EM', 'EMS', 'SMC', 'interpreter', 'latex',...
    'Fontsize', 10, 'Location', 'northwest');
pbaspect([1.5 1 1])

% ESS
1./sum(W.^2, 2)
% 9.8907e+03