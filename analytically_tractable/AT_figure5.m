% Comparison of convergence for EM, EMS and SMC with different values of
% smoothing

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
% initial variance of approximation of f
sigmaF1 = rand(1);

% set parameters
% EM/EMS
% number of iterations
Niter = 100;
% number of bins
Nbins = 100;
% bin centres for f
fBin = 1/(2*Nbins):1/Nbins:1;
% SMC
% number of particles
Nparticles = 1000;

%%% smoothing
% scale value
epsilon = linspace(0, 1, 100);

%%% KL
% points at which compare the true and estimated h
refY = linspace(0,1,100);

% bin centres for the support of h
hBin = 1/(2*Nbins):1/Nbins:1;
% discretisation of h
hDisc = h(hBin);
% discretisation of f
fDisc = f(fBin);
% discretisation of g
gDisc = zeros(Nbins);
for i=1:Nbins
   for j=1:Nbins
      gDisc(i, j) = g(hBin(j), hBin(i));       
   end    
end

% initial value
f0EM = normpdf(fBin, 0.5, sqrt(sigmaF1));
f0SMC = rand(1, Nparticles);

% STATISTICS
% mean
meanEMS = zeros(1, length(epsilon));
meanSMC = zeros(1, length(epsilon));
% variance
varEMS = zeros(1, length(epsilon));
varSMC = zeros(1, length(epsilon));
% MISE f
miseFEMS = zeros(1, length(epsilon));
miseFSMC = zeros(1, length(epsilon));
% divergence
divEMS = zeros(1, length(epsilon));
divSMC = zeros(1, length(epsilon));
% MISE h
miseHEMS = zeros(1, length(epsilon));
miseHSMC = zeros(1, length(epsilon));

% replications for SMC
Nrep = 1000;

% EMS and SMC
parfor i=1:length(epsilon)
    if(i == 1)
        % epsilon = 0 is EM
        Kmatrix = eye(length(fBin));
    else
        K = @(x, z) normpdf(z, x, epsilon(i));
        % smoothing matrix
        Kmatrix = smoothingMatrix(K, fBin);
    end
    % get exact variance for h_n(y)
    [~, exactVarianceF] = AT_exact_variance_recursion(sigmaH, sigmaG, ...
        sigmaF1, epsilon(i)^2, Niter);
    % get exact values of variance for h_{n}(y)
    exactVarianceH = exactVarianceF + sigmaG;
    fEMS = ems(gDisc, hDisc, Niter, Kmatrix, f0EM);
    % last time step
    fEMS = fEMS(Niter, :);
    % diagnostics
    [meanEMS(i), varEMS(i), ~, miseFEMS(i)] = diagnosticsF(f, fBin, fEMS);
    [miseHEMS(i), divEMS(i)] = diagnosticsH(h, g, fBin, fEMS, refY);
    % replicate SMC Nrep times
    meanSMCrep = zeros(1, Nrep);
    varSMCrep = zeros(1, Nrep);
    miseFSMCrep = zeros(1, Nrep);
    divSMCrep = zeros(1, Nrep);
    miseHSMCrep = zeros(1, Nrep);
    for k=1:Nrep
        y = 0.5 + sqrt(0.043^2 + 0.045^2) * randn(10^4, 1);
        M = min(Nparticles, length(y));
        [x, W] = smc_AT_exact_potential(Nparticles, Niter, ...
            epsilon(i), exactVarianceH, f0SMC, y, M);
        fSMC = ksdensity(x(Niter,:), fBin, 'weight', W(Niter,:),...
            'Bandwidth', epsilon(i), 'Function', 'pdf');
        % diagnostics
        [meanSMCrep(k), varSMCrep(k), ~, miseFSMCrep(k)] =...
            diagnosticsF(f, fBin, fSMC);
        [miseHSMCrep(k), divSMCrep(k)] = diagnosticsH(h, g, fBin, fSMC, refY);
    end
    divSMC(i) = mean(divSMCrep);
    miseHSMC(i) = mean(miseHSMCrep);
    miseFSMC(i) = mean(miseFSMCrep);
    meanSMC(i) = mean(meanSMCrep);
    varSMC(i) = mean(varSMCrep);
end

% plot
close all;
figure(1);
subplot(221)
plot(epsilon, varEMS, '-.r', 'LineWidth', 3)
hold on
plot(epsilon(2:end), varSMC(2:end), ':b', 'LineWidth', 3)
title('Variance')
pbaspect([1.5 1 1])
legend('EMS', 'SMC', 'interpreter', 'latex', 'FontSize', ...
    20, 'Location', 'southeast');
hold off;
subplot(222)
plot(epsilon, miseFEMS, '-.r', 'LineWidth', 3)
hold on
plot(epsilon(2:end), miseFSMC(2:end), ':b', 'LineWidth', 3)
title('MISE f')
pbaspect([1.5 1 1])
subplot(223)
plot(epsilon, miseHEMS, '-.r', 'LineWidth', 3)
hold on
plot(epsilon(2:end), miseHEMS(2:end), ':b', 'LineWidth', 3)
title('MISE h')
pbaspect([1.5 1 1])
subplot(224)
plot(epsilon, abs(divEMS), '-.r', 'LineWidth', 3)
hold on
plot(epsilon(2:end), abs(divSMC(2:end)), ':b', 'LineWidth', 3)
title('KL')
pbaspect([1.5 1 1])