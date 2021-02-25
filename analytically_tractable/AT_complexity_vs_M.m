%%% Computational complexity as function of M

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

% set parameters
% number of iterations
Niter = 100;
% number of particles
N = 1000;
% scale
epsilon = 10^(-3/2);
% initial variance of approximation of f
sigmaF1 = rand(1);
% grid on support of f
refX = linspace(0, 1, 1000);

%%% EXACT POTENTIAL
% get exact variance for f_{n}(y)
[~, exactVarianceF] = AT_exact_variance_recursion(sigmaH, sigmaG, ...
    sigmaF1, epsilon^2, Niter);
% get exact values of variance for h_{n}(y)
exactVarianceH = exactVarianceF + sigmaG;
% starting distribution
x0 = 0.5 + sqrt(sigmaF1).*randn(1, N);

% values of M
M = [1:9 10:10:90 100:100:N];
% number of repetitions
Nrep = 1000;
miseExact = zeros(1, length(M));
miseApproximated = zeros(1, length(M));
runtimeExact = zeros(1, length(M));
runtimeApproximated = zeros(1, length(M));
SDmiseExact = zeros(1, length(M));
SDmiseApproximated = zeros(1, length(M));

parfor i=1:length(M)
    miseExactRep = zeros(1, Nrep);
    miseApproximatedRep = zeros(1, Nrep);
    runtimeExactRep = zeros(1, Nrep);
    runtimeApproximatedRep = zeros(1, Nrep);
    for k=1:Nrep
        % sample from h
        y = 0.5 + sqrt(0.043^2 + 0.045^2) * randn(10^3, 1);
        % Exact
        tstartExact = tic;
        [xExact, WExact] = smc_AT_exact_potential(N, Niter, epsilon,...
            exactVarianceH, x0, y, M(i));
        runtimeExactRep(k) = toc(tstartExact);
        bwExact = sqrt(epsilon^2 + optimal_bandwidthESS(xExact(Niter, :), WExact(Niter, :))^2);
        [KDEyExact, KDExExact] = ksdensity(xExact(Niter, :), 'weight', WExact(Niter, :), ...
            'Bandwidth', bwExact, 'Function', 'pdf');
        [~, ~, ~, miseExactRep(k)] = diagnosticsF(f, KDExExact, KDEyExact);
        % Approximated
        tstartApproximated = tic;
        [xApproximated, WApproximated] = smc_AT_approximated_potential(N, Niter, epsilon,...
            x0, y, M(i));
        runtimeApproximatedRep(k) = toc(tstartApproximated);
        bwApproximated = sqrt(epsilon^2 + ...
            optimal_bandwidthESS(xApproximated(Niter, :), WApproximated(Niter, :))^2);
        [KDEyApproximated, KDExApproximated] = ksdensity(xApproximated(Niter, :),...
            'weight', WApproximated(Niter, :), 'Bandwidth', bwApproximated,...
            'Function', 'pdf');
        [~, ~, ~, miseApproximatedRep(k)] = diagnosticsF(f, KDExApproximated, KDEyApproximated);
    end
    % means
    miseExact(i) = mean(miseExactRep);
    miseApproximated(i) = mean(miseApproximatedRep);
    runtimeExact(i) = mean(runtimeExactRep);
    runtimeApproximated(i) = mean(runtimeApproximatedRep);
    % standard deviations
    SDmiseExact(i) = std(miseExactRep);
    SDmiseApproximated(i) = std(miseApproximatedRep);
end

% plots
close all;
figure(1);
loglog(M, runtimeExact, 'ro', 'MarkerFaceColor', 'red', 'Linewidth', 3)
hold on
loglog(M, runtimeApproximated, 'bo', 'MarkerFaceColor', 'blue', 'Linewidth', 3)
legend('Exact', 'Approximated', 'Interpreter', 'latex',...
    'Fontsize', 15, 'Location', 'northwest');
figure(2);
errorbar(M, miseExact, 2*SDmiseExact, 'ro', 'MarkerFaceColor', 'red', 'Linewidth', 3)
hold on
errorbar(M, miseApproximated, 2*SDmiseApproximated, 'bo', 'MarkerFaceColor', 'blue', 'Linewidth', 3)
legend('Exact', 'Approximated', 'Interpreter', 'latex',...
    'Fontsize', 15, 'Location', 'northwest');
set(gca, 'Xscale', 'log')