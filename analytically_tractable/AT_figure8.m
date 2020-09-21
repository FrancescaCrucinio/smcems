% Mean and MSE for analytically tractable example
% Different values of M are compared

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
% SMC
% number of iterations
Niter = 100;
% number of particles
N = 1000;
% smoothing scale
epsilon = 1e-03;
% initial variance of approximation of f
sigmaF1 = rand(1);
% coordinate at which KDE is evaluated
refX = linspace(0, 1, 1000);

% exact variance for f_{n}(y)
[~, exactVarianceFK] = AT_exact_variance_recursion(sigmaH, sigmaG, ...
    sigmaF1, epsilon^2, Niter);
% exact variance for h_{n}(y)
exactVarianceHK = exactVarianceFK + sigmaG;
% starting distribution
x0 = 0.5 + sqrt(sigmaF1).*randn(1, N);

% number of repetitions
Nrep = 1000;
% KDE
KDEyExact10 = zeros(Nrep, length(refX));
KDEyApproximated10 = zeros(Nrep, length(refX));
runtimeExact10 = zeros(Nrep, 1);
runtimeApproximated10 = zeros(Nrep, 1);
KDEyExact100 = zeros(Nrep, length(refX));
KDEyApproximated100 = zeros(Nrep, length(refX));
runtimeExact100 = zeros(Nrep, 1);
runtimeApproximated100 = zeros(Nrep, 1);
KDEyExactN = zeros(Nrep, length(refX));
KDEyApproximatedN = zeros(Nrep, length(refX));
runtimeExactN = zeros(Nrep, 1);
runtimeApproximatedN = zeros(Nrep, 1);

parfor i=1:Nrep
   % run SMC
     % M=10
    tstart10E = tic;
    [xExact10, WExact10] = smc_AT_exact_potential(N, Niter, epsilon,...
        exactVarianceHK, x0, 10);
    runtimeExact10(i) = toc(tstart10E);
    bwE10 = sqrt(epsilon^2 + ...
        optimal_bandwidthESS(xExact10(Niter, :), WExact10(Niter, :))^2);
    KDEyExact10(i, :) = ksdensity(xExact10(Niter, :), refX,...
        'weight', WExact10(Niter, :), 'Bandwidth', bwE10, 'Function', 'pdf');
    tstart10A = tic;
    [x10, W10] = smc_AT_approximated_potential(N, Niter, epsilon, ...
        x0, 10);
    runtimeApproximated10(i) = toc(tstart10A);
    bwA10 = sqrt(epsilon^2 + ...
        optimal_bandwidthESS(x10(Niter, :), W10(Niter, :))^2);
    KDEyApproximated10(i, :) = ksdensity(x10(Niter, :), refX,...
        'weight', W10(Niter, :), 'Bandwidth', bwA10, 'Function', 'pdf');
    % M=100
    tstart100E = tic;
    [xExact100, WExact100] = smc_AT_exact_potential(N, Niter, epsilon,...
        exactVarianceHK, x0, 100);
    runtimeExact100(i) = toc(tstart100E);
    bwE100 = sqrt(epsilon^2 + ...
        optimal_bandwidthESS(xExact100(Niter, :), WExact100(Niter, :))^2);
    KDEyExact100(i, :) = ksdensity(xExact100(Niter, :), refX,...
        'weight', WExact100(Niter, :), 'Bandwidth', bwE100, 'Function', 'pdf');
    tstart100A = tic;
    [x100, W100] = smc_AT_approximated_potential(N, Niter, epsilon, ...
        x0, 100);
    runtimeApproximated100(i) = toc(tstart100A);
    bwA100 = sqrt(epsilon^2 + ...
        optimal_bandwidthESS(x100(Niter, :), W100(Niter, :))^2);
    KDEyApproximated100(i, :) = ksdensity(x100(Niter, :), refX,...
        'weight', W100(Niter, :), 'Bandwidth', bwA100, 'Function', 'pdf');
    % M=N
    tstartNE = tic;
    [xExactN, WExactN] = smc_AT_exact_potential(N, Niter, epsilon,...
        exactVarianceHK, x0, N);
    runtimeExactN(i) = toc(tstartNE);
    bwEN = sqrt(epsilon^2 + ...
        optimal_bandwidthESS(xExactN(Niter, :), WExactN(Niter, :))^2);
    KDEyExactN(i, :) = ksdensity(xExactN(Niter, :), refX,...
        'weight', WExactN(Niter, :), 'Bandwidth', bwEN, 'Function', 'pdf');
    tstartNA = tic;
    [xN, WN] = smc_AT_approximated_potential(N, Niter, epsilon, ...
        x0, N);
    runtimeApproximatedN(i) = toc(tstartNA);
    bwAN = sqrt(epsilon^2 + ...
        optimal_bandwidthESS(xN(Niter, :), WN(Niter, :))^2);
    KDEyApproximatedN(i, :) = ksdensity(xN(Niter, :), refX,...
        'weight', WN(Niter, :), 'Bandwidth', bwAN, 'Function', 'pdf');
end
% MSE
mseExact10 = var(KDEyExact10 - f(refX), 1);
mseApproximated10 = var(KDEyApproximated10 - f(refX), 1);
mseExact100 = var(KDEyExact100 - f(refX), 1);
mseApproximated100 = var(KDEyApproximated100 - f(refX), 1);
mseExactN = var(KDEyExactN - f(refX), 1);
mseApproximatedN = var(KDEyApproximatedN - f(refX), 1);
% plot
close all;
figure(1);
fplot(f, '-k', [0, 1], 'Linewidth', 4);
hold on
plot(refX, mean(KDEyExact10, 1), '-', 'color', [0 0 0.5451], 'Linewidth', 3);
plot(refX, mean(KDEyApproximated10, 1), '-', 'color', [0.5020 0 0], 'Linewidth', 3);
plot(refX, mean(KDEyExact100, 1), '--b', 'Linewidth', 3);
plot(refX, mean(KDEyApproximated100, 1), '--r', 'Linewidth', 3);
plot(refX, mean(KDEyExactN, 1), ':', 'color', [0 0.4470 0.7410], 'Linewidth', 3);
plot(refX, mean(KDEyApproximatedN, 1), ':', 'color', [1 0.5490 0], 'Linewidth', 3);
xlabel('$x$', 'FontSize', 20, 'interpreter', 'latex');
ylabel('$f(x)$', 'FontSize', 20, 'interpreter', 'latex');
legend('$f(x)$', 'E - 10', 'A - 10', 'E - 100', 'A - 100', 'E - N', 'A - N',...
    'Interpreter', 'latex', 'Fontsize', 15, 'Location', 'northwest');
pbaspect([1.5 1 1])
figure(2);
plot(refX, mseExact10,  '-', 'color', [0 0 0.5451], 'Linewidth', 3);
hold on
plot(refX, mseApproximated10, '-', 'color', [0.5020 0 0], 'Linewidth', 3);
plot(refX, mseExact100, '--b', 'Linewidth', 3);
plot(refX, mseApproximated100, '--r', 'Linewidth', 3);
plot(refX, mseExactN, ':', 'color', [0 0.4470 0.7410], 'Linewidth', 3);
plot(refX, mseApproximatedN, ':', 'color', [1 0.5490 0], 'Linewidth', 3);
xlabel('$x$', 'FontSize', 20, 'interpreter', 'latex');
legend('E - 10', 'A - 10', 'E - 100', 'A - 100', 'E - N', 'A - N',...
    'Interpreter', 'latex', 'Fontsize', 15, 'Location', 'northwest');
pbaspect([1.5 1 1])
% runtime
'M = 10'
mean(runtimeExact10)
mean(runtimeApproximated10)
'M = 100'
mean(runtimeExact100)
mean(runtimeApproximated100)
'M = N'
mean(runtimeExactN)
mean(runtimeApproximatedN)