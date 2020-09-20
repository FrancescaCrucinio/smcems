%%% POSITRON EMISSION TOMOGRAPHY

% set seed
rng('default');

% Shepp Logan phantom
pixels = 128;
% phantom
P = phantom('Modified Shepp-Logan', pixels);
P(P<0) = 0;
% projections on equally spaced angles
phi = linspace(0, 360, pixels);
% get projections for each phi and corresponding offset
[R, xi] = radon(P, phi);
% add Poisson noise
noisyR = imnoise(R*1e-12, 'poisson');

%%% set parameters
% SMC
% number of iterations
Niter = 100;
% number of particles
N = 20000;
% smoothing parameter
epsilon = 1e-03;
% variance of normal describing alignment
sigma = 0.02;

% run SMC
tstart = tic;
[x, y, W] = smc_pet(N, Niter, epsilon, phi, xi, noisyR, sigma);
toc(tstart);
% set coordinate system over image
% x is in [-0.7, 0.7]
evalX = linspace(-0.7 + 1/pixels, 0.7 - 1/pixels, pixels);
% y is in [-0.7, 0.7]
evalY = linspace(-0.7 + 1/pixels, 0.7 - 1/pixels, pixels);
% build grid with this coordinates
RevalX = repmat(evalX, pixels, 1);
eval =[RevalX(:) repmat(evalY, 1, pixels)'];

%%% plot
% select which steps to show
showIter = [1, 5, 10, 15, 20, 50, 70, Niter];
Npic = 8;
% mise
mse = zeros(1, Npic);

for n=1:Npic
    % bandwidth
    bw1 = sqrt(epsilon^2 + optimal_bandwidthESS(x(showIter(n), :), W(showIter(n), :))^2);
    bw2 = sqrt(epsilon^2 + optimal_bandwidthESS(y(showIter(n), :), W(showIter(n), :))^2);
    lambda = ksdensity([x(showIter(n), :)' y(showIter(n), :)'], eval, 'weight', ...
        W(showIter(n), :), 'Bandwidth', [bw1 bw2], 'Function', 'pdf');
    KDEn = reshape(lambda, [pixels, pixels]);
    KDEn = flipud(mat2gray(KDEn));
    figure(1);
    subplot(2, 4, n);
    imshow(KDEn, [])
    colormap(gca,hot);
    pbaspect([1 1 1])
    title(['Iteration ' num2str(showIter(n))],...
        'interpreter', 'latex', 'FontSize', 10);
    hold off;
    % MISE
    mse(n) = immse(P, KDEn);
    % relative error
    figure(2);
    subplot(2, 4, n)
    error = abs(KDEn - P);
    positive = (P>0);
    error(positive) = error(positive)./P(positive);
    imshow(error, [])
    colormap(gca,hot);
    pbaspect([1 1 1])
    title(['Iteration ' num2str(showIter(n))],...
        'interpreter', 'latex', 'FontSize', 10);
end
'MSE'
mse

% ESS
1./sum(W.^2, 2)
% 1.4081e+04
