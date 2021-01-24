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
% normalise xi
xi = xi/max(xi);
% convert phi to radiants
phi = deg2rad(phi);
% add Poisson noise
noisyR = imnoise(R*1e-12, 'poisson');
% normalise R
noisyR = noisyR/max(noisyR, [], 'all');
%%% set parameters
% SMC
% number of iterations
Niter = 10;
% number of particles
N = 100000;
% smoothing parameter
epsilon = 1e-03;
% variance of normal describing alignment
sigma = 0.02;
% run SMC
tstart = tic;
[x, y, W, iter_stop] = smc_pet(N, Niter, epsilon, phi, xi, noisyR, sigma, 1e-5, 5); 
toc(tstart);
% set coordinate system over image
% x is in [-0.8, 0.8]
evalX = linspace(-0.75 + 1/pixels, 0.75 - 1/pixels, pixels);
% y is in [-0.8, 0.8]
evalY = linspace(-0.75 + 1/pixels, 0.75 - 1/pixels, pixels);
% build grid with this coordinates
RevalX = repmat(evalX, pixels, 1);
eval =[RevalX(:) repmat(evalY, 1, pixels)'];

%%% plot
% select which steps to show
showIter = [1, 5, 10, 15, 37, 50, 70, Niter];
Npic = length(showIter);
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
    %subplot(2, 4, n);
    imshow(KDEn, [])
    colormap(gca,hot);
    pbaspect([1 1 1])
    % title(['Iteration ' num2str(showIter(n))],...
    %    'interpreter', 'latex', 'FontSize', 10);
    hold off;
    filename = sprintf('%s%d', 'pet', n);
    printEps(gcf, filename)
    % MISE
    mse(n) = immse(P, KDEn);
    % relative error
    figure(2);
    %subplot(2, 4, n)
    error = abs(KDEn - P);
    positive = (P>0);
    error(positive) = error(positive)./P(positive);
    imshow(error, [])
    colormap(gca,hot);
    pbaspect([1 1 1])
    % title(['Iteration ' num2str(showIter(n))],...
    %    'interpreter', 'latex', 'FontSize', 10);
    filename = sprintf('%s%d', 'pet_relative_error', n);
    printEps(gcf, filename)
end
'MSE'
mse

% ESS
1./sum(W(iter_stop, :).^2, 2)
% 1.4081e+04
