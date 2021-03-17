%%% MOTION DEBLURRING

% set seed
rng('default');
% read original image
I = imread('BC.png');
% image dimension
pixels = size(I(:,:,1));

%%% set parameters
% MOTION
% speed of constant motion
b = 128;
% standard deviation of normal
sigma = 0.02;
% SMC/RL
% numer of time steps
Niter = 100;
% number of particles
N = 5000;
% smoothing parameter
epsilon = 1e-03;

%%% create blurred image
% blurredI = blurred_image(I, b, sigma);
% or upload one
blurredI = imread('BCblurred.png');
blurredI = im2double(blurredI(:, :, 1));

% add noise
noisyI = multiplicative_noise(blurredI, 0.1, 0.5);

%%% run RL
% create point spread function (PSF) for constant speed motion
PSF = fspecial('motion', b, 0); 
rl = deconvlucy(noisyI, PSF, Niter);

%%% run SMC
% outputs particle locations and weights
tstart = tic;
[x, y, W] = smc_deblurring(N, Niter, epsilon, noisyI, sigma, b);
% kernel density estimator
% set coordinate system over image
% x is in [-1, 1]
evalX = linspace(-1 + 1/pixels(2), 1 - 1/pixels(2), pixels(2));
% y is in [-0.5, 0.5]
evalY = linspace(0.5 - 1/pixels(1), -0.5 + 1/pixels(1), pixels(1));
% build grid with this coordinates
RevalX = repmat(evalX, pixels(1), 1);
eval =[RevalX(:) repmat(evalY, 1, pixels(2))'];
% bandwidth
bw1 = sqrt(epsilon^2 + optimal_bandwidthESS(x(Niter, :), W(Niter, :))^2);
bw2 = sqrt(epsilon^2 + optimal_bandwidthESS(y(Niter, :), W(Niter, :))^2);
lambda = ksdensity([x(Niter, :)' y(Niter, :)'], eval, ...
    'Weight', W(Niter, :), 'Bandwidth', [bw2, bw1],...
    'Function', 'pdf', 'Support', [-1 -0.5; 1 0.5]);
lambda = reshape(lambda', [pixels(1), pixels(2)]);
toc(tstart)
%%% plot
figure(2);
subplot(221)
imshow(I);
subplot(222)
imshow(noisyI);
subplot(223)
imshow(rl)
subplot(224)
imshow(lambda);

%%% reconstruction comparison
% get original image in gray levels
I = mat2gray(I(:, :, 1));
'MSE'
immse(I, rl)
immse(I, lambda)


% histograms for match distance
% original image
[Icounts, IX] = imhist(I);
% RL
[rlcounts, rlX] = imhist(rl);
% SMC
[lambdacounts, lambdaX] = imhist(lambda);
% cdfs
cdfI = cumsum(Icounts);
cdfI = cdfI./sum(cdfI);
cdfrl = cumsum(rlcounts);
cdfrl = cdfrl./sum(cdfrl);
cdflambda = cumsum(lambdacounts);
cdflambda = cdflambda./sum(cdflambda);
'Match distance'
norm(cdfI - cdfrl, 1)
norm(cdfI - cdflambda, 1)


% ESS
1./sum(W.^2, 2)