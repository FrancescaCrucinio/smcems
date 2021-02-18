% 2D reconstructions for SMC and EMS on toy Gaussian mixture model

% set seed
rng('default');

% number of dimensions
p = 2;
% build mixture of Gaussians
mu = [0.3*ones(1, p); 0.7*ones(1, p)];
sigmaF = cat(3, 0.07^2*ones(1, p), 0.1^2*ones(1, p));
weights = [1/3, 2/3];
gmF = gmdistribution(mu, sigmaF, weights);
sigmaG = 0.15;
sigmaH = sigmaF + sigmaG^2*eye(p);
gmH = gmdistribution(mu, sigmaH, weights);

% discretisation grid
Ndarrays = cell(1, p);
grid1d = linspace(0, 1, 200);
dx = grid1d(2) - grid1d(1);
[Ndarrays{:}] = ndgrid(grid1d); 
eval_grid = zeros(200^p, p);
for i=1:p
    eval_grid(:, i) = Ndarrays{i}(:);
end
fDisc = pdf(gmF, eval_grid);
fDisc = reshape(fDisc, [200, 200]);

% set parameters
% number of iterations
Niter = 20;
% number of bins/particles
Nbins = 200;
Nparticles = Nbins^p;
% smoothing parameter
epsilon = 1e-03;

% grid for EMS
Ndarrays = cell(1, p);
[Ndarrays{:}] = ndgrid(linspace(0, 1, Nbins)); 
eval = zeros(Nbins^p, p);
for i=1:p
    eval(:, i) = Ndarrays{i}(:);
end
% discretisation of h
hDisc = pdf(gmH, eval);
% sample from h
hSample = random(gmH, 10^6);

% initial distribution
f0 = ones(Nbins^p, 1)/Nbins^p;
x0 = rand(Nparticles, p);

% EMS
tstart = tic;
EMSres = ems_p(hDisc, p, eval, Niter, epsilon, f0, sigmaG);
EMS = reshape(EMSres, [Nbins, Nbins]);
EMS = repelem(EMS, 200/Nbins, 200/Nbins);
'EMS'
toc(tstart)
dx^2*norm(fDisc - EMS)^2
% SMC
tstart = tic;
[x, W] = smc_p_dim_gaussian_mixture(Nparticles, Niter, epsilon, x0, hSample, sigmaG);
% KDE
bw1 = sqrt(epsilon^2 + optimal_bandwidthESS(x(:, 1), W)^2);
bw2 = sqrt(epsilon^2 + optimal_bandwidthESS(x(:, 2), W)^2);
lambda = ksdensity([x(:, 1) x(:, 2)], eval_grid, 'weight', ...
    W, 'Bandwidth', [bw1 bw2], 'Function', 'pdf');
SMC_EMS = reshape(lambda, [200, 200]);
'SMC'
toc(tstart)
dx^2*norm(fDisc - SMC_EMS)^2

% plots
close;
subplot(2, 2, 1)
surface(grid1d, grid1d, fDisc)
shading interp
pbaspect([1 1 1])
subplot(2, 2, 2)
surface(grid1d, grid1d, SMC_EMS)
shading interp
pbaspect([1 1 1])
subplot(2, 2, 3)
surface(grid1d, grid1d, EMS)
shading interp
pbaspect([1 1 1])