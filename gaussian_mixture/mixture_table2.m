% Comparison of reconstruction for EM, EMS, IB and SMC

% set seed
rng('default');

% f, h, g are mixtures of Gaussians
h = @(y) 2*normpdf(y, 0.3, sqrt(0.043^2 + 0.045^2))./3 + ...
    normpdf(y, 0.5, sqrt(0.015^2 + 0.045^2))./3;
g = @(x,y) normpdf(y, x, 0.045);
f = @(x) normpdf(x, 0.3, 0.015)/3 + normpdf(x, 0.5, 0.043)*2/3;
sigG = 0.045;
varG = sigG^2;

% set parameters
% EM/EMS/IB
% number of iterations
Niter = 100;
% number of bins
Nbins = 100;
% bin centers
fBin = 1/(2*Nbins):1/Nbins:1;
dx = fBin(2) - fBin(1);
% number of samples for IB/SMC
nsamples = 1000;
% SMC
% number of particles
Nparticles = [500, 1000, 5000];
% smoothing
% scale for smoothing kernel
epsilon = 1e-3;
% smoothing kernel
K = @(x1, x2) normpdf(x2, x1, epsilon);
% get smoothing matrix
Kmatrix = smoothingMatrix(K, fBin);
% smoothing matrix from Silverman 1990 3.2.2
S = smoothingSilverman(Nbins, 3);
% coordinates in Y to compute KL divergence
refY = linspace(0, 1, 1000);
% discretisation of h
hDisc = h(fBin);
% discretisation of g
gDisc = zeros(Nbins);
for i=1:Nbins
   for j=1:Nbins
      gDisc(i, j) = g(fBin(j), fBin(i));       
   end    
end

% number of repetitions
Nrep = 100;
% percentile of MSE to compare smoothness
q = 0.95;
% diagnostics
% mse
EMmse = zeros(Nrep, length(fBin));
EMSKmse = zeros(Nrep, length(fBin));
EMSJmse = zeros(Nrep, length(fBin));
SMCmse500 = zeros(Nrep, length(fBin));
SMCmse1000 = zeros(Nrep, length(fBin));
SMCmse5000 = zeros(Nrep, length(fBin));
IBmse = zeros(Nrep, length(fBin));
DKDEpimse = zeros(Nrep, length(fBin));
DKDEcvmse = zeros(Nrep, length(fBin));
% mean, variance, mise, kl
EMstats = zeros(Nrep, 4);
EMSKstats = zeros(Nrep, 4);
EMSJstats = zeros(Nrep, 4);
SMCstats500 = zeros(Nrep, 4);
SMCstats1000 = zeros(Nrep, 4);
SMCstats5000 = zeros(Nrep, 4);
IBstats = zeros(Nrep, 4);
DKDEpistats = zeros(Nrep, 4);
DKDEcvstats = zeros(Nrep, 4);
% runtimes
EMtime = zeros(Nrep, 1);
EMSKtime = zeros(Nrep, 1);
EMSJtime = zeros(Nrep, 1);
SMCtime500 = zeros(Nrep, 1);
SMCtime1000 = zeros(Nrep, 1);
SMCtime5000 = zeros(Nrep, 1);
IBtime = zeros(Nrep, 1);
DKDEpitime = zeros(Nrep, 1);
DKDEcvtime = zeros(Nrep, 1);

parfor j=1:Nrep
    % random starting point
    f0EM = rand(Nbins, 1);
    f0SMC500 = rand(Nparticles(1), 1);
    f0SMC1000 = rand(Nparticles(2), 1);
    f0SMC5000 = rand(Nparticles(3), 1);
    % EM
    tstart = tic;
    EMres = em(gDisc, hDisc, Niter, f0EM);
    EMtime(j) = toc(tstart);
    [EMstats(j, :), EMmse(j, :)] = ...
        diagnostics(f, h, g, fBin, EMres(Niter, :), refY);
%     EMS - Silverman
    tstart = tic;
    EMSJres = ems(gDisc, hDisc, Niter, S, f0EM);
    EMSJtime(j) = toc(tstart);
    [EMSJstats(j, :), EMSJmse(j, :)] = ...
        diagnostics(f, h, g, fBin, EMSJres(Niter, :), refY);
%     IB    
    tstart = tic;
%     samples from h
    obs = Ysample_gaussian_mixture(nsamples);
%     KDE for h
    h_ib = ksdensity(obs, fBin, 'width', 0.02);
    IBres = em(gDisc, h_ib, Niter, f0EM);
    IBtime(j) = toc(tstart);
    [IBstats(j, :), IBmse(j, :)] = ...
        diagnostics(f, h, g, fBin, IBres(Niter, :), refY);
%     SMC - Nparticles = 500
    tstart = tic;
    [x500, W500] = smc_gaussian_mixture(Nparticles(1), Niter, epsilon,...
        f0SMC500, obs);
    SMCtime500(j) = toc(tstart);
%     KDE
%     bandwidth
    bw500 = sqrt(epsilon^2 + optimal_bandwidthESS(x500(Niter, :), W500(Niter, :))^2);
    KDEy500 = ksdensity(x500(Niter, :), fBin, 'weight', W500(Niter, :), ...
        'Bandwidth', bw500, 'Function', 'pdf');
    [SMCstats500(j, :), SMCmse500(j, :)] = ...
        diagnostics(f, h, g, fBin, KDEy500, refY);
%     SMC - Nparticles = 1000
    tstart = tic;
    [x1000, W1000] = smc_gaussian_mixture(Nparticles(2), Niter, epsilon,...
        f0SMC1000, obs);
    SMCtime1000(j) = toc(tstart);
%     KDE
%     bandwidth
    bw1000 = sqrt(epsilon^2 + optimal_bandwidthESS(x1000(Niter, :), W1000(Niter, :))^2);
    KDEy1000 = ksdensity(x1000(Niter, :), fBin, 'weight', W1000(Niter, :), ...
        'Bandwidth', bw1000, 'Function', 'pdf');
    [SMCstats1000(j, :), SMCmse1000(j, :)] = ...
        diagnostics(f, h, g, fBin, KDEy1000, refY);
%     SMC - Nparticles = 5000
    tstart = tic;
    [x5000, W5000] = smc_gaussian_mixture(Nparticles(3), Niter, epsilon,...
        f0SMC5000, obs);
    SMCtime5000(j) = toc(tstart);
%     KDE
%     bandwidth
    bw5000 = sqrt(epsilon^2 + optimal_bandwidthESS(x5000(Niter, :), W5000(Niter, :))^2);
    KDEy5000 = ksdensity(x5000(Niter, :), fBin, 'weight', W5000(Niter, :), ...
        'Bandwidth', bw5000, 'Function', 'pdf');
    [SMCstats5000(j, :), SMCmse5000(j, :)] = ...
        diagnostics(f, h, g, fBin, KDEy5000, refY);
%     EMS - K
%     smoothing kernel
    K = @(x1, x2) normpdf(x2, x1, bw500);
%     smoothing matrix
    Kmatrix = smoothingMatrix(K, fBin);
    tstart = tic;
    EMSKres = ems(gDisc, hDisc, Niter, Kmatrix, f0EM);
    EMSKtime(j) = toc(tstart);
    [EMSKstats(j, :), EMSKmse(j, :)] = ...
        diagnostics(f, h, g, fBin, EMSKres(Niter, :), refY);
%       DKDE - pi
    tstart = tic;
    hPI = PI_deconvUknownth4(obs, 'norm', varG, sigG);
    fdecPI = fdecUknown(fBin, obs, hPI, 'norm', sigG, dx);
    DKDEpitime(j) = toc(tstart);
    [DKDEpistats(j, :), DKDEpimse(j, :)] = ...
        diagnostics(f, h, g, fBin, fdecPI, refY);
    %       DKDE - cv
    tstart = tic;
    hCV = CVdeconv(obs, 'norm', sigG);
    fdecCV = fdecUknown(fBin, obs, hCV, 'norm', sigG, dx);
    DKDEcvtime(j) = toc(tstart);
    [DKDEcvstats(j, :), DKDEcvmse(j, :)] = ...
        diagnostics(f, h, g, fBin, fdecCV, refY);
end
format long
% create table with statistics
resTable = zeros(9, 6);
% mean, variance, mise, kl
resTable(1, [1:3, 5]) = mean(EMstats, 1);
resTable(2, [1:3, 5]) = mean(EMSKstats, 1);
resTable(3, [1:3, 5]) = mean(EMSJstats, 1);
resTable(4, [1:3, 5]) = mean(IBstats, 1);
resTable(5, [1:3, 5]) = mean(SMCstats500, 1);
resTable(6, [1:3, 5]) = mean(SMCstats1000, 1);
resTable(7, [1:3, 5]) = mean(SMCstats5000, 1);
resTable(8, [1:3, 5]) = mean(DKDEpistats, 1);
resTable(9, [1:3, 5]) = mean(DKDEcvstats, 1);
% mse percentile
resTable(1, 4) = quantile(mean(EMmse, 1), q);
resTable(2, 4) = quantile(mean(EMSJmse, 1), q);
resTable(3, 4) = quantile(mean(EMSKmse, 1), q);
resTable(4, 4) = quantile(mean(IBmse, 1), q);
resTable(5, 4) = quantile(mean(SMCmse500, 1), q);
resTable(6, 4) = quantile(mean(SMCmse1000, 1), q);
resTable(7, 4) = quantile(mean(SMCmse5000, 1), q);
resTable(8, 4) = quantile(mean(DKDEpimse, 1), q);
resTable(9, 4) = quantile(mean(DKDEcvmse, 1), q);
% time
resTable(1, 6) = mean(EMtime, 1);
resTable(2, 6) = mean(EMSKtime, 1);
resTable(3, 6) = mean(EMSJtime, 1);
resTable(4, 6) = mean(IBtime, 1);
resTable(5, 6) = mean(SMCtime500, 1);
resTable(6, 6) = mean(SMCtime1000, 1);
resTable(7, 6) = mean(SMCtime5000, 1);
resTable(8, 6) = mean(DKDEpitime, 1);
resTable(9, 6) = mean(DKDEcvtime, 1);
% log runtime 
resTable(:, 6) = log(resTable(:, 6));
% write table
dlmwrite('EM_EMS_IB_SMC_Table',resTable,'delimiter', '&',...
    'newline', 'pc')
save('mixture_table_12Jan2021.mat')