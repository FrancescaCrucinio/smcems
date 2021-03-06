% Influence of dimensionality for SMC and EMS

% set seed
rng('default');

% number of dimensions
p = 5;
% build mixture of Gaussians
mu = [0.3*ones(1, p); 0.7*ones(1, p)];
variance = [0.07^2; 0.1^2];
sigmaF = cat(3, variance(1)*ones(1, p), variance(2)*ones(1, p));
weights = [1/3, 2/3];
gmF = gmdistribution(mu, sigmaF, weights);
sigmaG = 0.15;
sigmaH = sigmaF + sigmaG^2*eye(p);
gmH = gmdistribution(mu, sigmaH, weights);

% moments of mixture
m = weights*mu(:, 1);
second_moment = weights(1)*sigmaF(:, 1, 1) + weights(2)*sigmaF(:, 1, 2) + weights*mu(:, 1).^2;
v = second_moment - m.^2;
% probability of lower quadrant
p_quadrant = ((normcdf(0.5, 0.3, 0.07)-normcdf(0, 0.3, 0.07)) + 2*(normcdf(0.5, 0.7, 0.1)-normcdf(0, 0.7, 0.1)))/3;
p_quadrant = p_quadrant^p;
% probability of circle of radius r around the mode [0.3]^p
r = 0.15;
% sample from f
fSample = random(gmF, 10^6);
p_circle = sum(sum(fSample - mu(1, 1), 2).^2 <= r)/10^6;
% set parameters
% number of iterations
Niter = 30;
% number of bins/particles
Nparticles = [10^2 10^3 10^4];
Nbins = ceil(Nparticles.^(1/p));
% scale for SMC smoothing
epsilon = 1e-03;

% number of replications
Nrep = 100;
% execution times
EMStime = zeros(Nrep, length(Nbins));
SMCtime = zeros(Nrep, length(Nbins));
% DKDEtime = zeros(Nrep, length(Nbins));
% mise
EMSstats = zeros(4, length(Nbins), Nrep);
SMCstats = zeros(4, length(Nbins), Nrep);
% DKDEstats = zeros(5, length(Nbins), Nrep);

parfor index=1:length(Nparticles)
    % discretisation grid for EMS
    Ndarrays = cell(1, p);
    [Ndarrays{:}] = ndgrid(1/(2*Nbins(index)):1/Nbins(index):1); 
    eval = zeros(Nbins(index)^p, p);
    for i=1:p
        eval(:, i) = Ndarrays{i}(:);
    end
    % discretisation of h
    hDisc = pdf(gmH, eval);
    % indices in lower quadrant
    lq = logical(prod((eval <= 0.5) & (eval >= 0), 2));
    % indices in circle
    c = logical(sum((eval - mu(1, 1)).^2, 2) <= r);
    for j=1:Nrep
        % sample from h
        hSample = random(gmH, 10^5);
        % initial distribution
        f0 = rand(Nbins(index)^p, 1);
        x0 = rand(Nparticles(index), p);
        % EMS
        tstart = tic;
        EMSres = ems_p(hDisc, p, eval, Niter, epsilon, f0, sigmaG);
        EMStime(j, index) = toc(tstart);
        
        % SMC
        tstart = tic;
        [x, W] = smc_p_dim_gaussian_mixture(Nparticles(index), Niter, epsilon, x0, hSample, sigmaG);
        SMCtime(j, index) = toc(tstart);
        
        % moments & probability
        mHat = zeros(2, p);
        vHat = zeros(2, p);
        for i=1:p
            mHat(1, i) = sum(W.*x(:, i))/sum(W);
            mHat(2, i) = sum(EMSres.*eval(:, i))/sum(EMSres);
            vHat(1, i) = sum(W.*x(:, i).^2)/sum(W) - mHat(1, i)^2;
            vHat(2, i) = sum(EMSres.*eval(:, i).^2)/sum(EMSres) - mHat(2, i)^2;
        end
        pEMSquadrant = sum(EMSres(lq))/sum(EMSres);
        indices_q = logical(prod((x <= 0.5 & x>= 0), 2));
        pSMCquadrant = sum(W(indices_q));
        pEMScircle = sum(EMSres(c))/sum(EMSres);
        indices_c = (sum((x - mu(1, 1)).^2, 2) <= r);
        pSMCcircle = sum(W(indices_c));
        EMSstats(:, index, j) = [mean(mHat(2, :)) mean(vHat(2, :)) pEMSquadrant pEMScircle];
        SMCstats(:, index, j) = [mean(mHat(1, :)) mean(vHat(1, :)) pSMCquadrant pSMCcircle];
    end
end
EMSstatsSTD = zeros(4, length(Nbins), Nrep);
SMCstatsSTD = zeros(4, length(Nbins), Nrep);
EMSstatsSTD(1, :, :) = EMSstats(1, :, :) - m;
EMSstatsSTD(2, :, :) = EMSstats(2, :, :) - v;
EMSstatsSTD(3, :, :) = EMSstats(3, :, :) - p_quadrant;
EMSstatsSTD(4, :, :) = EMSstats(4, :, :) - p_circle;
SMCstatsSTD(1, :, :) = SMCstats(1, :, :) - m;
SMCstatsSTD(2, :, :) = SMCstats(2, :, :) - v;
SMCstatsSTD(3, :, :) = SMCstats(3, :, :) - p_quadrant;
SMCstatsSTD(4, :, :) = SMCstats(4, :, :) - p_circle;
% create table
format long
resTable = zeros(2*length(Nbins), 5);
resTable(1:2:5, 1:4) = mean(EMSstatsSTD.^2, 3)';
resTable(1:2:5, 5) = mean(EMStime, 1);
resTable(2:2:6, 1:4) = mean(SMCstatsSTD.^2, 3)';
resTable(2:2:6, 5) = mean(SMCtime, 1);
% log runtime 
resTable(:, 5) = log(resTable(:, 5));
% write table
dlmwrite('p_dim',resTable,'delimiter', '&', 'newline', 'pc')

