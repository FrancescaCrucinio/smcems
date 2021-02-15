% Influence of dimensionality for SMC and EMS

% set seed
rng('default');

% number of dimensions
p = 2;
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
third_moment = weights*(mu(:, 1).^3 + 3*mu(:, 1).*variance);
s = (third_moment - 3*m.*v-m.^3)./v.^(3/2);
fourth_moment = weights*(mu(:, 1).^4 + 6*mu(:, 1).^2.*variance + 3*variance.^2);
k = (fourth_moment - 4*third_moment.*m + 6*second_moment.*m.^2 - 3*m.^4)./v.^2;
% probability of lower quadrant
p_quadrant = ((normcdf(0.5, 0.3, 0.07)-normcdf(0, 0.3, 0.07)) + 2*(normcdf(0.5, 0.7, 0.1)-normcdf(0, 0.7, 0.1)))/3;
p_quadrant = p_quadrant^p;

% set parameters
% number of iterations
Niter = 50;
% number of bins/particles
Nbins = [10 50];
% scale for SMC smoothing
epsilon = 1e-03;

% number of replications
Nrep = 2;
% execution times
EMStime = zeros(Nrep, length(Nbins));
SMCtime = zeros(Nrep, length(Nbins));
DKDEtime = zeros(Nrep, length(Nbins));
% mise
EMSstats = zeros(5, length(Nbins), Nrep);
SMCstats = zeros(5, length(Nbins), Nrep);
DKDEstats = zeros(5, length(Nbins), Nrep);

for index=1:length(Nbins)
    % number of particles
    Nparticles = Nbins(index)^p;
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
    for j=1:Nrep
        % sample from h
        hSample = random(gmH, 10^5);
        % initial distribution
        f0 = rand(Nbins(index)^p, 1);
        x0 = rand(Nparticles, p);
        % EMS
        tstart = tic;
        EMSres = ems_p(hDisc, p, eval, Niter, epsilon, f0, sigmaG);
        EMStime(j, index) = toc(tstart);
        
        % SMC
        tstart = tic;
        [x, W] = smc_p_dim_gaussian_mixture(Nparticles, Niter, epsilon, x0, hSample, sigmaG);
        SMCtime(j, index) = toc(tstart);
        
        % moments & probability
        mHat = zeros(2, p);
        vHat = zeros(2, p);
        sHat = zeros(2, p);
        kHat = zeros(2, p);
        
        for i=1:p
            mHat(1, i) = sum(W.*x(:, i))/sum(W);
            mHat(2, i) = sum(EMSres.*eval(:, i))/sum(EMSres);
            vHat(1, i) = sum(W.*x(:, i).^2)/sum(W) - mHat(1, i)^2;
            vHat(2, i) = sum(EMSres.*eval(:, i).^2)/sum(EMSres) - mHat(2, i)^2;
            sHat(1, i) = sum(W.*(x(:, i) - mHat(1, i)).^3)/(sum(W)*vHat(1, i)^(3/2));
            sHat(2, i) = sum(EMSres.*(eval(:, i) - mHat(2, i)).^3)/(sum(EMSres)*vHat(2, i)^(3/2));
            kHat(1, i) = sum(W.*(x(:, i) - mHat(1, i)).^4)/(sum(W)*vHat(1, i)^2);
            kHat(2, i) = sum(EMSres.*(eval(:, i) - mHat(2, i)).^4)/(sum(EMSres)*vHat(2, i)^2);
        end
        pEMS = sum(EMSres(lq))/sum(EMSres);
        pSMC = sum(prod((x <= 0.5 & x>= 0), 2))/Nparticles;
        EMSstats(:, index, j) = [mean(mHat(2, :)) mean(vHat(2, :)) mean(sHat(2, :)) mean(kHat(2, :)) pEMS];
        SMCstats(:, index, j) = [mean(mHat(1, :)) mean(vHat(1, :)) mean(sHat(1, :)) mean(kHat(1, :)) pSMC];
    end
end
EMSstatsSTD(1, :, :) = EMSstats(1, :, :) - m;
EMSstatsSTD(2, :, :) = EMSstats(2, :, :) - v;
EMSstatsSTD(3, :, :) = EMSstats(3, :, :) - s;
EMSstatsSTD(4, :, :) = EMSstats(4, :, :) - k;
EMSstatsSTD(5, :, :) = EMSstats(5, :, :) - p_quadrant;
SMCstatsSTD(1, :, :) = SMCstats(1, :, :) - m;
SMCstatsSTD(2, :, :) = SMCstats(2, :, :) - v;
SMCstatsSTD(3, :, :) = SMCstats(3, :, :) - s;
SMCstatsSTD(4, :, :) = SMCstats(4, :, :) - k;
SMCstatsSTD(5, :, :) = SMCstats(5, :, :) - p_quadrant;
% create table
format long
resTable = zeros(2*length(Nbins), 6);
resTable([1 3], 1:5) = mean(EMSstatsSTD.^2, 3)';
resTable([1 3], 6) = mean(EMStime, 1);
resTable([2 4], 1:5) = mean(SMCstatsSTD.^2, 3)';
resTable([2 4], 6) = mean(SMCtime, 1);
% log runtime 
resTable(:, 6) = log(resTable(:, 6));
% write table
dlmwrite('p_dim',resTable,'delimiter', '&',...
    'newline', 'pc')
% save('2dim5Feb2021.mat')

