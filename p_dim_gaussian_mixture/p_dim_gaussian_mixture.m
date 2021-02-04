% Influence of dimensionality for SMC and EMS

% set seed
rng('default');

% number of dimensions
p = 2;
% build mixture of Gaussians
mu = [0.3*ones(1, p); 0.5*ones(1, p)];
sigmaF = cat(3, 0.015^2*ones(1, p), 0.043^2*ones(1, p));
weights = [1/3, 2/3];
gmF = gmdistribution(mu, sigmaF, weights);
sigmaG = 0.045^2*eye(p);
sigmaH = sigmaF + sigmaG;
gmH = gmdistribution(mu, sigmaH, weights);

% moments of mixture
m = weights*mu;
second_moment = weights(1)*sigmaF(:, :, 1) + weights(2)*sigmaF(:, :, 2) + weights*mu.^2;
v = second_moment - m.^2;
third_moment = (0.0272025 + 2*0.127773)*ones(1, p)/3;
s = (third_moment - 3*m.*v-m.^3)./v.^(3/2);
fourth_moment = (0.00822165 + 2*0.0652838)*ones(1, p)/3;
k = (fourth_moment - 4*third_moment.*m + 6*second_moment.*m.^2 - 3*m.^4)./v.^2;
% probability of lower quadrant
p_quadrant = ((normcdf(0.5, 0.3, 0.015)-normcdf(0, 0.3, 0.015)) + 2*(normcdf(0.5, 0.5, 0.043)-normcdf(0, 0.3, 0.043)))/3;
p_quadrant = p_quadrant^p;

% set parameters
% number of iterations
Niter = 50;
% number of bins/particles
Nbins = [10 50];
% scale for SMC smoothing
epsilon = 1e-02;

% number of replications
Nrep = 10;
% execution times
EMStime = zeros(Nrep, length(Nbins));
SMCtime = zeros(Nrep, length(Nbins));
DKDEtime = zeros(Nrep, length(Nbins));
% mise
EMSstats = zeros(5, length(Nbins), Nrep);
SMCstats = zeros(5, length(Nbins), Nrep);
DKDEstats = zeros(5, length(Nbins), Nrep);

parfor index=1:length(Nbins)
    % number of particles
    Nparticles = Nbins(index)^p;
    % discretisation grid for EMS
    Ndarrays = cell(1, p);
    [Ndarrays{:}] = ndgrid(linspace(0, 1, Nbins(index))); 
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
        EMSres = ems_p(hDisc, p, eval, Niter, epsilon, f0);
        EMStime(j, index) =toc(tstart);
        
        % SMC
        tstart = tic;
        [x, W] = smc_p_dim_gaussian_mixture(Nparticles, Niter, epsilon, x0, hSample);
        SMCtime(j, index) = toc(tstart);
        
        % DKDE
        % sample from h
        y = random(gmH, Nparticles*10);
        tstart = tic;
        DKDEres = DKDE_p_dim(Nbins(index), y, 0.045);
        DKDEtime(j, index) = toc(tstart);
        
        % moments & probability
        mHat = zeros(3, p);
        vHat = zeros(3, p);
        sHat = zeros(3, p);
        kHat = zeros(3, p);
        
        for i=1:p
            mHat(1, i) = sum(W.*x(:, i))/sum(W);
            mHat(2, i) = sum(EMSres.*eval(:, i))/sum(EMSres);
            mHat(3, i) = sum(DKDEres.*eval(:, i))/sum(DKDEres);
            vHat(1, i) = sum(W.*x(:, i).^2)/sum(W) - mHat(1, i)^2;
            vHat(2, i) = sum(EMSres.*eval(:, i).^2)/sum(EMSres) - mHat(2, i)^2;
            vHat(3, i) = sum(DKDEres.*eval(:, i).^2)/sum(DKDEres) - mHat(3, i)^2;
            sHat(1, i) = sum(W.*(x(:, i) - mHat(1, i)).^3)/(sum(W)*vHat(1, i)^(3/2));
            sHat(2, i) = sum(EMSres.*(eval(:, i) - mHat(2, i)).^3)/(sum(EMSres)*vHat(2, i)^(3/2));
            sHat(3, i) = sum(DKDEres.*(eval(:, i) - mHat(3, i)).^3)/(sum(DKDEres)*vHat(3, i)^(3/2));
            kHat(1, i) = sum(W.*(x(:, i) - mHat(1, i)).^4)/(sum(W)*vHat(1, i)^2);
            kHat(2, i) = sum(EMSres.*(eval(:, i) - mHat(2, i)).^4)/(sum(EMSres)*vHat(2, i)^2);
            kHat(3, i) = sum(DKDEres.*(eval(:, i) - mHat(3, i)).^4)/(sum(DKDEres)*vHat(3, i)^2);
        end
        mHat = mHat - mean(m);
        vHat = vHat - mean(v);
        sHat = sHat - mean(s);
        kHat = kHat - mean(k);
        pEMS = sum(EMSres(lq))/sum(EMSres) - p_quadrant;
        pDKDE = sum(DKDEres(lq))/sum(DKDEres) - p_quadrant;
        pSMC = sum(prod((x <= 0.5 & x>= 0), 2))/Nparticles - p_quadrant;
        EMSstats(:, index, j) = [mean(mHat(2, :)) mean(vHat(2, :)) mean(sHat(2, :)) mean(kHat(2, :)) pEMS];
        SMCstats(:, index, j) = [mean(mHat(1, :)) mean(vHat(1, :)) mean(sHat(1, :)) mean(kHat(1, :)) pSMC];
        DKDEstats(:, index, j) = [mean(mHat(3, :)) mean(vHat(3, :)) mean(sHat(3, :)) mean(kHat(3, :)) pDKDE];
    end
end

% create table
format long
resTable = zeros(3*length(Nbins), 6);
resTable([1 4], 1:5) = mean(EMSstats.^2, 3)';
resTable([1 4], 6) = mean(EMStime, 1);
resTable([2 5], 1:5) = mean(SMCstats.^2, 3)';
resTable([2 5], 6) = mean(SMCtime, 1);
resTable([3 6], 1:5) = mean(DKDEstats.^2, 3)';
resTable([3 6], 6) = mean(DKDEtime, 1);
% log runtime 
resTable(:, 6) = log(resTable(:, 6));
% write table
dlmwrite('p_dim',resTable,'delimiter', '&',...
    'newline', 'pc')
