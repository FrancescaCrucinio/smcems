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
Nbins = [10, 50, 100];
% scale for SMC smoothing
epsilon = 1e-03;

% number of replications
Nrep = 100;
% execution times
EMStime = zeros(Nrep, length(Nbins));
SMCtime = zeros(Nrep, length(Nbins));
% mise
EMSstats = zeros(5, Nrep, length(Nbins));
SMCstats = zeros(5, Nrep, length(Nbins));

for index=1:length(Nbins)
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
    % discretisation of g
    gDisc = zeros(length(eval));
    for i=1:length(eval)
       for j=1:length(eval)
          gDisc(i, j) = mvnpdf(eval(i, :), eval(j, :), 0.045^2*eye(p));       
       end    
    end
    % indices in lower quadrant
    lq = logical(prod((eval <= 0.5) & (eval >= 0), 2));
    for k=1:Nrep
        % sample from h
        hSample = random(gmH, 10^5);
        % smoothing matrix
        Kmatrix = smoothingMatrix_p(epsilon, eval);
        % initial distribution
        f0 = rand(Nbins(index)^p, 1);
        x0 = rand(Nparticles, p);
        % EMS
        tstart = tic;
        EMres = ems_p(gDisc, hDisc, Niter, Kmatrix, f0);
        EMStime(k, index) =toc(tstart);
        
        %SMC
        tstart = tic;
        [x, W] = smc_p_dim_gaussian_mixture(Nparticles, Niter, epsilon, x0, hSample);
        SMCtime(k, index) =toc(tstart);
        
        % moments & probability
        mSMC = zeros(1, p);
        vSMC = zeros(1, p);
        sSMC = zeros(1, p);
        kSMC = zeros(1, p);
        mEMS = zeros(1, p);
        vEMS = zeros(1, p);
        sEMS = zeros(1, p);
        kEMS = zeros(1, p);
        for i=1:p
            mSMC(i) = sum(W.*x(:, i))/sum(W);
            mEMS(i) = sum(EMres.*eval(:, i))/sum(EMres);
            vSMC(i) = sum(W.*x(:, i).^2)/sum(W) - mSMC(i)^2;
            vEMS(i) = sum(EMres.*eval(:, i).^2)/sum(EMres) - mEMS(i)^2;
    %         for j=1:(i-1)
    %             vSMC(i, j) = sum(W.*x(:, i).*x(:, j))/sum(W) - mSMC(i)*mSMC(j);
    %             vEMS(i, j) = sum(EMres.*eval(:, i).*eval(:, j))/sum(EMres) - mEMS(i)*mEMS(j);
    %         end
            sSMC(i) = sum(W.*(x(:, i) - mSMC(i)).^3)/(sum(W)*vSMC(i)^(3/2));
            sEMS(i) = sum(EMres.*(eval(:, i) - mEMS(i)).^3)/(sum(EMres)*vEMS(i)^(3/2));
            kSMC(i) = sum(W.*(x(:, i) - mSMC(i)).^4)/(sum(W)*vSMC(i)^2);
            kEMS(i) = sum(EMres.*(eval(:, i) - mEMS(i)).^4)/(sum(EMres)*vEMS(i)^2);
        end
        EMSstats(1, k, index) = mean(mEMS);
        SMCstats(1, k, index) = mean(mSMC);
        EMSstats(2, k, index) = mean(vEMS);
        SMCstats(2, k, index) = mean(vSMC);
        EMSstats(3, k, index) = mean(sEMS);
        SMCstats(3, k, index) = mean(sSMC);
        EMSstats(4, k, index) = mean(kEMS);
        SMCstats(4, k, index) = mean(kSMC);
        EMSstats(5, k, index) = sum(EMres(lq))/sum(EMres);
        SMCstats(5, k, index) = sum(prod((x <= 0.5 & x>= 0), 2))/Nparticles;
    end
end
