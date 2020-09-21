% Comparison of MISE and runtime for deterministic and stochastic
% discretisation of EMS

% set seed
rng('default');

% f, h, g are Normals
h = @(y) 2*normpdf(y, 0.3, sqrt(0.043^2 + 0.045^2))./3 + ...
    normpdf(y, 0.5, sqrt(0.015^2 + 0.045^2))./3;
g = @(x,y) normpdf(y, x, 0.045);
f = @(x) normpdf(x, 0.3, 0.015)/3 + normpdf(x, 0.5, 0.043)*2/3;

% set paremeters
% number of iterations
Niter = 100;
% number of bins/particles
N = [100, 500, 1000, 5000, 10000];
% number of samples for IB
nsamples = 1000;
% scale for SMC smoothing
epsilon = 1e-03;
% smoothing kernel
K = @(x1, x2) normpdf(x2, x1, epsilon);

% number of replications
Nrep = 1000;
% execution times
EMStime = zeros(Nrep, length(N));
SMCtime = zeros(Nrep, length(N));
% mise
EMSstats = zeros(Nrep, length(N));
SMCstats = zeros(Nrep, length(N));

parfor index=1:length(N)
    for k=1:Nrep
        % bin centers
        fBin = 1/(2*N(index)):1/N(index):1;
        % discretisation of h
        hDisc = h(fBin);
        % discretisation of g
        gDisc = zeros(N(index));
        for i=1:N(index)
           for j=1:N(index)
              gDisc(i, j) = g(fBin(j), fBin(i));       
           end    
        end
        % random starting point
        f0EM = rand(N(index), 1);
        f0SMC = rand(N(index), 1);
        % SMC
        tstart = tic;
        [x, W] = smc_gaussian_mixture(N(index), Niter, epsilon,...
            f0SMC);
        SMCtime(k, index) = toc(tstart);
        % KDE
    	% bandwidth
        bw = sqrt(epsilon^2 + optimal_bandwidthESS(x(Niter, :), W(Niter, :))^2);
        KDEy = ksdensity(x(Niter, :), fBin, 'weight', W(Niter, :), ...
            'Bandwidth', bw, 'Function', 'pdf');
        [~, ~, ~, SMCstats(k, index)] = diagnosticsF(f, fBin, KDEy);
        % EMS
        % smoothing kernel
        K = @(x1, x2) normpdf(x2, x1, bw);
        % smoothing matrix
        Kmatrix = smoothingMatrix(K, fBin);
        tstart = tic;
        EMSres = ems(gDisc, hDisc, Niter, Kmatrix, f0EM);
        EMStime(k, index) = toc(tstart);
        [~, ~, ~, EMSstats(k, index)] = diagnosticsF(f, fBin, EMSres(Niter, :));
    end
end

SMCruntime = mean(SMCtime, 1);
EMSruntime = mean(EMStime, 1);
SMCmise = mean(SMCstats, 1);
EMSmise = mean(EMSstats, 1);

close all;
h = zeros(2 + length(N),1);
h(1) = semilogx(SMCruntime, SMCmise, ':b', 'LineWidth', 3);
hold on
h(2) = semilogx(EMSruntime, EMSmise, '-.r', 'LineWidth', 3);
markers = {'o' 's' 'd' 'x' 'p'};
for i=1:length(N)
    h(2 + i) = semilogx(SMCruntime(i), SMCmise(i), strcat(markers{i} ,'k'),...
        'MarkerFaceColor', 'k', 'LineWidth', 4);
    semilogx(SMCruntime(i), SMCmise(i), strcat(markers{i} ,'b'),...
        'MarkerFaceColor', 'b', 'LineWidth', 4)
    semilogx(EMSruntime(i), EMSmise(i), strcat(markers{i} ,'r'),...
        'MarkerFaceColor', 'r', 'LineWidth', 4)
end
legend(h, 'SMC', 'EMS', ['N = ' num2str(N(1))], ['N = ' num2str(N(2))], ...
    ['N = ' num2str(N(3))], ['N = ' num2str(N(4))], ['N = ' num2str(N(5))], ...
    'interpreter', 'latex', 'FontSize', ...
    10, 'Location', 'southwest');
pbaspect([1.5 1 1])