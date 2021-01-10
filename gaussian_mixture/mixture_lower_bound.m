% Dependence of MISE on lower bound 

% set seed
rng('default');

% f, h, g are Normals
h = @(y) 2*normpdf(y, 0.3, sqrt(0.043^2 + 0.045^2))./3 + ...
    normpdf(y, 0.5, sqrt(0.015^2 + 0.045^2))./3;
g = @(x,y) normpdf(y, x, 0.045);
f = @(x) normpdf(x, 0.3, 0.015)/3 + normpdf(x, 0.5, 0.043)*2/3;
ftilde = @(x, a) 3*f(x) * nc_gaussian_mixture(a, x, 0.045)./...
    (nc_gaussian_mixture(a, 0.3, sqrt(0.015^2 + 0.045^2)) + 2*nc_gaussian_mixture(a, 0.5, sqrt(0.043^2 + 0.045^2)));
% set paremeters
% number of iterations
Niter = 100;
% number of particles
Nparticles = [100, 500, 1000];
% scale for SMC smoothing
epsilon = 1e-03;
% bin centres
KDEx = linspace(0, 1, 100);
% lower bound 
a = linspace(0.2, 1, 10);
% number of replications
Nrep = 100;
% mise
SMCmise = zeros(length(Nparticles), length(a));

parfor index=1:length(a)
    SMCmiseN = zeros(length(Nparticles), 1);
    for N=1:length(Nparticles)
        SMCmiseRep = zeros(Nrep, 1);
        for k=1:Nrep
            % initial distribution
            f0SMC = 0.4-a(index)+2*a(index)*rand(Nparticles(N), 1);
            % sample from truncated h
            hSample = Ysample_gaussian_mixture_trunc(10^5, a(index));
            % SMC
            [x, W] = smc_gaussian_mixture_trunc(Nparticles(N), Niter, epsilon, a(index), f0SMC, hSample);
            % KDE
            % bandwidth
            bw = sqrt(epsilon^2 + optimal_bandwidthESS(x(Niter, :), W(Niter, :))^2);
            KDEy = ksdensity(x(Niter, :), KDEx, 'weight', W(Niter, :), ...
                'Bandwidth', bw, 'Function', 'pdf');
            SMCmiseRep(k) = var(f(KDEx) - KDEy, 1);
        end
        SMCmiseN(N) = mean(SMCmiseRep);
    end
    SMCmise(:, index) = SMCmiseN;
end
close all;
plot(a, SMCmise, 'LineWidth', 4)
legend(['N = ' num2str(Nparticles(1))], ['N = ' num2str(Nparticles(2))], ...
    ['N = ' num2str(Nparticles(3))],...
    'interpreter', 'latex', 'FontSize', ...
    10, 'Location', 'best');
pbaspect([1.5 1 1])
printEps(gcf, 'mixture_lower_bound.eps')