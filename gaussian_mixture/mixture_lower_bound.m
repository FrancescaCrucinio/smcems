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
SMCmiset = zeros(length(Nparticles), length(a));
SMCmise = zeros(length(Nparticles));
parfor N=1:length(Nparticles)
    SMCmiseRept = zeros(Nrep, length(a));
    SMCmiseRep = zeros(Nrep, 1);
    for k=1:Nrep
        for index=1:length(a)      
            % initial distribution
            f0SMC = 0.4-a(index)+2*a(index)*rand(Nparticles(N), 1);
            % sample from truncated h
            hSample = Ysample_gaussian_mixture_trunc(10^5, a(index));
            % SMC - truncated
            [x, W] = smc_gaussian_mixture_trunc(Nparticles(N), Niter, epsilon, a(index), f0SMC, hSample);
            % KDE
            % bandwidth
            bw = sqrt(epsilon^2 + optimal_bandwidthESS(x(Niter, :), W(Niter, :))^2);
            KDEy = ksdensity(x(Niter, :), KDEx, 'weight', W(Niter, :), ...
                'Bandwidth', bw, 'Function', 'pdf');
            SMCmiseRept(k, index) = var(f(KDEx) - KDEy, 1);
        end
        % initial distribution
        f0SMC = rand(Nparticles(N), 1);
        % sample from  h
        hSample = Ysample_gaussian_mixture(10^5);
        % SMC - truncated
        [x, W] = smc_gaussian_mixture(Nparticles(N), Niter, epsilon, f0SMC, hSample);
        % KDE
        % bandwidth
        bw = sqrt(epsilon^2 + optimal_bandwidthESS(x(Niter, :), W(Niter, :))^2);
        KDEy = ksdensity(x(Niter, :), KDEx, 'weight', W(Niter, :), ...
            'Bandwidth', bw, 'Function', 'pdf');
        SMCmiseRep(k) = var(f(KDEx) - KDEy, 1);
    end
    SMCmiset(N, :) = mean(SMCmiseRept, 1);
    SMCmise(N) = mean(SMCmiseRep);
end
close all;
plot(a, SMCmiset(1, :), '-.k', 'LineWidth', 4);
hold on
yline(SMCmise(1),  'LineWidth', 4, 'color', 'black')
p = plot(a, SMCmiset, '-.', 'LineWidth', 4);
colors = get(p, 'Color');
for i=1:length(Nparticles)
    yline(SMCmise(i),  'LineWidth', 4, 'color', colors{i})
end
legend('with LB', 'no LB', ['N = ' num2str(Nparticles(1))], ['N = ' num2str(Nparticles(2))], ...
    ['N = ' num2str(Nparticles(3))],...
    'interpreter', 'latex', 'FontSize', ...
    10, 'Location', 'eastoutside');
pbaspect([1.5 1 1])
% printEps(gcf, 'mixture_lower_bound.eps')
