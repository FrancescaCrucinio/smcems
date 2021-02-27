% Comparison of convergence for EM, EMS and SMC

% set seed
rng('default');

% variances for f, g, h
sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaG + sigmaF;
% f, g, h are Normals
h = @(y) normpdf(y, 0.5, sqrt(sigmaH));
g = @(x,y) normpdf(y, x, sqrt(sigmaG));
f = @(x) normpdf(x, 0.5, sqrt(sigmaF));

% set parameters
% EM/EMs
% number of iterations
Niter = 100;
% number of bins
Nbins = 100;
% bin centers
fBin = 1/(2*Nbins):1/Nbins:1;
% smoothing kernel
epsilon = 1e-01;
K = @(x, z) normpdf(z, x, epsilon);
Kmatrix = smoothingMatrix(K, fBin);
% SMC
% number of particles
Nparticles = 1000;

% coordinates at which compute KL divergence
refY = linspace(0,1,100);
refX = linspace(0,1,100);

% discretisation of h
hBin = 1/(2*Nbins):1/Nbins:1;
hDisc = h(hBin);
% discretisation of g
gDisc = zeros(Nbins);
for i=1:Nbins
   for j=1:Nbins
      gDisc(i, j) = g(hBin(j), hBin(i));       
   end    
end

%%% initial values
f0EM = zeros(3, Nbins);
f0SMC = zeros(3, Nparticles);
% delta, uniform and fixed point
% value at which the Dirac delta is concentrated
delta = 0.5;
f0EM(1, :) = delta*ones(1, Nbins);
f0EM(2, :) = rand(1, Nbins);
f0EM(3, :) = 0.5 + sqrt(sigmaH).*randn(Nbins,1);
f0SMC(1, :) = delta*ones(1, Nparticles);
f0SMC(2, :) = rand(1, Nparticles);
f0SMC(3, :) = 0.5 + sqrt(sigmaH).*randn(Nparticles,1);
% results
fEM = zeros(3, Nbins);
fEMS = zeros(3, Nbins);
fSMC = zeros(3, Nbins);
% set divergence path for different initial values
divEM = zeros(3, Niter);
divEMS = zeros(3, Niter);
divSMC = zeros(3, Niter);
% run EM, EMS and SMC
for i=1:size(f0EM, 1)
    % EM
    EMres = em(gDisc, hDisc, Niter, f0EM(i, :));
    fEM(i, :) = EMres(Niter, :);
    % EMS
    EMSres = ems(gDisc, hDisc, Niter, Kmatrix, f0EM(i, :));
    fEMS(i, :) = EMSres(Niter, :);
    y = 0.5 + sqrt(0.043^2 + 0.045^2) * randn(10^4, 1);
    M = min(Nparticles, length(y));
    % SMC
    [x, W] = smc_AT_approximated_potential(Nparticles, Niter, epsilon,...
        f0SMC(i, :), y, M);   
    % KL
    for n=1:Niter
        % EM
        [~, divEM(i, n)] = diagnosticsH(h, g, fBin, EMres(n, :), refY);
        % EMS
        [~, divEMS(i, n)] = diagnosticsH(h, g, fBin, EMSres(n, :), refY);
        % SMC
        bw = sqrt(epsilon^2 + optimal_bandwidthESS(x(n, :), W(n, :))^2);
        KDEy = ksdensity(x(n, :), fBin, 'weight', W(n, :), ...
            'Function', 'pdf', 'Bandwidth', bw); 
        [~, divSMC(i, n)] = diagnosticsH(h, g, fBin, KDEy, refY);
    end
end

close all;
figure(1);
for i=1:size(f0EM, 1)
    plot(1:Niter, divEM(i, :), '--', 'Linewidth', 3)
    hold on
    plot(1:Niter, divEMS(i, :), ':', 'Linewidth', 3)
    plot(1:Niter, divSMC(i, :), 'Linewidth', 3)
end
set(gca, 'Xscale', 'log')
xlabel('log10(iteration)', 'FontSize', 10, 'interpreter', 'latex');
ylabel('KL($h$, $\hat{h}$)', 'interpreter', 'latex', 'FontSize', 10);
pbaspect([1.5 1 1])
plot_names = {['EM - $\delta_{' num2str(delta) '}$'] ...
    ['EMS - $\delta_{' num2str(delta) '}$']...
    ['SMC - $\delta_{' num2str(delta) '}$']...
    'EM - Unif([0,1])'...
    'EMS - Unif([0,1])'...
    'SMC - Unif([0,1])'...
    'EM - $f(x)$'...
    'EMS - $f(x)$'...
    'SMC - $f(x)$'};
legendHandle = legend(plot_names, 'interpreter', 'latex', 'FontSize', 9, ...
    'Orientation', 'horizontal', 'NumColumns', 3, 'Location', 'SouthOutside');
hold off;