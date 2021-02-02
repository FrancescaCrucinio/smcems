% SMC for PET reconstruction
% OUTPUTS
% 1/2 - particle locations
% 3 - particle weights
% 4 - iteration at which stopping criterion is satisfied
% INPUTS
% 'N' number of particles
% 'maxIter' maximum number of iterations
% 'epsilon' scale parameter for the gaussian smoothing kernel
% 'phi' degrees at which projections are taken
% 'xi' offset of projections
% 'R' Radon transform
% 'sigma' standard deviation for Normal describing alignment
% 'tolerance' tolerance for stopping rule
% 'm' width of moving average

function[x1, x2, W, iter_stop] = smc_pet(N, maxIter, epsilon, phi, xi, R, sigma, m)    
    
    % sample from the sinogram R
    pixels = length(phi);
    hSample = pinky(xi, phi, R', 10^6);
    % 2D array to store the x-coordinate of the particles for each time step
    x1 = zeros(maxIter, N);
    % 2D array to store the y-coordinate of the particles for each time step
    x2 = zeros(maxIter, N);
    % 2D array to store the weights of the particles for each time step
    W = zeros(maxIter, N);
    % sample random particles in [-0.75, 0.75]^2 for time step n = 1
    x1(1,:) = 1.5*rand(1, N) - 0.75;
    x2(1,:) = 1.5*rand(1, N) - 0.75;
    % uniform weights at time step n = 1
    W(1,:) = ones(1, N)/N;

    % KDE
    % set coordinate system over image
    % x is in [-0.75, 0.75]
    evalX1 = linspace(-0.75 + 1/pixels, 0.75 - 1/pixels, pixels);
    % y is in [-0.75, 0.75]
    evalX2 = linspace(-0.75 + 1/pixels, 0.75 - 1/pixels, pixels);
    % build grid with this coordinates
    RevalX = repmat(evalX1, pixels, 1);
    eval =[RevalX(:) repmat(evalX2, 1, pixels)'];
    % grid for reconstructed h
    delta1 = phi(2) - phi(1);
    delta2 = xi(2) - xi(1);
    % bandwidth
    bw1 = sqrt(epsilon^2 + optimal_bandwidthESS(x1(1, :), W(1, :))^2);
    bw2 = sqrt(epsilon^2 + optimal_bandwidthESS(x2(1, :), W(1, :))^2);
    KDE = ksdensity([x1(1, :)' x2(1, :)'], eval, 'weight', ...
        W(1, :), 'Bandwidth', [bw1 bw2], 'Function', 'pdf');
    % variance
    PETvar = zeros(1, maxIter);
    PETvar(1) = var(KDE, 1);
    % reconstruction of h
    hatHNew = Hreconstruction_pet(phi, xi, sigma, eval, KDE);
    % stopping rule
    iter_stop = 100;
        
    'Start SMC'
    for n=2:maxIter
        hatHOld = hatHNew;
        % ESS
        ESS=1/sum(W(n-1,:).^2);
        %%%%%% RESAMPLING
        if(ESS < N/2)
            ['Resampling at Iteration ' num2str(n)]
            indices = mult_resample(W(n-1,:), N);
            x1(n,:) = x1(n-1, indices);
            x2(n,:) = x2(n-1, indices);
            W(n,:) = 1/N;
        else
            x1(n,:) = x1(n-1,:);
            x2(n,:) = x2(n-1,:);	
            W(n,:) = W(n-1,:);
        end
                
        % Markov kernel
        x1(n,:) = x1(n,:) + epsilon * randn(1,N);
        x2(n,:) = x2(n,:) + epsilon * randn(1,N);
        
        % Compute h^N_{n} one for each sample from R
        yIndex = randsample(1:length(hSample), N, true);
        y = hSample(yIndex, :);
        hN = zeros(N,1);
        for j=1:N
             hN(j) = sum(W(n,:) .*  normpdf(x1(n-1,:)*cos(y(j,2)) + ...
                 x2(n-1,:)*sin(y(j,2)) - y(j,1), 0, sigma));
        end
        
        % update weights
        for i=1:N
            g = normpdf(x1(n,i)*cos(y(:,2)) + ...
                      x2(n,i)*sin(y(:,2)) - y(:,1), 0, sigma);
            % potential at time n
            potential = sum(g./ hN);
            % update weights
            W(n,i) = W(n,i) .* potential;
        end
        % normalise weights
        W(n,:) = W(n,:) ./ sum(W(n,:));
        % KDE
        % bandwidth
        bw1 = sqrt(epsilon^2 + optimal_bandwidthESS(x1(n, :), W(n, :))^2);
        bw2 = sqrt(epsilon^2 + optimal_bandwidthESS(x2(n, :), W(n, :))^2);
        KDE = ksdensity([x1(n, :)' x2(n, :)'], eval, 'weight', ...
            W(n, :), 'Bandwidth', [bw1 bw2], 'Function', 'pdf');
        % variance
        PETvar(n) = var(KDE, 1);
        % L2norm
        KDE = reshape(KDE, [pixels, pixels]);
        KDE = flipud(mat2gray(KDE));
        KDE = KDE(:);
        hatHNew = Hreconstruction_pet(phi, xi, sigma, eval, KDE);
        % stopping rule
        if(n>=m)
            L2norm = delta1*delta2*sum((hatHNew - hatHOld).^2, 'all');
            moving_var = var(PETvar((n-m+1):n));
            if(L2norm<=moving_var)
                iter_stop = n;
                ['Stop at iteration ' num2str(n)]
                % uncomment to exit algorithm as soon as tolerance reached
                % break
            end
        end
    end
    'SMC Finished'
end
