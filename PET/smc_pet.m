% SMC for PET reconstruction
% OUTPUTS
% 1/2 - particle locations
% 3 - particle weights
% INPUTS
% 'N' number of particles
% 'Niter' number of time steps
% 'epsilon' scale parameter for the gaussian smoothing kernel
% 'phi' degrees at which projections are taken
% 'xi' offset of projections
% 'R' Radon transform
% 'sigma' standard deviation for Normal describing alignment

function[x, y, W] = smc_pet(N, Niter, epsilon, phi, xi, R, sigma)    
    % convert phi to radiants
    phi = deg2rad(phi);
    % normalise xi
    xi = xi/max(xi);
    
    % 2D array to store the x-coordinate of the particles for each time step
    x = zeros(Niter, N);
    % 2D array to store the y-coordinate of the particles for each time step
    y = zeros(Niter, N);
    % 2D array to store the weights of the particles for each time step
    W = zeros(Niter, N);
    % sample random particles in [-0.7, 0.7]^2 for time step n = 1
    x(1,:) = 1.4 * rand(1, N) - 0.7;
    y(1,:) = 1.4 * rand(1, N) - 0.7;
    % uniform weights at time step n = 1
    W(1,:) = ones(1, N)/N;

    'Start SMC'
    for n=2:Niter
        % sample from the sinogram R
        hSample = pinky(xi, phi, R', N);
        % ESS
        ESS=1/sum(W(n-1,:).^2);
        %%%%%% RESAMPLING
        if(ESS < N/2)
            ['Resampling at Iteration ' num2str(n)]
            indices = mult_resample(W(n-1,:), N);
            x(n,:) = x(n-1, indices);
            y(n,:) = y(n-1, indices);
            W(n,:) = 1/N;
        else
            x(n,:) = x(n-1,:);
            y(n,:) = y(n-1,:);	
            W(n,:) = W(n-1,:);
        end
                
        % Markov kernel
        x(n,:) = x(n,:) + epsilon * randn(1,N);
        y(n,:) = y(n,:) + epsilon * randn(1,N);
        
        % Compute h^N_{n} one for each sample from R
        hN = zeros(N,1);
        for j=1:N
             hN(j) = sum(W(n,:) .*  normpdf(x(n-1,:)*cos(hSample(j,2)) + ...
                 y(n-1,:)*sin(hSample(j,2)) - hSample(j,1), 0, sigma));
        end
        
        % update weights
        for i=1:N
            g = normpdf(x(n,i)*cos(hSample(:,2)) + ...
                      y(n,i)*sin(hSample(:,2)) - hSample(:,1), 0, sigma);
            % potential at time n
            potential = sum(g./ hN);
            % update weights
            W(n,i) = W(n,i) .* potential;
        end
        % normalise weights
        W(n,:) = W(n,:) ./ sum(W(n,:));
    end
    'SMC Finished'
end
