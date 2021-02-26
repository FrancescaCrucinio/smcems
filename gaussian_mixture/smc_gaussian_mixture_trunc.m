% SMC for gaussian mixture example with truncation
% OUTPUTS
% 1 - particle locations
% 2 - particle weights
% INPUTS
% 'N' number of particles
% 'Niter' number of time steps
% 'epsilon' standard deviation for Gaussian smoothing kernel
% 'a' truncation interval
% 'x0' user selected initial distribution.
% 'hSample' sample from h

function[x, W] = smc_gaussian_mixture_trunc(N, Niter, epsilon, a, x0, hSample)
    % initialise a matrix x storing the particles at each time step
    x = zeros(Niter,N);
    % initialise a matrix W storing the weights at each time step
    W = zeros(Niter,N);
    x(1, :) = x0;
    
    % uniform weights at time n = 1
    W(1, :) = ones(1, N)/N;
    % number of samples from h(y) to draw at each iteration
    M = min(N, length(hSample));
    for n=2:Niter
        % ESS
        ESS=1/sum(W(n-1,:).^2);
        %%%%%% RESAMPLING
        if(ESS < N/2)
            x(n,:) = x(n-1,mult_resample(W(n-1,:), N));
            W(n,:) = 1/N;
        else
            x(n,:) = x(n-1,:);	    
            W(n,:) = W(n-1,:);
        end
        
        % Markov kernel
        proposal = x(n,:) + epsilon*randn(1, N);
        accept = proposal < 0.4+a & proposal > 0.4-a;
        x(n,accept) = proposal(accept);
        
        % Compute h^N_{n}
       y = randsample(hSample, M, false);
        hN = zeros(length(y),1);
        for j=1:length(y)
            hN(j) = mean(W(n,:) .* normpdf(y(j),x(n,:),0.045)./nc_gaussian_mixture(a, x(n,:), 0.045));
        end

        % update weights
        for i=1:N
            g = normpdf(y,x(n,i),0.045)./nc_gaussian_mixture(a, x(n,i), 0.045);
            % potential at time n
            potential = mean(g ./ hN);
            % update weight
            W(n,i) = W(n,i) * potential;
        end
        % normalise weights
        W(n,:) = W(n,:) ./ sum(W(n,:));
    end
end