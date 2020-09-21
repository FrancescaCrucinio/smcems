% SMC for analytically tractable example (exact potential)
% OUTPUTS
% 1 - particle locations
% 2 - particle weights
% INPUTS
% 'N' number of particles
% 'Niter' number of time steps
% 'epsilon' standard deviation for Gaussian smoothing kernel
% 'exactVarianceH' vector of exact variances for h_{n-1}(y)
% 'x0' user selected initial distribution
% 'M' number of samples from h(y) to be drawn at each iteration

function[x, W] = ...
    smc_AT_exact_potential(N, Niter, epsilon, exactVarianceH, x0, M)

    % initialise a matrix x storing the particles
    x = zeros(Niter,N);
    % initialise a matrix W storing the weights
    W = zeros(Niter,N);
    % initial distribution is given as input:
    x(1, :) = x0;
    % uniform weights at time n = 1
    W(1, :) = ones(1, N)/N;
    
    for n=2:Niter
        % get samples from h(y)
        y = 0.5 + sqrt(0.043^2 + 0.045^2) * randn(M, 1);
        %%%%%% RESAMPLING
        x(n,:) = x(n-1,mult_resample(W(n-1,:), N));
        W(n,:) = 1/N;
                              
        % Markov kernel
        x(n, :) = x(n, :) + epsilon*randn(1, N);

        % update weights
        for i=1:N
            potential = sum( sqrt(exactVarianceH(n))/0.045 * ...
                exp(-(y - x(n, i)).^2/(2*0.045^2) + ...
                (y - 0.5).^2/(2*exactVarianceH(n))));
            % update weight
            W(n, i) = potential;
        end
        % normalise weights
        W(n, :) = W(n, :)./ sum(W(n, :));
    end
end