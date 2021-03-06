% SMC for gaussian mixture exmaple
% OUTPUTS
% 1 - particle locations
% 2 - particle weights
% INPUTS
% 'N' number of particles
% 'Niter' number of time steps
% 'epsilon' standard deviation for Gaussian smoothing kernel
% 'x0' initial distribution.
% 'hSample' sample from h
% 'sigmaG' standard deviation of g

function[xNew, W] = smc_p_dim_gaussian_mixture(N, Niter, epsilon, x0, hSample, sigmaG)
    % initial distribution
    xOld = x0;
    % number of dimensions
    p = size(x0, 2);
    % uniform weights at time n = 1
    W = ones(N, 1)/N;
    % number of samples to draw from h(y)
    M = min(N, size(hSample, 1));
    for n=2:Niter
        % ESS
        ESS=1/sum(W.^2);
        %%%%% RESAMPLING
        if(ESS < N/2)
            xNew = xOld(mult_resample(W, N), :);
            W = ones(N, 1)/N;
        else
            xNew = xOld;
        end
        
        % Markov kernel
        xNew = xNew + epsilon*randn(N, p);
        
        % Compute h^N_{n}
        yIndex = randsample(1:size(hSample, 1), M, false);
        y = hSample(yIndex, :);
        hN = zeros(size(y, 1),1);
        for j=1:size(y, 1)
            hN(j) = mean(W .* mvnpdf(xNew, y(j, :), sigmaG^2*eye(p)));
        end

        % update weights
        for i=1:N
            g = mvnpdf(xNew(i, :), y, sigmaG^2*eye(p));
            % potential at time n
            potential = mean(g ./ hN);
            % update weight
            W(i) = W(i) * potential;
        end
        % normalise weights
        W = W / sum(W);
        xOld = xNew;
    end
end