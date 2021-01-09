% SMC for gaussian mixture exmaple
% OUTPUTS
% 1 - particle locations
% 2 - particle weights
% INPUTS
% 'N' number of particles
% 'Niter' number of time steps
% 'epsilon' standard deviation for Gaussian smoothing kernel
% 'x0' (optional) user selected initial distribution.
% Default = Uniform on [0, 1]
% 'hSample' (optional) sample from h

function[x, W] = smc_gaussian_mixture(N, Niter, epsilon, varargin)
    % initialise a matrix x storing the particles at each time step
    x = zeros(Niter,N);
    % initialise a matrix W storing the weights at each time step
    W = zeros(Niter,N);
    % sample random particles for time n = 1
    % if the initial distribution is given as input:
    if(nargin>=4)
        x(1, :) = varargin{1};
    else
        x(1, :) = rand(1, N);
    end
    if(nargin==5)
        hSample = varargin{2};
    else
        hSample = Ysample_gaussian_mixture(N);
    end
    % uniform weights at time n = 1
    W(1, :) = ones(1, N)/N;

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
        x(n,:) = x(n,:) + epsilon*randn(1, N);
        
        % Compute h^N_{n}
        y = randsample(hSample, N, true);
        hN = zeros(length(y),1);
        for j=1:length(y)
            hN(j) = sum(W(n,:) .* normpdf(y(j),x(n,:),0.045));
        end

        % update weights
        for i=1:N
            g = normpdf(y,x(n,i),0.045);
            % potential at time n
            potential = sum(g ./ hN);
            % update weight
            W(n,i) = W(n,i) * potential;
        end
        % normalise weights
        W(n,:) = W(n,:) ./ sum(W(n,:));
    end
end