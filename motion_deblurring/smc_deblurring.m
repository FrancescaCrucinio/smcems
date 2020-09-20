% SMC for motion deblurring
% OUTPUTS
% 1/2 - particle locations
% 3 - particle weights
% INPUTS
% 'N' number of particles
% 'Niter' number of time steps
% 'epsilon' scale parameter for the gaussian kernel
% 'I' blurred and noisy image
% 'sigma' standard deviation for Normal approximating Dirac delta
% 'b' velocity of motion

function[x, y, W] = smc_deblurring(N, Niter, epsilon, I, sigma, b)     
    % get dimension of image
    pixels = size(I);
    % normalize velocity
    b = b/300;
    % x is in [-1, 1]
    Xin = linspace(-1 + 1/pixels(2), 1 - 1/pixels(2), pixels(2))';
    % y is in [-0.5, 0.5]  
    Yin = linspace(0.5 - 1/pixels(1), -0.5 + 1/pixels(1), pixels(1))';
    % 2D array to store the x-coordinate of the particles for each time step
    x = zeros(Niter, N);
    % 2D array to store the y-coordinate of the particles for each time step
    y = zeros(Niter, N);
    % sample random particles for x in [-1, 1] for time step n = 1
    x(1,:) = 2 * rand(1, N) - 1;
    % sample random particles for y in [-0.5, 0.5] for time step n = 1
    y(1,:) = rand(1, N) - 0.5;
    % 2D array to store the weights of the particles for each time step
    W = zeros(Niter, N);
    % uniform weights
    W(1,:) = ones(1, N)/N;
    
    'Start SMC'
    for n=2:Niter
        % get N samples from blurred image
        hSample = pinky(Yin, Xin', I', N);
        % ESS
        ESS=1/sum(W(n-1,:).^2);
        %%% RESAMPLING
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

        % compute h^N_{n} for each y_j
        hN = zeros(N,1);
        for j=1:N
             hN(j) = sum(W(n,:) .* normpdf(hSample(j,1) -y(n,: ), 0, sigma).* ...
                 (x(n,:) - hSample(j,2) <= b/2 & x(n,:) - hSample(j,2) >= -b/2)./b);
        end
        
        % apply Markov kerne
        x(n, :) = x(n,:) + epsilon * randn(1,N);
        y(n, :) = y(n,:) + epsilon * randn(1,N);

        % update weights
        for i=1:N
            g = normpdf(hSample(:,1) - y(n,i), 0, sigma).* ...
                (x(n,i) - hSample(:,2) <= b/2 & x(n,i) - hSample(:,2) >= -b/2)/b;
            % potential at time n
            potential = sum(g ./ hN);
            % check for division by 0
            potential(isnan(potential)) = 0;
            % update weight
            W(n,i) = W(n,i) .* potential;
        end
        % normalise weights
        W(n,:) = W(n,:) ./ sum(W(n,:));
    end
    'SMC Finished'
end
