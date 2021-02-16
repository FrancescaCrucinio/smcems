% EMS algorithm for Poisson distributed data
% OUTPUTS
% 1 - estimate of f
% INPUTS
% 'hDisc' discretized data distribution h(y)
% 'p' dimension
% 'eval' p-dimensional grid
% 'Niter' number of iterations
% 'epsilon' smoothing parameter
% 'f0' initial distribution
% 'sigmaG' standard deviation of g

function f = ems_p(hDisc, p, eval, Niter, epsilon, f0, sigmaG) 
    % get dimension of unknown function f
    M = length(f0);
    % initial distribution
    f = f0;
    % smoothing
    for b=1:M
        weights = mvnpdf(eval, eval(b, :), epsilon^2*eye(p));
        weights = weights./sum(weights);
        f(b) = f' * weights;
    end
    for t=2:Niter
        % update the denominator
        den = zeros(1, M);
        for d=1:M
        den(d) = f' * mvnpdf(eval, eval(d, :), sigmaG^2*eye(p));
        end
        for b=1:M
            % numerator
            f(b) = f(b)*sum(hDisc .* mvnpdf(eval(b, :), eval, sigmaG^2*eye(p))./den');
        end
        % smooth
        for b=1:M
            weights = mvnpdf(eval, eval(b, :), epsilon^2*eye(p));
            weights = weights./sum(weights);
            f(b) = f' * weights;
        end
    end
end