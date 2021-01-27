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

function f = ems_p(hDisc, p, eval, Niter, epsilon, f0) 
    % get dimension of unknown function f
    M = length(f0);
    % initial distribution
    f = f0;
    
    for t=2:Niter
        f_temp = f;
        % update the denominator
        den = zeros(1, M);
        for d=1:M
        den(d) = f_temp' * mvnpdf(eval, eval(d, :), 0.045^2*eye(p));
        end
        for b=1:M
            % numerator
            f_temp(b) = f_temp(b)*sum((hDisc .* mvnpdf(eval(b, :), eval, 0.045^2*eye(p)))./den');
        end
        % smooth f
        for b=1:M
            f(b) = f_temp' * mvnpdf(eval, eval(b, :), epsilon^2*eye(p));
        end
    end
end