% EMS algorithm for Poisson distributed data
% OUTPUTS
% 1 - estimate of f
% INPUTS
% 'g' matrix corresponding to the kernel g(y | x)
% 'h' discretized data distribution h(y)
% 'maxStep' number of steps to iterate
% 'S' smoothing matrix
% 'f0' initial distribution

function [f_final] = ems_p(gDisc, hDisc, Niter, S, f0) 
    % get dimension of unknown function f
    M = length(f0);
    % initialize vector to store the values of f
    f = zeros(Niter, M);
    % initial distribution
    f(1, :) = f0;
    f(1, :) = f(1, :)*S;
    
    % compute the numerator of the EM iterative formula
    num = hDisc' .* gDisc;
    for t=2:Niter
        % update the denominator
        den = gDisc * f(t-1,:)';
        % update f
        f(t, :) = f(t-1, :) .* sum(num./den);
        % smooth f
        f(t, :) = f(t, :)*S ;
    end
    f_final = f(Niter, :)';
end