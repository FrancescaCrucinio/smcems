% EMS algorithm for Poisson distributed data
% OUTPUTS
% 1 - estimate of f
% INPUTS
% 'g' matrix corresponding to the kernel g(y | x)
% 'h' discretized data distribution h(y)
% 'maxStep' number of steps to iterate
% 'S' smoothing matrix
% 'varargin' (optional) user selected initial distribution.
% Default = 1/number of bins

function [f] = ems(g, h, maxStep, S, varargin) 
    % get dimension of unknown function f
    M = size(g, 2);
    % initialize vector to store the values of f
    f = zeros(maxStep, M);
    % set starting approximation for each bin
    % if the initial distribution is given as input:
    if(nargin==5)
        f(1, :) = varargin{1};
    else
        f(1, :) = ones(1, M)/M;
    end
    % smooth f
    f(1, :) = f(1, :)*S;
    
    % compute the numerator of the EM iterative formula
    num = h' .* g;
    for t=2:maxStep
        % update the denominator
        den = g * f(t-1,:)';
        % update f
        f(t, :) = f(t-1, :) .* sum(num./den);
        % smooth f
        f(t, :) = f(t, :)*S ;
    end
end