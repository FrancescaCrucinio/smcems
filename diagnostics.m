% Combine diagnostics for f and for h
% OUTPUTS
% 1 - mean
% 2 - variance
% 3 - Mean Integrated Squared Error for f
% 4 - Kullback Leibler divergence
% INPUTS
% 'f' true f (function handle)
% 'h' true h (function handle)
% 'g' mixing kernel (function handle)
% 'KDEx' points in the domain of f at which the approximated f
% and the true f are compared
% 'KDEy' approximated f
% 'refY' points in the domain of h at which the approximated h
% and the true h are compared
function[out1, difference] = diagnostics(f, h, g, KDEx, KDEy, refY)
    [m, v, difference, misef] = diagnosticsF(f, KDEx, KDEy);
    [~, div] = diagnosticsH(h, g, KDEx, KDEy, refY);
    out1 = [m, v, misef, div];
end