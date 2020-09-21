% Analytically Tractable example - variance of EMS approximation at time n
% OUTPUTS
% 1 - variance of recursion with no smoothing
% 2 - variance of recursion with smoothing
% INPUTS
% 'sigmaH' variance of h
% 'sigmaG' variance of g
% 'sigmaF1' variance of starting approximation of f
% 'sigmaK' variance of smoothing kernel
% 'Niter' number of iterations of the recursion
function[sigmaF, sigmaFK] = AT_exact_variance_recursion(sigmaH, ...
    sigmaG, sigmaF1, sigmaK, Niter)
% recursion with no smoothing (EM)
sigmaF = zeros(1, Niter);
sigmaF(1) = sigmaF1;
% recursion with smoothing 
sigmaFK = zeros(1, Niter);
sigmaFK(1) = sigmaF1;
% corresponding value for the variance of h_1(y)
% no smoothing
sigmaHatH = zeros(1, Niter);
sigmaHatH(1) = sigmaF(1) + sigmaG;
% with smoothing
sigmaHatHK = zeros(1, Niter);
sigmaHatHK(1) = sigmaFK(1) + sigmaG;
% iterate
for n=2:Niter
    % precompute common quantities
    alpha = sigmaG*(sigmaF(n-1) + sigmaG - sigmaH);
    beta = sigmaH*(sigmaF(n-1) + sigmaG);
    % no smoothing
    sigmaF(n) = (sigmaH * sigmaG * sigmaF(n-1) * (alpha + beta) * ...
        sigmaHatH(n-1))/(alpha * beta * sigmaF(n-1) + sigmaG * sigmaH * ...
        (alpha + beta) * sigmaHatH(n-1));
    % with smoothing
    sigmaFK(n) = (2 * sigmaK * sigmaG^2 + ...
        2 * sigmaK * sigmaFK(n-1) * sigmaG + ...
        sigmaFK(n-1) * sigmaG^2 + ...
        sigmaG * sigmaFK(n-1)^2 + ...
        sigmaH * sigmaFK(n-1)^2)/sigmaHatHK(n-1)^2;
    % update sigmaH
    % no smoothing
    sigmaHatH(n) = sigmaF(n) + sigmaG;
    % with smoothing
    sigmaHatHK(n) = sigmaFK(n) + sigmaG;
end
end