% Optimal bandwidth for Gaussian kernel with effective sample size (ESS)
% OUTPUTS
% 1 - bandwidth
% INPUTS
% 'x' particle locations
% 'W' particle weights

function[bw] = optimal_bandwidthESS(x, W)
% ESS
ESS=sum(W.^2);

% standard deviation (weighted)
s = std(x, W);

% bandwidth
bw = 1.06*s*ESS^(1/5);
end