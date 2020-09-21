% Smoothing matrix for EMS
% described in 3.2.2 of "A smoothed EM approach to indirect estimation
% problems", Silverman et al. (1990)
% OUTPUTS
% 1 - smoothing matrix
% INPUTS
% 'M' matrix dimension
% 'J' number of points to use for smoothing

function [S] = smoothingSilverman(M, J)

j = 0.5*(J-1);
S = diag(nchoosek(2*j, j)*ones(1,M));
for r=1:j
    S = S + diag(nchoosek(2*j, j+r)*ones(1,M-r), r) + ...
         diag(nchoosek(2*j, j-r)*ones(1,M-r), -r);
end
% normalise
S = 2^(-2*j)*S;
rowsum = sum(S, 1);
unit = find(rowsum ~= 1);
for i=unit
    S(i, i) = S(i, i) + 1 - rowsum(i);
end
end