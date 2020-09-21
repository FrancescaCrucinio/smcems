% Smoothing matrix from discretisation of smoothing kernel
% OUTPUTS
% 1 - smoothing matrix
% INPUTS
% 'K' smoothing kernel
% 'refX' discretisation gris
function [S] = smoothingMatrix(K, refX)
    % matrix dimension
    M = length(refX);
    S = zeros(M);
    % discretise K
    for k=1:M
       for j=k:M
          S(k, j) = K(refX(j), refX(k));       
       end    
    end    
    % add symmetric part
    S = S + S' - diag(diag(S));
    % normalise rows
    S = S./sum(S, 1);
    % normalise columns
    S = S./sum(S, 2);
end