% Recontruction of h for PET
% OUTPUTS
% 1 - KL divergence between observed sinogram and reconstructed sinogram
% from PET image reconstruction
% INPUTS
% 'R' observed sinogram
% 'phi' degrees at which projections are taken
% 'xi' offset of projections
% 'sigma' standard deviation for Normal describing alignment
% 'KDEx' pixel locations for PET image
% 'KDEy' pixel color for PET image

function hatH = Hreconstruction_pet(phi, xi, sigma, KDEx, KDEy)
    dx = KDEx(1, 2) - KDEx(2, 2);
    % convolution of approximated of and g
    hatH = zeros(length(xi), length(phi));
    % convolution with approximated f
    % this gives the approximated value
    for i=1:length(xi)
        for j=1:length(phi)
            hatH(i, j) = dx^2*sum(normpdf(KDEx(:, 1) * cos(phi(j)) +...
                KDEx(:, 2) * sin(phi(j)) - xi(i), 0, sigma).* KDEy);
        end
    end
end