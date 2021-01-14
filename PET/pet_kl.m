% Kullback-Leilbler divergence for PET reconstruction
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

function div = pet_kl(R, phi, xi, sigma, KDEx, KDEy)
    delta1 = phi(2) - phi(1);
    delta2 = xi(2) - xi(1);
    % logarithm of true h
    trueHlog = log(R);
    trueHlog(~isfinite(trueHlog)) = 0;
    % approximated value
    hatH = zeros(length(xi), length(phi));
    % convolution with approximated f
    % this gives the approximated value
    for i=1:length(xi)
        for j=1:length(phi)
            hatH(i, j) = sum(normpdf(KDEx(:, 1) * cos(phi(j)) +...
                KDEx(:, 2) * sin(phi(j)) - xi(i), 0, sigma).* KDEy);
        end
    end
    hatH = hatH/max(hatH, [], 'all');
    % compute log of hatH
    hatHlog = log(hatH);
    hatHlog(~isfinite(hatHlog)) = 0;
    div = delta1*delta2*sum(R.*(trueHlog - hatHlog), 'all');
end