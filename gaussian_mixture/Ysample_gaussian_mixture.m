% Sample y from the gaussian mixture model
% OUTPUTS
% 1 - sample from h
% INPUTS
% 'M' number of samples 

function [y] = Ysample_gaussian_mixture(M)
% Mixture
yl = rand(M,1) > 1/3;
% mean
ym = 0.3 + 0.2 * yl;
% variance
yv = zeros(size(ym));
for i=1:length(yl)
    if(yl(i))
	  yv(i) = 0.043^2 + 0.045^2;
    else
	  yv(i) = 0.015^2 + 0.045^2;
    end
end
% traslated standard normal
y = ym + sqrt(yv).*randn(M,1);
end

