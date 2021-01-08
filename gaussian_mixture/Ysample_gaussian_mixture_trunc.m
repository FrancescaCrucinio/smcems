% Sample y from the gaussian mixture model truncated t [0.5-a, 0.5+a]
% OUTPUTS
% 1 - sample from h
% INPUTS
% 'M' number of samples 
% 'a' width of interval

function [y] = Ysample_gaussian_mixture_trunc(M, a)
% Mixture
s = 1/3 * (normcdf((a+0.4-0.3)/sqrt(0.015^2 + 0.045^2)) - normcdf((0.4-a-0.3)/sqrt(0.015^2 + 0.045^2))) + ...
    2/3 * (normcdf((a+0.4-0.5)/sqrt(0.043^2 + 0.045^2)) - normcdf((0.4-a-0.5)/sqrt(0.043^2 + 0.045^2)));
p = 1/3 * (normcdf((a+0.4-0.3)/sqrt(0.015^2 + 0.045^2)) - normcdf((0.4-a-0.3)/sqrt(0.015^2 + 0.045^2)))/s;
yl = rand(M,1) > p;
% sample
y = zeros(M, 1);
for i=1:length(y)
    if(yl(i))
	  y(i) = 0.5 + sqrt(0.043^2 + 0.045^2) * norminv(normcdf(0.4 - a, 0.5, sqrt(0.043^2 + 0.045^2)) + ...
             rand(1)*(normcdf(0.4 + a, 0.5, sqrt(0.043^2 + 0.045^2)) - normcdf(0.4 - a, 0.5, sqrt(0.043^2 + 0.045^2))));
    else
	  y(i) = 0.3 + sqrt(0.015^2 + 0.045^2) * norminv(normcdf(0.4 - a, 0.3, sqrt(0.015^2 + 0.045^2)) + ...
             rand(1)*(normcdf(0.4 + a, 0.3, sqrt(0.015^2 + 0.045^2)) - normcdf(0.4 - a, 0.3, sqrt(0.015^2 + 0.045^2))));
    end
end
end

