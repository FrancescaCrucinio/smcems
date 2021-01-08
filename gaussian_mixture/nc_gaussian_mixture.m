% Compute normalising constant for gaussian mixture example
function C = nc_gaussian_mixture(a, mu, sigma)
C = 0.5*(erf((a+0.4-mu)./(sigma*sqrt(2))) - erf((0.4-a-mu)./(sigma*sqrt(2))));
end