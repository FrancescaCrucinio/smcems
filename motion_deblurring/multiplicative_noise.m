% Multiplicative noise for a perfectly observed images
% Section 6.2 of "Experiments with maximum likelihood
% method for image motion deblurring"
% Lee and Vardi 1994
% OUTPUTS
% 1 - noisy image
% INPUTS
% 'image' noise free image
% 'alpha' amount of noise
% 'beta' probability of adding noise

function[image] = multiplicative_noise(image, alpha, beta)
% get dimension of image
pixels = size(image);
% noise
unif = rand(pixels);
mult_matrix = (unif <= beta/2)*(1-alpha) + ...
    (beta/2 < unif <= 1-beta/2) + (1-beta/2 < unif <= 1)*(1+alpha);
image = image.*mult_matrix;
end