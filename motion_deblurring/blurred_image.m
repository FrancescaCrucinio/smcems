% Blur caused by constant speed motion
% OUTPUTS
% 1 - blurred image
% INPUTS
% 'Imagef' original sharp image
% 'b' speed of motion in horizontal direction
% 'sigma' variance of gaussian describing motion in vertical direction

function [Imageh] = blurred_image(Imagef, b, sigma)
% image as double
Imagef = im2double(Imagef(:,:,1));
% dimension of image
pixels = size(Imagef);
% normalize velocity
b = b/300;
% create empty image
Imageh = zeros(pixels);
% set coordinate system over image
% x is in [-1, 1]
evalX = linspace(-1 + 1/pixels(2), 1 - 1/pixels(2), pixels(2));
% y is in [-0.5, 0.5]
evalY = linspace(0.5 - 1/pixels(1), -0.5 + 1/pixels(1), pixels(1));
% build grid with this coordinates
[gridX, gridY] = meshgrid(evalX, evalY);

for i=1:pixels(2)
    u = evalX(i);
    for j=1:pixels(1)
        v = evalY(j);
        % new pixel value after blur
        Imageh(j, i) = sum(Imagef .* normpdf(v, gridY, sigma) .*...
            unifpdf(gridX - u, -b/2, b/2), 'all');
    end
end

% normalize image
Imageh = mat2gray(Imageh);
end