% Diagnostics for approximations of f
% OUTPUTS
% 1 - mean
% 2 - variance
% 3 - mean squared error
% 4 - mean integrated squared error
% INPUTS
% 'f' true f (function handle)
% 'x' sample points in the domain of f
% 'y' estimated value of f at sample points

function[m, v, difference, mise] = diagnosticsF(f, x, y)

% mean
m = sum(y.*x)/sum(y);
% variance
v = sum(y.*(x.^2))/sum(y) - m^2;
% exact f
trueF = f(x);
% compute MISE for f
difference = (trueF - y).^2;
mise = sum(difference)/length(difference);
end