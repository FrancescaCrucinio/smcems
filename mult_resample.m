% Multinomial resampling
% OUTPUTS
% 1- indices of resampled particles
% INPUTS
% 'W' vector of weights
% 'N' number of particles to sample (in most cases N = length(W))

function [indices] = mult_resample(W, N)
% make sure W is a vector
W = W(:);
% vector to store number of offsprings
indices = zeros(N, 1);

% start inverse transfor method
s = W(1);
u = sort(rand(N, 1));
j = 1;
for i=1:N
while(s < u(i))
  j = j+1;
  s = s + W(j);
end
indices(i) = j;
end
end