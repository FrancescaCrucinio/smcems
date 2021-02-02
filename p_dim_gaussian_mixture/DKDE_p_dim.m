function fXdecUKfinal = DKDE_p_dim(Nbins, W, sigmaG)

% dimension
p = size(W, 2);
n = size(W, 1);

% grid
xx = linspace(0, 1, Nbins);
dx = xx(2) - xx(1);
% characteristic function of g
phiG = @(t) exp(-sigmaG^2.*t.^2/2);

% phiK: Fourier transform of the kernel K
phiK = @(t) (1-t.^2).^3;


% range of t-values in [-1, 1]
deltat = .0002;
t = (-1:deltat:1)';
t = reshape(t,length(t),1);

fXdecUK = zeros(Nbins, p);
for i=1:p
    h = PI_deconvUknownth4(W(:, i), "norm", sigmaG^2, sigmaG);
    OO = outerop(t/h, W(:, i), '*');
    phiGth = phiG(t/h);
    % empirical characteristic function of W
    rehatphiX = sum(cos(OO),2)./phiGth/n;
    imhatphiX = sum(sin(OO),2)./phiGth/n;
    
    xt = outerop(t/h, xx,'*');

    % DKDE estimator
    fXdecUKtemp = cos(xt).*repmat(rehatphiX,1,Nbins)+sin(xt).*repmat(imhatphiX,1,Nbins);
    fXdecUK(:, i) = sum(fXdecUKtemp.*repmat(phiK(t),1,Nbins), 1)/h;
end

fXdecUKfinal = outerop(fXdecUK(:, 1), fXdecUK(:, 2), '*');
for i=3:p
    fXdecUKfinal = outerop(fXdecUKfinal(:), fXdecUK(:, i));
end
fXdecUKfinal = fXdecUKfinal(:)*(deltat/(2*pi))^p;
% make sure all values are positive
fXdecUKfinal(fXdecUKfinal<0)=0*fXdecUKfinal(fXdecUKfinal<0);
% scale to 1
fXdecUKfinal=fXdecUKfinal/sum(fXdecUKfinal)/dx^p;
end