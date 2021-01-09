function hCV = CVdeconv(W,errortype,sigU)

%Author: Aurore Delaigle
%compute CV bandwidth for kernel deconvolution estimator as in Stefanski and Carroll (1990)
%Stefanski, L., Carroll, R.J. (1990). Deconvolutingkernel density estimators. Statistics 2, 169–184.
%Delaigle, A. and I. Gijbels (2004). Practical bandwidth selection in deconvolution kernel density estimation, Computational Statistics and Data Analysis, 45, 249-267

%W: vector of contaminated data
%errortype: 'Lap' for Laplace errors and 'norm' for normal errors. For other error distributions, simply redefine phiU below 
%sigU: parameter of Laplace or normal errors used only to define phiU.


% ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
%								WARNINGS:
% ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
%The kernel you use must be the same as the kernel defined in the function fdecUknown.m
%
%If you change the kernel you have to chage muK2, RK and the range of t-values (these must correspond to the support of phiK)
%
%In case of multiple bandwidth solutions, by default this code takes the largest solution: you can change this to your preferred way of breaking ties.
%Often if you plot CV you will see that the first few solutions seem unreasonable (CV fluctuates widely). You can take the first minimum that looks reasonable.
%
% ------------------------------------------------------------------------------------------------------------------------------------------------------------------------



% --------------------------------------------------------
% Preliminary calculations and initialisation of functions
% --------------------------------------------------------


%Default values of phiU(t)=characteristic function of the errors
%If you want to consider another error type, simply replace phiU by the characteristic function of your error type

if strcmp(errortype,'Lap')==1
	phiU=@(t) 1./(1+sigU^2*t.^2);
elseif strcmp(errortype,'norm')==1
	phiU = @(t) exp(-sigU^2*t.^2/2);
end

%phiK: Fourier transform of the kernel K. You can change this is you wish to use another kernel but make sure 
%you change the range of t-values, which should correspond to the support of phiK
phiK = @(t) (1-t.^2).^3;

%second moment \int x^2 K(x) dx  of the kernel K
muK2 = 6;
%integral \int K^2(x) dx
RK=1024/3003/pi;


%Range of t-values (must correspond to the domain of phiK)
deltat = .0002;
t = (-1:deltat:1);
t=reshape(t,length(t),1);


%Sample size
n = length(W);



%Define hgrid, the grid of h values where to search for a solution: you can change the default grid if no solution is found in this grid.
maxh=(max(W)-min(W))/10;

%NR bandwidth of the KDE estimator using the same kernel as above
hnaive=((8*sqrt(pi)*RK/3/muK2^2)^0.2)*sqrt(var(W))*n^(-1/5);

hgrid=hnaive/3:(maxh-hnaive/3)/100:maxh;
lh = length(hgrid);
hgrid=reshape(hgrid,1,lh);




%Quantities that will be needed several times in the computations below
toverh=t*(1./hgrid);
phiU2=phiU(toverh).^2;
phiKt=phiK(t);



% --------------------------------------------------------
%Compute CV criterion
% --------------------------------------------------------


CVcrit=0*hgrid;
longh=length(hgrid);

OO=outerop(t,W,'*');


for j=1:longh
	h=hgrid(j);

	%Estimate the square of the norm of the empirical characteristic function of W
	rehatphiW=sum(cos(OO/h),2)/n;
	imhatphiW=sum(sin(OO/h),2)/n;
	normhatphiW2=rehatphiW.^2+imhatphiW.^2;
	
	%Compute CV
	CVcrit(j)=sum(phiKt./phiU2(:,j).*(normhatphiW2.*((n-1)*phiKt-2*n)+2));
	CVcrit(j)=CVcrit(j)/h;
end


indh=find((CVcrit(2:(longh-1))<CVcrit(1:(longh-2)))&(CVcrit(2:(longh-1))<CVcrit(3:(longh))));


if length(indh)<1
	hCV=find(CVcrit==min(CVcrit))
else
	%In case of multiple solutions, take the largest bandwidth: you can change this to your preferred way of breaking ties
	hCV = max(hgrid(indh));
end
