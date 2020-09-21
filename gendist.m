function T = gendist(P,N,M,varargin)
%The function gendist(P,N,M) takes in a positive vector P whose values
%form a discrete probability distribution for the indices of P. The
%function outputs an N x M matrix of integers corresponding to the indices
%of P chosen at random from the given underlying distribution.
%
%P will be normalized, if it is not normalized already. Both N and M must
%be greater than or equal to 1.  The optional parameter 'plot' creates a
%plot displaying the input distribution in red and the generated points as
%a blue histogram.

%check for input errors
if and(nargin~=3,nargin~=4)
    error('Error:  Invalid number of input arguments.')
end

if min(P)<0
    error('Error:  All elements of first argument, P, must be positive.')
end

if size(P,1)>1
    P=P';
end

if or(N<1,M<1)
    error('Error:  Output matrix dimensions must be greater than or equal to one.')
end

%normalize P
Pnorm=[0 P]/sum(P);

%create cumlative distribution
Pcum=cumsum(Pnorm);

%create random matrix
N=round(N);
M=round(M);
R=rand(1,N*M);

%calculate T output matrix
V=1:length(P);
[~,inds] = histc(R,Pcum); 
T = V(inds);

%shape into output matrix
T=reshape(T,N,M);

%if desired, output plot
if nargin==4
    if strcmp(varargin{1},'plot')
        Pfreq=N*M*P/sum(P);
        figure;
        hold on
        hist(T(T>0),1:length(P))
        plot(Pfreq,'r-o')
        ylabel('Frequency')
        xlabel('P-vector Index')
        axis tight
        box on
    end
end