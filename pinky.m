%Tristan Ursell
%2D Random Number Generator for a Given Discrete Distribution
%March 2012
%
%[sample]=pinky(Xin,Yin,dist_in,varargin);
%
%'Xin' is a vector specifying the equally spaced values along the x-axis.
%
%'Yin' is a vector specifying the equally spaced values along the y-axis.
%
%'dist_in' (dist_in > 0) is a matrix with dimensions length(Yin) x 
%length(Xin), whose values specify a 2D discrete probability distribution.
%The distribution does not need to be normalized.
%
%'res' (res > 1) is a multiplicative increase in the resolution of
%chosen random number, using cubic spline interpolation of the values in
%'dist_in'.  Using the 'res' option can significantly slow down the code,
%due to the computational costs of interpolation, but allows one to
%generate more continuous values from the distribution.
%
% 'howmany' is the number of samples to draw
%
%[sample] is the output 2 x howmany matrix of random numbers consistent with dist_in.


function [sample]=pinky(Xin,Yin,dist_in,howmany,varargin)

%check input
if length(size(dist_in))>2
    error('The input must be a N x M matrix.')
end

%check sizes
[sy,sx]=size(dist_in);
if or(length(Xin)~=sx,length(Yin)~=sy)
    error('Dimensions of input vectors and input matrix must match.')
end

%check values
if any(dist_in(:)<0)
    error('All input probability values must be positive.')
end

%get res
if nargin==5
    res=varargin{1};
    if res<=1
        error('The resolution factor (res) must be an integer greater than one.')
    end
elseif nargin~=4
    error('Incorrect number of input arguments.')
end
    
%create column distribution and pick random number
col_dist=sum(dist_in,1);

%pick column distribution type
if nargin==4
    %if no res parameter, simply update X/Yin2
    col_dist=col_dist/sum(col_dist);
    Xin2=Xin;
    Yin2=Yin;
else
    %generate new, higher res input vectors
    Xin2=linspace(min(Xin),max(Xin),round(res*length(Xin)));
    Yin2=linspace(min(Yin),max(Yin),round(res*length(Yin)));
    
    %generate interpolated column-sum distribution
    col_dist=interp1(Xin,col_dist,Xin2,'pchip');
    
    %check to make sure interpolated values are positive
    if any(col_dist<0)
        col_dist=abs(col_dist);
        warning('Interpolation generated negative probability values.')
    end
    col_dist=col_dist/sum(col_dist);
end

%generate random value index
ind1=gendist(col_dist,howmany,1);

%save first value
x0 = Xin2(ind1);
y0 = zeros(howmany,1);
for i=1:howmany
    [val_temp,ind_temp]=sort((x0(i)-Xin).^2);
    if val_temp<eps %if we land on an original value
        row_dist=dist_in(:,ind_temp);
    else %if we land inbetween, perform linear interpolation
        low_val=min(ind_temp(1:2));
        high_val=max(ind_temp(1:2));
    
        Xlow=Xin(low_val);
        Xhigh=Xin(high_val);
    
        w1=1-(x0(i)-Xlow)/(Xhigh-Xlow);
        w2=1-(Xhigh-x0(i))/(Xhigh-Xlow);
    
        row_dist=w1*dist_in(:,low_val) + w2*dist_in(:,high_val);
    end
    % pick column distribution type
    %if nargin==4
        row_dist=row_dist/sum(row_dist);
%     else
%         row_dist=interp1(Yin,row_dist,Yin2,'pchip');
%         row_dist=row_dist/sum(row_dist);
%     end

    %generate random value index
    ind2=gendist(row_dist,1,1);
    %save first value
    y0(i)=Yin2(ind2);
end
sample = [x0, y0];
end