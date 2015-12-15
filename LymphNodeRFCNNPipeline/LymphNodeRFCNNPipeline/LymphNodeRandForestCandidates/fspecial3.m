function h = fspecial3(type,siz)
%FSPECIAL3 Create predefined 3-D filters.
%   H = FSPECIAL3(TYPE,SIZE) creates a 3-dimensional filter H of the
%   specified type and size. Possible values for TYPE are:
%
%     'average'    averaging filter
%     'ellipsoid'  ellipsoidal averaging filter
%     'gaussian'   Gaussian lowpass filter
%     'laplacian'  Laplacian operator
%     'log'        Laplacian of Gaussian filter
%
%   The default SIZE is [5 5 5]. If SIZE is a scalar then H is a 3D cubic
%   filter of dimension SIZE^3.
%
%   H = FSPECIAL3('average',SIZE) returns an averaging filter H of size
%   SIZE. SIZE can be a 3-element vector specifying the dimensions in
%   H or a scalar, in which case H is a cubic array.
%
%   H = FSPECIAL3('ellipsoid',SIZE) returns an ellipsoidal averaging filter.
%
%   H = FSPECIAL3('gaussian',SIZE) returns a centered Gaussian lowpass
%   filter of size SIZE with standard deviations defined as
%   SIZE/(4*sqrt(2*log(2))) so that FWHM equals half filter size
%   (http://en.wikipedia.org/wiki/FWHM). Such a FWHM-dependent standard
%   deviation yields a congruous Gaussian shape (what should be expected
%   for a Gaussian filter!).
%
%   H = FSPECIAL3('laplacian') returns a 3-by-3-by-3 filter approximating
%   the shape of the three-dimensional Laplacian operator. REMARK: the
%   shape of the Laplacian cannot be adjusted. An infinite number of 3D
%   Laplacian could be defined. If you know any simple formulation allowing
%   one to control the shape, please contact me.
%
%   H = FSPECIAL3('log',SIZE) returns a rotationally symmetric Laplacian of
%   Gaussian filter of size SIZE with standard deviation defined as
%   SIZE/(4*sqrt(2*log(2))).
%
%   Class Support
%   -------------
%   H is of class double.
%
%   Example
%   -------
%      I = single(rand(80,40,20));
%      h = fspecial3('gaussian',[9 5 3]); 
%      Inew = imfilter(I,h,'replicate');
%       
%   See also IMFILTER, FSPECIAL.
%
%   -- Damien Garcia -- 2007/08

error(nargchk(1,2,nargin))
type = lower(type);

if nargin==1
        siz = 5;
end

if numel(siz)==1
    siz = round(repmat(siz,1,3));
elseif numel(siz)~=3
    error('Number of elements in SIZ must be 1 or 3')
else
    siz = round(siz(:)');
end

switch type
    
    case 'average'
        h = ones(siz)/prod(siz);
        
    case 'gaussian'
        sig = siz/(4*sqrt(2*log(2)));
        siz   = (siz-1)/2;
        [x,y,z] = ndgrid(-siz(1):siz(1),-siz(2):siz(2),-siz(3):siz(3));
        h = exp(-(x.*x/2/sig(1)^2 + y.*y/2/sig(2)^2 + z.*z/2/sig(3)^2));
        h = h/sum(h(:));
        
    case 'ellipsoid'
        R = siz/2;
        R(R==0) = 1;
        h = ones(siz);
        siz = (siz-1)/2;
        [x,y,z] = ndgrid(-siz(1):siz(1),-siz(2):siz(2),-siz(3):siz(3));
        I = (x.*x/R(1)^2+y.*y/R(2)^2+z.*z/R(3)^2)>1;
        h(I) = 0;
        h = h/sum(h(:));
        
    case 'laplacian'
        h = zeros(3,3,3);
        h(:,:,1) = [0 3 0;3 10 3;0 3 0];
        h(:,:,3) = h(:,:,1);
        h(:,:,2) = [3 10 3;10 -96 10;3 10 3];
        
    case 'log'
        sig = siz/(4*sqrt(2*log(2)));
        siz   = (siz-1)/2;
        [x,y,z] = ndgrid(-siz(1):siz(1),-siz(2):siz(2),-siz(3):siz(3));
        h = exp(-(x.*x/2/sig(1)^2 + y.*y/2/sig(2)^2 + z.*z/2/sig(3)^2));
        h = h/sum(h(:));
        arg = (x.*x/sig(1)^4 + y.*y/sig(2)^4 + z.*z/sig(3)^4 - ...
            (1/sig(1)^2 + 1/sig(2)^2 + 1/sig(3)^2));
        h = arg.*h;
        h = h-sum(h(:))/prod(2*siz+1);
        
    otherwise
        error('Unknown filter type.')
        
end
        
        
        