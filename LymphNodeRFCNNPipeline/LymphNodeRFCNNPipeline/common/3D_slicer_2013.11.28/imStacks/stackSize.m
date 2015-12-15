function dim = stackSize(img, varargin)
%STACKSIZE  Compute the size of a 3D stack in [x y z] form
%
%   SIZ = stackSize(IMG)
%   Return the size of the stack, in [x, y, z] order, for a grayscale or a
%   color image.
%   In the case of a 3D color image, return only the 3 physical dimension
%   of the image.
%
%   DIM = stackSize(IMG, DIR)
%   Return the size of the stack in the specified direction.
%   DIR=1 corresponds to X axis (second direction of matlab array)
%   DIR=2 corresponds to Y axis (first direction of matlab array)
%   DIR=3 corresponds to Z axis (third or fourth direction of matlab array)
%
%   DIM = stackSize(IMG, RESOL)
%   Return the size of the stack in physical units. RESOL is a 1-by-3 row
%   vector containing resolution in x, y and z directions respectively.
%
%
%   Example
%   % compute size of a portion of MRI image
%     img = analyze75read(analyze75info('brainMRI.hdr'));
%     img2 = img(1:50, :, :);   % keep 50 rows, in the y-direction
%     stackSize(img2)
%     ans =
%        128    50    27
%   % the number of voxels in y-direction is given as second parameter.
%
%   % return only the number of rows, i.e. the number of voxels in the
%   % Y-direction
%     ny = stackSize(img2, 2)
%     ny =
%         50
%
%   See also
%   stackExtent, size, ndims
%
%
% ------
% Author: David Legland
% e-mail: david.legland@grignon.inra.fr
% Created: 2010-07-01,    using Matlab 7.9.0.529 (R2009b)
% Copyright 2010 INRA - Cepia Software Platform.

% size of matlab array
dim = size(img);

% keep physical dimension, and swap x-y dimensions
if length(dim)>3
    dim = dim([2 1 4]);
else
    dim = dim([2 1 3]);
end

if ~isempty(varargin)
    var = varargin{1};
    
    if isscalar(var)
        % in case a single direction is asked, select it
        dim = dim(varargin{1});
        
    elseif length(var) == 3
        % if resolution is specified, compute size in physical units
        dim = dim .* var;
    end
end
