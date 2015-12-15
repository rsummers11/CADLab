function rgb = computeSliceRGB(slice, displayRange, lut)
%COMPUTESLICERGB Convert slice data to renderable slice
%
%   RGB = computeSliceRGB(SLICE)
%
%   RGB = computeSliceRGB(SLICE, DISPLAYRANGE, LUT)
%
%   Example
%     computeSliceRGB
%
%   See also
%
%
% ------
% Author: David Legland
% e-mail: david.legland@grignon.inra.fr
% Created: 2011-11-07,    using Matlab 7.9.0.529 (R2009b)
% Copyright 2011 INRA - Cepia Software Platform.

% RGB slices are returned directly
if ndims(slice) > 2
    rgb = slice;
    return;
end

% forces computation of extreme values
if isempty(displayRange) || islogical(slice)
    displayRange = [0 max(slice(:))];
end

% converts to uint8, rescaling data between 0 and max value
displayRange = double(displayRange);
extent = displayRange(2) - displayRange(1);
slice = uint8((double(slice) - displayRange(1)) * 255 / extent);

% eventually apply a LUT
if ~isempty(lut)
    if ischar(lut)
        lut = feval(lut, 256);
    end
    
    lutMax = max(lut(:));
    dim = size(slice);
    rgb = zeros([dim 3], 'uint8');
    
    % compute each channel
    for c = 1:size(lut, 2)
        res = zeros(size(slice));
        for i = 0:size(lut,1)-1
            res(slice==i) = lut(i+1, c);
        end
        rgb(:,:,c) = uint8(res * 255 / lutMax);
    end
    
else
    % if no LUT, simply use gray equivalent
    rgb = repmat(slice, [1 1 3]);
end

