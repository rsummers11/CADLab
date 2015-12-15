function res = stackSlice(img, dir, slice)
%STACKSLICE Extract a planar slice from a 3D image
%
%   SLICE = stackSlice(IMG, DIR, INDEX)
%   IMG is either a 3D or a 4D (3D+color) image.
%   DIR is 1, 2 or 3, or 'x', 'y', 'z'
%   INDEX is the slice index, between 1 and the number of voxels in the DIR
%   direction (using physical indexing).
%
%   If the input image has size NY-by-NX-by-NZ (gray scale image), or
%   NY-by-NX-by-3-by-NZ (color image), then the resulting slice has size: 
%   - NZ-by-NY(-by-3) if DIR is 1 or DIR is 'x'
%   - NX-by-NZ(-by-3) if DIR is 2 or DIR is 'y'
%   - NY-by-NX(-by-3) if DIR is 3 or DIR is 'z'
%   
%   The functions returns a planar image with the same type as the
%   original. 
%
%
%   Example
%   % Display 3 slices of a MRI head
%     img = analyze75read(analyze75info('brainMRI.hdr'));
%     figure(1); clf; hold on;
%     for i=1:3
%         subplot(3, 1, i);
%         imshow(stackSlice(img, 'x', 20+20*i)');
%         set(gca, 'ydir', 'normal')
%     end
%
%   See also
%
%
%   ------
%   author: David Legland, david.legland(at)grignon.inra.fr
%   INRA - Cepia Software Platform
%   Created: 2007-08-14,    using Matlab 7.4.0.287 (R2007a)
%   http://www.pfl-cepia.inra.fr/index.php?page=slicer
%   Licensed under the terms of the new BSD license, see file license.txt

%   HISTORY
%   2010-12-02 code cleanup, fix dimension order of output slices
%   2011-04-26 use xyz convention for slice index


% image size
dim = size(img);

% convert to an index between 1 and 3, in xyz order
dir = parseAxisIndex(dir);

% check dimension of input array
nd = length(dim);
if nd < 3 || nd > 4
    error('Input array should have dimension 3 or 4');
end

% Extract either a grayscale or a color slice
if nd == 3
    % gray-scale image
    switch dir
        case 1
            % X-slice, result is a YZ slice with size NZ-by-NY
            res = squeeze(img(:, slice, :))';
        case 2
            % Y-slice, result is a ZX slice with size NX-by-NZ
            res = squeeze(img(slice, :, :));
        case 3
            % Z-slice, result is a XY slice with size NY-by-NX
            res = img(:, :, slice);
        otherwise
            error('Direction should be comprised between 1 and 3');
    end

else 
    % color images: keep channel dim as third dimension, and put slice
    % dimension as last dimension.
    switch dir
        case 1
            % X-slice, result is a YZ color slice with size NZ-by-NY-by-3
            res = squeeze(permute(img(:, slice, :, :), [4 1 3 2]));
        case 2
            % Y-slice, result is a ZX color slice with size NX-by-NZ-by-3
            res = squeeze(permute(img(slice, :, :,:), [2 4 3 1]));
        case 3
            % Z-slice, result is a XY color slice with size NY-by-NX-by-3
            res = img(:, :, :, slice);
        otherwise
            error('Direction should be comprised between 1 and 3');
    end
end
