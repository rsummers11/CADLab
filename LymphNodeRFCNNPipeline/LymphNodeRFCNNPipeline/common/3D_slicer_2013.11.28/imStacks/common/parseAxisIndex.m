function axis = parseAxisIndex(axis)
%PARSEAXISINDEX Parse argument and return axis number between 1 and 3
%
%   AXIS = parseAxisIndex(AXIS)
%   If AXIS is a number, checks it is comprised between 1 and 3, and return
%   it. 
%   If AXIS is a string, try to parse 'x', 'y', or 'z', and return the
%   corresponding index.
%   AXIS index is given as physical index:
%   AXIS=1 corresponds to X axis, and to dimension 2 in matlab array
%   AXIS=2 corresponds to Y axis, and to dimension 1 in matlab array
%   AXIS=3 corresponds to Z axis, and to dimension 3 (or 4 for 3D color
%       image) in matlab array 
%
%   Example
%   % Return array index corresponding to x-direction, i.e. 1
%   ind = parseAxisIndex('x')
%   ind = 
%         1
%
%   See also
%
%
% ------
% Author: David Legland
% e-mail: david.legland@grignon.inra.fr
% Created: 2010-10-20,    using Matlab 7.9.0.529 (R2009b)
% Copyright 2010 INRA - Cepia Software Platform.

if isnumeric(axis)
    % check index bounds
    if axis<1 || axis>3
        error('Axis index should be comprised between 1 and 3');
    end
    
elseif ischar(axis)
    % Try to parse the axis string
    if strcmpi(axis, 'x')
        axis = 1;
    elseif strcmpi(axis, 'y')
        axis = 2;
    elseif strcmpi(axis, 'z');
        axis = 3;
    else
        error('Unknown direction string. Use ''x'', ''y'' or ''z''');
    end
    
else
    % Otherwise throw an error
    error('Unknown type for axis');
end
