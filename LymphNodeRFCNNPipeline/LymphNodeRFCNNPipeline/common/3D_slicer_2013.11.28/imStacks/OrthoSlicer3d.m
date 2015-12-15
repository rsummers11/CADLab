classdef OrthoSlicer3d < handle
%ORTHOSLICER3D Display 3D interactive orthoslicer
%
%   OrthoSlicer3d(IMG)
%   Displays an interactive 3D orthoslicer of a 3D image. 
%
%   OrthoSlicer3d(..., NAME, VALUE)
%   Specifies one or more display options as name-value parameter pairs.
%   Available parameter names are:
%   'position'      the position of the slices intersection point, given in
%           pixels as a 1-by-3 row vector.
%   'spacing'       specifies the size of voxel elements. VALUE is a 1-by-3
%           row vector containing spacing in x, y and z direction.
%   'origin'        specifies coordinate of first voxel in user space
%   'displayRange'  the values of min and max gray values to display. The
%           default behaviour is to use [0 255] for uint8 images, or to
%           compute bounds such as 95% of the voxels are converted to visible
%           gray levels for other image types.
%   'colormap'      the name of the colormap used for displaying grayscale
%           values. Available values are 'jet', 'hsv', 'gray'...
%
%   Example
%   % Explore human brain MRI
%     metadata = analyze75info('brainMRI.hdr');
%     I = analyze75read(metadata);
%     OrthoSlicer3d(I);
%  
%   % add some setup
%     figure;
%     OrthoSlicer3d(I, 'Position', [60 40 5], 'spacing', [1 1 2.5], ...
%           'displayRange', [0 90]);
%
%   See also
%   Slicer
%
% ------
% Author: David Legland
% e-mail: david.legland@grignon.inra.fr
% Created: 2012-03-20,    using Matlab 7.9.0.529 (R2009b)
% Copyright 2012 INRA - Cepia Software Platform.

properties
    % reference image
    imageData;
    
    % type of image. Can be one of {'grayscale', 'color', 'vector'}
    imageType;
    
    % physical size of the reference image (1-by-3 row vector, in xyz order) 
    imageSize;
    
    % extra info for image, such as the result of imfinfo
    imageInfo;

    % the position of the intersection point of the three slices, as index
    % in xyz ordering
    position;
    
    % extra info for image, such as the result of imfinfo
    imageName;

    % used to adjust constrast of the slice
    displayRange;
    
    % Look-up table for display of uint8 images (default is empty)
    lut             = '';
    
    % calibraton information for image
    voxelOrigin     = [0 0 0];
    voxelSize       = [1 1 1];
    voxelSizeUnit   = '';
    
    % shortcut for avoiding many tests. Should be set to true when either
    % voxelOrigin, voxelsize or voxelSizeUnit is different from its default
    % value.
    calibrated = false;
    
    % list of handles to the widgets
    handles;
    
    % for managing slice dragging
    startRay;
    startIndex;
   
    draggedSlice;
    sliceIndex;
end


%% Constructors
methods
    function this = OrthoSlicer3d(img, varargin)
        
        % call parent constructor
        this = this@handle();
        
        this.handles = struct();

        % Setup image data and type
        this.imageData = img;
        this.imageType = 'grayscale';
        
        % compute size, and detect RGB
        dim = size(img);
        nd = length(dim);
        
        % check image type
        if nd > 3
            % Image is 3D color or vector
            
            % extreme values
            valMin = min(img(:));
            valMax = max(img(:));
            
            % determines image nature
            if dim(3) ~= 3 || valMin < 0 || (isfloat(img) && valMax > 1)
                this.imageType = 'vector';
            else
                this.imageType = 'color';
            end
            
            % keep only spatial dimensions
            dim = dim([1 2 4]);
        end
        
        % convert to use dim(1)=x, dim(2)=y, dim(3)=z
        dim = dim([2 1 3]);
        this.imageSize = dim;
        
        % eventually compute grayscale extent
        if ~strcmp(this.imageType, 'color')
            [mini maxi] = computeGrayScaleExtent(this);
            this.displayRange  = [mini maxi];
        end
        
        % default slice index is in the middle of the stack
        pos                 = ceil(dim / 2);
        this.position       = pos;

        
        parsesInputArguments();
        

        % handle to current figure;
        hFig = gcf();
        this.handles.figure = hFig;
        
        % figure settings
        hold on;

        % display three orthogonal slices
        pos = this.position;
        this.handles.sliceYZ = createSlice3d(this, 1, pos(1));
        this.handles.sliceXZ = createSlice3d(this, 2, pos(2));
        this.handles.sliceXY = createSlice3d(this, 3, pos(3));

        % set up mouse listener
        set(this.handles.sliceYZ, 'ButtonDownFcn', @this.startDragging);
        set(this.handles.sliceXZ, 'ButtonDownFcn', @this.startDragging);
        set(this.handles.sliceXY, 'ButtonDownFcn', @this.startDragging);
        
        
        % extract spatial calibration
        spacing = this.voxelSize;
        origin = this.voxelOrigin;
        box = stackExtent(this.imageData, spacing, origin);
        
        % position of orthoslice center
        xPos = (pos(1) - 1) * spacing(1) + origin(1);
        yPos = (pos(2) - 1) * spacing(2) + origin(2);
        zPos = (pos(3) - 1) * spacing(3) + origin(3);
        
        % show orthogonal lines
        this.handles.lineX = line(...
            box(1:2), [yPos yPos], [zPos zPos], ...
            'color', 'r');
        this.handles.lineY = line(...
            [xPos xPos], box(3:4), [zPos zPos], ...
            'color', 'g');
        this.handles.lineZ = line(...
            [xPos xPos], [yPos yPos], box(5:6), ...
            'color', 'b');

        % show frames around each slice
        xmin = box(1);
        xmax = box(2);
        ymin = box(3);
        ymax = box(4);
        zmin = box(5);
        zmax = box(6);
        this.handles.frameXY = line(...
            [xmin xmin xmax xmax xmin], ...
            [ymin ymax ymax ymin ymin], ...
            [zPos zPos zPos zPos zPos], ...
            'color', 'k');
        this.handles.frameXZ = line(...
            [xmin xmin xmax xmax xmin], ...
            [yPos yPos yPos yPos yPos], ...
            [zmin zmax zmax zmin zmin], ...
            'color', 'k');
        this.handles.frameYZ = line(...
            [xPos xPos xPos xPos xPos], ...
            [ymin ymin ymax ymax ymin], ...
            [zmin zmax zmax zmin zmin], ...
            'color', 'k');
        
        % setup display
        %view([-20 30]);
        view([-30 30]);
        axis equal;
 
        function parsesInputArguments()
            % iterate over couples of input arguments to setup display
            while length(varargin) > 1
                param = varargin{1};
                switch lower(param)
                    case 'slice'
                        % setup initial slice
                        pos = varargin{2};
                        this.sliceIndex = pos(1);
                        
                    case 'position'
                        % setup position of the intersection point (pixels)
                        this.position = varargin{2};
                        
                    % setup of image calibration    
                    case 'spacing'
                        this.voxelSize = varargin{2};
                        if ~this.calibrated
                            this.voxelOrigin = [0 0 0];
                        end
                        this.calibrated = true;
                        
                    case 'origin'
                        this.voxelOrigin = varargin{2};
                        this.calibrated = true;
                        
                    case 'unitname'
                        this.voxelSizeUnit = varargin{2};
                        this.calibrated = true;
                        
                    % setup display calibration
                    case 'displayrange'
                        this.displayRange = varargin{2};
                    case {'colormap', 'lut'}
                        this.lut = varargin{2};
                        
                    otherwise
                        error(['Unknown parameter name: ' param]);
                end
                varargin(1:2) = [];
            end
        end % function parseInputArguments

    end
end


%% member functions
methods

    function hs = createSlice3d(this, dim, index, varargin)
        %CREATESLICE3D Show a moving 3D slice of an image


        % extract the slice
        slice = stackSlice(this.imageData, dim, this.position(dim));
        
        % compute equivalent RGB image
        slice = computeSliceRGB(slice, this.displayRange, this.lut);

        % size of the image
        siz = this.imageSize;

        % extract slice coordinates
        switch dim
            case 1
                % X Slice

                % compute coords of u and v
                vy = ((0:siz(2)) - .5);
                vz = ((0:siz(3)) - .5);
                [ydata zdata] = meshgrid(vy, vz);

                % coord of slice supporting plane
                lx = 0:siz(1);
                xdata = ones(size(ydata)) * lx(index);

            case 2
                % Y Slice

                % compute coords of u and v
                vx = ((0:siz(1)) - .5);
                vz = ((0:siz(3)) - .5);
                [zdata xdata] = meshgrid(vz, vx);

                % coord of slice supporting plane
                ly = 0:siz(2);
                ydata = ones(size(xdata)) * ly(index);

            case 3
                % Z Slice

                % compute coords of u and v
                vx = ((0:siz(1)) - .5);
                vy = ((0:siz(2)) - .5);
                [xdata ydata] = meshgrid(vx, vy);

                % coord of slice supporting plane
                lz = 0:siz(3);
                zdata = ones(size(xdata)) * lz(index);

            otherwise
                error('Unknown stack direction');
        end
        
        % initialize transform matrix from index coords to physical coords
        dcm = diag([this.voxelSize 1]);
        %dcm(4, 1:3) = this.voxelOrigin;
        
        % transform coordinates from image reference to spatial reference
        hdata = ones(1, numel(xdata));
        trans = dcm(1:3, :) * [xdata(:)'; ydata(:)'; zdata(:)'; hdata];
        xdata(:) = trans(1,:) + this.voxelOrigin(1);
        ydata(:) = trans(2,:) + this.voxelOrigin(2);
        zdata(:) = trans(3,:) + this.voxelOrigin(3);


        % global parameters for surface display
        params = [{'facecolor', 'texturemap', 'edgecolor', 'none'}, varargin];

        % display voxel values in appropriate reference space
        hs = surface(xdata, ydata, zdata, slice, params{:});

        % setup user data of the slice
        data.dim = dim;
        data.index = index;
        set(hs, 'UserData', data);

    end

    function updateLinesPosition(this)
        
        dim = this.imageSize;
        spacing = this.voxelSize;
        origin = this.voxelOrigin;
        
        xdata = (0:dim(1)) * spacing(1) + origin(1);
        ydata = (0:dim(2)) * spacing(2) + origin(2);
        zdata = (0:dim(3)) * spacing(3) + origin(3);
        
        pos = this.position;
        xPos = xdata(pos(1));
        yPos = ydata(pos(2));
        zPos = zdata(pos(3));
        
        % show orthogonal lines
        set(this.handles.lineX, 'ydata', [yPos yPos]);
        set(this.handles.lineX, 'zdata', [zPos zPos]);
        set(this.handles.lineY, 'xdata', [xPos xPos]);
        set(this.handles.lineY, 'zdata', [zPos zPos]);
        set(this.handles.lineZ, 'xdata', [xPos xPos]);
        set(this.handles.lineZ, 'ydata', [yPos yPos]);
    end

    function updateFramesPosition(this)
        dim = this.imageSize;
        spacing = this.voxelSize;
        origin = this.voxelOrigin;
        
        xdata = (0:dim(1)) * spacing(1) + origin(1);
        ydata = (0:dim(2)) * spacing(2) + origin(2);
        zdata = (0:dim(3)) * spacing(3) + origin(3);
        
        pos = this.position;
        xPos = xdata(pos(1));
        yPos = ydata(pos(2));
        zPos = zdata(pos(3));

        set(this.handles.frameXY, 'zdata', repmat(zPos, 1, 5));
        set(this.handles.frameXZ, 'ydata', repmat(yPos, 1, 5));
        set(this.handles.frameYZ, 'xdata', repmat(xPos, 1, 5));

    end
    
    function startDragging(this, src, event) %#ok<INUSD>
        %STARTDRAGGING  One-line description here, please.
        %
    
        
        % store data for creating ray
        this.startRay   = get(gca, 'CurrentPoint');
        
        % find current index
        data = get(src, 'UserData');
        dim = data.dim;
        this.startIndex = this.position(dim);
                
        this.draggedSlice = src;

        % set up listeners for figure object
        hFig = gcbf();
        set(hFig, 'WindowButtonMotionFcn', @this.dragSlice);
        set(hFig, 'WindowButtonUpFcn', @this.stopDragging);
    end

    function stopDragging(this, src, event) %#ok<INUSD>
        %STOPDRAGGING  One-line description here, please.
        %

        % remove figure listeners
        hFig = gcbf();
        set(hFig, 'WindowButtonUpFcn', '');
        set(hFig, 'WindowButtonMotionFcn', '');

        % reset slice data
        this.startRay = [];
        this.draggedSlice = [];
        
        % update display
        drawnow;
    end


    function dragSlice(this, src, event) %#ok<INUSD>
        %DRAGSLICE  One-line description here, please.
        %
        
        % Extract slice data
        hs      = this.draggedSlice;
        data    = get(hs, 'UserData');

        % basic checkup
        if isempty(this.startRay)
            return;
        end

        % dimension in xyz
        dim = data.dim;

        
        % initialize transform matrix from index coords to physical coords
        dcm = diag([this.voxelSize 1]);
        
        % compute the ray corresponding to current slice normal
        center = (this.position .* this.voxelSize) + this.voxelOrigin;
        sliceNormal = [center; center+dcm(1:3, dim)'];

        % Project start ray on slice-axis
        alphastart = posProjRayOnRay(this, this.startRay, sliceNormal);

        % Project current ray on slice-axis
        currentRay = get(gca, 'CurrentPoint');
        alphanow = posProjRayOnRay(this, currentRay, sliceNormal);

        % compute difference in positions
        slicediff = alphanow - alphastart;

        index = this.startIndex + round(slicediff);
        index = min(max(1, index), stackSize(this.imageData, data.dim));
        this.sliceIndex = index;
        
        this.position(data.dim) = index;


        % extract slice corresponding to current index
        slice = stackSlice(this.imageData, data.dim, this.sliceIndex);

        % convert to renderable RGB
        slice = computeSliceRGB(slice, this.displayRange, this.lut);

        % setup display data
        set(hs, 'CData', slice);


        % the mesh used to render image has one element more, to enclose all pixels
        meshSize = [size(slice, 1) size(slice, 2)] + 1;

        spacing = this.voxelSize;
        origin = this.voxelOrigin;

        switch data.dim
            case 1
                xpos = (this.sliceIndex - 1) * spacing(1) + origin(1);
                xdata = ones(meshSize) * xpos;
                set(hs, 'xdata', xdata);
                
            case 2
                ypos = (this.sliceIndex - 1) * spacing(2) + origin(2);
                ydata = ones(meshSize) * ypos;
                set(hs, 'ydata', ydata);
                
            case 3
                zpos = (this.sliceIndex - 1) * spacing(3) + origin(3);
                zdata = ones(meshSize) * zpos;
                set(hs, 'zdata', zdata);
                
            otherwise
                error('Unknown stack direction');
        end

        % update display
        updateLinesPosition(this);
        updateFramesPosition(this);
        drawnow;
    end


    function alphabeta = computeAlphaBeta(this, a, b, s) %#ok<MANU>
        dab = b - a;
        alphabeta = pinv([s'*s -s'*dab ; dab'*s -dab'*dab]) * [s'*a dab'*a]';
    end

    function pos = posProjRayOnRay(this, ray1, ray2) %#ok<MANU>
        % ray1 and ray2 given as 2-by-3 arrays
        
        u = ray1(2,:) - ray1(1,:);
        v = ray2(2,:) - ray2(1,:);
        w = ray1(1,:) - ray2(1,:);
        
        a = dot(u, u, 2);
        b = dot(u, v, 2);
        c = dot(v, v, 2);
        d = dot(u, w, 2);
        e = dot(v, w, 2);
        
        pos = (a*e - b*d) / (a*c - b^2);
    end
end

methods
    %% Some methods for image manipulation (should be factorized)
    function [mini maxi] = computeGrayScaleExtent(this)
        % compute grayscale extent of this inner image
        
        if isempty(this.imageData)
            mini = 0; 
            maxi = 1;
            return;
        end
        
        % check image data type
        if isa(this.imageData, 'uint8')
            % use min-max values depending on image type
            mini = 0;
            maxi = 255;
            
        elseif islogical(this.imageData)
            % for binary images, the grayscale extent is defined by the type
            mini = 0;
            maxi = 1;
            
        elseif strcmp(this.imageType, 'vector')
            % case of vector image: compute max of norm
            
            dim = size(this.imageData);
            
            norm = zeros(dim([1 2 4]));
            
            for i = 1:dim(3);
                norm = norm + squeeze(this.imageData(:,:,i,:)) .^ 2;
            end
            
            mini = 0;
            maxi = sqrt(max(norm(:)));
            
        else
            % for float images, display 99 percents of dynamic
            [mini maxi] = computeGrayscaleAdjustement(this, .01);            
        end
    end
    
    function [mini maxi] = computeGrayscaleAdjustement(this, alpha)
        % compute grayscale range that maximize vizualisation
        
        if isempty(this.imageData)
            mini = 0; 
            maxi = 1;
            return;
        end
        
        % use default value for alpha if not specified
        if nargin == 1
            alpha = .01;
        end
        
        % sort values that are valid (avoid NaN's and Inf's)
        values = sort(this.imageData(isfinite(this.imageData)));
        n = length(values);

        % compute values that enclose (1-alpha) percents of all values
        mini = values( floor((n-1) * alpha/2) + 1);
        maxi = values( floor((n-1) * (1-alpha/2)) + 1);

        % small control to avoid mini==maxi
        minDiff = 1e-12;
        if abs(maxi - mini) < minDiff
            % use extreme values in image
            mini = values(1);
            maxi = values(end);
            
            % if not enough, set range to [0 1].
            if abs(maxi - mini) < minDiff
                mini = 0;
                maxi = 1;
            end
        end
    end
    
end % end of image methods

%% Methods for text display
methods
    function display(this)
        % display a resume of the slicer structure
       
        % determines whether empty lines should be printed or not
        if strcmp(get(0, 'FormatSpacing'), 'loose')
            emptyLine = '\n';
        else
            emptyLine = '';
        end
        
        % eventually add space
        fprintf(emptyLine);
        
        % get name to display
        objectname = inputname(1);
        if isempty(objectname)
            objectname = 'ans';
        end
        
        % display object name
        fprintf('%s = \n', objectname);
        
        fprintf(emptyLine);
        
        fprintf('OrthoSlicer3d object, containing a %d x %d x %d %s image.\n', ...
            this.imageSize, this.imageType);
        
        % calibration information for image
        if this.calibrated
            fprintf('  Voxel spacing = [ %g %g %g ] %s\n', ...
                this.voxelSize, this.voxelSizeUnit');
            fprintf('  Image origin  = [ %g %g %g ] %s\n', ...
                this.voxelOrigin, this.voxelSizeUnit');
        end

        fprintf(emptyLine);
        
    end
end

end