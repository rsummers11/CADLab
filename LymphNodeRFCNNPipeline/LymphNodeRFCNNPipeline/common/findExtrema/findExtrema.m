function [minima,maxima] = findExtrema(array,kernel,strict)

%function [minima,maxima] = findExtrema(array,kernel,strict)
%
%PURPOSE:   findExtrema.m is designed to find the extrema points (minima 
%			and maxima) from an N-dimensional array. It employs the 
%			optimization recommended by Steve Eddins on Stack Overflow, 
%			making it much more computationally efficient than most 
%			implementations. See the thread for details. 
%			(http://stackoverflow.com/questions/1856197/how-can-i-find-local-maxima-in-an-image-in-matlab)
%
%DEPENDENCIES:
%			This code requires the Image Processing Toolbox from MathWorks.
%
%INPUTS:
%			array
%				The only required input, all others are optional. This
%				should be the n-dimensional array (matrix, surface, time
%				series) for which all local extrema should be found. 
%			kernel
%				An optional input, the default specifies to check all
%				neighboring points (row, column, and diagonal) in two
%				dimensions (2D). By including this as an optional input,
%				this program can easily be run using different connectivity
%				or extended to additional dimensions. For instance, the
%				2D kernel [0 1 0; 1 0 1; 0 1 0] only considers adjacent
%				points on rows and columns, not diagonals, to be neighbors.
%				A standard N-D kernel could be made by:
%					kernelDimensions = 3*ones(N);
%					kernel = ones(kernelDimensions);
%					centerPoint = 2*ones(N);
%					kernel(centerPoint) = 0;
%			strict
%				An optional input, default value true. If true, a point is
%				only considered a local minimum (maximum) if it is 
%				strictly less (greater) than all its neighbors, where 
%				neighbors are defined by the kernel. If false, a point is 
%				considered a local minimum (maximum) if it is less 
%				(greater) than or equal to all its neighbors. 
%
%OUTPUTS:
%			minima
%				This will be a logical array identical in size to the input
%				array. Locations of local minima will be true (1) while all
%				other locations will be false (0). The notes indicate how
%				to obtain other commonly desired outputs. 
%			maxima
%				This will be a logical array identical in size to the input
%				array. Locations of local maxima will be true (1) while all
%				other locations will be false (0). The notes indicate how
%				to obtain other commonly desired outputs. 
%
%LIMITATIONS:	The code employed does not handle input arrays which
%				include NaN values. If ignoring NaN values is desired, this
%				code can be used, but obtaining both the minima and the
%				maxima will require two run; each run will only have one of
%				the outpute valid. 
%
%				To obtain the maxima accurately, simply run on the 
%				temporary array tmpArray using the following additional 
%				code. 
%					tmpArray = array;
%					tmpArray(isnan(tmpArray)) == min(tmpArray(:))-1;
%				To obtain the minima accurately, simply run on the 
%				temporary array tmpArray using the following additional 
%				code. 
%					tmpArray = array;
%					tmpArray(isnan(tmpArray)) == max(tmpArray(:))+1;
%
%NOTES:		The behavior at edges bears mentioning, as these points lack
%			well defined neighbors. This program considers a point to be a
%			local maximum if it is greater than all neighbors which exist
%			in the array. Thus for example, if the input array was the 1x1 
%			array [-1], that point would be considered a local maximum. 
%
%			If desired, obtaining a list of the local maxima 
%			(sortedMaxima) and their indices (sortedIndices) in descending 
%			order is readily accomplished by after this program, running
%			the following code:
%				indices = find(maxima(:));
%				values = array(maxima);
%				tmp = sortrows([values indices],-1);
%				sortedMaxima = tmp(:,1);
%				sortedIndices = tmp(:,2);
%
%Written by Stephen M. Anthony based upon Steve Eddins comments to a thread
%at Stack Overflow
%http://stackoverflow.com/questions/1856197/how-can-i-find-local-maxima-in-an-image-in-matlab

if nargin<2
	%The default kernel will determine whether the point is a local maximum
	%including adjacent points on the row, column, and diagonals.
	kernel = [1 1 1; 1 0 1; 1 1 1];
end
comparisonMatrix = imdilate(array,kernel);
if nargin<3 || strict
	%By default, only strict maxima are found, points that are larger than
	%all their neighbors
	minima = array < comparisonMatrix;
	maxima = array > comparisonMatrix;
else
	%Points found need not be strict local maxima, they need only be equal
	%to or greater than their neighboring points
	minima = array <= comparisonMatrix;
	maxima = array >= comparisonMatrix;
end