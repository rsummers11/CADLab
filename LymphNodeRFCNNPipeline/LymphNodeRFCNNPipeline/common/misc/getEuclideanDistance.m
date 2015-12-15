function d = getEuclideanDistance(C,x,y,z)
    % [c, cidx] = FINDCLOSESTCENTRELINEPOINT(C,x,y,z)
    % or
    % [c, cidx] = FINDCLOSESTCENTRELINEPOINT(C,point)
    if nargin==2
        z = x(3);
        y = x(2); 
        x = x(1);
    end
    
    d = [C(:,1)-x, C(:,2)-y, C(:,3)-z];
    d = sqrt( sum(d.^2,2) );
    