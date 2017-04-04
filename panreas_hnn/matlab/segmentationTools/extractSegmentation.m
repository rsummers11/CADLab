function [V, idx] = extractSegmentation(I,r)

    if numel(r)==1
        rx = r;
        ry = r;
        rz = r;
    elseif numel(r) == 3
        rx = r(1);
        ry = r(2);
        rz = r(3);
    else
        error('numel(r) needs to be 1 or 2!')
    end

    siz = size(I);
    [x, y, z] = findnd(I>0);
    
    x1 = min(x)-rx;
    if x1<1
        x1 = 1;
    end
    x2 = max(x)+rx;
    if (x2>siz(1))
        x2 = siz(1);
    end
    
    y1 = min(y)-ry;
    if y1<1
        y1 = 1;
    end    
    y2 = max(y)+ry;    
    if (y2>siz(2))
        y2 = siz(2);
    end    

    z1 = min(z)-rz;
    if z1<1
        z1 = 1;
    end    
    z2 = max(z)+rz;        
    if (z2>siz(3))
        z2 = siz(3);
    end    

    
    idx = [x1 x2; y1 y2; z1 z2];
    
    V = logical(I(x1:x2,y1:y2,z1:z2)>0);
    