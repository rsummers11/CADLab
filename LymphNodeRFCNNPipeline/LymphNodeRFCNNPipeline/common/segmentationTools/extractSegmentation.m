function [V, idx] = extractSegmentation(I,r)

    siz = size(I);

    XY = sum(I,3);
    X = sum(XY,1);
    Y = sum(XY,2);
    
    Z = sum(sum(permute(I,[3,2,1]),3),2);
    
    x1 = find(X>0,1,'first')-r;
    if x1<1
        x1 = 1;
    end
    x2 = find(X>0,1,'last')+r;
    if (x2>siz(2))
        x2 = siz(2);
    end
    
    y1 = find(Y>0,1,'first')-r;
    if y1<1
        y1 = 1;
    end    
    y2 = find(Y>0,1,'last')+r;    
    if (y2>siz(1))
        y2 = siz(1);
    end    

    z1 = find(Z>0,1,'first')-r;
    if z1<1
        z1 = 1;
    end    
    z2 = find(Z>0,1,'last')+r;        
    if (z2>siz(3))
        z2 = siz(3);
    end    

    
    idx = [x1 x2; y1 y2; z1 z2];
    
    V = logical(I(y1:y2,x1:x2,z1:z2)>0);
    