function I = erodeColon(I,r)
    
    [V, idx] = extractSegmentation(I,r);

    SE = strelSphere(r);
    
    disp(sprintf('morpholocigal erosion with radius %d ...',r))
    
    O = imerode(V,SE);
    
    I(idx(2,1):idx(2,2),idx(1,1):idx(1,2),idx(3,1):idx(3,2)) = O;