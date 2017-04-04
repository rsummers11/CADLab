function I = erodeColon(I,r)
    
    if any(I(:))
        [V, idx] = extractSegmentation(I,r);

        SE = strelSphere(r);

        fprintf('erodeColon: morpholocigal erosion with radius %d ...\n',r)

        O = imerode(V,SE);

        I(idx(1,1):idx(1,2),idx(2,1):idx(2,2),idx(3,1):idx(3,2)) = O;        
    else
        warning('erodeColon: there is no object in image!')
    end
    