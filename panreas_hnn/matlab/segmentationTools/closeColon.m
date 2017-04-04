function I = closeColon(I,r)
    
    if sum(I(:)) ~= 0 && r>0
        [V, idx] = extractSegmentation(I,r);

        SE = strelSphere(r);

        disp(['morpholocigal closing with radius: ', num2str(r)])

        O = imclose(V,SE);

        I(idx(1,1):idx(1,2),idx(2,1):idx(2,2),idx(3,1):idx(3,2)) = O;
    end
    