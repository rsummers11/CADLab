function [I, SE] = dilateColon(I,r)
    
    if sum(I(:)) ~= 0
        if numel(size(I))==3
            [V, idx] = extractSegmentation(I,r);

            SE = strelSphere(r);

            disp(['morpholocigal dilation with radius: ', num2str(r)])

            O = imdilate(V,SE);

            I(idx(1,1):idx(1,2),idx(2,1):idx(2,2),idx(3,1):idx(3,2)) = O;
        elseif numel(size(I))==2
            if r==1
                SE = ones(r*2+1);
            else
                SE = strel('disk',r,8);
            end
            I = imdilate(I,SE);
        else    
            error('dilateColon: %d dimensions not supported!',numel(size(I)))
        end
    end
