function SE = strelSphere(r)

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

if any(r>2)
    [X,Y,Z] = meshgrid(1:2*rx+1,1:2*ry+1,1:2*rz+1);
    SE = sqrt( (X-rx-1).^2 + (Y-ry-1).^2 + (Z-rz-1).^2 ) < sqrt(rx^2 + ry^2 + rz^2);
elseif r==2
        SE = ones(3,3,3);
elseif r==0
        disp('Using 0 structure element...')
        SE = 0;
else
    disp('using 6-connected structure element...')
    SE = zeros(3,3,3);
    SE(2,2,:) = 1;
    SE(2,:,2) = 1;
    SE(:,2,2) = 1;
end