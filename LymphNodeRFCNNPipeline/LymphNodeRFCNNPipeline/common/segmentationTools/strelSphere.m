function SE = strelSphere(r)

if r>2
    [X,Y,Z] = meshgrid(1:2*r+1,1:2*r+1,1:2*r+1);
    SE = sqrt( (X-r-1).^2 + (Y-r-1).^2 + (Z-r-1).^2 ) < r;
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