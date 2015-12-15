function Iout = rotate90(I,dim,k)
% I = ROTATE(I)
% dim: rotation axis

    if dim==3
        Iout = zeros(size(I,2),size(I,1),size(I,3));
        for z = 1:size(I,3)
            Iout(:,:,z) = rot90(I(:,:,z),k);
        end
    else
        error('dim not implemented')
    end