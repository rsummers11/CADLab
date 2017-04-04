function SAME = compareImages(size1,vdim1,size2,vdim2)

    SAME = true;
    if any(size1(1:3)-size2(1:3))
        warning('Image and mask sizes are not the same');
        SAME = false;
    end
    if any(vdim1(1:3)-vdim2(1:3))
        warning('Image and mask pixdims are not the same');
        SAME = false;
    end   

    if numel(size1) ~= numel(size2)
        warning('number of image dimensions are not the same!: %d vs. %d, but 3D dimesions fit', numel(size1), numel(size2))
    end    
   