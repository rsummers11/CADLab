function [L0, LE] = findLargestConnComponent(I,r)

    %L0: opened largest component
    %LE: connected object with largest volume after eroding
  
    if ~exist('r','var')
        r = 0;
    end
    
    if r > 0
        I = erodeColon(logical(I),r);
    else
        disp(' no errosion to find largest connected component.');
    end

    I = uint8(I>0);

    [Lab, num] = bwlabeln(I,6);
    if num==0
        error(' There is no object in the image!')
    end
    disp('Number of objects in segmentation:')
    disp(num)
    

    % Find Large intestine as segment with greatest volume
%     Volumes = zeros(num,1);
%     for i=1:num
%         Volumes(i) = sum(Lab(:)==i);
%     end
 
    % MUCH FASTER!
    Volumes = zeros(num,1);
    STATS = regionprops(Lab, 'Area');
    for i=1:num
        Volumes(i) = STATS(i).Area;
    end


    LE = uint8(Lab==find(Volumes==max(Volumes),1,'first')); 
    %sortVolumes = sort(Volumes);
    %LE = uint8(Lab==find(Volumes==sortVolumes(end)) | Lab==find(Volumes==sortVolumes(end-2))); % the first two largest objects
    
    L0 = LE;
    if r>0
        L0 = dilateColon(logical(L0),r);
    end
    
    