function [L0, LE, Lab, num] = findLargestConnComponent(I,r,CONN)

    %L0: opened largest component
    %LE: connected object with largest volume after eroding
    %Lab: Labelled connected objects after eroding
    %num: number connected objects after eroding
  
    if ~isnumeric(I) && ~islogical(I)
        error('Input must be a matrix!')
    end

    if ~exist('CONN','var')
        CONN = 6;
    end
    
    if r > 0
        I = erodeColon(logical(I),r);
    end

    I = uint8(I>0);

    [Lab, num] = bwlabeln(I,CONN);
    if num>0
        
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
    
    else
        warning(' There is no object in the image!')
        L0 = I;
        LE = I;
    end
    
    