function [ATERY, limits] = getAterialRegion(V,t_low_atery,t_high_atery, t_low_roi, t_high_roi, rDilate)
    %% PARAMS
    zExtFactor = 1/3;
    rErode = 1; % erode a little to find ateria
    rDilate = rErode + rDilate; % dilate more to get Search Region
    rErodeCleaning = 3 * rErode; % just to get one fully connected search region

    %% RUN
    vsiz = size(V);
    NX = vsiz(1);
    NY = vsiz(2);
    NZ = vsiz(3);

    disp('  find aterial region...')
    ATERY = V > t_low_atery & V < t_high_atery;
    ATERY = erodeColon(ATERY,rErode);
    
    
    visiblePercentage = 0.01;
    CONN = 6;     
     
    Nvx = prod(vsiz); 
    disp('  find connected components...')
    tic
    [L, N] = bwlabeln(ATERY,CONN);
    STATS = regionprops(L,'Area','BoundingBox','Centroid');
    AREA = zeros(N,1);
    ZEXTEND = zeros(N,1);
    CENTROIDS = zeros(N,3); 
    for i = 1:N    
        AREA(i,1) = STATS(i).Area;
        ZEXTEND(i,1) = STATS(i).BoundingBox(6); % width in 3rd dim (z)
        CENTROIDS(i,:) = STATS(i).Centroid;
    end
    idx = find(100*AREA/Nvx > visiblePercentage); 
    zExtend = ZEXTEND(idx);
    idx(zExtend<(zExtFactor*max(zExtend)) | zExtend==NZ) = [];
    zCentroids = CENTROIDS(idx,3);
    [minZ, minZidx] = min(zCentroids); % assume aorta has a lower center of gravity than spine blood
    idx = idx(minZidx);
    Nidx = numel(idx);
    
    ATERY = L==idx;
    
    fprintf(' dilate by %d voxels to get Search Region\n', rDilate)
    ATERY = dilateColon(ATERY,rDilate); % dilate more to get Search Region

    ATERY(V < t_low_roi | V > t_high_roi) = 0; % remove certain intensities not of interest  
    ATERY = findLargestConnComponent(ATERY,rErodeCleaning);
    %figure;
    %showSurface(ATERY,4);    
    
    % Get region limits
    abdstats = regionprops(ATERY, 'BoundingBox');
    if numel(abdstats) ~= 1
        error(' could not find rib cage!');
    end
    ul_corner = abdstats.BoundingBox(1:3);
    width = abdstats.BoundingBox(4:6);
    x1 = ceil(ul_corner(1));
    y1 = ceil(ul_corner(2));
    z1 = ceil(ul_corner(3)); % z1 is bottom of lungs (above)!
    x2 = floor(x1 + width(1) -1);
    y2 = floor(y1 + width(2) -1);
    z2 = floor(z1 + width(3) -1);    
    
    limits = [x1,x2; y1,y2; z1,z2];        
%     
%     
%     figure;
%     showSurface(ATERYdil,4);
%     
%     ATERYregion = ATERYdil & (~ATERY);
%     figure
%     showSurface(ATERYregion,4);
%     
%     
%     disp('  find connected components...')
%     LBACK = bwlabeln(BACKGROUND,26);
%     % remove objects on sides
%     idx1 = unique(reshape(LBACK(1,:,:),[],1)); 
%     idx2 = unique(reshape(LBACK(end,:,:),[],1));
%     idx3 = unique(reshape(LBACK(:,1,:),[],1));
%     idx4 = unique(reshape(LBACK(:,end,:),[],1));    
%     idx = unique([idx1; idx2; idx3; idx4]); 
%     %idx = unique([idx1; idx2; idx3; idx4; idx5; idx6]); 
%     % remove objectes that touch top and botton (bed)
%     idx5 = unique(reshape(LBACK(:,:,1),[],1)); 
%     idx6 = unique(reshape(LBACK(:,:,end),[],1));            
%     idxC = intersect(idx5,idx6);
%     idx = unique([idx; idxC]);
%     N = numel(idx);
%     for i = 1:N
%         fprintf(' removing region %d of %d...\n', i, N);
%         BACKGROUND(LBACK==idx(i)) = 0;
%     end
%     % find lungs as largest object
%     ATERY = findLargestConnComponent(BACKGROUND,0);    
%     
%     % connect lungs
%     [lx, ly, lz] = findnd(LUNGS>0);
%     if min(lz) > NZ/2
%         lung_at_end = true
%     else
%         lung_at_end = false
%     end    
%     lrangex = max(lx)-min(lx);
%     lrangey = max(ly)-min(ly);
%     if lrangex < lrangey
%         largest_dim = 1
%     else
%         largest_dim = 2
%     end
%     disp('  fill space between lungs...')
%     for v = 1:vsiz(largest_dim)
%         for z = 1:NZ
%             if largest_dim==1
%                 V1 = find( squeeze(LUNGS(v,:,z))>0, 1, 'first');
%                 V2 = find( squeeze(LUNGS(v,:,z))>0, 1, 'last');
%                 if ~isempty(V1) && ~isempty(V2)
%                     LUNGS(v,V1:V2,z) = 1;
%                 end                              
%             else
%                 V1 = find( squeeze(LUNGS(:,v,z))>0, 1, 'first');
%                 V2 = find( squeeze(LUNGS(:,v,z))>0, 1, 'last');
%                 if ~isempty(V1) && ~isempty(V2)
%                     LUNGS(V1:V2,v,z) = 1;
%                 end                
%             end
%         end
%         if mod(v,100)==0
%             fprintf('  fill ray %d of %d...\n', v, vsiz(largest_dim));
%         end        
%     end    
%     
% 
% %%%    LUNGS = lung_segmentation(V);
%     
%     % get abdominal region
%     disp('  project lungs downwards...')    
%     BELOW_LUNGS = zeros(vsiz,'uint8');
%     for x = 1:NX
%         for y = 1:NY
%             if lung_at_end
%                 z1 = find( LUNGS(x,y,:)>0, 1, 'first');
%                 if ~isempty(z1)
%                     BELOW_LUNGS(x,y,1:z1) = 1;
%                 end                
%             else
%                 z1 = find( LUNGS(x,y,:)>0, 1, 'last');
%                 if ~isempty(z1)
%                     BELOW_LUNGS(x,y,z1:end) = 1;
%                 end                
%             end
%         end
%         if mod(x,100)==0
%             fprintf('  fill ray %d of %d...\n', x, NX);
%         end        
%     end
%     
%     abdstats = regionprops(BELOW_LUNGS, 'BoundingBox');
%     if numel(abdstats) ~= 1
%         error(' could not find rib cage!');
%     end
%     ul_corner = abdstats.BoundingBox(1:3);
%     width = abdstats.BoundingBox(4:6);
%     x1 = ceil(ul_corner(1));
%     y1 = ceil(ul_corner(2));
%     z1 = ceil(ul_corner(3)); % z1 is bottom of lungs (above)!
%     x2 = floor(x1 + width(1) -1);
%     y2 = floor(y1 + width(2) -1);
%     z2 = floor(z1 + width(3) -1);    
%     
%     limits = [x1,x2; y1,y2; z1,z2];    
% 
%     % threshold intensity range of interes
%     V_SR = V(x1:x2,y1:y2,z1:z2);
%     BONES = fillHolesInSlices(V_SR,t_high,'greater');
%     BACKGROUND = fillHolesInSlices(V_SR,t_low,'smaller');   
%     SR_cropped = (~BACKGROUND) & (~BONES) & BELOW_LUNGS(x1:x2,y1:y2,z1:z2);
%     
%     SR = zeros(vsiz,'uint8');
%     SR(x1:x2,y1:y2,z1:z2) = findLargestConnComponent(SR_cropped,0);
%      
% %     % find abdominal regions within rib cage
% %     
% %     V_SR = V(x1:x2,y1:y2,z1:z2);
% 
% %     
% %     if skip_voxels>1 
% %         SR_cropped(1:skip_voxels:end,1:skip_voxels:end,:) = 0;
% %     end
% 
%     
% 
% %     RIBCAGE = findLargestConnComponent(BONES,0);    
% %     ribstats = regionprops(RIBCAGE, 'BoundingBox');
% %     if numel(ribstats) ~= 1
% %         error(' could not find rib cage!');
% %     end
% %     ul_corner = ribstats.BoundingBox(1:3);
% %     width = ribstats.BoundingBox(4:6);
% %     x1 = ceil(ul_corner(1));
% %     y1 = ceil(ul_corner(2));
% %     z1 = ceil(ul_corner(3)); % z1 is bottom of lungs (above)!
% %     x2 = floor(x1 + width(1) -1);
% %     y2 = floor(y1 + width(2) -1);
% %     z2 = floor(z1 + width(3) -1);
% %     fprintf(' Selecting slices %d to %d from %d.\n',z1,z2,NZ)    
% %     z1 = z1_lungs;
