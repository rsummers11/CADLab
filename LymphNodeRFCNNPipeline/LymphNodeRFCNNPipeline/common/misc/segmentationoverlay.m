function segmentationoverlay(V,M,vdim)

    if ~exist('vdim','var')
        vdim = [1 1 1];
    end

    vsiz = size(V);
    if isempty(M)
        M = zeros(vsiz,'uint8');
    end    
    msiz = size(M);
    
    if any((vsiz-msiz)~=0)
        error(' Image and segmentation must be the same size!');
    end
    
    sortV = sort(V(:));
    N = numel(sortV);
    
    amin = double(sortV(round(0.1*N))); % 10th percentile
    amax = double(sortV(round(0.9*N))); % 90th percentile
    
    NX = vsiz(1);
    NY = vsiz(2);
    NZ = vsiz(3);

    % center on segmentation
    if sum(M(:))>0
        [lx, ly, lz] = findnd(M>0);
        MeanX = round(mean(lx));
        MeanY = round(mean(ly));
        MeanZ = round(mean(lz));
    else
        MeanX = round(NX/2);
        MeanY = round(NY/2);
        MeanZ = round(NZ/2);
    end
        
	try
    figure_fullscreen;
    subplot(2,2,1)
    VS = mat2gray(V(:,:,MeanZ),[amin amax]);
    MS = M(:,:,MeanZ);
    imshow(imoverlay(VS,MS>0,[1 0 0]))
    axis square
    subplot(2,2,2)
    VS = mat2gray(squeeze(V(:,MeanY,:)),[amin amax]);
    MS = squeeze(M(:,MeanY,:));
    imshow(imoverlay(VS,MS>0,[1 0 0]))    
    axis square
    subplot(2,2,3)
    VS = mat2gray(squeeze(V(MeanX,:,:)),[amin amax]);
    MS = squeeze(M(MeanX,:,:));
    imshow(imoverlay(VS,MS>0,[1 0 0]))    
    axis square
    subplot(2,2,4)
    title(sprintf('Segmentation [%d,%d,%d]',NX,NY,NZ),'FontSize',15)    
    showSurface(M,4,vdim);    
    shg
    end
	