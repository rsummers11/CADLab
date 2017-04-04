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
    
    amin = double(sortV(round(0.75*N))); % >50% of image is air
    amax = double(sortV(round(0.99*N))); % 99th percentile
    
    NX = vsiz(1);
    NY = vsiz(2);
    NZ = vsiz(3);
    
    [x, y, z] = findnd(M>0);
    x = round(mean(x));
    y = round(mean(y));
    z = round(mean(z));
    
    figure_fullscreen;
    subplot(2,2,1)
    imshow(imoverlay(mat2gray(V(:,:,z),[amin amax]),...
        edge(M(:,:,z))>0,[1 0 0]))
    axis square
    subplot(2,2,2)
    imshow(imoverlay(mat2gray(squeeze(V(:,y,:))',[amin amax]),...
        edge(squeeze(M(:,y,:)))'>0,[1 0 0]));
    axis square
    subplot(2,2,3)
    imshow(imoverlay(mat2gray(squeeze(V(x,:,:))',[amin amax]),...
        edge(squeeze(M(x,:,:)))'>0,[1 0 0]));
    axis square
    subplot(2,2,4)
    title(sprintf('Segmentation [%d,%d,%d]',NX,NY,NZ),'FontSize',15)    
    showSurface(M,4,vdim);    
    shg
    