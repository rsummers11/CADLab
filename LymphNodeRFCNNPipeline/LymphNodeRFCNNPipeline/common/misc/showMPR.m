function showMPR(V,vdim,displayRange,origin,colormapstyle,showAxesLabel)    

    if ~exist('vdim','var')
        vdim = [1 1 1];
    end    
    if ~exist('displayRange','var')
       displayRange = [min(V(:)), max(V(:))];
    end
    if ~exist('origin','var')
        origin = [0 0 0];
    end    
    if ~exist('colormapstyle','var')
        colormapstyle = 'gray';
    end
    if ~exist('showAxesLabel','var')
        showAxesLabel = false;
    end

    % using 3D slicer toolbox
    if strcmpi(colormapstyle,'gray')
        cmap = gray(256);
    elseif strcmpi(colormapstyle,'jet')
        cmap = jet(256);
    else
        error('No such colormap style (use gray or jet): %s !\n',...
            colormapstyle);
    end
        
    OrthoSlicer3d(V, ...
        'Origin', origin, 'Spacing', vdim, ...
        'DisplayRange', displayRange, 'ColorMap', cmap);

    if showAxesLabel
        xlabel('X axis');
        ylabel('Y Axis');
        zlabel('Z Axis');
    else
        axis off
    end
    
%% "thick" slices
%     siz = size(V);
%     siz2 = round(siz/2);
% 
%     figure
%     subplot(1,3,1)
%     imagesc(1:siz(1)*vdim(1),1:siz(2)*vdim(2),V(:,:,siz2(3))');
%     title('xy')
%     colormap gray;
%     axis equal,axis off;
%     subplot(1,3,2)
%     imagesc(1:siz(1)*vdim(1),1:siz(3)*vdim(3),squeeze(V(:,siz2(3),:))');
%     title('xz')
%     colormap gray;
%     axis equal,axis off;
%     subplot(1,3,3)
%     imagesc(1:siz(2)*vdim(2),1:siz(3)*vdim(3),squeeze(V(siz2(3),:,:))');
%     title('yz')
%     colormap gray;
%     axis equal,axis off;    
%     shg
    