function [no_surface, fv] = showSurface(I,RD,dim,RP,TF,fc,ec,light,lim,use_vtk_indixing,Translation)
%
% SHOWSURFACE(I,RD,dim,RP,TF,fc,ec)
%
% I: Binary volume of the segmented object
% RD: Reduction factor
% dim: voxel dimensions
% RP: Reduce patches
% TF: Transparacy
% fc: facet colour (e.g. 'red','blue',...)
% ec: edge colour (e.g. 'red','blue',...)
% 
% Example: showSurface(X,5,[1 2 3],10000)
%
% Author: Holger Roth
% Date: 12-05-2009

set(0,'DefaultFigureRenderer','opengl') 

 if (nargin==0)
     error(['Usage: showSurface(Image,',...
            'ReduceVolumeFactor,',...
            'dim, ',...
            'ReducePatchFactor, ',...
            'TransperanceFactor, ',...
            'FaceColour, ',...
            'EdgeColour, '])
 end

 if nargin<11
     Translation = [0 0 0];
     if nargin<10
         use_vtk_indixing = true;
         if nargin<9
             lim = true;
             if nargin<8
                 light = true;
                if nargin<7
                    ec = 'none';
                    if nargin<6
                        fc = 'colon';
                        if nargin<5
                            TF = 1.0;
                            if nargin<4;
                                RP = 1;
                                if nargin<3;
                                    dim = [1 1 1];
                                    if nargin<2
                                        RD  = 1;
                                    end
                                end
                            end
                        end
                    end
                end
             end
         end
     end
 end

   if strcmpi(fc,'colon')
        fc = [225 116 80]/256;
   end

   % figure
    % Volume reduction
    if RD>=2
        I = reducevolume(I,RD);
    end
    
    % find voxel location arrays
     %siz = size(I);
%     [xs ys zs] = get_vox_locs(siz,dim);
%     [X Y Z] = ndgrid(xs,-ys,zs);
    
    %ys = [1:siz(1)]*RD;
    %xs = [1:siz(2)]*RD;
    %zs = [1:siz(3)]*RD;    
    
    %[X Y Z] = ndgrid(ys,xs,zs);
    
    % Compute isosurface
    %fv = isosurface(Y,X,Z,I,0);
    
    I = single(I);
    fv = isosurface(uint8(255*I/max(I(:))),0,'noshare'); % 'noshare' doesnot create shared vertices. This is faster, but produces a largerset of vertices.
    if isempty(fv.vertices)
        warning('Could not extract surface!')
        no_surface = true;
        return
    end
    fv = reducepatch(fv,RP);
    
    if use_vtk_indixing
        fv.vertices(:,1) = (fv.vertices(:,1)-1)*dim(2)*RD;  % VTK indexing starting from 0.0 mm
        fv.vertices(:,2) = (fv.vertices(:,2)-1)*dim(1)*RD;  % VTK indexing starting from 0.0 mm
        fv.vertices(:,3) = (fv.vertices(:,3)-1)*dim(3)*RD;  % VTK indexing starting from 0.0 mm 
    else
        fv.vertices(:,1) = fv.vertices(:,1)*dim(2)*RD;  % VTK indexing starting from 0.0 mm
        fv.vertices(:,2) = fv.vertices(:,2)*dim(1)*RD;  % VTK indexing starting from 0.0 mm
        fv.vertices(:,3) = fv.vertices(:,3)*dim(3)*RD;  % VTK indexing starting from 0.0 mm     
    end
    
    if any(Translation~=0)
        fv.vertices = fv.vertices + ones(size(fv.vertices,1),1)*Translation;
    end
    
    % Plot
    %fc = 'red'; %[0.0 0.0 0.75]; % facet colour
    %ec = 'none'; %[0.5 0.5 0.5]; % edge colour
    P = patch(fv);
    %isonormals(X,Y,Z,I,P)
    set(P,'FaceColor',fc,'EdgeColor',ec);
    daspect([1 1 1])
    view(3); axis vis3d
    hold on
    
    %camlight('headlight')
    %camlight('left')
    %camlight('right')        
    %camlight(0,0) % front
    %camlight(90,0)
    %camlight(180,0) % back
    %camlight(270,0) 
    
    

    %lighting gouraud%calculates the vertex normals and interpolates linearly 
    %across the faces. Select this method to view curved surfaces.
    %lighting phong %interpolates the vertex normals across each face and calculates the reflectance at each pixel. Select this choice to view curved
    %surfaces. Phong lighting generally produces better results than
    %Gouraud lighting, but it takes longer to render.
    lighting gouraud %faster than phong

    if light        
        camlight(0,0) % front
        camlight(180,0) % back    
    end
    
    if lim            
        xlim([min(fv.vertices(:,1)),max(fv.vertices(:,1))])
        ylim([min(fv.vertices(:,2)),max(fv.vertices(:,2))]) 
        if min(fv.vertices(:,3))~=max(fv.vertices(:,3))
            zlim([min(fv.vertices(:,3)),max(fv.vertices(:,3))])    
        end
    end    

    xlabel('x')
    ylabel('y')
    zlabel('z')    
    grid on
    
    % Transparacy
    if TF<1.0
        alpha(P,TF)
    end
    
   % shg
    view(-20,10)
    
    no_surface = false;
