function Imagefolder2volume(folder_name,format,volume_name,vdim)
    folder_name = strrep(folder_name,'"','');
    
    r = rdir([folder_name,filesep,'*',format]);       
    
    NZ = numel(r);
    hw = waitbar(0,sprintf('Processing frame %d of %d',0,NZ));
    warning off;
    slice = imread(r(1).name);
    [NX, NY, ~] = size(slice);
    V = zeros(NX,NY,NZ);
    for i = 1:NZ
        V(:,:,i) = rgb2gray(imread(r(i).name));
        
        waitbar(i/NZ,hw,sprintf('Processing frame %d of %d',i,NZ));
    end
    warning on;

    write_nifti_volume(V,vdim,volume_name);
    close(hw)
    disp('...done converting.')
    
    