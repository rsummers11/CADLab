function [v vsize vdim filename hdr] = read_nifti_volume(filename,permute_on)
% [v vsize vdim filename hdr] = read_nifti_volume(filename,permute_on)

tic
if nargin < 2
    permute_on = false;
    if nargin < 1
      [filename,pathname] = uigetfile({'*.nii';'*.nii.gz'},'Load nifti volume file') ;

      if filename ==0
        error( ' No file selected.' )
      end 

      filename = [pathname filename] ;
    end
    
    % check for permute and get file gui
    if nargin == 1
        if islogical(filename)
            permute_on = filename;
            
            [filename,pathname] = uigetfile('*.nii','Load nifti volume file') ;

            if filename ==0
                error([ ' No file selected.' ])
            end 

            filename = [pathname filename] ;
        end
        % else keep filename
    end    
end

filename = cleanFileName(filename);

if exist('rdir')
    disp('loading image from ')
    rdir(filename)
else
    disp(['loading image from ', filename])
end

% support compressed nifti (.nii.gz)
compressed_file = false;
if strcmpi(filename(end-2:end),'.gz')
    compressed_file = true;    
    if isunix
        %%tmp_filename = [filename,'_tmp-',getRandomString(50),'.nii'];
        disp(['gunzip -f ',filename]);
        unix(['gunzip -f ',filename]);
        filename = filename(1:end-3);
    else
        curr_dir = pwd;
        [filepath, name, ext] = fileparts(filename);
        if ~isempty(filepath)
            cd(filepath);          
        end
        gunzip(filename);      
        filename = name;
    end
end

nii = load_untouch_nii(filename);

v = nii.img;
vsize = size(v);
vdim = nii.hdr.dime.pixdim(2:numel(vsize)+1);
hdr = nii.hdr;

if permute_on
    warning('swapping x and y dimension to synchronize with ITK/VTK')
    v = permute(v,[2 1 3]);
    vdim = [vdim(2) vdim(1) vdim(3)];
    vsize = [vsize(2) vsize(1) vsize(3)];
    
    warning('also flipping each slice ud/lr')
    for i = 1:vsize(3)
        v(:,:,i) = fliplr(flipud(v(:,:,i)));
    end
end

% support compressed nifti (.nii.gz)
if compressed_file
    if isunix
        disp(['gzip -f "',filename,'"']);
        unix(['gzip -f "',filename,'"']);
        filename = [filename,'.gz'];
    else
%        gzip(filename);
        [status, result] = dos(['del "',filepath,filesep,filename,'"'],'-echo'); % delete unzipped file       
        if status ~= 0
            error(result);
        end
        filename = [filepath,filesep,filename,'.gz'];
        cd(curr_dir)
    end
end

toc
