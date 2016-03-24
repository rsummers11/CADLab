function [hdr, filetype, fileprefix, machine] = read_nifti_hdr(filename)

    filename = strrep(filename,'"','');
    filename = strrep(filename,' ','');
    
    % support compressed nifti (.gz)
    compressed_file = false;
    if strcmpi(filename(end-2:end),'.gz')
        compressed_file = true;    
        if isunix
            filename = strrep(filename,'&','\&');
            filename = strrep(filename,'(','\(');
            filename = strrep(filename,')','\)');            
            filename = strrep(filename,'''','\''');       
            disp(['gunzip -f ',filename]);
            unix(['gunzip -f ',filename]);
            filename = filename(1:end-3);
            filename = strrep(filename,'\&','&');
            filename = strrep(filename,'\(','(');
            filename = strrep(filename,'\)',')');      
            filename = strrep(filename,'\''','''');       
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

%    [hdr, filetype, fileprefix, machine] = load_nii_hdr(filename);
   [hdr,filetype,fileprefix,machine] = load_nii_hdr(filename);

   if filetype == 0
      hdr = load_untouch0_nii_hdr(fileprefix,machine);
      ext = [];
   else
      hdr = load_untouch_nii_hdr(fileprefix,machine,filetype);

      %  Read the header extension
      %
      ext = load_nii_ext(filename);
   end
    
    % support compressed nifti (.gz)
    if compressed_file
        if isunix
            filename = strrep(filename,'&','\&');
            filename = strrep(filename,'(','\(');
            filename = strrep(filename,')','\)');
            filename = strrep(filename,'''','\''');       
            disp(['gzip -f ',filename]);
            unix(['gzip -f ',filename]);
            filename = strrep(filename,'\&','&');
            filename = strrep(filename,'\(','(');
            filename = strrep(filename,'\)',')');
            filename = strrep(filename,'\''','''');       
            filename = [filename,'.gz'];
        else
    %        gzip(filename);
            [status, result] = dos(['del ',filepath,filesep,filename],'-echo'); % delete unzipped file       
            if status ~= 0
                error(result);
            end
            filename = [filepath,filesep,filename,'.gz'];
            cd(curr_dir)
        end
    end 
