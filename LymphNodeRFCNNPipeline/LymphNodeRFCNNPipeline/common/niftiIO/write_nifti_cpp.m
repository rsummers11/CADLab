function write_nifti_cpp(v, header_name,filename)

% v: control point posistions
% header_name: need to use existing header from nifty_reg
% filename: where to save control points

if nargin < 4
    permute_on = false;

    if nargin < 3
      [filename,pathname] = uiputfile({'*.nii';'*.nii.gz'},'Save nifti volume as...') ;

      if filename ==0
        error(' No file selected.')
      end 

      filename = [pathname filename] ;

      if nargin < 2
          vdim = [1 1 1];%ones(ndims(v),1);
      end
    end
end

filename = strrep(filename,'"','');

% support compressed nifti (.nii.gz)
compressed_file = false;
if strcmpi(filename(end-2:end),'.gz')
    if ~strcmp(computer,'PCWIN')
        compressed_file = true;
        filename = filename(1:end-3);
    else
        error('COMPRESSED NIFTI (.nii.gz) NOT SUPPORTED UNDER WINDOWS!')
    end
end

% if permute_on
%     warning('swapping x and y dimension to synchronize with ITK/VTK')
%     v = permute(v,[2 1 3]);
% end

% type_name = class(v);
% 
% switch type_name
%     case 'logical'
%         v = uint8(v);
%         typeID = 2;
%     case 'int8'
%         typeID = 256;
%     case 'uint8'
%         typeID = 2;
%     case 'int16'
%         typeID = 4;
%     case 'uint16'
%         typeID = 512;
%     case 'int32'
%         typeID = 8;
%     case 'uint32'
%         typeID = 768;
%     case 'single'
%         typeID = 16;
%     case 'double'
%         typeID = 64;
%     otherwise
%         error('data type not supported');
% end
% 
% nii = make_nii(v,vdim,[],typeID);

nii = load_untouch_nii(header_name);
nii.img = v;
%nii.hdr.dime.dim = [5 size(v,1) size(v,2) size(v,3) 1 2 1 1];
%nii.hdr.dime.pixdim = [1 vdim(1) vdim(2) vdim(3) 1 1 1 1];

save_untouch_nii(nii,filename);

% support compressed nifti (.nii.gz)
if compressed_file
    disp(['gzip -f ',filename]);
    unix(['gzip -f ',filename]);
end
