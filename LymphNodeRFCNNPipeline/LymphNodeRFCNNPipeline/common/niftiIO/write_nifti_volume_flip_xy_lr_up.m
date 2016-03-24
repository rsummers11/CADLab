function write_nifti_volume_flip_xy_lr_up(v, vdim, filename)

if nargin < 3
  [filename,pathname] = uiputfile('*.nii','Save nifti volume as...') ;

  if filename ==0
    error(' No file selected.')
  end 

  filename = [pathname filename] ;
  
  if nargin < 2
      vdim = [1 1 1];%ones(ndims(v),1);
  end
end

for i = 1:size(v,3)
    v(:,:,i) = fliplr(flipud(v(:,:,i)));
end

type_name = class(v);

switch type_name
    case 'logical'
        v = uint8(v);
        typeID = 2;
    case 'int8'
        typeID = 256;
    case 'uint8'
        typeID = 2;
    case 'int16'
        typeID = 4;
    case 'uint16'
        typeID = 512;
    case 'int32'
        typeID = 8;
    case 'uint32'
        typeID = 768;
    case 'single'
        typeID = 16;
    case 'double'
        typeID = 64;
    otherwise
        error('data type not supported');
end

nii = make_nii(v,vdim,[],typeID);

save_nii(nii,filename);