function write_nifti_volume(v, vdim, filename, hdr, permute_on)

if nargin < 5
    permute_on = false;
    if nargin < 4
        hdr = [];

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
end

filename = cleanFileName(filename);

% support compressed nifti (.nii.gz)
compressed_file = false;
if strcmpi(filename(end-2:end),'.gz')
    compressed_file = true;
    filename = filename(1:end-3);
end

if permute_on
    warning('swapping x and y dimension to synchronize with ITK/VTK')
    v = permute(v,[2 1 3]);
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
if ~isempty(hdr)
    disp('... replacing nifty header.');
    nii.hdr = hdr;
    nii.hdr.dime.datatype = typeID;
   switch nii.hdr.dime.datatype
   case 2,
      nii.hdr.dime.bitpix = 8;  precision = 'uint8';
   case 4,
      nii.hdr.dime.bitpix = 16; precision = 'int16';
   case 8,
      nii.hdr.dime.bitpix = 32; precision = 'int32';
   case 16,
      nii.hdr.dime.bitpix = 32; precision = 'float32';
   case 32,
      nii.hdr.dime.bitpix = 64; precision = 'float32';
   case 64,
      nii.hdr.dime.bitpix = 64; precision = 'float64';
   case 128
      nii.hdr.dime.bitpix = 24;  precision = 'uint8';
   case 256 
      nii.hdr.dime.bitpix = 8;  precision = 'int8';
   case 511
      nii.hdr.dime.bitpix = 96;  precision = 'float32';
   case 512 
      nii.hdr.dime.bitpix = 16; precision = 'uint16';
   case 768 
      nii.hdr.dime.bitpix = 32; precision = 'uint32';
   case 1792,
      nii.hdr.dime.bitpix = 128; precision = 'float64';
   otherwise
      error('Datatype is not supported by make_nii.');
   end  
   
   if numel(vdim)==4 % update assuming 4D volume
       disp(' update header assuming 4D volume.')
       nii.hdr.dime.dim(1) = 4;
       nii.hdr.dime.dim(2:5) = size(v);
       nii.hdr.dime.pixdim(1) = 4;
       nii.hdr.dime.pixdim(2:5) = vdim;
       
       nii.hdr.hist.srow_z(end) = vdim(4);
   end
else
    disp('... using empty nifty header.');
end

outdir = fileparts(filename);
if ~isdir(outdir)
    mkdir(outdir);
end

save_nii(nii,filename);

% support compressed nifti (.nii.gz)
if compressed_file
    if ispc
        zipFilenames = gzip(filename);
        if ~isempty(zipFilenames)
            dos(['del ',filename],'-echo');
        end
    else
        disp(['gzip -f ',filename]);
        unix(['gzip -f ',filename]);
    end
    
    disp(['Saved nii file in ',filename,'.gz'])
else
    disp(['Saved nii file in ',filename])
end


