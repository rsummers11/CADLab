%  Return time frame of a NIFTI dataset. Support both *.nii and 
%  *.hdr/*.img file extension. If file extension is not provided,
%  *.hdr/*.img will be used as default. 
%
%  It is a lightweighted "load_nii_hdr", and is equivalent to
%  hdr.dime.dim(5)
%  
%  Usage: [ total_scan ] = get_nii_frame(filename)
%
%  filename - NIFTI file name.
%
%  Returned values:
%
%  total_scan - total number of image scans for the time frame
%
%  NIFTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
function [ total_scan ] = get_nii_frame(fileprefix)

   if ~exist('fileprefix','var'),
      error('Usage: [ hdr, fileprefix, machine ] = get_nii_frame(filename)');
   end

   if ~exist('machine','var'), machine = 'ieee-le'; end

   new_ext = 0;

   if findstr('.nii',fileprefix)
      new_ext = 1;
      fileprefix = strrep(fileprefix,'.nii','');
   end

   if findstr('.hdr',fileprefix)
      fileprefix = strrep(fileprefix,'.hdr','');
   end

   if findstr('.img',fileprefix)
      fileprefix = strrep(fileprefix,'.img','');
   end

   if new_ext
      fn = sprintf('%s.nii',fileprefix);

      if ~exist(fn)
         msg = sprintf('Cannot find file "%s.nii".', fileprefix);
         error(msg);
      end
   else
      fn = sprintf('%s.hdr',fileprefix);

      if ~exist(fn)
         msg = sprintf('Cannot find file "%s.hdr".', fileprefix);
         error(msg);
      end
   end

   fid = fopen(fn,'r',machine);
    
   if fid < 0,
      msg = sprintf('Cannot open file %s.',fn);
      error(msg);
   else
      hdr = read_header(fid);
      fclose(fid);
   end
   
   if hdr.sizeof_hdr ~= 348
      % first try reading the opposite endian to 'machine'
      switch machine,
      case 'ieee-le', machine = 'ieee-be';
      case 'ieee-be', machine = 'ieee-le';
      end
        
      fid = fopen(fn,'r',machine);
        
      if fid < 0,
         msg = sprintf('Cannot open file %s.',fn);
         error(msg);
      else
         hdr = read_header(fid);
         fclose(fid);
      end
   end

   if hdr.sizeof_hdr ~= 348
      % Now throw an error
      msg = sprintf('File "%s" is corrupted.',fn);
      error(msg);
   end

   total_scan = hdr.dim(5);

   return;					% get_nii_frame


%---------------------------------------------------------------------
function [ dsr ] = read_header(fid)

    fseek(fid,0,'bof');
    dsr.sizeof_hdr    = fread(fid,1,'int32')';  % should be 348!

    fseek(fid,40,'bof');
    dsr.dim           = fread(fid,8,'int16')';

    return;					% read_header

