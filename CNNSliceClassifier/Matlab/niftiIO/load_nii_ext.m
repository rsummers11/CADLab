%  Load NIFTI header extension after its header is loaded using load_nii_hdr.
%
%  Usage: ext = load_nii_ext(filename)
%
%  filename - NIFTI file name.
%
%  Returned values:
%
%  ext - Structure of NIFTI header extension, which includes num_ext,
%       and all the extended header sections in the header extension.
%       Each extended header section will have its esize, ecode, and
%       edata, where edata can be plain text, xml, or any raw data
%       that was saved in the extended header section.
%
%  NIFTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
function ext = load_nii_ext(fileprefix)

   if ~exist('fileprefix','var'),
      error('Usage: ext = load_nii_ext(filename)');
   end

   machine = 'ieee-le';
   new_ext = 0;

   if findstr('.nii',fileprefix)
      new_ext = 1;
      %fileprefix = strrep(fileprefix,'.nii',''); % deletes .nii if
      %accidently in middle of string
      nii_idx = max(findstr('.nii',fileprefix)); % only delete the last incidence
      fileprefix(nii_idx:end) = [];      
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
   vox_offset = 0;
    
   if fid < 0,
      msg = sprintf('Cannot open file %s.',fn);
      error(msg);
   else
      fseek(fid,0,'bof');

      if fread(fid,1,'int32') == 348
         if new_ext
            fseek(fid,108,'bof');
            vox_offset = fread(fid,1,'float32');
         end

         ext = read_extension(fid, vox_offset);
         fclose(fid);
      else
         fclose(fid);

         %  first try reading the opposite endian to 'machine'
         %
         switch machine,
         case 'ieee-le', machine = 'ieee-be';
         case 'ieee-be', machine = 'ieee-le';
         end

         fid = fopen(fn,'r',machine);

         if fid < 0,
            msg = sprintf('Cannot open file %s.',fn);
            error(msg);
         else
            fseek(fid,0,'bof');

            if fread(fid,1,'int32') ~= 348

               %  Now throw an error
               %
               msg = sprintf('File "%s" is corrupted.',fn);
               error(msg);
            end

            if new_ext
               fseek(fid,108,'bof');
               vox_offset = fread(fid,1,'float32');
            end

            ext = read_extension(fid, vox_offset);
            fclose(fid);
         end
      end
   end

   return                                       % load_nii_ext


%---------------------------------------------------------------------
function ext = read_extension(fid, vox_offset)

   ext = [];

   if vox_offset
      end_of_ext = vox_offset;
   else
      fseek(fid, 0, 'eof');
      end_of_ext = ftell(fid);
   end

   if end_of_ext > 352
      fseek(fid, 348, 'bof');
      ext.extension = fread(fid,4)';
   end

   if isempty(ext) | ext.extension(1) == 0
      ext = [];
      return;
   end

   i = 1;

   while(ftell(fid) < end_of_ext)
      ext.section(i).esize = fread(fid,1,'int32');
      ext.section(i).ecode = fread(fid,1,'int32');
      ext.section(i).edata = char(fread(fid,ext.section(i).esize-8)');
      i = i + 1;
   end

   ext.num_ext = length(ext.section);

   return                                               % read_extension

