%  Collapse multiple single-scan NIFTI files into a multiple-scan NIFTI file
%
%  Usage: collapse_nii_scan(scan_file_pattern, [collapsed_filename], [scan_file_folder])
%
%  Here, scan_file_pattern should look like: 'myscan_0*.img'
%  If collapsed_filename is omit, 'multi_scan.nii' will be used
%  If scan_file_folder is omit, current file folder will be used
%
%  The order of volumes in the collapsed file will be the order of 
%  corresponding filenames for those selected scan files.
%
%  NIFTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
function collapse_nii_scan(scan_pattern, fileprefix, scan_path)

   if ~exist('fileprefix','var'), fileprefix = 'multi_scan.nii'; end
   if ~exist('scan_path','var'), scan_path = pwd; end

   filetype = 1;

   %  Note: fileprefix is actually the filename you want to save
   %   
   if findstr('.nii',fileprefix)
      filetype = 2;
      fileprefix = strrep(fileprefix,'.nii','');
   end
   
   if findstr('.hdr',fileprefix)
      fileprefix = strrep(fileprefix,'.hdr','');
   end
   
   if findstr('.img',fileprefix)
      fileprefix = strrep(fileprefix,'.img','');
   end

   pnfn = fullfile(scan_path, scan_pattern);

   file_lst = dir(pnfn);
   flist = {file_lst.name};
   flist = flist(:);

   nii = load_untouch_nii(flist{1});
   nii.hdr.dime.dim(5) = length(flist);

   if nii.hdr.dime.dim(1) < 4
      nii.hdr.dime.dim(1) = 4;
   end

   hdr = nii.hdr;

   if isfield(nii,'ext') & ~isempty(nii.ext)
      ext = nii.ext;
      [ext, esize_total] = verify_nii_ext(ext);
   else
      ext = [];
   end

   switch double(hdr.dime.datatype),
   case   1,
      hdr.dime.bitpix = int16(1 ); precision = 'ubit1';
   case   2,
      hdr.dime.bitpix = int16(8 ); precision = 'uint8';
   case   4,
      hdr.dime.bitpix = int16(16); precision = 'int16';
   case   8,
      hdr.dime.bitpix = int16(32); precision = 'int32';
   case  16,
      hdr.dime.bitpix = int16(32); precision = 'float32';
   case  32,
      hdr.dime.bitpix = int16(64); precision = 'float32';
   case  64,
      hdr.dime.bitpix = int16(64); precision = 'float64';
   case 128,
      hdr.dime.bitpix = int16(24); precision = 'uint8';
   case 256 
      hdr.dime.bitpix = int16(8 ); precision = 'int8';
   case 512 
      hdr.dime.bitpix = int16(16); precision = 'uint16';
   case 768 
      hdr.dime.bitpix = int16(32); precision = 'uint32';
   case 1024
      hdr.dime.bitpix = int16(64); precision = 'int64';
   case 1280
      hdr.dime.bitpix = int16(64); precision = 'uint64';
   case 1792,
      hdr.dime.bitpix = int16(128); precision = 'float64';
   otherwise
      error('This datatype is not supported');
   end

   if filetype == 2
      fid = fopen(sprintf('%s.nii',fileprefix),'w');
      
      if fid < 0,
         msg = sprintf('Cannot open file %s.nii.',fileprefix);
         error(msg);
      end
      
      hdr.dime.vox_offset = 352;

      if ~isempty(ext)
         hdr.dime.vox_offset = hdr.dime.vox_offset + esize_total;
      end

      hdr.hist.magic = 'n+1';
      save_untouch_nii_hdr(hdr, fid);

      if ~isempty(ext)
         save_nii_ext(ext, fid);
      end
   else
      fid = fopen(sprintf('%s.hdr',fileprefix),'w');
      
      if fid < 0,
         msg = sprintf('Cannot open file %s.hdr.',fileprefix);
         error(msg);
      end
      
      hdr.dime.vox_offset = 0;
      hdr.hist.magic = 'ni1';
      save_untouch_nii_hdr(hdr, fid);

      if ~isempty(ext)
         save_nii_ext(ext, fid);
      end
      
      fclose(fid);
      fid = fopen(sprintf('%s.img',fileprefix),'w');
   end

   if filetype == 2 & isempty(ext)
      skip_bytes = double(hdr.dime.vox_offset) - 348;
   else
      skip_bytes = 0;
   end

   if skip_bytes
      fwrite(fid, ones(1,skip_bytes), 'uint8');
   end

   glmax = -inf;
   glmin = inf;

   for i = 1:length(flist)
      nii = load_untouch_nii(flist{i});

      if double(hdr.dime.datatype) == 128

         %  RGB planes are expected to be in the 4th dimension of nii.img
         %
         if(size(nii.img,4)~=3)
            error(['The NII structure does not appear to have 3 RGB color planes in the 4th dimension']);
         end

         nii.img = permute(nii.img, [4 1 2 3 5 6 7 8]);
      end

      %  For complex float32 or complex float64, voxel values
      %  include [real, imag]
      %
      if hdr.dime.datatype == 32 | hdr.dime.datatype == 1792
         real_img = real(nii.img(:))';
         nii.img = imag(nii.img(:))';
         nii.img = [real_img; nii.img];
      end

      if nii.hdr.dime.glmax > glmax
         glmax = nii.hdr.dime.glmax;
      end

      if nii.hdr.dime.glmin < glmin
         glmin = nii.hdr.dime.glmin;
      end

      fwrite(fid, nii.img, precision);
   end

   hdr.dime.glmax = round(glmax);
   hdr.dime.glmin = round(glmin);

   if filetype == 2
      fseek(fid, 140, 'bof');
      fwrite(fid, hdr.dime.glmax, 'int32');
      fwrite(fid, hdr.dime.glmin, 'int32');
   else
      fid2 = fopen(sprintf('%s.hdr',fileprefix),'w');

      if fid2 < 0,
         msg = sprintf('Cannot open file %s.hdr.',fileprefix);
         error(msg);
      end

      save_untouch_nii_hdr(hdr, fid2);

      if ~isempty(ext)
         save_nii_ext(ext, fid2);
      end

      fclose(fid2);
   end

   fclose(fid);

   return;					% collapse_nii_scan

