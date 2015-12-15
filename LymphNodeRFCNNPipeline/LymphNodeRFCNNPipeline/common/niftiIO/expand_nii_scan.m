%  Expand a multiple-scan NIFTI file into multiple single-scan NIFTI files
%
%  Usage: expand_nii_scan(multi_scan_filename, [img_idx], [path_to_save])
%
%  NIFTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
function expand_nii_scan(filename, img_idx, newpath)

   if ~exist('newpath','var'), newpath = pwd; end
   if ~exist('img_idx','var'), img_idx = 1:get_nii_frame(filename); end

   for i=img_idx
      nii_i = load_untouch_nii(filename, i);

      fn = [nii_i.fileprefix '_' sprintf('%04d',i)];
      pnfn = fullfile(newpath, fn);

      save_untouch_nii(nii_i, pnfn);
   end

   return;					% expand_nii_scan

