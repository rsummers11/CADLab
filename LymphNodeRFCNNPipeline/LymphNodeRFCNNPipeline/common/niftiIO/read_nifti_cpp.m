function [cpg refpoints nii] = read_nifti_cpp(filename)

if nargin < 1
  [filename,pathname] = uigetfile('*.nii','Load nifti CPP file') ;

  if filename ==0
    error([ ' No file selected.' ])
  end 

  filename = [pathname filename] ;
end

filename = strrep(filename,' ','');
filename = strrep(filename,'"','');

nii = load_untouch_nii(filename);

cpg = nii.img;
cnd = ndims(cpg)-1;
refpoints = zeros(2,cnd);
refpoints(1,:) = -nii.hdr.dime.pixdim(2:cnd+1);
refpoints(2,:) = (nii.hdr.dime.dim(2:cnd+1)-2).*nii.hdr.dime.pixdim(2:cnd+1);

if exist('rdir')
    disp('read cp from ')
    rdir(filename)
else
    disp(['read cp from ',filename])
end
