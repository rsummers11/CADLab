function result_file = getBody(filename,results_dir)

%% params
t_table = -500; %HU
rTable = 1;

%% RUN
if ~isdir(results_dir)
    mkdir(results_dir);
end

[~,case_name] = fileparts(filename);
case_name = strrep(case_name,'.nii','');

[V, vsiz, vdim, ~, hdr] = read_nifti_volume(filename);

%% RUN
B = findLargestConnComponent(V>t_table,rTable);

result_file = [results_dir,filesep,case_name,'_body.nii.gz'];

segmentationoverlay(V,B,vdim)
camup([0 0 -1])
view(180,30)

write_nifti_volume(B,vdim,result_file,hdr);
