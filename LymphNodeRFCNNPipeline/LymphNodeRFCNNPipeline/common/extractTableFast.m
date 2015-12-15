%% remove table from image
function result_file = extractTableFast(filename,results_dir)

%% params
t_body = -200; %HU
rTable = 3;

%% RUN
fprintf('Remove table with t_body > %g...\n', t_body)
if ~isdir(results_dir)
    mkdir(results_dir);
end

[~,case_name] = fileparts(filename);
case_name = strrep(case_name,'.nii','');

[V, vsiz, vdim, ~, hdr] = read_nifti_volume(filename);

%% RUN
B = findLargestConnComponent(V>t_body,rTable);
V(~B) = min(V(:)); % remove table

result_file = [results_dir,filesep,case_name,'_body.nii.gz'];
write_nifti_volume(V,vdim,result_file,hdr);
