% This assumes dicom images have already been converted to nifti
indir = '/media/TCIA/Pancreas-CT/NII'

out = '/media/pancreas_hnn_exp'

%% RUN
r = rdir([indir,'/**/*_Pancreas.nii.gz']);
N = numel(r)

DSC = zeros(N,1);
for i = 1:N
    gtname = r(i).name;
    imagename_or_dicomdir = strrep(gtname,'_Pancreas','');
    [~, patient] = fileparts(clearExtension(imagename));
    outroot = [out,filesep,patient]
    
    run_pancreas_hnn
    
    bb_vol_diffs(i,1) = bb_vol_diffs1; 
    bb_recall(i,1) = bb_recall1; 
    bb_Dice_coef(i,1) = bb_Dice_coef1;
    
    bb_vol_diffs(i,2) = bb_vol_diffs2; 
    bb_recall(i,2) = bb_recall2; 
    bb_Dice_coef(i,2) = bb_Dice_coef2;
    
    close all
end

