%% Input/output (comment this if using run_batch.m)
%% TCIA Pancreas-CT
imagename_or_dicomdir = '/media/TCIA/DICOM/DOI/PANCREAS_0001'
gtname = '/media/TCIA/TCIA_pancreas_labels-02-05-2017/label0082.nii.gz' % can be empty string!
outroot = '/media/TCIA/Results/PANCREAS_0082' % use absolute path for results!


%% Common params
PLANE = {'ax','co','sa'};

minWin = -160;
maxWin = +240;

apps_build_dir = 'nihApps-release';
global itkExtractImageSlices_exe
global itkExtractBoundingBoxRegion_exe
global CAFFE_LD_LIBRARY_PATH


apps_build_dir = [pwd,filesep,'nihApps-release'];
itkExtractImageSlices_exe = [apps_build_dir,'/itkExtractImageSlices'];
itkExtractBoundingBoxRegion_exe = [apps_build_dir,'/itkExtractBoundingBoxRegion'];
DicomSeriesReadImageWrite2_exe = [apps_build_dir,'/itkDicomSeriesReadImageWrite2'];

CAFFE_LD_LIBRARY_PATH='/home/rothhr/torch/install/lib:/usr/local/cuda-8.0/lib64:/usr/local/cuda/extras/CUPTI/lib64';

caffe_stage1_hed_dir = [pwd,filesep,'/hed-globalweight/stage1_model'];
caffe_stage2_hed_dir = [pwd,filesep,'/hed-globalweight/stage2_model'];

%%%%%%%%%%%%%%%%%% RUN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath([pwd,filesep,'matlab']))

%% Create output info
pwd_dir = pwd;
[~, basename] = fileparts(imagename_or_dicomdir);
basename = clearExtension(basename);
fprintf('Processing %s ...\n',basename)

%% Check if dicom directory is given:
if isdir(imagename_or_dicomdir)
    if ~isdir(outroot)
        mkdir(outroot)
    end
    imagename = [outroot,filesep,basename,'.nii.gz'];
    dicom2volume(DicomSeriesReadImageWrite2_exe,imagename_or_dicomdir,imagename);
else
    imagename = imagename_or_dicomdir;
end
    
%% use patient's body as initial candidate region
body_dir = [outroot,'/body'];
bodyname = getBody(imagename,body_dir);

%% 1st Stage 
outroot1 = [outroot,filesep,'stage1'];
[bb_name, stage1_prob_volumename, bodycroppedimage, bodycroppedbody, bodycroppedgt, bb_vol_diffs1, bb_recall1, bb_Dice_coef1] = ...
                        predictStage(caffe_stage1_hed_dir,outroot1,imagename,basename,bodyname,PLANE,minWin,maxWin, gtname);

%% 2nd Stage 
outroot2 = [outroot,filesep,'stage2'];
[final_bb_name, stage2_prob_volumename, ~, ~, ~, bb_vol_diffs2, bb_recall2, bb_Dice_coef2] = ...
                        predictStage(caffe_stage2_hed_dir,outroot2,bodycroppedimage,basename,bb_name,PLANE,minWin,maxWin, bodycroppedgt);
                    
cd(pwd_dir)
