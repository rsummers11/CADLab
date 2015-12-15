%% Optional hardcoded input (comment out for GUI dialog for input/output)
%input_dicom_dir = 'data\CTimage';
%output_dir = 'C:\tmp\CTimage_fastTest5';
input_dicom_dir = '';
output_dir = '';

%% RUN
install_LymphNodeRFCNNPipeline;

%% COMMON PARAMS
Cpp_Exe_Folder=itk_dir;

%% RANDOM FOREST CANDIDATE GENERATION
optRF.itkConvertIndexToPhysicalPointsExe = [Cpp_Exe_Folder,filesep,'itkConvertIndexToPhysicalPoints.exe'];
optRF.SearchRegionAtlas_dir = [pwd,filesep,'data\SearchRegionAtlas'];
optRF.NiftyRegAppsDir = nifty_prebuild_apps_dir;
optRF.save_flag = true;
optRF.t_prob = 0.25; % only keep probabilities >= t_prob

optRF.randForestFilename = [pwd,filesep,'trained_Forests\RF_regression.mat'];
optRF.randForestModelName = 'model_reg_f5';

%SEARCH REGION PARAMETERS
optRF.t_low = 700; 
optRF.t_high = 1250;
optRF.skip_voxels = 1;
optRF.rErodeDilate = 0; %1 erode (negative value) or dilate (positive value) search region after atlas-based segmentation

%FEATURE PARAMETERS
optRF.n = 5; % filter size
optRF.N_feature_display = 2000;        

%% IMAGE PATCH GENERATION FOR CNN

optCNNimages.itkResampleRegionOfInterestExe = [Cpp_Exe_Folder,filesep,'itkResampleRegionOfInterestFromCoordinateList.exe'];
optCNNimages.pointList_suffix = '_physicalPoints.txt';
optCNNimages.outputLabel = 'CADe';
optCNNimages.t_lower='-100';%[HU]
optCNNimages.t_upper='200';%[HU] 
optCNNimages.displacementFactorRandomTranslations='1.5';%'3.0';%[mm]
optCNNimages.ratio = 1;
optCNNimages.numberRandomTranslations=2;
optCNNimages.numberRandomRotations=2;
optCNNimages.scales = 45; %mm
optCNNimages.numberROIvoxels = 32; 
optCNNimages.interpolationType='BSPLINE';
optCNNimages.transformType='XYZ';
%FASTER OPTIONS->
    %optCNNimages.numberRandomTranslations=5;
    %optCNNimages.numberRandomRotations=5;
    %optCNNimages.scales = 45; %mm
    %optCNNimages.scales = [30, 45]; %mm
    %optCNNimages.interpolationType='LINEAR';
%->FASTER OPTIONS

%% CNN PREDICTIONS    
optCNNpredictions.ConvNetSrcFolder = convnent_dir;
optCNNpredictions.CNNmodel = [pwd,filesep,'trained_pyconvnet\train_batch23_AxCoSa_balanced\fc512-11pct\1200.5']; % JUST ABDOMEN
optCNNpredictions.NumberBatches = 10*numel(optCNNimages.scales); % makes sure all images are included
optCNNpredictions.UseMultiview = true;
optCNNpredictions.ImageSize = optCNNimages.numberROIvoxels;
optCNNpredictions.ImageSearchString = '_AxCoSa.png';
optCNNpredictions.ImageChannels = 3;
optCNNpredictions.BatchStartIndex = 1;    
optCNNpredictions.operating_probability = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.80, 0.85, 0.90, 0.95];
optCNNpredictions.showMPRpredictions = false;
optCNNpredictions.unique_roi_nifti_str = '_t00000_r00000.nii.gz';
optCNNpredictions.itkGetRegionOfInterestCentersFromList_exe = [Cpp_Exe_Folder,filesep,'itkGetRegionOfInterestCentersFromList.exe'];
  
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                    RUN CADe                                      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

% check some if trained RF and ConvNet exist
if ~exist(optRF.randForestFilename,'file')
    error(' optRF.randForestFilename = %s does not exist!', optRF.randForestFilename);
end
if ~exist(optCNNpredictions.CNNmodel,'file')
    error(' optCNNpredictions.CNNmodel = %s does not exist!', optCNNpredictions.CNNmodel);
end

%% read DICOM and get ouput dir
if ~exist('input_dicom_dir','var') || isempty(input_dicom_dir) || ~ischar(input_dicom_dir)
    disp('Please select DICOM dir (with one volume only!)')
    input_dicom_dir = uigetdir([pwd,filesep,'data',filesep,'CTimage'],'Please select DICOM dir (with one volume only!)');
end    
disp(['   ',input_dicom_dir,' selected.'])

if ~exist('output_dir','var') || isempty(output_dir) || ~ischar(output_dir)
    disp('Please select OUTPUT dir')
    output_dir = uigetdir([pwd,filesep,'data'],'Please select OUTPUT dir');
    disp(['   ',output_dir,' selected.'])
end
[~, outname] = fileparts(output_dir); % folder with dicom slices
OutFileName = [output_dir, filesep, outname, '.nii'];
dicom2volume([Cpp_Exe_Folder,filesep,'itkDicomSeriesReadImageWrite2.exe'],input_dicom_dir,OutFileName) %% recompile and change exe!!!!

%% run CADe
Result_dir = LymphNodeRFCNNPipeline(OutFileName,optRF,optCNNimages,optCNNpredictions);

%% move all relevant outputs to result dir
r = rdir([output_dir,filesep,'**\*CNN*.mps']);
for i = 1:numel(r)
    [~, filename, fileext] = fileparts(r(i).name);
    move_to_filename = [Result_dir,filesep,filename, fileext];
    if ~exist(move_to_filename,'file') % cannot copy onto itself
        fprintf(' moving %s to %s ...\n', [filename, fileext], Result_dir);
        movefile(r(i).name,move_to_filename)
    end
end

r = rdir([output_dir,filesep,'**\*_SearchRegion.nii.gz']); % RF search region
for i = 1:numel(r)
    [~, filename, fileext] = fileparts(r(i).name);
    move_to_filename = [Result_dir,filesep,filename, fileext];
    if ~exist(move_to_filename,'file') % cannot copy onto itself
        fprintf(' moving %s to %s ...\n', [filename, fileext], Result_dir);
        movefile(r(i).name,move_to_filename)
    end
end

r = rdir([output_dir,filesep,'**\*_probabilities_RF.nii.gz']); % RF probs
for i = 1:numel(r)
    [~, filename, fileext] = fileparts(r(i).name);
    move_to_filename = [Result_dir,filesep,filename, fileext];
    if ~exist(move_to_filename,'file') % cannot copy onto itself
        fprintf(' moving %s to %s ...\n', [filename, fileext], Result_dir);
        movefile(r(i).name,move_to_filename)
    end
end

r = rdir([output_dir,filesep,'*.nii']); % CT image
for i = 1:numel(r)
    [~, filename, fileext] = fileparts(r(i).name);
    move_to_filename = [Result_dir,filesep,filename, fileext];
    if ~exist(move_to_filename,'file') % cannot copy onto itself
        fprintf(' moving %s to %s ...\n', [filename, fileext], Result_dir);
        movefile(r(i).name,move_to_filename)
    end
end

disp('*********************************************************************')
fprintf(' moved all final results to %s\n Have a nice day!\n', Result_dir);
disp('*********************************************************************')



    