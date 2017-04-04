function [bb_name, prediction_prob_volumename, croppedimage, cropped, croppedgt, bb_vol_diffs, bb_recall, bb_Dice_coef] = ...
              predictStage(caffe_python_dir,outroot,imagename,basename,bbname,PLANE,minWin,maxWin,gtname)
global itkExtractImageSlices_exe
global itkExtractBoundingBoxRegion_exe
global CAFFE_LD_LIBRARY_PATH
pwd_dir = pwd;

%% Crop around bounding box
bb_dir = fileparts(bbname);
croppedimage = [bb_dir,filesep,basename,'_cropped.nii.gz'];
cropped = [clearExtension(bbname),'_cropped.nii.gz'];
command = [itkExtractBoundingBoxRegion_exe, ' ', ...
           imagename, ' ', ...
           bbname, ' ', ...
           croppedimage,' ', ...
           cropped,' ', ...
           ];
%disp(command)
[status, result] = dos(command,'-echo');
if status ~= 0 
    error(result)
end

if ~isempty(gtname)
    croppedgt = [bb_dir,filesep,basename,'_Pancreas_cropped.nii.gz'];
    cropped = [clearExtension(bbname),'_cropped.nii.gz'];
    command = [itkExtractBoundingBoxRegion_exe, ' ', ...
               gtname, ' ', ...
               bbname, ' ', ...
               croppedgt,' ', ...
               cropped,' ', ...
               ];
    %disp(command)
    [status, result] = dos(command,'-echo');
    if status ~= 0 
        error(result)
    end
else
    croppedgt = '';
end

%% EXTRACT SLICES
predictionsDir = [outroot,filesep,'prediction'];
for i = 1:numel(PLANE)
    if strcmpi(PLANE{i},'ax')
        AXIS = 'z';
    elseif strcmpi(PLANE{i},'co')
        AXIS = 'x';
    elseif strcmpi(PLANE{i},'sa')
        AXIS = 'y';
    else
        error('No such plane/axis combination: %s !', PLANE{i})
    end    
    
    outputDir = [predictionsDir,filesep,'img_',PLANE{i},filesep];  
    
    command = [itkExtractImageSlices_exe, ' ', ...
               croppedimage, ' ', ...
               outputDir, ' ', ...
               num2str(minWin), ' ', ...
               num2str(maxWin), ' ', ...
               '-1111111 ', ...  % NOT USED IN ITK CODE!!!!!!!!!!!!!!
               '.png ', ...
               AXIS...
               ];

    %disp(command)
    [status, result] = dos(command,'-echo');
    if status ~= 0 
        error(result)
    end    
    
    if strcmpi(PLANE{i},'ax')
        warning('transposing images! (for PNG view in browser)')
        rimg = rdir([outputDir,'*.png'])
        for j = 1:numel(rimg)
           I = imread(rimg(j).name);
           imwrite(flipud(I),rimg(j).name);
        end
%     elseif strcmpi(PLANE{i},'co') CORONAL WORKS
%         warning('transposing images! (for PNG view in browser)')
%         rimg = rdir([outputDir,'*.png'])
%         for j = 1:numel(rimg)
%            I = imread(rimg(j).name);
%            imwrite(fliplr(rot90(I')),rimg(j).name);
%         end        
    elseif strcmpi(PLANE{i},'sa')
        warning('transposing images! (for PNG view in browser)')
        rimg = rdir([outputDir,'*.png'])
        for j = 1:numel(rimg)
           I = imread(rimg(j).name);
           imwrite(fliplr(I'),rimg(j).name);
        end
    end      
end

%% Deploy prediction HED
cd(caffe_python_dir)
MATLAB_LD_LIB_PATH = getenv('LD_LIBRARY_PATH');
setenv('LD_LIBRARY_PATH',CAFFE_LD_LIBRARY_PATH);
command = ['python ../deploy_hed_pancreas.py ',predictionsDir]
%disp(command)
[status, result] = dos(command,'-echo');
%if status ~= 0 % IGNORE ERROR. CAFFE DOES NOT CLEAN MEMORY CORRECTLY...
%    error(result)
%end    
cd(pwd_dir)
setenv('LD_LIBRARY_PATH',MATLAB_LD_LIB_PATH);

%% Construct prediction volume
for i = 1:numel(PLANE)
    disp(PLANE{i})
    inputSlicesDir = [predictionsDir,filesep,'predict_hed_pancreas_',upper(PLANE{i}),'_mask_iter_100000_globalweight'];
    [V, vsize, vdim, ~, vhdr] = read_nifti_volume( convertImageSlicesToVolume_hed(inputSlicesDir,croppedimage,PLANE{i}) );
    eval(['p',upper(PLANE{i}),' = V;']);
end
% mean of max. majority
P = sort( cat(4,pAX,pCO,pSA), 4);
P = squeeze(mean(P(:,:,:,2:end),4));

outsuffix = 'meanmaxAxCoSa';
prediction_prob_volumename = [predictionsDir,filesep,basename,'_',outsuffix,'.nii.gz'];
write_nifti_volume(P,vdim,prediction_prob_volumename,vhdr);

%% FIT BOUNDING BOX TO PREDICTION VOLUME
bbDir = [outroot,filesep,'largestObjBB']; 
[bb_name, bb_vol_diffs, bb_recall, bb_Dice_coef] = fit_evaluate_largest_obj_bounding_box(prediction_prob_volumename,croppedgt,bbDir);
